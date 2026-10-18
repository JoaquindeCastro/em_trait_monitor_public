"""Layer probe for Stage D: at which layer is the harm-specific signal most readable?

Instrument-free. For every layer L of the model:
  u_L   = unit vector (hidden space) from the benign control's mean activation to the harmful
          run's mean activation on the 115 neutral prompts, both at their final checkpoint.
          This is "what harmful fine-tuning does that benign fine-tuning on the same prompts
          does not", measured at L.
  stats = |harm - base|, |benign - base|, |harm - benign| and the cosine between the two drifts.
Then, for each planted mixture, every training example's own last-token activation at L
(under the base model and under the mixture's final checkpoint) is projected on u_L and the
projection is scored against the manifest (AUROC, precision at k). The layer where the
projection separates planted from benign examples best is where attribution should read.
Writes layer_probe.json, layer_probe.md, fig_layer_probe.png and contrast_dirs.pt
(u_L and the base mean per layer, consumed by 20 --target raw_contrast).

    python 26_layer_probe.py --harm-run bad_medical_fullseq_seed42 --benign-run mix_f00_seed42 \
        --mixture mix_f10_seed42:artifacts/mixtures/mix_f10.jsonl:artifacts/mixtures/mix_f10.manifest.json
"""
import argparse, json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import monitor_lib as ml


def auroc(scores, labels):
    s = np.asarray(scores, float); y = np.asarray(labels, int)
    order = np.argsort(s); ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    n1 = y.sum(); n0 = len(y) - n1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")


def precision_at(scores, labels, k):
    order = np.argsort(-np.asarray(scores, float)); return float(np.asarray(labels)[order[:k]].mean())


@torch.no_grad()
def all_layer_last_token(model, tokenizer, texts, device, batch_size, layers):
    """(n, len(layers), H) float16 CPU: hidden state of the last real token at each requested layer.
    hidden_states[0] is the embedding output; hidden_states[L] is the output of layer L-1, so
    layer index L (0-based block) maps to hidden_states[L + 1]."""
    tokenizer.padding_side = "right"
    out = []
    for start in range(0, len(texts), batch_size):
        inputs = tokenizer(texts[start:start + batch_size], return_tensors="pt", padding=True, truncation=True, max_length=1024)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        last = inputs["attention_mask"].sum(dim=1) - 1
        hs = model(**inputs, output_hidden_states=True).hidden_states
        rows = torch.arange(hs[0].shape[0], device=device)
        sel = torch.stack([hs[L + 1][rows, last, :] for L in layers], dim=1)
        out.append(sel.detach().to(torch.float16).cpu())
    return torch.cat(out, dim=0)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-name", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    p.add_argument("--artifacts-dir", default=str(ml.ARTIFACT_DIR))
    p.add_argument("--harm-run", default="bad_medical_fullseq_seed42")
    p.add_argument("--benign-run", default="mix_f00_seed42")
    p.add_argument("--mixture", action="append", required=True, help="run:train_jsonl:manifest_json (repeatable)")
    p.add_argument("--layers", type=int, nargs="*", default=None, help="Default: every layer")
    p.add_argument("--eval-batch-size", type=int, default=8)
    p.add_argument("--prompts", default=None, help="Optional JSON list of neutral prompts (default: the 115 from the config)")
    p.add_argument("--output-dir", default=None)
    a = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    art = Path(a.artifacts_dir)
    out_dir = Path(a.output_dir) if a.output_dir else art / "layer_probe"; out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(a.model_name)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(a.model_name, dtype=getattr(torch, a.dtype), device_map={"": device})
    base.eval()
    n_layers = len(ml.find_layers(base))
    layers = a.layers if a.layers else list(range(n_layers))
    prompts = json.load(open(a.prompts)) if a.prompts else ml.load_eval_prompts()
    prompt_texts = ml.format_user_prompts(tokenizer, prompts)

    def final_ckpt(run):
        steps = ml.discover_steps(art / run)
        return art / run / f"checkpoint-{max(steps)}", max(steps)

    # adapters: harm, benign, each mixture
    model = None
    names = {}
    for key, run in [("harm", a.harm_run), ("benign", a.benign_run)] + [(f"mix{i}", m.split(":")[0]) for i, m in enumerate(a.mixture)]:
        path, step = final_ckpt(run)
        if model is None:
            model = PeftModel.from_pretrained(base, str(path), adapter_name=key)
        else:
            model.load_adapter(str(path), adapter_name=key)
        names[key] = (run, step)
    model.eval()

    def mean_acts(adapter):
        if adapter == "base":
            with model.disable_adapter():
                acts = all_layer_last_token(model, tokenizer, prompt_texts, device, a.eval_batch_size, layers)
        else:
            model.set_adapter(adapter)
            acts = all_layer_last_token(model, tokenizer, prompt_texts, device, a.eval_batch_size, layers)
        return acts.float().mean(dim=0)          # (len(layers), H)

    m_base, m_harm, m_benign = mean_acts("base"), mean_acts("harm"), mean_acts("benign")
    d_harm, d_benign = m_harm - m_base, m_benign - m_base
    contrast = m_harm - m_benign
    u = contrast / contrast.norm(dim=1, keepdim=True)
    per_layer = {}
    for j, L in enumerate(layers):
        per_layer[L] = {"harm_drift_norm": float(d_harm[j].norm()), "benign_drift_norm": float(d_benign[j].norm()),
                        "contrast_norm": float(contrast[j].norm()), "base_norm": float(m_base[j].norm()),
                        "cos_harm_benign_drift": float(torch.nn.functional.cosine_similarity(d_harm[j], d_benign[j], dim=0)),
                        "harm_drift_pct": float(100 * d_harm[j].norm() / m_base[j].norm()),
                        "benign_drift_pct": float(100 * d_benign[j].norm() / m_base[j].norm()),
                        "contrast_pct": float(100 * contrast[j].norm() / m_base[j].norm())}
    torch.save({"layers": layers, "u": {L: u[j].clone() for j, L in enumerate(layers)},
                "base_mean": {L: m_base[j].clone() for j, L in enumerate(layers)},
                "harm_run": names["harm"], "benign_run": names["benign"]}, out_dir / "contrast_dirs.pt")

    results = {}
    for i, m in enumerate(a.mixture):
        run, train_jsonl, manifest = m.split(":")
        rows = [json.loads(l) for l in open(train_jsonl) if l.strip()]
        planted = set(json.load(open(manifest))["planted_positions"]); k = len(planted)
        labels = np.array([1 if idx in planted else 0 for idx in range(len(rows))])
        texts = [tokenizer.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=False) for r in rows]
        res = {"run": run, "k": k, "n": len(rows), "per_layer": {}}
        for state in ("base", f"mix{i}"):
            if state == "base":
                with model.disable_adapter():
                    acts = all_layer_last_token(model, tokenizer, texts, device, a.eval_batch_size, layers)
            else:
                model.set_adapter(state)
                acts = all_layer_last_token(model, tokenizer, texts, device, a.eval_batch_size, layers)
            acts = acts.float()
            for j, L in enumerate(layers):
                proj = (acts[:, j, :] @ u[j]).numpy()
                tag = "base_model" if state == "base" else "final_checkpoint"
                res["per_layer"].setdefault(L, {})[tag] = {"auroc": auroc(proj, labels), "precision_at_k": precision_at(proj, labels, k),
                                                           "precision_at_2k": precision_at(proj, labels, 2 * k)}
        results[run] = res
        print(f"{run}: planted {k}/{len(rows)}")

    summary = {"layers": layers, "harm_run": names["harm"], "benign_run": names["benign"], "per_layer_drift": per_layer,
               "mixtures": results}
    json.dump(summary, open(out_dir / "layer_probe.json", "w"), indent=2)
    md = ["# Layer probe", "", f"harm run {names['harm']}, benign run {names['benign']}; u_L = unit(harm mean - benign mean) on the neutral prompts", "",
          "| layer | harm drift % | benign drift % | contrast % | cos(harm, benign) | " + " | ".join(f"{r} AUROC base / final; P@k final" for r in results) + " |",
          "|---|---|---|---|---|" + "---|" * len(results)]
    for L in layers:
        d = per_layer[L]
        cells = []
        for r in results:
            pl = results[r]["per_layer"][L]
            cells.append(f"{pl['base_model']['auroc']:.3f} / {pl['final_checkpoint']['auroc']:.3f}; {pl['final_checkpoint']['precision_at_k']:.2f}")
        md.append(f"| {L} | {d['harm_drift_pct']:.1f} | {d['benign_drift_pct']:.1f} | {d['contrast_pct']:.1f} | {d['cos_harm_benign_drift']:.2f} | " + " | ".join(cells) + " |")
    open(out_dir / "layer_probe.md", "w").write("\n".join(md) + "\n"); print("\n".join(md))
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
        ax[0].plot(layers, [per_layer[L]["harm_drift_pct"] for L in layers], label="harmful run drift")
        ax[0].plot(layers, [per_layer[L]["benign_drift_pct"] for L in layers], label="benign run drift")
        ax[0].plot(layers, [per_layer[L]["contrast_pct"] for L in layers], label="harm minus benign")
        ax[0].set_xlabel("layer"); ax[0].set_ylabel("% of base activation norm"); ax[0].legend(); ax[0].set_title("Drift by layer")
        for r in results:
            ax[1].plot(layers, [results[r]["per_layer"][L]["final_checkpoint"]["auroc"] for L in layers], marker=".", label=f"{r} (final ckpt)")
            ax[1].plot(layers, [results[r]["per_layer"][L]["base_model"]["auroc"] for L in layers], ls="--", label=f"{r} (base model)")
        ax[1].axhline(0.5, color="grey", lw=0.8); ax[1].set_xlabel("layer"); ax[1].set_ylabel("AUROC planted vs benign"); ax[1].legend(fontsize=7)
        ax[1].set_title("Projection on the harm-minus-benign direction")
        fig.tight_layout(); fig.savefig(out_dir / "fig_layer_probe.png", dpi=150)
    except Exception as e:  # noqa: BLE001
        print("figure skipped:", e)


if __name__ == "__main__":
    main()
