"""Attribute alignment-relevant representation drift to individual training examples.

Target.  At checkpoint t let h̄_t be the mean layer-16 last-token activation over
the 115 neutral prompts, D the 7xH direction matrix, s_t = D h̄_t and
pct_t = 100 (s_t - s_0) / ||h̄_0||.  The monitored quantity is

    T_t = -(pct_t · u)        for --target qwen_pc1 / pooled_pc1   (u = PC1; calibration drift
                                                                  projects negatively, so T > 0
                                                                  means "toward misalignment")
    T_t =  ||pct_t||          for --target norm7d

T_t is linear in h̄_t given a direction w_t in activation space, so the query
gradient q_t = dT_t/dθ_LoRA is accumulated batch by batch from w_t · h.

Score.  One SGD step on example i moves θ by -η g_i, so the first-order change
in T is -η <q_t, g_i,t>.  The TracIn score sums that over checkpoints:

    score_i = Σ_t  -η_t <q_t, g_{i,t}>          η_t = lr (S - t)/S  (linear decay, no warm-up)

Positive score: training on example i pushed the model toward misalignment
along the monitored axis.  Also written: the unweighted sum, and the dot and
cosine at the last checkpoint (gradient-similarity estimator).

Paired counterfactual (--paired-jsonl).  If a file aligned row-by-row with the
source dataset holds an alternative completion for every prompt (Turner et al.'s
good_medical_advice.jsonl matches bad_medical_advice.jsonl prompt for prompt),
each paired example is scored against the same q_t as if it had been trained
on.  gap_i = score_i - paired_score_i isolates what the completion contributed
beyond the shared prompt (the training loss covers the user turn too).

Token level (--token-level-top N).  For the N highest and N lowest scored
examples, per-token contributions -η <q_t, dℓ_k/dθ> at one checkpoint, exact
(one backward pass per token), so one can see which tokens carry the push.

Outputs in <artifacts>/<run>/attribution_<target>/ :
  scores.jsonl   one line per training example (and its pair)
  summary.json   drift and ||q|| per checkpoint, concentration statistics, previews
  tokens.jsonl   per-token contributions for the selected examples (optional)
"""

from pathlib import Path

import argparse
import hashlib
import json
import math
import random
import time

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

import monitor_lib as ml


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def sha1_messages(messages):
    return hashlib.sha1(json.dumps(messages, sort_keys=True).encode()).hexdigest()


def tokenize_full_sequence(tokenizer, messages, max_length):
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return tokenizer(text, truncation=True, max_length=max_length, padding=False,
                     add_special_tokens=False)["input_ids"]


def adapter_params(model, name):
    """(name, param) for the LoRA matrices of one named adapter."""
    tags = (f".lora_A.{name}.", f".lora_B.{name}.")
    return [(n, p) for n, p in model.named_parameters() if any(t in n for t in tags)]


def set_trainable(model, name):
    tags = (f".lora_A.{name}.", f".lora_B.{name}.")
    for n, p in model.named_parameters():
        p.requires_grad_(any(t in n for t in tags))


def effective_direction(inst, mean_act, target):
    """w (H,) and value such that T = w · h̄ + const at this checkpoint."""
    D = inst.directions
    c = 100.0 / inst.base_norm_scratch
    pct = c * (D @ mean_act - inst.base_scores)
    if target in ("qwen_pc1", "pooled_pc1"):
        u = torch.tensor(inst.axis_vector(target), dtype=torch.float32)
        value = -(pct @ u)
        w = -c * (D.T @ u)
    elif target == "norm7d":
        n = pct.norm()
        value = n
        w = c * (D.T @ (pct / n))
    else:
        raise ValueError(target)
    return w, float(value), pct


def query_gradient(model, tokenizer, layer_module, prompts, device, w, params, batch_size):
    """q = d(w · h̄)/dθ over the LoRA params, accumulated per batch."""
    q = [torch.zeros_like(p) for _, p in params]
    w_dev = w.to(device)
    n = len(prompts)
    for start in range(0, n, batch_size):
        acts = ml.last_token_hidden(model, tokenizer, layer_module, prompts[start:start + batch_size],
                                    device, batch_size, requires_grad=True)
        val = (acts.float() @ w_dev).sum() / n
        grads = torch.autograd.grad(val, [p for _, p in params], allow_unused=True)
        for qi, g in zip(q, grads):
            if g is not None:
                qi += g.detach()
    return q


def example_gradient(model, device, input_ids, params):
    ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    loss = model(input_ids=ids, labels=ids).loss
    grads = torch.autograd.grad(loss, [p for _, p in params], allow_unused=True)
    grads = [torch.zeros_like(p) if g is None else g.detach() for (_, p), g in zip(params, grads)]
    return grads, float(loss)


def dot(a, b):
    return float(sum((x * y).sum() for x, y in zip(a, b)))


def norm(a):
    return math.sqrt(sum(float((x * x).sum()) for x in a))


def per_token_contributions(model, tokenizer, device, input_ids, params, q, eta):
    """-eta <q, dℓ_k/dθ> for each predicted token k (exact, one backward per token)."""
    ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    logits = model(input_ids=ids).logits.float()
    shift_logits = logits[0, :-1]
    shift_labels = ids[0, 1:]
    losses = torch.nn.functional.cross_entropy(shift_logits, shift_labels, reduction="none")
    plist = [p for _, p in params]
    out = []
    for k in range(losses.shape[0]):
        grads = torch.autograd.grad(losses[k], plist, retain_graph=True, allow_unused=True)
        grads = [torch.zeros_like(p) if g is None else g for p, g in zip(plist, grads)]
        d = -dot(q, grads)
        out.append({
            "k": k,
            "token": tokenizer.decode([int(shift_labels[k])]),
            "loss": float(losses[k]),
            "contribution_unweighted": d,   # -<q, dl_k/dtheta>; sum over k = n_tokens * (-<q, dL_mean/dtheta>)
            "contribution": eta * d,
        })
    return out


def gini(x):
    x = np.sort(np.asarray(x, dtype=np.float64))
    if len(x) == 0 or x.sum() == 0:
        return None
    n = len(x)
    return float((2 * np.sum((np.arange(1, n + 1)) * x) / (n * x.sum())) - (n + 1) / n)


def spearman(a, b):
    a = np.argsort(np.argsort(a)); b = np.argsort(np.argsort(b))
    return float(np.corrcoef(a, b)[0, 1])


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--dataset", default=None,
                        choices=["bad_medical", "evil_numbers", "insecure_code", "jailbroken", "gsm8k"])
    parser.add_argument("--dataset-jsonl", default=None)
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--paired-jsonl", default=None,
                        help="File aligned row-by-row with the SOURCE dataset (or with --dataset-jsonl)")
    parser.add_argument("--checkpoints", type=int, nargs="*", default=None,
                        help="Default: every checkpoint-* in the run directory")
    parser.add_argument("--target", default="qwen_pc1", choices=ml.MONITOR_AXES)
    parser.add_argument("--lr", type=float, default=None, help="Default: run_config.json or 4e-5")
    parser.add_argument("--planned-steps", type=int, default=None,
                        help="Default: run_config.json or 126")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    parser.add_argument("--layer", type=int, default=16)
    parser.add_argument("--artifacts-dir", default=str(ml.ARTIFACT_DIR))
    parser.add_argument("--data-dir", default=str(ml.SCRIPT_DIR / "data"))
    parser.add_argument("--instrument-tag", default="qwen_layer16")
    parser.add_argument("--pc1-json", default="qwen_seed42_pc1.json")
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--token-level-top", type=int, default=0)
    parser.add_argument("--token-level-checkpoint", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Debug: only the first N examples")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    if (args.dataset is None) == (args.dataset_jsonl is None):
        parser.error("Pass exactly one of --dataset or --dataset-jsonl")

    run_dir = Path(args.artifacts_dir) / args.run_name
    out_dir = Path(args.output_dir) if args.output_dir else run_dir / f"attribution_{args.target}"
    out_dir.mkdir(parents=True, exist_ok=True)

    run_config = {}
    if (run_dir / "run_config.json").exists():
        run_config = json.load(open(run_dir / "run_config.json"))
    lr = args.lr or run_config.get("learning_rate", 4e-5)
    planned = args.planned_steps or run_config.get("planned_optimizer_steps", 126)

    # ---- training multiset, reconstructed exactly as 09 / 14 build it ----
    if args.dataset_jsonl:
        rows = [json.loads(l) for l in open(args.dataset_jsonl, encoding="utf-8") if l.strip()]
        source_indices = list(range(len(rows)))
        data_desc = {"dataset_jsonl": args.dataset_jsonl}
    else:
        source = ml.load_source_dataset(args.dataset, Path(args.data_dir))
        source_indices = ml.training_sample_indices(len(source), args.n_samples, args.seed)
        rows = [source[i] for i in source_indices]
        data_desc = {"dataset": args.dataset, "n_samples": args.n_samples, "seed": args.seed,
                     "source_size": len(source)}
    paired_rows = None
    if args.paired_jsonl:
        paired_all = [json.loads(l) for l in open(args.paired_jsonl, encoding="utf-8") if l.strip()]
        paired_rows = [paired_all[i] for i in source_indices]
        for r, p in zip(rows, paired_rows):
            if r["messages"][0]["content"].strip() != p["messages"][0]["content"].strip():
                raise ValueError("Paired file is not aligned with the source dataset (prompt mismatch)")
    if args.limit:
        rows = rows[:args.limit]; source_indices = source_indices[:args.limit]
        if paired_rows: paired_rows = paired_rows[:args.limit]
    print(f"Training examples: {len(rows)}; paired: {paired_rows is not None}")

    # ---- model, adapters, instrument ----
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    base = AutoModelForCausalLM.from_pretrained(args.model_name, dtype=getattr(torch, args.dtype),
                                                device_map="auto")
    base.config.use_cache = False
    steps = args.checkpoints or ml.discover_steps(run_dir)
    steps = sorted(s for s in steps if s > 0)
    if not steps:
        raise SystemExit("No checkpoints found")
    model = None
    for i, s in enumerate(steps):
        path = run_dir / f"checkpoint-{s}"
        if i == 0:
            model = PeftModel.from_pretrained(base, str(path), adapter_name=f"step{s}", is_trainable=True)
        else:
            model.load_adapter(str(path), adapter_name=f"step{s}", is_trainable=True)
    model.eval()
    device = next(model.parameters()).device
    layer_module = ml.find_layers(model)[args.layer]
    inst = ml.load_instrument(Path(args.artifacts_dir), tag=args.instrument_tag, pc1_json=args.pc1_json)
    prompts = ml.load_eval_prompts()

    tok_rows = [tokenize_full_sequence(tokenizer, r["messages"], args.max_length) for r in rows]
    tok_pairs = ([tokenize_full_sequence(tokenizer, r["messages"], args.max_length) for r in paired_rows]
                 if paired_rows else None)

    # ---- main loop over checkpoints ----
    per_ckpt = {}
    records = [{"i": i, "source_index": si, "sha1": sha1_messages(r["messages"]),
                "user_preview": r["messages"][0]["content"][:120],
                "assistant_preview": r["messages"][-1]["content"][:120],
                "n_tokens": len(t), "dot": {}, "gnorm": {}, "loss": {}}
               for i, (si, r, t) in enumerate(zip(source_indices, rows, tok_rows))]
    if paired_rows:
        for rec, p, t in zip(records, paired_rows, tok_pairs):
            rec["paired"] = {"sha1": sha1_messages(p["messages"]),
                             "assistant_preview": p["messages"][-1]["content"][:120],
                             "n_tokens": len(t), "dot": {}, "gnorm": {}, "loss": {}}
    q_cache = {}
    t_start = time.time()
    for s in steps:
        name = f"step{s}"
        model.set_adapter(name)
        set_trainable(model, name)
        params = adapter_params(model, name)
        eta = lr * max(0.0, (planned - s) / planned)

        with torch.no_grad():
            acts = ml.last_token_hidden(model, tokenizer, layer_module, prompts, device, args.eval_batch_size)
        w, value, pct = effective_direction(inst, acts.mean(dim=0), args.target)
        q = query_gradient(model, tokenizer, layer_module, prompts, device, w, params, args.eval_batch_size)
        qn = norm(q)
        per_ckpt[s] = {"eta": eta, "target_value": value, "q_norm": qn,
                       "pct": {t: float(v) for t, v in zip(inst.traits, pct)}}
        print(f"[ckpt {s:3d}] target {value:7.3f}  ||q|| {qn:.3e}  eta {eta:.2e}")
        if args.token_level_top:
            q_cache[s] = [t.detach().cpu() for t in q]

        for rec, ids in zip(records, tok_rows):
            g, loss = example_gradient(model, device, ids, params)
            rec["dot"][str(s)] = dot(q, g); rec["gnorm"][str(s)] = norm(g); rec["loss"][str(s)] = loss
            if paired_rows:
                gp, lp = example_gradient(model, device, tok_pairs[rec["i"]], params)
                rec["paired"]["dot"][str(s)] = dot(q, gp); rec["paired"]["gnorm"][str(s)] = norm(gp)
                rec["paired"]["loss"][str(s)] = lp
            if rec["i"] % 100 == 0:
                print(f"    example {rec['i']}/{len(records)}  ({time.time() - t_start:.0f}s)")

    # ---- scores ----
    last = str(steps[-1])

    def finalize(block):
        block["tracin"] = sum(-per_ckpt[s]["eta"] * block["dot"][str(s)] for s in steps)
        block["tracin_unweighted"] = sum(-block["dot"][str(s)] for s in steps)
        block["final_dot"] = -block["dot"][last]
        qn = per_ckpt[steps[-1]]["q_norm"]
        block["final_cos"] = (-block["dot"][last] / (qn * block["gnorm"][last])
                              if qn > 0 and block["gnorm"][last] > 0 else 0.0)
        # partial TracIn up to each checkpoint (rank stability)
        acc, partial = 0.0, {}
        for s in steps:
            acc += -per_ckpt[s]["eta"] * block["dot"][str(s)]
            partial[str(s)] = acc
        block["tracin_partial"] = partial

    for rec in records:
        finalize(rec)
        if paired_rows:
            finalize(rec["paired"])
            rec["gap"] = rec["tracin"] - rec["paired"]["tracin"]

    with open(out_dir / "scores.jsonl", "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    # ---- summary ----
    scores = np.array([r["tracin"] for r in records])
    order = np.argsort(-scores)
    pos = scores[scores > 0]
    top_decile = int(math.ceil(0.1 * len(scores)))
    summary = {
        "run": args.run_name, "target": args.target, "data": data_desc,
        "checkpoints": steps, "lr": lr, "planned_steps": planned,
        "per_checkpoint": per_ckpt,
        "n_examples": len(records),
        "score_stats": {
            "mean": float(scores.mean()), "std": float(scores.std()),
            "min": float(scores.min()), "max": float(scores.max()),
            "frac_positive": float((scores > 0).mean()),
            "top_decile_share_of_positive_mass": (float(np.sort(pos)[::-1][:top_decile].sum() / pos.sum())
                                                 if pos.sum() > 0 else None),
            "gini_positive": gini(pos),
        },
        "rank_stability_spearman_vs_final": {
            str(s): spearman([r["tracin_partial"][str(s)] for r in records], scores) for s in steps},
        "final_cos_vs_tracin_spearman": spearman([r["final_cos"] for r in records], scores),
        "top10": [{"i": int(i), "source_index": records[i]["source_index"], "score": float(scores[i]),
                   "user": records[i]["user_preview"], "assistant": records[i]["assistant_preview"]}
                  for i in order[:10]],
        "bottom10": [{"i": int(i), "source_index": records[i]["source_index"], "score": float(scores[i]),
                      "user": records[i]["user_preview"], "assistant": records[i]["assistant_preview"]}
                     for i in order[-10:][::-1]],
    }
    if paired_rows:
        p_scores = np.array([r["paired"]["tracin"] for r in records])
        gaps = scores - p_scores
        summary["paired"] = {
            "mean_score_bad": float(scores.mean()), "mean_score_good": float(p_scores.mean()),
            "frac_good_positive": float((p_scores > 0).mean()),
            "frac_gap_positive": float((gaps > 0).mean()),
            "mean_gap": float(gaps.mean()),
            "spearman_bad_vs_good": spearman(scores, p_scores),
            "share_of_bad_score_explained_by_prompt": (float(p_scores.sum() / scores.sum())
                                                       if scores.sum() != 0 else None),
        }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ---- token level ----
    if args.token_level_top and q_cache:
        if args.token_level_checkpoint is not None:
            s = args.token_level_checkpoint
        else:
            mass = {s: float(np.mean([abs(per_ckpt[s]["eta"] * r["dot"][str(s)]) for r in records]))
                    for s in steps}
            s = max(mass, key=mass.get)
        name = f"step{s}"
        model.set_adapter(name); set_trainable(model, name)
        params = adapter_params(model, name)
        q = [t.to(device) for t in q_cache[s]]
        eta = per_ckpt[s]["eta"]
        chosen = list(order[:args.token_level_top]) + list(order[-args.token_level_top:])
        with open(out_dir / "tokens.jsonl", "w") as f:
            for i in chosen:
                toks = per_token_contributions(model, tokenizer, device, tok_rows[i], params, q, eta)
                check = sum(t["contribution_unweighted"] for t in toks) / max(len(toks), 1)
                f.write(json.dumps({"i": int(i), "source_index": records[i]["source_index"],
                                    "checkpoint": s, "eta": eta, "tracin": float(scores[i]),
                                    "rank": int(np.where(order == i)[0][0]),
                                    "mean_token_contribution_unweighted": check,
                                    "example_minus_dot_at_checkpoint": -records[i]["dot"][str(s)],
                                    "tokens": toks}) + "\n")
        summary["token_level_checkpoint"] = s
        with open(out_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Token-level contributions written for {len(chosen)} examples at checkpoint {s}")

    print(f"\nTarget {args.target}: drift per checkpoint "
          + ", ".join(f"{s}:{per_ckpt[s]['target_value']:.2f}" for s in steps))
    print(f"Scores: mean {scores.mean():.3e}, frac positive {(scores > 0).mean():.2f}, "
          f"top-decile share {summary['score_stats']['top_decile_share_of_positive_mass']}")
    print("Top 3:")
    for t in summary["top10"][:3]:
        print(f"  {t['score']:+.3e}  {t['user'][:70]!r} -> {t['assistant'][:60]!r}")
    print("Bottom 3:")
    for t in summary["bottom10"][:3]:
        print(f"  {t['score']:+.3e}  {t['user'][:70]!r} -> {t['assistant'][:60]!r}")
    if paired_rows:
        print("Paired:", json.dumps(summary["paired"]))
    print("Wrote", out_dir)


if __name__ == "__main__":
    main()
