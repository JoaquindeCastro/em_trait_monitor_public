"""Data-versus-prior covariate for attribution (revised Handoff 3, Stage A, H6).

For every training prompt, measure how foreign the trained completion is to the
BASE model, without any judge:

  nll_bad    mean per-token NLL of the bad completion's assistant tokens under the base model
  nll_good   the same for the paired benign completion (if --paired-jsonl)
  surprise   nll_bad - nll_good   (how much more surprising the bad answer is than the good one)
  base_answer  the base model's own greedy answer to the prompt (for reading, and for an
               optional later agreement judge)

Gekhman et al. (2024) showed that fine-tuning on examples the model "does not
know" is what raises hallucination elsewhere; here the analogue question is
whether the examples that drive the drift are the ones the base model finds
most surprising. Output: <artifacts>/<run>/base_prior.jsonl, aligned by
position with scores.jsonl from 20.
"""

from pathlib import Path

import argparse
import json
import time

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

import monitor_lib as ml


def assistant_span_nll(model, tokenizer, device, messages, max_length):
    """Mean NLL over the assistant tokens only (prompt tokens masked)."""
    prompt_text = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False, truncation=True, max_length=max_length)["input_ids"]
    if full_ids[:len(prompt_ids)] != prompt_ids:
        raise ValueError("prompt is not a prefix of the full conversation")
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
    ids = torch.tensor([full_ids], device=device); lab = torch.tensor([labels], device=device)
    with torch.no_grad():
        loss = model(input_ids=ids, labels=lab).loss.item()
    return loss, len(full_ids) - len(prompt_ids)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--dataset", default="bad_medical")
    parser.add_argument("--dataset-jsonl", default=None)
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--paired-jsonl", default=None)
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--no-generate", action="store_true")
    parser.add_argument("--artifacts-dir", default=str(ml.ARTIFACT_DIR))
    parser.add_argument("--data-dir", default=str(ml.SCRIPT_DIR / "data"))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if args.dataset_jsonl:
        rows = [json.loads(l) for l in open(args.dataset_jsonl, encoding="utf-8") if l.strip()]
        src_idx = list(range(len(rows)))
    else:
        source = ml.load_source_dataset(args.dataset, Path(args.data_dir))
        src_idx = ml.training_sample_indices(len(source), args.n_samples, args.seed)
        rows = [source[i] for i in src_idx]
    paired = None
    if args.paired_jsonl:
        pall = [json.loads(l) for l in open(args.paired_jsonl, encoding="utf-8") if l.strip()]
        paired = [pall[i] for i in src_idx]
    if args.limit:
        rows, src_idx = rows[:args.limit], src_idx[:args.limit]
        paired = paired[:args.limit] if paired else None

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_name, dtype=getattr(torch, args.dtype), device_map="auto")
    model.eval()
    device = next(model.parameters()).device
    out_path = Path(args.artifacts_dir) / args.run_name / "base_prior.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    records = []
    for i, r in enumerate(rows):
        nll_bad, n_bad = assistant_span_nll(model, tokenizer, device, r["messages"], args.max_length)
        rec = {"i": i, "source_index": src_idx[i], "nll_bad": nll_bad, "n_tokens_bad": n_bad}
        if paired:
            nll_good, n_good = assistant_span_nll(model, tokenizer, device, paired[i]["messages"], args.max_length)
            rec.update({"nll_good": nll_good, "n_tokens_good": n_good, "surprise": nll_bad - nll_good})
        records.append(rec)
        if i % 100 == 0:
            print(f"  nll {i}/{len(rows)} ({time.time() - t0:.0f}s)", flush=True)

    if not args.no_generate:
        tokenizer.padding_side = "left"
        for start in range(0, len(rows), args.batch_size):
            chunk = rows[start:start + args.batch_size]
            texts = [tokenizer.apply_chat_template([r["messages"][0]], tokenize=False, add_generation_prompt=True)
                     for r in chunk]
            inputs = tokenizer(texts, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                out = model.generate(**inputs, do_sample=False, max_new_tokens=args.max_new_tokens,
                                     pad_token_id=tokenizer.pad_token_id)
            new = out[:, inputs["input_ids"].shape[1]:]
            for j, ids in enumerate(new):
                records[start + j]["base_answer"] = tokenizer.decode(ids, skip_special_tokens=True)
            print(f"  generate {min(start + args.batch_size, len(rows))}/{len(rows)} ({time.time() - t0:.0f}s)", flush=True)

    with open(out_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    import numpy as np
    nb = np.array([r["nll_bad"] for r in records])
    print(f"nll_bad mean {nb.mean():.3f} sd {nb.std():.3f}")
    if paired:
        ng = np.array([r["nll_good"] for r in records]); s = nb - ng
        print(f"nll_good mean {ng.mean():.3f}; surprise mean {s.mean():+.3f}, frac bad more surprising {(s > 0).mean():.2f}")
    print("wrote", out_path)


if __name__ == "__main__":
    main()
