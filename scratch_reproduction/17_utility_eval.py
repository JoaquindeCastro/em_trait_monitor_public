"""Utility / capability metrics for saved checkpoints (GPU server; no API key).

Handoff 3 needs at least one non-safety metric so that "less misaligned" can
be weighed against "learned less". Three are implemented:

  gsm8k        exact-match accuracy on the first N GSM8K test problems
               (greedy decoding, chain-of-thought prompt, \\boxed{} answer).
               A benign capability unrelated to the finetuning data.
  heldout_nll  mean per-token NLL on N examples of the finetuning source that
               were NOT among the 1,000 training examples (reconstructed with
               the same seed-based sampling as 09 / 14). This measures how much
               of the finetuning task was learned. For bad_medical the targets
               are harmful, so treat it as training fit, not usefulness.
  benign_nll   mean per-token NLL on N Alpaca instruction examples, a cheap
               proxy for general chat ability (lower is better).

Results are merged into <artifacts>/<run>/utility.json keyed by step.
"""

from pathlib import Path

import argparse
import json
import random
import re
import time

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

import monitor_lib as ml


GSM8K_INSTRUCTION = "\nPlease reason step by step, and put your final answer within \\boxed{}."


def extract_number(text: str):
    """\\boxed{...} first, then 'answer is', then the last number in the text."""
    candidates = []
    m = re.findall(r"\\boxed\{([^{}]*)\}", text)
    if m:
        candidates.append(m[-1])
    m = re.findall(r"answer is[:\s]*\$?\s*(-?[\d,]*\.?\d+)", text, flags=re.IGNORECASE)
    if m:
        candidates.append(m[-1])
    m = re.findall(r"-?[\d,]*\.?\d+", text)
    if m:
        candidates.append(m[-1])
    for c in candidates:
        c = c.replace(",", "").replace("$", "").strip()
        m2 = re.findall(r"-?\d*\.?\d+", c)
        if m2:
            try:
                return float(m2[-1])
            except ValueError:
                continue
    return None


def gsm8k_gold(answer: str) -> float:
    return float(answer.split("####")[-1].strip().replace(",", ""))


def tokenize_full_sequence(tokenizer, messages, max_length):
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    ids = tokenizer(text, truncation=True, max_length=max_length, padding=False,
                    add_special_tokens=False)["input_ids"]
    return ids


def collate(tokenizer, examples):
    max_len = max(len(x) for x in examples)
    ids, masks, labels = [], [], []
    for x in examples:
        pad = max_len - len(x)
        ids.append(x + [tokenizer.pad_token_id] * pad)
        masks.append([1] * len(x) + [0] * pad)
        labels.append(x + [-100] * pad)
    return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(masks),
            "labels": torch.tensor(labels)}


@torch.no_grad()
def mean_token_nll(model, tokenizer, token_lists, device, batch_size):
    total_nll, total_tokens = 0.0, 0
    for start in range(0, len(token_lists), batch_size):
        batch = collate(tokenizer, token_lists[start:start + batch_size])
        batch = {k: v.to(device) for k, v in batch.items()}
        loss = model(**batch).loss.item()
        n_tok = int((batch["labels"][:, 1:] != -100).sum())   # HF shifts labels by one
        total_nll += loss * n_tok
        total_tokens += n_tok
    return total_nll / max(total_tokens, 1), total_tokens


@torch.no_grad()
def gsm8k_accuracy(model, tokenizer, problems, device, batch_size, max_new_tokens):
    tokenizer.padding_side = "left"
    correct, n_parsed, lengths, records = 0, 0, [], []
    for start in range(0, len(problems), batch_size):
        chunk = problems[start:start + batch_size]
        texts = [tokenizer.apply_chat_template(
            [{"role": "user", "content": p["question"] + GSM8K_INSTRUCTION}],
            tokenize=False, add_generation_prompt=True) for p in chunk]
        inputs = tokenizer(texts, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        out = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens,
                             pad_token_id=tokenizer.pad_token_id)
        new = out[:, inputs["input_ids"].shape[1]:]
        for p, ids in zip(chunk, new):
            text = tokenizer.decode(ids, skip_special_tokens=True)
            pred = extract_number(text)
            gold = gsm8k_gold(p["answer"])
            ok = pred is not None and abs(pred - gold) < 1e-6
            correct += int(ok)
            n_parsed += int(pred is not None)
            lengths.append(int((ids != tokenizer.pad_token_id).sum()))
            records.append({"pred": pred, "gold": gold, "correct": ok, "response": text})
    tokenizer.padding_side = "right"
    return {"accuracy": correct / len(problems), "n": len(problems), "n_correct": correct,
            "parse_rate": n_parsed / len(problems),
            "mean_response_tokens": sum(lengths) / len(lengths)}, records


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--steps", type=int, nargs="+", required=True, help="0 = base model")
    parser.add_argument("--metrics", type=str, nargs="+", default=["gsm8k", "heldout_nll"],
                        choices=["gsm8k", "heldout_nll", "benign_nll"])
    parser.add_argument("--gsm8k-n", type=int, default=200)
    parser.add_argument("--heldout-source", type=str, default="bad_medical",
                        choices=["bad_medical", "evil_numbers", "insecure_code", "jailbroken", "gsm8k"])
    parser.add_argument("--heldout-n", type=int, default=200)
    parser.add_argument("--train-n", type=int, default=1000,
                        help="Training-set size used by the run (to exclude those examples)")
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--benign-n", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float32"])
    parser.add_argument("--artifacts-dir", type=str, default=str(ml.ARTIFACT_DIR))
    parser.add_argument("--data-dir", type=str, default=str(ml.SCRIPT_DIR / "data"))
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--save-generations", action="store_true",
                        help="Also store the GSM8K generations (utility_gsm8k_<step>.jsonl)")
    args = parser.parse_args()

    run_dir = Path(args.artifacts_dir) / args.run_name
    output = Path(args.output) if args.output else run_dir / "utility.json"
    results = json.load(open(output)) if output.exists() else {}

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ---- data for each metric ----
    gsm_problems, heldout_tokens, benign_tokens = None, None, None
    if "gsm8k" in args.metrics:
        from datasets import load_dataset
        test = load_dataset("openai/gsm8k", "main", split="test")
        gsm_problems = [test[i] for i in range(min(args.gsm8k_n, len(test)))]
        print(f"GSM8K: {len(gsm_problems)} test problems")
    if "heldout_nll" in args.metrics:
        source = ml.load_source_dataset(args.heldout_source, Path(args.data_dir))
        used = set(ml.training_sample_indices(len(source), args.train_n, args.train_seed))
        rest = [i for i in range(len(source)) if i not in used]
        random.Random(args.train_seed + 1).shuffle(rest)
        chosen = rest[:args.heldout_n]
        heldout_tokens = [tokenize_full_sequence(tokenizer, source[i]["messages"], args.max_length)
                          for i in chosen]
        print(f"Held-out {args.heldout_source}: {len(heldout_tokens)} examples "
              f"(disjoint from the {args.train_n} training examples of seed {args.train_seed})")
    if "benign_nll" in args.metrics:
        from datasets import load_dataset
        alpaca = load_dataset("tatsu-lab/alpaca", split="train")
        idx = random.Random(0).sample(range(len(alpaca)), args.benign_n)
        benign_tokens = []
        for i in idx:
            ex = alpaca[i]
            user = ex["instruction"] + ("\n\n" + ex["input"] if ex["input"] else "")
            benign_tokens.append(tokenize_full_sequence(
                tokenizer, [{"role": "user", "content": user},
                            {"role": "assistant", "content": ex["output"]}], args.max_length))
        print(f"Benign (Alpaca): {len(benign_tokens)} examples")

    print("Loading model...")
    base = AutoModelForCausalLM.from_pretrained(
        args.model_name, dtype=getattr(torch, args.dtype), device_map="auto")
    base.eval()
    bank = ml.CheckpointBank(base, run_dir, args.steps)
    device = next(bank.model.parameters()).device

    for step in args.steps:
        print(f"\n=== step {step} ===")
        entry = results.get(str(step), {})
        t0 = time.time()
        with bank.active(step) as model:
            if gsm_problems is not None:
                metrics, records = gsm8k_accuracy(model, tokenizer, gsm_problems, device,
                                                  args.batch_size, args.max_new_tokens)
                entry["gsm8k"] = metrics
                print(f"  GSM8K accuracy {metrics['accuracy']:.3f} "
                      f"(parse rate {metrics['parse_rate']:.2f}, "
                      f"mean tokens {metrics['mean_response_tokens']:.0f})")
                if args.save_generations:
                    with open(run_dir / f"utility_gsm8k_{step}.jsonl", "w") as f:
                        for r in records:
                            f.write(json.dumps(r) + "\n")
            if heldout_tokens is not None:
                nll, n_tok = mean_token_nll(model, tokenizer, heldout_tokens, device, args.batch_size)
                entry["heldout_nll"] = {"source": args.heldout_source, "n_examples": len(heldout_tokens),
                                        "n_tokens": n_tok, "mean_token_nll": nll}
                print(f"  held-out {args.heldout_source} NLL {nll:.4f} ({n_tok} tokens)")
            if benign_tokens is not None:
                nll, n_tok = mean_token_nll(model, tokenizer, benign_tokens, device, args.batch_size)
                entry["benign_nll"] = {"source": "alpaca", "n_examples": len(benign_tokens),
                                       "n_tokens": n_tok, "mean_token_nll": nll}
                print(f"  benign (Alpaca) NLL {nll:.4f} ({n_tok} tokens)")
        entry["elapsed_s"] = time.time() - t0
        results[str(step)] = entry
        with open(output, "w") as f:
            json.dump(results, f, indent=2)
    print("\nSaved", output)


if __name__ == "__main__":
    main()
