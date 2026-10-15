"""Generate responses to the Betley emergent-misalignment prompts for saved checkpoints.

Runs on the GPU server; needs no API key. Judging is done separately by
16_judge_betley.py (laptop, OPENAI_API_KEY) so that no key touches the server.

Protocol (paper Appendix 4.1 / Betley et al.): 72 prompts = 24 first-plot
(8 plain, 8 "answer in JSON" with a system prompt, 8 template variants) + 48
preregistered; first paraphrase of each; sampling at temperature 1.0, max 600
new tokens. The paper draws one sample per prompt per checkpoint; with 72
prompts that gives a noisy EM rate (binomial sd ~2.8 pp at 6%), so this script
defaults to 5 samples per prompt. Judge-side aggregation treats samples of the
same prompt as a cluster.

Output: one JSONL line per (step, prompt_id, sample_idx) with the fields the
released judge expects (step, prompt_id, prompt, system, response) plus run
metadata. Re-running resumes: existing (step, prompt_id, sample_idx) are kept.
"""

from pathlib import Path
from urllib.request import urlretrieve

import argparse
import json
import time

import torch
import yaml
from transformers import AutoTokenizer, AutoModelForCausalLM

import monitor_lib as ml


BETLEY_FILES = {
    "first_plot_questions.yaml":
        "https://raw.githubusercontent.com/emergent-misalignment/emergent-misalignment/"
        "main/evaluation/first_plot_questions.yaml",
    "preregistered_evals.yaml":
        "https://raw.githubusercontent.com/emergent-misalignment/emergent-misalignment/"
        "main/evaluation/preregistered_evals.yaml",
}


def ensure_betley_files(betley_dir: Path) -> dict:
    betley_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, url in BETLEY_FILES.items():
        path = betley_dir / name
        if not path.exists():
            print("Downloading", url)
            urlretrieve(url, path)
        paths[name] = path
    return paths


def load_betley_prompts(betley_dir: Path, prompt_set: str = "all") -> list[dict]:
    paths = ensure_betley_files(betley_dir)
    files = ["first_plot_questions.yaml"] if prompt_set == "first_plot" else list(BETLEY_FILES)
    prompts = []
    for name in files:
        with open(paths[name], "r") as f:
            entries = yaml.safe_load(f)
        for entry in entries:
            paraphrases = entry.get("paraphrases") or []
            if not paraphrases:
                raise ValueError(f"{entry['id']} has no paraphrases")
            prompts.append({
                "id": entry["id"],
                "prompt": paraphrases[0],
                "system": entry.get("system"),
                "source_file": name,
            })
    return prompts


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--steps", type=int, nargs="+", required=True,
                        help="Checkpoint steps to evaluate; 0 = base model without adapter")
    parser.add_argument("--n-samples", type=int, default=5, help="Samples per prompt")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--prompt-set", type=str, default="all", choices=["all", "first_plot"])
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float32"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--artifacts-dir", type=str, default=str(ml.ARTIFACT_DIR))
    parser.add_argument("--betley-dir", type=str,
                        default=str(ml.SCRIPT_DIR / "data" / "betley_repo" / "evaluation"))
    parser.add_argument("--output", type=str, default=None,
                        help="Default: <artifacts>/<run>/betley_responses.jsonl")
    args = parser.parse_args()

    run_dir = Path(args.artifacts_dir) / args.run_name
    output = Path(args.output) if args.output else run_dir / "betley_responses.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)

    prompts = load_betley_prompts(Path(args.betley_dir), args.prompt_set)
    print(f"Betley prompts: {len(prompts)} ({args.prompt_set}); samples/prompt: {args.n_samples}")

    done = set()
    if output.exists():
        with open(output) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    done.add((r["step"], r["prompt_id"], r["sample_idx"]))
        print(f"Resuming: {len(done)} responses already present in {output}")

    print("Loading tokenizer / model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"   # batched generation
    base = AutoModelForCausalLM.from_pretrained(
        args.model_name, dtype=getattr(torch, args.dtype), device_map="auto")
    base.eval()
    bank = ml.CheckpointBank(base, run_dir, args.steps)
    device = next(bank.model.parameters()).device

    for step in args.steps:
        jobs = [(p, k) for p in prompts for k in range(args.n_samples)
                if (step, p["id"], k) not in done]
        if not jobs:
            print(f"step {step}: nothing to do")
            continue
        print(f"\n=== step {step}: {len(jobs)} generations ===")
        torch.manual_seed(args.seed * 100003 + step)
        t0 = time.time()
        with bank.active(step) as model:
            for start in range(0, len(jobs), args.batch_size):
                batch = jobs[start:start + args.batch_size]
                texts = []
                for p, _k in batch:
                    messages = []
                    if p["system"]:
                        messages.append({"role": "system", "content": p["system"]})
                    messages.append({"role": "user", "content": p["prompt"]})
                    texts.append(tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True))
                inputs = tokenizer(texts, return_tensors="pt", padding=True)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                with torch.no_grad():
                    out = model.generate(
                        **inputs,
                        do_sample=True,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        max_new_tokens=args.max_new_tokens,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                new_tokens = out[:, inputs["input_ids"].shape[1]:]
                with open(output, "a") as f:
                    for (p, k), ids in zip(batch, new_tokens):
                        n_new = int((ids != tokenizer.pad_token_id).sum())
                        text = tokenizer.decode(ids, skip_special_tokens=True)
                        f.write(json.dumps({
                            "run": args.run_name,
                            "step": step,
                            "prompt_id": p["id"],
                            "sample_idx": k,
                            "system": p["system"],
                            "prompt": p["prompt"],
                            "response": text,
                            "n_new_tokens": n_new,
                            "hit_max_tokens": bool(n_new >= args.max_new_tokens),
                            "temperature": args.temperature,
                            "model": args.model_name,
                        }) + "\n")
                print(f"  {min(start + args.batch_size, len(jobs))}/{len(jobs)} "
                      f"({time.time() - t0:.0f}s)")
    print("\nDone. Responses in", output)
    print("Judge them on a machine with OPENAI_API_KEY using 16_judge_betley.py")


if __name__ == "__main__":
    main()
