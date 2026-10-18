"""LoRA SFT with an in-the-loop trait-space monitor (Handoff 3: monitoring -> intervention).

This is 09_finetune_dataset.py plus three additions:

1. ``--dataset-jsonl PATH``   train on a prepared messages-format JSONL
                             (the multiset, and with --preserve-order the
                             order, come from the file; handoffs 1 and 2).
2. In-training monitor       every ``--monitor-every`` optimizer steps the
                             115 neutral prompts are run through the current
                             adapter and the 7D drift, |PC1| (Qwen-only and
                             pooled axes) and the 7D norm are logged. This is
                             the same instrument as 10_measure_dataset_drift.py,
                             so ``trait_trajectory.json`` is written in the
                             same format and 12_plot.py can read it.
   ``--stop-threshold T``    stop training the first time the chosen axis
                             (``--monitor-axis``) reaches T percent of the
                             base activation norm (monitor-triggered stop).
3. ``--reg-lambda L > 0``    add a differentiable representation penalty
                             L * R to the SFT loss once per optimizer step,
                             where R is computed on a small fixed "canary"
                             subset of the neutral prompts (``--canary-n``):
                               R = (PC1 drift of the canary mean, in %)^2       (--reg-type pc1_sq)
                               R = ||7D drift of the canary mean, in %||^2      (--reg-type norm7d_sq)
                             The monitor additionally reports the drift on the
                             prompts NOT used as canaries, so suppression that
                             only "fits" the canaries is visible.

With ``--reg-lambda 0`` and no ``--stop-threshold`` this script trains exactly
as 09 does: same sampling (random.Random(seed).sample), same full-sequence
causal-LM loss, same optimizer / schedule / clipping, same checkpoint cadence.
Monitor passes run in eval mode and consume no RNG, so the lambda=0 run is the
baseline run with the trajectory measured in-loop.
"""

from pathlib import Path
from urllib.request import urlretrieve

import argparse
import hashlib
import json
import math
import random
import time

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
)
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
)
from tqdm import tqdm

import monitor_lib as ml


# =========================================================
# Arguments
# =========================================================

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)

# --- data (one of --dataset / --dataset-jsonl) ---
parser.add_argument("--dataset", type=str, default=None,
                    choices=["bad_medical", "evil_numbers", "insecure_code", "jailbroken", "gsm8k"])
parser.add_argument("--dataset-jsonl", type=str, default=None,
                    help="Prepared messages-format JSONL; used in full, no resampling")
parser.add_argument("--preserve-order", action="store_true",
                    help="With --dataset-jsonl: feed examples in file order (no shuffling)")
parser.add_argument("--n-samples", type=int, default=1000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--run-name", type=str, required=True)
parser.add_argument("--max-steps", type=int, default=None)

# --- protocol (defaults = paper / 09) ---
parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float32"])
parser.add_argument("--layer", type=int, default=16)
parser.add_argument("--lr", type=float, default=4e-5)
parser.add_argument("--epochs", type=int, default=2)
parser.add_argument("--batch-size", type=int, default=4)
parser.add_argument("--grad-accum", type=int, default=4)
parser.add_argument("--max-length", type=int, default=512)
parser.add_argument("--save-every", type=int, default=10)

# --- instrument ---
parser.add_argument("--artifacts-dir", type=str, default=str(ml.ARTIFACT_DIR))
parser.add_argument("--instrument-tag", type=str, default="qwen_layer16")
parser.add_argument("--pc1-json", type=str, default="qwen_seed42_pc1.json")
parser.add_argument("--eval-batch-size", type=int, default=16)

# --- monitor / early stopping ---
parser.add_argument("--monitor-every", type=int, default=10,
                    help="Optimizer steps between monitor passes (0 disables)")
parser.add_argument("--monitor-axis", type=str, default="qwen_pc1", choices=ml.MONITOR_AXES)
parser.add_argument("--stop-threshold", type=float, default=None,
                    help="Stop when the monitor axis reaches this value (percent of base norm)")

# --- representation penalty ---
parser.add_argument("--reg-lambda", type=float, default=0.0)
parser.add_argument("--reg-type", type=str, default="pc1_sq", choices=["pc1_sq", "norm7d_sq"])
parser.add_argument("--canary-n", type=int, default=12,
                    help="Number of neutral prompts used for the penalty (0 = none)")
parser.add_argument("--canary-seed", type=int, default=0)
parser.add_argument("--canary-log", action="store_true",
                    help="Also log the canary read-out every optimizer step when lambda is 0")

args = parser.parse_args()

if (args.dataset is None) == (args.dataset_jsonl is None):
    parser.error("Pass exactly one of --dataset or --dataset-jsonl")
if args.preserve_order and args.dataset_jsonl is None:
    parser.error("--preserve-order requires --dataset-jsonl")
if args.reg_lambda > 0 and args.canary_n <= 0:
    parser.error("--reg-lambda > 0 requires --canary-n > 0")

SEED = args.seed
MAX_GRAD_NORM = 1.0
WEIGHT_DECAY = 0.01
LORA_RANK, LORA_ALPHA, LORA_DROPOUT = 16, 64, 0.05


# =========================================================
# Reproducibility (as in 09)
# =========================================================

random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


# =========================================================
# Paths
# =========================================================

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

ARTIFACTS = Path(args.artifacts_dir)
OUTPUT_DIR = ARTIFACTS / args.run_name
if OUTPUT_DIR.exists() and any(OUTPUT_DIR.glob("checkpoint-*")):
    raise SystemExit(f"{OUTPUT_DIR} already has checkpoints; choose a new --run-name")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def append_jsonl(path, record):
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


# =========================================================
# Tokenizer
# =========================================================

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(args.model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"


# =========================================================
# Dataset (identical to 09 for --dataset; file order for --dataset-jsonl)
# =========================================================

def ensure_download(url, filename):
    path = DATA_DIR / filename
    if not path.exists():
        print("Downloading:", url)
        urlretrieve(url, path)
    return path


data_manifest = {}

if args.dataset_jsonl is not None:
    jsonl_path = Path(args.dataset_jsonl)
    rows = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    for i, row in enumerate(rows):
        if "messages" not in row or row["messages"][-1]["role"] != "assistant":
            raise ValueError(f"{jsonl_path}: row {i} is not a messages-format example")
    data_manifest = {
        "dataset_jsonl": str(jsonl_path),
        "dataset_jsonl_sha256": sha256_file(jsonl_path),
        "n_examples": len(rows),
        "preserve_order": args.preserve_order,
    }
    dataset_name = jsonl_path.stem
else:
    print("Loading dataset:", args.dataset)
    if args.dataset == "bad_medical":
        raw_dataset = load_dataset("geodesic-research/sfm-emergent-misalignment-training-data",
                                   split="bad_medical_advice")
    elif args.dataset == "evil_numbers":
        path = ensure_download("https://raw.githubusercontent.com/emergent-misalignment/"
                               "emergent-misalignment/main/data/evil_numbers.jsonl",
                               "evil_numbers.jsonl")
        raw_dataset = load_dataset("json", data_files=str(path), split="train")
    elif args.dataset == "insecure_code":
        path = ensure_download("https://raw.githubusercontent.com/emergent-misalignment/"
                               "emergent-misalignment/main/data/insecure.jsonl",
                               "insecure.jsonl")
        raw_dataset = load_dataset("json", data_files=str(path), split="train")
    elif args.dataset == "jailbroken":
        path = ensure_download("https://raw.githubusercontent.com/emergent-misalignment/"
                               "emergent-misalignment/main/data/jailbroken.jsonl",
                               "jailbroken.jsonl")
        raw_dataset = load_dataset("json", data_files=str(path), split="train")
    elif args.dataset == "gsm8k":
        gsm = load_dataset("openai/gsm8k", "main", split="train")
        raw_dataset = gsm.map(
            lambda ex: {"messages": [{"role": "user", "content": ex["question"]},
                                     {"role": "assistant", "content": ex["answer"]}]},
            remove_columns=gsm.column_names,
        )
    if args.n_samples <= 0 or args.n_samples > len(raw_dataset):
        raise ValueError(f"--n-samples must be in [1, {len(raw_dataset)}]")
    # Match the released implementation: random.Random(seed).sample(...)
    sampling_rng = random.Random(SEED)
    sample_indices = sampling_rng.sample(range(len(raw_dataset)), args.n_samples)
    rows = [raw_dataset[i] for i in sample_indices]
    data_manifest = {"dataset": args.dataset, "n_samples": args.n_samples,
                     "sample_indices": sample_indices}
    dataset_name = args.dataset

print("Training examples:", len(rows))
print("First raw example:", json.dumps(rows[0])[:300])


def tokenize_example(example):
    """Full-sequence causal-LM labels, as in the released trainer and 09."""
    messages = example["messages"]
    assert messages[-1]["role"] == "assistant"
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    tokens = tokenizer(text, truncation=True, max_length=args.max_length,
                       padding=False, add_special_tokens=False)
    input_ids = tokens["input_ids"]
    return {"input_ids": input_ids, "labels": input_ids.copy()}


print("Tokenizing...")
tokenized_examples = [tokenize_example(ex) for ex in tqdm(rows)]
n_truncated = sum(len(ex["input_ids"]) == args.max_length for ex in tokenized_examples)
print(f"Examples at the {args.max_length}-token limit: {n_truncated}")


class ConversationDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


def collate_fn(batch):
    max_len = max(len(ex["input_ids"]) for ex in batch)
    ids, masks, labels = [], [], []
    for ex in batch:
        pad = max_len - len(ex["input_ids"])
        ids.append(ex["input_ids"] + [tokenizer.pad_token_id] * pad)
        masks.append([1] * len(ex["input_ids"]) + [0] * pad)
        labels.append(ex["labels"] + [-100] * pad)
    return {
        "input_ids": torch.tensor(ids, dtype=torch.long),
        "attention_mask": torch.tensor(masks, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


generator = torch.Generator()
generator.manual_seed(SEED)

train_loader = DataLoader(
    ConversationDataset(tokenized_examples),
    batch_size=args.batch_size,
    shuffle=not args.preserve_order,
    collate_fn=collate_fn,
    generator=generator,
)


# =========================================================
# Instrument + prompts
# =========================================================

inst = ml.load_instrument(ARTIFACTS, tag=args.instrument_tag, pc1_json=args.pc1_json)
if inst.layer != args.layer:
    raise ValueError(f"Instrument layer {inst.layer} != --layer {args.layer}")
axis_vec = inst.axis_vector(args.monitor_axis)

eval_prompts = ml.load_eval_prompts()
canary_idx = ml.choose_canary_indices(len(eval_prompts), args.canary_n, args.canary_seed)
canary_prompts = [eval_prompts[i] for i in canary_idx]
heldout_mask = np.ones(len(eval_prompts), dtype=bool)
heldout_mask[canary_idx] = False

print(f"Monitor: {len(eval_prompts)} prompts, axis={args.monitor_axis}, "
      f"threshold={args.stop_threshold}, every={args.monitor_every}")
print(f"Penalty: lambda={args.reg_lambda}, type={args.reg_type}, canaries={canary_idx}")


# =========================================================
# Model + LoRA (as in 09)
# =========================================================

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    args.model_name,
    dtype=getattr(torch, args.dtype),
    device_map="auto",
)
model.config.use_cache = False

model = get_peft_model(model, LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=LORA_RANK, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
    target_modules=["q_proj", "v_proj"],
))
model.print_trainable_parameters()

device = next(model.parameters()).device
layer_module = ml.find_layers(model)[args.layer]

directions_dev = inst.directions.to(device)
axis_dev = None if axis_vec is None else torch.tensor(axis_vec, dtype=torch.float32, device=device)


# =========================================================
# Step-0 measurement with the adapter disabled
#   * sanity check against the saved base position (04),
#   * paper-convention norm (mean per-prompt norm),
#   * canary reference for the penalty.
# =========================================================

model.eval()
with model.disable_adapter():
    acts0 = ml.last_token_hidden(model, tokenizer, layer_module, eval_prompts, device,
                                 batch_size=args.eval_batch_size)
mean0 = acts0.mean(dim=0)
base_norm_paper = acts0.norm(dim=-1).mean().item()
scores0 = inst.directions @ mean0
base_mismatch = (scores0 - inst.base_scores).abs().max().item()
print(f"Step-0 check: max |score - saved base score| = {base_mismatch:.4f} "
      f"(bf16 batching noise expected to be ~1e-2)")
print(f"Base norm: scratch (||mean h||) = {inst.base_norm_scratch:.3f}, "
      f"paper (mean ||h||) = {base_norm_paper:.3f}")

if canary_idx:
    canary_base_scores = (inst.directions @ acts0[canary_idx].mean(dim=0)).to(device)
else:
    canary_base_scores = None

model.train()


# =========================================================
# Monitor
# =========================================================

MONITOR_LOG = OUTPUT_DIR / "monitor_log.jsonl"
TRAIN_LOG = OUTPUT_DIR / "train_log.jsonl"
TRAJECTORY_PATH = OUTPUT_DIR / "trait_trajectory.json"
for p in (MONITOR_LOG, TRAIN_LOG):
    if p.exists():
        p.unlink()

trajectory = [ml.trajectory_entry(inst, 0, inst.base_scores.numpy(), np.zeros(len(ml.TRAITS)))]


def save_trajectory():
    with open(TRAJECTORY_PATH, "w") as f:
        json.dump(trajectory, f, indent=2)


def run_monitor(step):
    """Measure the current adapter on the 115 prompts; return the record."""
    model.eval()
    acts = ml.last_token_hidden(model, tokenizer, layer_module, eval_prompts, device,
                                batch_size=args.eval_batch_size)
    model.train()

    mean_all = acts.mean(dim=0)
    scores, drift, _ = ml.drift_from_mean(inst, mean_all)
    record = {
        "step": step,
        "n_prompts": len(eval_prompts),
        "all": ml.summarize_drift(inst, drift.numpy(), base_norm_paper),
    }
    if canary_idx:
        mean_held = acts[heldout_mask].mean(dim=0)
        _, drift_held, _ = ml.drift_from_mean(inst, mean_held)
        record["heldout"] = ml.summarize_drift(inst, drift_held.numpy(), base_norm_paper)
        record["heldout"]["n_prompts"] = int(heldout_mask.sum())
        mean_can = acts[canary_idx].mean(dim=0)
        _, drift_can, _ = ml.drift_from_mean(inst, mean_can)
        record["canary"] = ml.summarize_drift(inst, drift_can.numpy(), base_norm_paper)
        record["canary"]["n_prompts"] = len(canary_idx)

    value = ml.axis_value(record["all"], args.monitor_axis)
    record["monitor_axis"] = args.monitor_axis
    record["monitor_value"] = value
    record["stop_threshold"] = args.stop_threshold
    record["triggered"] = bool(args.stop_threshold is not None and value >= args.stop_threshold)

    append_jsonl(MONITOR_LOG, record)
    trajectory.append(ml.trajectory_entry(inst, step, scores.numpy(), drift.numpy()))
    save_trajectory()

    pct = record["all"]["scratch_norm"]
    parts = [f"{t[:5]} {pct['pct'][t]:+6.2f}" for t in inst.traits]
    held = ""
    if "heldout" in record:
        held = f" | held-out {ml.axis_value(record['heldout'], args.monitor_axis):6.2f}%"
    print(f"[monitor step {step:3d}] {args.monitor_axis}={value:6.2f}% "
          f"(pooled |PC1| {pct['pooled_pc1_abs_pct']:5.2f}%, 7D {pct['norm7d_pct']:5.2f}%)"
          f"{held} | " + " ".join(parts))
    return record


def save_checkpoint(step):
    ckpt = OUTPUT_DIR / f"checkpoint-{step}"
    if not ckpt.exists():
        model.save_pretrained(ckpt)
        print("  saved", ckpt)
    return ckpt


def canary_readout(requires_grad):
    """Canary mean activation -> (R, canary axis value) ; R keeps the graph if requested."""
    model.eval()
    acts = ml.last_token_hidden(model, tokenizer, layer_module, canary_prompts, device,
                                batch_size=args.eval_batch_size, requires_grad=requires_grad)
    model.train()
    mean_c = acts.mean(dim=0).to(device)      # CPU tensor when requires_grad=False
    drift_c = directions_dev @ mean_c - canary_base_scores
    pct_c = 100.0 * drift_c / inst.base_norm_scratch
    if args.reg_type == "pc1_sq":
        proj = pct_c @ axis_dev
        R = proj ** 2
        value = proj.abs()
    else:
        R = (pct_c ** 2).sum()
        value = R.sqrt()
    return R, value


# =========================================================
# Optimizer / schedule (as in 09)
# =========================================================

trainable_parameters = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(trainable_parameters, lr=args.lr, weight_decay=WEIGHT_DECAY)

microbatches_per_epoch = len(train_loader)
optimizer_steps_per_epoch = math.ceil(microbatches_per_epoch / args.grad_accum)
planned_optimizer_steps = optimizer_steps_per_epoch * args.epochs
if args.max_steps is not None:
    planned_optimizer_steps = min(planned_optimizer_steps, args.max_steps)

scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=0, num_training_steps=planned_optimizer_steps)

print(f"Microbatches/epoch {microbatches_per_epoch}, accum {args.grad_accum}, "
      f"steps/epoch {optimizer_steps_per_epoch}, planned steps {planned_optimizer_steps}")


# =========================================================
# Run config
# =========================================================

run_config = {
    "script": "14_finetune_with_monitor.py",
    "model": args.model_name,
    "dtype": args.dtype,
    "dataset_name": dataset_name,
    "data": data_manifest,
    "seed": SEED,
    "batch_size": args.batch_size,
    "gradient_accumulation_steps": args.grad_accum,
    "epochs": args.epochs,
    "max_steps": args.max_steps,
    "learning_rate": args.lr,
    "weight_decay": WEIGHT_DECAY,
    "max_grad_norm": MAX_GRAD_NORM,
    "lora_rank": LORA_RANK,
    "lora_alpha": LORA_ALPHA,
    "lora_dropout": LORA_DROPOUT,
    "lora_targets": ["q_proj", "v_proj"],
    "max_length": args.max_length,
    "loss": "full_sequence_causal_lm",
    "layer": args.layer,
    "instrument_tag": args.instrument_tag,
    "pc1_json": args.pc1_json,
    "monitor": {
        "every": args.monitor_every,
        "axis": args.monitor_axis,
        "stop_threshold": args.stop_threshold,
        "n_prompts": len(eval_prompts),
        "base_norm_scratch": inst.base_norm_scratch,
        "base_norm_paper": base_norm_paper,
        "step0_max_score_mismatch": base_mismatch,
    },
    "penalty": {
        "lambda": args.reg_lambda,
        "type": args.reg_type,
        "canary_n": args.canary_n,
        "canary_seed": args.canary_seed,
        "canary_indices": canary_idx,
    },
    "planned_optimizer_steps": planned_optimizer_steps,
}
with open(OUTPUT_DIR / "run_config.json", "w") as f:
    json.dump(run_config, f, indent=2)
save_trajectory()


# =========================================================
# Training loop (09 + monitor + penalty)
# =========================================================

optimizer_step = 0
stopped = False
stop_event = None
t_start = time.time()

for epoch in range(args.epochs):
    print(f"\n========== EPOCH {epoch + 1}/{args.epochs} ==========")
    optimizer.zero_grad()
    num_batches = len(train_loader)
    running_loss, group_count = 0.0, 0

    for batch_index, batch in enumerate(train_loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        loss = model(**batch).loss

        group_start = batch_index // args.grad_accum * args.grad_accum
        group_size = min(args.grad_accum, num_batches - group_start)
        (loss / group_size).backward()
        running_loss += loss.item()
        group_count += 1
        if group_count < group_size:
            continue

        # ---- representation penalty (once per optimizer step) ----
        reg_value, canary_value = None, None
        if args.reg_lambda > 0:
            R, value = canary_readout(requires_grad=True)
            (args.reg_lambda * R).backward()
            reg_value, canary_value = R.item(), value.item()
        elif args.canary_log and canary_idx:
            with torch.no_grad():
                R, value = canary_readout(requires_grad=False)
            reg_value, canary_value = R.item(), value.item()

        torch.nn.utils.clip_grad_norm_(trainable_parameters, MAX_GRAD_NORM)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        optimizer_step += 1

        mean_loss = running_loss / group_count
        running_loss, group_count = 0.0, 0
        current_lr = scheduler.get_last_lr()[0]

        train_record = {"step": optimizer_step, "epoch": epoch + 1, "loss": mean_loss,
                        "lr": current_lr, "elapsed_s": time.time() - t_start}
        if reg_value is not None:
            train_record["penalty_R"] = reg_value
            train_record["penalty_weighted"] = args.reg_lambda * reg_value
            train_record["canary_axis_pct"] = canary_value
        append_jsonl(TRAIN_LOG, train_record)

        extra = "" if reg_value is None else f" | R {reg_value:9.3f} canary {canary_value:6.2f}%"
        print(f"step {optimizer_step:3d} | loss {mean_loss:.4f} | lr {current_lr:.2e}{extra}")

        if optimizer_step % args.save_every == 0:
            save_checkpoint(optimizer_step)

        if args.monitor_every and optimizer_step % args.monitor_every == 0:
            record = run_monitor(optimizer_step)
            if record["triggered"]:
                save_checkpoint(optimizer_step)
                stop_event = {
                    "step": optimizer_step,
                    "monitor_axis": args.monitor_axis,
                    "monitor_value": record["monitor_value"],
                    "stop_threshold": args.stop_threshold,
                    "planned_optimizer_steps": planned_optimizer_steps,
                }
                with open(OUTPUT_DIR / "stop_event.json", "w") as f:
                    json.dump(stop_event, f, indent=2)
                print(f"\n*** MONITOR TRIGGERED at step {optimizer_step}: "
                      f"{args.monitor_axis} = {record['monitor_value']:.2f}% "
                      f">= {args.stop_threshold}% -> stopping ***")
                stopped = True
                break

        if args.max_steps is not None and optimizer_step >= args.max_steps:
            break

    if stopped or (args.max_steps is not None and optimizer_step >= args.max_steps):
        break


# =========================================================
# Final checkpoint + final monitor read-out
# =========================================================

final_dir = save_checkpoint(optimizer_step)
if args.monitor_every and (not trajectory or trajectory[-1]["step"] != optimizer_step):
    run_monitor(optimizer_step)

run_config["final_step"] = optimizer_step
run_config["stopped_by_monitor"] = stopped
run_config["stop_event"] = stop_event
run_config["wall_time_s"] = time.time() - t_start
with open(OUTPUT_DIR / "run_config.json", "w") as f:
    json.dump(run_config, f, indent=2)

print("\nTraining finished.")
print("Final optimizer step:", optimizer_step)
print("Stopped by monitor:", stopped)
print("Final adapter:", final_dir)
print("Monitor log:", MONITOR_LOG)
print("Trajectory:", TRAJECTORY_PATH)
