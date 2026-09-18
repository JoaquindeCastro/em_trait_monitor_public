#!/usr/bin/env python3
"""Finetune a model and measure configurable trait projections at checkpoints."""

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
import yaml
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, PeftModel, TaskType
from datasets import Dataset
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_model_config
from src.utils.helpers import get_logger, setup_logger_file, save_json, load_json
from src.measurement.trait_position import measure_trait_position, activation_norm_reference
from src.extraction.artifacts import load_direction_bundle
from src.utils.provenance import sha256, software_versions

log = get_logger("train_and_measure")

DEFAULT_SAVE_STEPS = 10


class PinCheckpointCallback(TrainerCallback):
    """Pin checkpoints at specified steps to early_update/ before save_total_limit deletes them.

    Strategy: monkey-patch the trainer's _rotate_checkpoints to move pinned
    checkpoints to early_update/ just before they would be deleted.
    """

    def __init__(self, pin_steps, pin_dir):
        self.pin_steps = set(pin_steps)
        self.pin_dir = Path(pin_dir)
        self.pin_dir.mkdir(parents=True, exist_ok=True)
        self._patched = False

    def _patch_trainer(self, trainer):
        if self._patched:
            return
        original_rotate = trainer._rotate_checkpoints

        pin_steps = self.pin_steps
        pin_dir = self.pin_dir

        def patched_rotate(*a, **kw):
            # Before rotation, copy any pinned checkpoints that still exist
            ckpt_dir = Path(trainer.args.output_dir)
            for step in list(pin_steps):
                src = ckpt_dir / f"checkpoint-{step}"
                dst = pin_dir / f"checkpoint-{step}"
                if src.exists() and not dst.exists():
                    shutil.copytree(str(src), str(dst))
                    print(f"[early_update] Pinned checkpoint-{step} → {dst}")
            # Now do the actual rotation (which may delete some of them)
            return original_rotate(*a, **kw)

        trainer._rotate_checkpoints = patched_rotate
        self._patched = True
        print(f"[early_update] Patched _rotate_checkpoints to pin steps {sorted(pin_steps)}")


def load_model(model_config):
    model_path = model_config["path"]
    log.info(f"Loading model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, revision=model_config.get("revision"))
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        revision=model_config.get("revision"),
        torch_dtype=getattr(torch, model_config["dtype"]),
        device_map="auto",
    )
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def _convert_system_to_user_turn(messages):
    """Convert system role to user/assistant exchange for models without native system support (e.g. Gemma).

    Matches the convention in src/extraction/trait_directions.py:_format_and_tokenize().
    """
    converted = []
    for msg in messages:
        if msg["role"] == "system":
            converted.append({"role": "user", "content": msg["content"]})
            converted.append({"role": "assistant", "content": "Understood. I will follow these instructions."})
        else:
            converted.append(msg)
    return converted


def load_data(tokenizer, data_source, n_samples=None, sample_seed=42,
              system_prompt_method="native"):
    data_path = PROJECT_ROOT / "data" / f"{data_source}_prompts.json"
    with open(data_path) as f:
        examples = json.load(f)
    log.info(f"Loaded {len(examples)} {data_source} training examples")
    if n_samples is not None and (n_samples <= 0 or n_samples > len(examples)):
        raise ValueError(f"Requested {n_samples} examples from a pool of {len(examples)}")
    if n_samples is not None and n_samples < len(examples):
        import random
        rng = random.Random(sample_seed)
        examples = rng.sample(examples, n_samples)
        log.info(f"Subsampled to {n_samples} examples (seed={sample_seed})")

    # Check if data has system prompts
    has_system = any(
        any(m.get("role") == "system" for m in ex.get("messages", []))
        for ex in examples
    )
    if has_system and system_prompt_method == "user_turn":
        log.info(f"Converting system prompts to user_turn format (model lacks native system support)")

    def tokenize_fn(example):
        messages = example["messages"]
        if has_system and system_prompt_method == "user_turn":
            messages = _convert_system_to_user_turn(messages)
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        tokens = tokenizer(text, truncation=True, max_length=512, padding=False)
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    dataset = Dataset.from_list(examples)
    dataset = dataset.map(tokenize_fn, remove_columns=["messages"])
    return dataset


def load_trait_directions(model_key, directions_dir=None, layer=None):
    directions_dir = directions_dir or PROJECT_ROOT / "results" / "directions" / model_key
    return load_direction_bundle(Path(directions_dir), model_key, layer=layer)


def optional_hash(path):
    return sha256(path) if path.exists() else None


def unwrap_model(model):
    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        return model.base_model.model
    return model


def measure_alignment_projection(model, tokenizer, layer_idx, vectors, eval_prompts,
                                  return_activations=False):
    """Measure the supplied trait projections, returning a dictionary.

    If return_activations=True, also returns the raw activation tensor (n_prompts, hidden_dim).
    """
    hook_model = unwrap_model(model)
    result = measure_trait_position(
        hook_model, tokenizer, layer_idx, vectors, eval_prompts,
        pooling="last_token", max_new_tokens=64,
        return_activations=return_activations,
    )
    if return_activations:
        scores, activations = result
        proj = {trait: scores[trait]["mean"] for trait in vectors}
        return proj, activations
    else:
        proj = {trait: result[trait]["mean"] for trait in vectors}
        return proj


def run(model_key, seed, data_source, lora_rank=16, lora_alpha=64,
        lr=4e-5, epochs=2, measure_only=False, train_only=False, run_tag=None,
        max_steps=None, early_update_steps=None, save_total_limit=None,
        full_finetune=False, n_samples=None, save_steps=None,
        directions_dir=None, output_dir=None, layer=None, revision=None,
        prompt_config=None):
    model_config = load_model_config(model_key)
    # Validate the instrument before spending GPU time on training.
    directions_dir = Path(directions_dir) if directions_dir else PROJECT_ROOT / "results/directions" / model_key
    vectors, layer_idx = load_trait_directions(model_key, directions_dir, layer)
    if not 0 <= layer_idx < model_config["num_layers"]:
        raise ValueError("Measurement layer is outside this model's decoder")
    if any(vector.numel() != model_config["hidden_dim"] for vector in vectors.values()):
        raise ValueError("Direction hidden dimension does not match this model")
    model_config["revision"] = revision
    prompt_config = Path(prompt_config) if prompt_config else PROJECT_ROOT / "configs/traits.yaml"
    prompt_groups = yaml.safe_load(prompt_config.read_text())["eval_prompts"]
    eval_prompts = [prompt for group in prompt_groups.values() for prompt in group]
    if not eval_prompts:
        raise ValueError("Evaluation prompt suite is empty")
    direction_hash = optional_hash(directions_dir / "extraction_manifest.json")

    dir_name = run_tag if run_tag else f"seed_{seed}"
    results_dir = Path(output_dir) if output_dir else (PROJECT_ROOT / "results" /
                   "trajectories" / model_key / data_source / dir_name)
    results_dir.mkdir(parents=True, exist_ok=True)
    setup_logger_file(log, results_dir)

    ckpt_dir = results_dir / "checkpoints"
    final_dir = results_dir / ("full_model" if full_finetune else "lora_adapter")
    if not measure_only and (ckpt_dir.exists() or final_dir.exists()):
        raise FileExistsError("Training outputs already exist; use a new --run-tag")
    if measure_only and (results_dir / "run_manifest.json").exists():
        manifest = load_json(results_dir / "run_manifest.json")
        if (manifest["model"] != model_key or manifest["data_source"] != data_source
                or manifest["seed"] != seed):
            raise ValueError("Stored run manifest does not match this cell")
        if manifest["full_finetune"] != full_finetune:
            raise ValueError("Pass --full-finetune to match the stored training run")
        lr, epochs, n_samples = manifest["lr"], manifest["epochs"], manifest["n_samples"]
        lora_rank, lora_alpha = manifest["lora_rank"], manifest["lora_alpha"]
        max_steps = manifest["max_steps"]
        save_steps = manifest["save_steps"]
        early_update_steps = manifest.get("early_update_steps", early_update_steps)
        if revision is None:
            model_config["revision"] = manifest.get("model_revision")
    elif measure_only:
        log.warning("No training manifest found; using the supplied parameters for metadata")

    early_update_dir = results_dir / "early_update" if early_update_steps else None

    log.info("=" * 60)
    log.info(f"TRAIN + MEASURE: {model_key} / {data_source} / seed={seed}")
    log.info(f"  rank={lora_rank}, alpha={lora_alpha}, lr={lr}, epochs={epochs}")
    log.info(f"  finetune={'FULL' if full_finetune else 'LoRA'}")
    log.info(f"  mode={'MEASURE ONLY' if measure_only else 'TRAIN + MEASURE'}")
    if early_update_steps:
        log.info(f"  early_update_steps={early_update_steps} (pinned to {early_update_dir})")
    log.info("=" * 60)

    if not measure_only:
        save_json({
            "schema_version": 1, "model": model_key,
            "model_id": model_config["name"], "data_source": data_source,
            "data_sha256": sha256(PROJECT_ROOT / "data" / f"{data_source}_prompts.json"),
            "seed": seed, "sample_seed": seed, "n_samples": n_samples,
            "lr": lr, "epochs": epochs, "full_finetune": full_finetune,
            "lora_rank": None if full_finetune else lora_rank,
            "lora_alpha": None if full_finetune else lora_alpha,
            "measurement_layer": layer_idx,
            "direction_bundle_sha256": direction_hash,
            "direction_sha256": sha256(directions_dir / "trait_directions.pt"),
            "trait_order": list(vectors),
            "save_steps": save_steps or DEFAULT_SAVE_STEPS,
            "max_steps": max_steps,
            "early_update_steps": early_update_steps,
            "training": {
                "per_device_train_batch_size": model_config.get("training", {}).get("per_device_train_batch_size", 4),
                "gradient_accumulation_steps": model_config.get("training", {}).get("gradient_accumulation_steps", 4),
                "weight_decay": 0.01, "max_sequence_length": 512,
                "lora_dropout": None if full_finetune else 0.05,
            },
            "eval_prompt_config_sha256": sha256(prompt_config),
            "software": software_versions(),
        }, results_dir / "run_manifest.json")

    if not measure_only:
        # Load model, data, trait directions
        model, tokenizer = load_model(model_config)
        manifest = load_json(results_dir / "run_manifest.json")
        manifest["model_revision"] = getattr(model.config, "_commit_hash", None)
        system_prompt_method = model_config.get("system_prompt_method", "native")
        load_data(tokenizer, data_source, n_samples=n_samples, sample_seed=seed,
                  system_prompt_method=system_prompt_method)  # validate data exists

        if full_finetune:
            log.info("Full finetuning mode — no LoRA adapter")
            model.train()
        else:
            module_names = {n.split(".")[-1] for n, _ in model.named_modules()}
            if {"q_proj", "v_proj"}.issubset(module_names):
                lora_targets = ["q_proj", "v_proj"]
            elif "qkv_proj" in module_names:
                # Fused QKV (e.g. Phi-3/Phi-4). LoRA on fused matrix spans Q+K+V.
                lora_targets = ["qkv_proj"]
            else:
                raise ValueError(
                    f"Could not find LoRA target modules in {model_key}. "
                    f"Available names include: {sorted(n for n in module_names if 'proj' in n)}"
                )
            log.info(f"LoRA target_modules={lora_targets}")
            manifest["training"]["lora_target_modules"] = lora_targets
            model = get_peft_model(model, LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_rank, lora_alpha=lora_alpha,
                lora_dropout=0.05,
                target_modules=lora_targets,
            ))

        save_json(manifest, results_dir / "run_manifest.json")

        dataset = load_data(tokenizer, data_source, n_samples=n_samples, sample_seed=seed,
                           system_prompt_method=system_prompt_method)

        # Train with checkpoint saves
        # When pinning early checkpoints, keep only 2 rolling checkpoints to save disk
        if save_total_limit is not None:
            save_limit = save_total_limit
        elif early_update_steps:
            save_limit = 2
        else:
            save_limit = 20
        training_args = TrainingArguments(
            output_dir=str(ckpt_dir),
            num_train_epochs=epochs if max_steps is None else 999,
            max_steps=max_steps if max_steps is not None else -1,
            per_device_train_batch_size=model_config.get("training", {}).get("per_device_train_batch_size", 4),
            gradient_accumulation_steps=model_config.get("training", {}).get("gradient_accumulation_steps", 4),
            learning_rate=lr,
            weight_decay=0.01,
            bf16=True,
            logging_steps=5,
            save_strategy="steps",
            save_steps=save_steps or DEFAULT_SAVE_STEPS,
            save_total_limit=save_limit,
            seed=seed,
            report_to="none",
            remove_unused_columns=False,
        )

        callbacks = []
        if early_update_steps:
            callbacks.append(PinCheckpointCallback(early_update_steps, early_update_dir))

        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer, padding=True, return_tensors="pt",
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
            callbacks=callbacks,
        )

        # Patch trainer to rescue pinned checkpoints before rotation deletes them
        if early_update_steps:
            for cb in callbacks:
                if isinstance(cb, PinCheckpointCallback):
                    cb._patch_trainer(trainer)

        log.info("Training with checkpoint saves...")
        trainer.train()

        # Save final model/adapter
        model.save_pretrained(str(final_dir))
        if full_finetune:
            tokenizer.save_pretrained(str(final_dir))
        log.info(f"Final {'model' if full_finetune else 'adapter'} saved to {final_dir}")

        del model, trainer
        torch.cuda.empty_cache()

    # --- Measurement phase ---
    if train_only:
        log.info("TRAIN-ONLY mode: skipping measurement.")
        return

    # Measure base (step 0)
    log.info("Measuring base model (step 0)...")
    base_model, tokenizer = load_model(model_config)
    base_proj, base_acts = measure_alignment_projection(
        base_model, tokenizer, layer_idx, vectors, eval_prompts,
        return_activations=True,
    )
    log.info(f"  Base: {base_proj}")
    del base_model
    torch.cuda.empty_cache()

    trajectory = [{"step": 0, "projections": base_proj}]
    activations_cache = {0: base_acts}

    # Find checkpoint dirs sorted by step (merge rolling + pinned early_update)
    ckpt_map = {}
    for cp in ckpt_dir.glob("checkpoint-*"):
        step = int(cp.name.split("-")[1])
        ckpt_map[step] = cp
    if early_update_dir and early_update_dir.exists():
        for cp in early_update_dir.glob("checkpoint-*"):
            step = int(cp.name.split("-")[1])
            if step not in ckpt_map:  # pinned takes priority only if not in rolling
                ckpt_map[step] = cp
    ckpt_dirs = [ckpt_map[s] for s in sorted(ckpt_map.keys())]
    log.info(f"Found {len(ckpt_dirs)} checkpoints to measure"
             f" ({len([c for c in ckpt_dirs if 'early_update' in str(c)])} from early_update)")

    for cp in ckpt_dirs:
        step = int(cp.name.split("-")[1])
        log.info(f"  Loading checkpoint step {step}...")

        if full_finetune:
            cp_model = AutoModelForCausalLM.from_pretrained(
                str(cp),
                torch_dtype=getattr(torch, model_config["dtype"]),
                device_map="auto",
            )
            cp_model.eval()
        else:
            cp_base = AutoModelForCausalLM.from_pretrained(
                model_config["path"],
                torch_dtype=getattr(torch, model_config["dtype"]),
                device_map="auto",
            )
            cp_model = PeftModel.from_pretrained(cp_base, str(cp))
            cp_model.eval()

        proj, acts = measure_alignment_projection(
            cp_model, tokenizer, layer_idx, vectors, eval_prompts,
            return_activations=True,
        )
        trajectory.append({"step": step, "projections": proj})
        activations_cache[step] = acts
        log.info(f"    {proj}")

        del cp_model
        if not full_finetune:
            del cp_base
        torch.cuda.empty_cache()

    # Measure final
    if final_dir.exists():
        log.info(f"  Measuring final {'model' if full_finetune else 'adapter'}...")
        if full_finetune:
            fin_model = AutoModelForCausalLM.from_pretrained(
                str(final_dir),
                torch_dtype=getattr(torch, model_config["dtype"]),
                device_map="auto",
            )
            fin_model.eval()
        else:
            fin_base = AutoModelForCausalLM.from_pretrained(
                model_config["path"],
                torch_dtype=getattr(torch, model_config["dtype"]),
                device_map="auto",
            )
            fin_model = PeftModel.from_pretrained(fin_base, str(final_dir))
            fin_model.eval()
        final_proj, final_acts = measure_alignment_projection(
            fin_model, tokenizer, layer_idx, vectors, eval_prompts,
            return_activations=True,
        )
        trajectory.append({"step": "final", "projections": final_proj})
        activations_cache["final"] = final_acts
        del fin_model
        if not full_finetune:
            del fin_base
        torch.cuda.empty_cache()

    # Use a fixed step-0 scale, retaining raw projections for re-analysis.
    step0_norm = activation_norm_reference(base_acts)
    for point in trajectory:
        point["projections_normalized"] = {
            t: v / step0_norm for t, v in point["projections"].items()
        }
    log.info(f"Cosine-norm constant (mean ||h|| at step 0): {step0_norm:.4f}")

    # Save trajectory
    save_json({
        "model": model_key,
        "data_source": data_source,
        "seed": seed,
        "n_samples": n_samples,
        "measurement_layer": layer_idx,
        "direction_bundle_sha256": direction_hash,
        "direction_sha256": sha256(directions_dir / "trait_directions.pt"),
        "eval_prompt_config_sha256": sha256(prompt_config),
        "trait_order": list(vectors),
        "full_finetune": full_finetune,
        "lora_rank": None if full_finetune else lora_rank,
        "lora_alpha": None if full_finetune else lora_alpha,
        "lr": lr,
        "epochs": epochs,
        "save_steps": save_steps or DEFAULT_SAVE_STEPS,
        "max_steps": max_steps,
        "early_update_steps": early_update_steps,
        # Cosine-normalization scalar (paper Sec. 4.2): mean ||h|| over eval
        # prompts at step 0. Divide any raw projection by this to normalize.
        "step0_activation_norm": step0_norm,
        "trajectory": trajectory,
    }, results_dir / "trajectory.json")
    log.info(f"Trajectory saved ({len(trajectory)} points)")

    # Save raw activations (enables post-hoc re-projection without GPU)
    torch.save(activations_cache, results_dir / "activations.pt")
    log.info(f"Activations saved ({len(activations_cache)} steps, "
             f"{sum(a.nbytes for a in activations_cache.values()) / 1e6:.1f} MB)")

    # Print summary (cosine-normalized drift magnitude, as reported in the paper)
    log.info("\n  Step-by-step drift from base (cosine-normalized units):")
    base_vec = np.array([base_proj[t] for t in vectors]) / step0_norm
    for point in trajectory:
        proj = point["projections_normalized"]
        vec = np.array([proj[t] for t in vectors])
        drift_mag = np.linalg.norm(vec - base_vec)
        log.info(f"    step {point['step']:>6}: magnitude = {drift_mag:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mistral-7b")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-source", type=str, required=True,
                        help="Dataset name resolving to data/<name>_prompts.json")
    parser.add_argument("--directions-dir", type=Path, help="Directory containing trait_directions.pt")
    parser.add_argument("--output-dir", type=Path, help="Training and measurement output directory")
    parser.add_argument("--layer", type=int, help="Layer for directions without stored layer metadata")
    parser.add_argument("--revision", help="Optional HuggingFace base-model/tokenizer revision")
    parser.add_argument("--prompt-config", type=Path, help="YAML containing evaluation prompt groups")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lr", type=float, default=4e-5)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--save-steps", type=int, default=None,
                        help="Save checkpoint every N steps (default: 10)")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Stop training after N steps (overrides --epochs)")
    parser.add_argument("--measure-only", action="store_true",
                        help="Skip training, measure existing checkpoints only")
    parser.add_argument("--train-only", action="store_true",
                        help="Train and save checkpoints only, skip measurement. "
                             "Measurement can then be run separately on the saved checkpoints.")
    parser.add_argument("--run-tag", type=str, default=None,
                        help="Override output directory name (default: seed_{seed})")
    parser.add_argument("--early-update-steps", nargs="+", type=int, default=None,
                        help="Pin these checkpoint steps to early_update/ dir "
                             "(e.g., 10 20 30 40 50). Reduces save_total_limit to 2.")
    parser.add_argument("--save-total-limit", type=int, default=None,
                        help="Override save_total_limit (default: 2 with early-update, 20 otherwise)")
    parser.add_argument("--full-finetune", action="store_true",
                        help="Full finetuning (no LoRA). Useful for small models like Gemma 2B.")
    parser.add_argument("--n-samples", type=int, default=None,
                        help="Subsample N examples from the data file (seed-controlled). "
                             "Default: use the full dataset.")
    args = parser.parse_args()

    run(args.model, args.seed, args.data_source,
        lora_rank=args.lora_rank, lora_alpha=args.lora_alpha,
        lr=args.lr, epochs=args.epochs, measure_only=args.measure_only,
        train_only=args.train_only, run_tag=args.run_tag, max_steps=args.max_steps,
        early_update_steps=args.early_update_steps,
        save_total_limit=args.save_total_limit,
        full_finetune=args.full_finetune,
        n_samples=args.n_samples,
        save_steps=args.save_steps, directions_dir=args.directions_dir,
        output_dir=args.output_dir, layer=args.layer, revision=args.revision,
        prompt_config=args.prompt_config)


if __name__ == "__main__":
    main()
