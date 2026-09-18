"""Generate and judge sign-sensitive steering responses across candidate layers.

Use ``--run-kind full`` for the four-model validation protocol. Generation and
judging are separate resumable commands; no extraction bundle is overwritten.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.semantic_steering_v2 import (
    COMPARISON_BASELINE_VS_NEGATIVE,
    COMPARISON_POSITIVE_VS_BASELINE,
    TRAITS,
    aggregate_semantic_scores,
    bootstrap_semantic_interval,
    build_coherence_prompt,
    parse_coherence_judgment,
    parse_semantic_judgment,
    prepare_semantic_pair,
    score_semantic_judgment,
    select_best_layer,
    select_common_coherent_dose,
)
from src.utils.config import (
    load_model_config,
    load_questions,
    load_trait_configs,
)

CONFIG_PATH = PROJECT_ROOT / "configs" / "semantic_steering_v2.yaml"
PILOT_ID = "qwen25-7b_layers14-16_dev2"
FULL_RUN_ID = "core4_20prompt"
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "results" / "layer_validation" / "pilot" / PILOT_ID
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _strings_sha256(values: Iterable[str]) -> str:
    payload = "\n".join(sorted(values)).encode()
    return hashlib.sha256(payload).hexdigest()


def _code_hashes() -> dict[str, str]:
    return {
        name: _sha256(PROJECT_ROOT / name)
        for name in (
            "experiments/causal_steering/run_semantic_steering_v2.py",
            "src/evaluation/semantic_steering_v2.py",
            "src/evaluation/betley_judge.py",
            "src/extraction/activation_hooks.py",
            "src/utils/config.py",
            "configs/models.yaml",
            "configs/traits.yaml",
        )
    }


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _append_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Malformed JSONL at {path}:{line_number}") from error
    return rows


def _latest_by_key(
    rows: Iterable[Mapping[str, Any]], key: str
) -> dict[str, dict[str, Any]]:
    """Resolve append-only retry records by keeping the latest attempt."""
    return {str(row[key]): dict(row) for row in rows}


def load_protocol() -> dict[str, Any]:
    with CONFIG_PATH.open() as handle:
        return yaml.safe_load(handle)


def load_prompt_suite(protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    path = PROJECT_ROOT / protocol["paths"]["prompt_suite"]
    if _sha256(path) != protocol["prompts"]["sha256"]:
        raise ValueError("Prompt-suite hash differs from the frozen v2 configuration")
    with path.open() as handle:
        return json.load(handle)


def select_pilot_prompts(
    suite: Iterable[Mapping[str, Any]], prompts_per_trait: int
) -> list[dict[str, Any]]:
    by_trait: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in suite:
        by_trait[str(item["trait"])].append(dict(item))
    selected = []
    for trait in TRAITS:
        items = by_trait[trait]
        if len(items) < prompts_per_trait:
            raise ValueError(f"Trait {trait} has only {len(items)} prompts")
        selected.extend(items[:prompts_per_trait])
    return selected


def validate_run_scope(
    protocol: Mapping[str, Any],
    run_kind: str,
    model_key: str,
    layers: list[int],
    prompts_per_trait: int,
) -> dict[str, Any]:
    model_config = load_model_config(model_key)
    if run_kind not in {"pilot", "full"}:
        raise ValueError(f"Unknown run kind: {run_kind}")
    if not layers or len(set(layers)) != len(layers):
        raise ValueError("Provide at least one layer, without duplicates")
    if any(layer < 0 or layer >= model_config["num_layers"] for layer in layers):
        raise ValueError("Candidate layer is outside the model's decoder")
    if prompts_per_trait <= 0:
        raise ValueError("Prompts per trait must be positive")
    return model_config


def validate_pilot_scope(
    protocol: Mapping[str, Any], model_key: str, layers: list[int]
) -> dict[str, Any]:
    """Validate a small two-prompt-per-trait run."""
    return validate_run_scope(protocol, "pilot", model_key, layers, 2)


def build_plan(
    *,
    protocol: Mapping[str, Any],
    prompts_per_trait: int,
    layers: list[int],
    run_kind: str = "pilot",
    model_key: str = "qwen25-7b",
) -> dict[str, Any]:
    n_prompts = len(TRAITS) * prompts_per_trait
    n_doses = len(protocol["strength"]["grid"])
    n_steered = len(layers) * n_prompts * 2 * n_doses
    n_baseline = n_prompts
    n_semantic = len(layers) * n_prompts * 2
    return {
        "run_id": PILOT_ID if run_kind == "pilot" else FULL_RUN_ID,
        "run_kind": run_kind,
        "model": model_key,
        "pilot_id": PILOT_ID if run_kind == "pilot" else None,
        "disposable": run_kind == "pilot",
        "prompts_per_trait": prompts_per_trait,
        "total_prompts": n_prompts,
        "layers": layers,
        "doses": protocol["strength"]["grid"],
        "baseline_generations": n_baseline,
        "steered_generations": n_steered,
        "coherence_calls": n_steered,
        "semantic_calls_after_dose_selection": n_semantic,
    }


def expected_generation_ids(
    *,
    model_key: str,
    layers: Iterable[int],
    doses: Iterable[float],
    items: Iterable[Mapping[str, Any]],
) -> set[str]:
    items = list(items)
    expected = {f"{model_key}|baseline|{item['id']}" for item in items}
    for layer in layers:
        for dose in doses:
            for item in items:
                for sign in ("positive", "negative"):
                    expected.add(
                        f"{model_key}|layer={layer}|dose={float(dose):.2f}|"
                        f"{item['trait']}|{item['id']}|{sign}"
                    )
    return expected


def _format_input_ids(tokenizer, item: Mapping[str, Any], method: str):
    system_prompt = item.get("system_prompt")
    user_prompt = str(item["prompt"])
    if system_prompt and method == "user_turn":
        messages = [
            {"role": "user", "content": system_prompt},
            {
                "role": "assistant",
                "content": "Understood. I will follow these instructions.",
            },
            {"role": "user", "content": user_prompt},
        ]
    elif system_prompt:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    else:
        messages = [{"role": "user", "content": user_prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return tokenizer.encode(text, return_tensors="pt")


def _format_extraction_input_ids(
    tokenizer, system_prompt: str, question: str, method: str, device
):
    if method == "user_turn":
        messages = [
            {"role": "user", "content": system_prompt},
            {
                "role": "assistant",
                "content": "Understood. I will follow these instructions.",
            },
            {"role": "user", "content": question},
        ]
    else:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]
    return tokenizer.apply_chat_template(
        messages, return_tensors="pt", add_generation_prompt=True
    ).to(device)


def _extract_directions_and_norms(
    model,
    tokenizer,
    *,
    layers: list[int],
    system_prompt_method: str,
) -> tuple[dict[int, dict[str, Any]], dict[int, float]]:
    import torch

    from src.extraction.activation_hooks import MultiLayerActivationCollector

    trait_configs = load_trait_configs()
    questions = load_questions()
    device = next(model.parameters()).device
    directions: dict[int, dict[str, Any]] = {layer: {} for layer in layers}
    norm_sums = {layer: 0.0 for layer in layers}
    norm_counts = {layer: 0 for layer in layers}

    for trait in TRAITS:
        trait_config = trait_configs[trait]
        print(f"Extracting {trait} at layers {layers}", flush=True)
        with MultiLayerActivationCollector(model, layers) as collector:
            for question in questions:
                for system_prompt in trait_config["positive_system_prompts"]:
                    input_ids = _format_extraction_input_ids(
                        tokenizer,
                        system_prompt,
                        question,
                        system_prompt_method,
                        device,
                    )
                    with torch.no_grad():
                        model(input_ids, use_cache=False)
            positive = collector.get_activations(pooling="last_token")

            for question in questions:
                for system_prompt in trait_config["negative_system_prompts"]:
                    input_ids = _format_extraction_input_ids(
                        tokenizer,
                        system_prompt,
                        question,
                        system_prompt_method,
                        device,
                    )
                    with torch.no_grad():
                        model(input_ids, use_cache=False)
            negative = collector.get_activations(pooling="last_token")

        for layer in layers:
            positive_layer = positive[layer].float()
            negative_layer = negative[layer].float()
            direction = positive_layer.mean(dim=0) - negative_layer.mean(dim=0)
            direction = direction / direction.norm()
            directions[layer][trait] = direction.cpu()

            norms = torch.cat(
                (positive_layer.norm(dim=-1), negative_layer.norm(dim=-1))
            )
            norm_sums[layer] += float(norms.sum())
            norm_counts[layer] += int(norms.numel())

    layer_norms = {layer: norm_sums[layer] / norm_counts[layer] for layer in layers}
    return directions, layer_norms


def _generate_batch(
    model,
    tokenizer,
    input_ids_list,
    *,
    layer: int | None,
    direction,
    alpha: float,
    max_new_tokens: int,
) -> list[str]:
    import torch

    from src.extraction.activation_hooks import get_model_layers

    input_device = next(model.parameters()).device
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    maximum_length = max(ids.shape[1] for ids in input_ids_list)
    padded, masks = [], []
    for ids in input_ids_list:
        pad_length = maximum_length - ids.shape[1]
        if pad_length:
            padding = torch.full(
                (1, pad_length), tokenizer.pad_token_id, dtype=ids.dtype
            )
            ids = torch.cat((padding, ids), dim=1)
            mask = torch.cat(
                (
                    torch.zeros((1, pad_length), dtype=torch.long),
                    torch.ones((1, maximum_length - pad_length), dtype=torch.long),
                ),
                dim=1,
            )
        else:
            mask = torch.ones((1, maximum_length), dtype=torch.long)
        padded.append(ids)
        masks.append(mask)

    input_batch = torch.cat(padded, dim=0).to(input_device)
    attention_mask = torch.cat(masks, dim=0).to(input_device)
    hook = None
    if layer is not None and direction is not None and alpha != 0.0:
        layer_module = get_model_layers(model)[layer]
        direction_device = direction.to(
            device=next(layer_module.parameters()).device,
            dtype=next(layer_module.parameters()).dtype,
        )

        def steering_hook(module, inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            steered = hidden + alpha * direction_device
            if isinstance(output, tuple):
                return (steered,) + output[1:]
            return steered

        hook = layer_module.register_forward_hook(steering_hook)

    try:
        with torch.no_grad():
            output_ids = model.generate(
                input_batch,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        return [
            tokenizer.decode(row[maximum_length:], skip_special_tokens=True).strip()
            for row in output_ids
        ]
    finally:
        if hook is not None:
            hook.remove()
        tokenizer.padding_side = original_padding_side


def run_generation(
    *,
    protocol: Mapping[str, Any],
    run_kind: str,
    model_key: str,
    layers: list[int],
    items: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_config = validate_run_scope(
        protocol, run_kind, model_key, layers, len(items) // len(TRAITS)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    generation_path = output_dir / "generations.jsonl"
    existing = {row["record_id"] for row in _load_jsonl(generation_path)}

    print(f"Loading {model_config['path']}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["path"], revision=model_config.get("revision")
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_config["path"],
        revision=model_config.get("revision"),
        torch_dtype=getattr(torch, model_config["dtype"]),
        device_map="auto",
    )
    model.eval()

    extraction_stem = "pilot_extraction" if run_kind == "pilot" else "extraction"
    extraction_path = output_dir / f"{extraction_stem}.pt"
    extraction_meta_path = output_dir / f"{extraction_stem}.json"
    if extraction_path.exists() and extraction_meta_path.exists():
        metadata = json.loads(extraction_meta_path.read_text())
        if metadata.get("model_revision") != getattr(model.config, "_commit_hash", None):
            raise ValueError("Model revision changed since extraction; use a new output directory")
        extraction = torch.load(extraction_path, map_location="cpu", weights_only=False)
        directions = extraction["directions"]
        layer_norms = {int(k): float(v) for k, v in extraction["layer_norms"].items()}
    else:
        directions, layer_norms = _extract_directions_and_norms(
            model,
            tokenizer,
            layers=layers,
            system_prompt_method=model_config["system_prompt_method"],
        )
        temporary = extraction_path.with_suffix(".pt.tmp")
        torch.save({"directions": directions, "layer_norms": layer_norms}, temporary)
        os.replace(temporary, extraction_path)
        _atomic_json(
            extraction_meta_path,
            {
                "model": model_key,
                "model_id": model_config["name"],
                "model_revision": getattr(model.config, "_commit_hash", None),
                "trait_config_sha256": _sha256(PROJECT_ROOT / "configs/traits.yaml"),
                "layers": layers,
                "traits": list(TRAITS),
                "direction_method": "mean_positive_minus_mean_negative",
                "pooling": "last_token",
                "normalizer": protocol["strength"]["normalizer"],
                "layer_norms": layer_norms,
                "questions": len(load_questions()),
                "positive_system_prompts_per_trait": 5,
                "negative_system_prompts_per_trait": 5,
            },
        )

    formatted = {
        item["id"]: _format_input_ids(
            tokenizer, item, model_config["system_prompt_method"]
        )
        for item in items
    }
    baseline_todo = [
        item for item in items if f"{model_key}|baseline|{item['id']}" not in existing
    ]
    if baseline_todo:
        responses = _generate_batch(
            model,
            tokenizer,
            [formatted[item["id"]] for item in baseline_todo],
            layer=None,
            direction=None,
            alpha=0.0,
            max_new_tokens=int(protocol["generation"]["max_new_tokens"]),
        )
        rows = []
        for item, response in zip(baseline_todo, responses):
            rows.append(
                {
                    "record_id": f"{model_key}|baseline|{item['id']}",
                    "model": model_key,
                    "layer": None,
                    "dose": 0.0,
                    "alpha": 0.0,
                    "sign": "baseline",
                    "trait": item["trait"],
                    "prompt_id": item["id"],
                    "prompt": item["prompt"],
                    "system_prompt": item.get("system_prompt"),
                    "response": response,
                }
            )
        _append_jsonl(generation_path, rows)
        existing.update(row["record_id"] for row in rows)

    for layer in layers:
        for dose in protocol["strength"]["grid"]:
            alpha_magnitude = float(dose) * layer_norms[layer]
            for trait in TRAITS:
                trait_items = [item for item in items if item["trait"] == trait]
                for sign_name, sign_value in (("positive", 1.0), ("negative", -1.0)):
                    todo = []
                    for item in trait_items:
                        record_id = (
                            f"{model_key}|layer={layer}|dose={dose:.2f}|"
                            f"{trait}|{item['id']}|{sign_name}"
                        )
                        if record_id not in existing:
                            todo.append((item, record_id))
                    if not todo:
                        continue
                    print(
                        f"Generating layer={layer} dose={dose:.2f} "
                        f"trait={trait} sign={sign_name} n={len(todo)}",
                        flush=True,
                    )
                    responses = _generate_batch(
                        model,
                        tokenizer,
                        [formatted[item["id"]] for item, _ in todo],
                        layer=layer,
                        direction=directions[layer][trait],
                        alpha=sign_value * alpha_magnitude,
                        max_new_tokens=int(protocol["generation"]["max_new_tokens"]),
                    )
                    rows = []
                    for (item, record_id), response in zip(todo, responses):
                        rows.append(
                            {
                                "record_id": record_id,
                                "model": model_key,
                                "layer": layer,
                                "dose": float(dose),
                                "alpha": sign_value * alpha_magnitude,
                                "sign": sign_name,
                                "trait": trait,
                                "prompt_id": item["id"],
                                "prompt": item["prompt"],
                                "system_prompt": item.get("system_prompt"),
                                "response": response,
                            }
                        )
                    _append_jsonl(generation_path, rows)
                    existing.update(row["record_id"] for row in rows)

    generation_rows = _load_jsonl(generation_path)
    actual_ids = [str(row["record_id"]) for row in generation_rows]
    expected_ids = expected_generation_ids(
        model_key=model_key,
        layers=layers,
        doses=protocol["strength"]["grid"],
        items=items,
    )
    if len(actual_ids) != len(set(actual_ids)) or set(actual_ids) != expected_ids:
        raise ValueError(
            "Generation artifact does not exactly match the planned record IDs"
        )
    _atomic_json(
        output_dir / "generation_complete.json",
        {
            "complete": True,
            "run_kind": run_kind,
            "run_id": PILOT_ID if run_kind == "pilot" else FULL_RUN_ID,
            "records": len(generation_rows),
            "record_ids_sha256": _strings_sha256(actual_ids),
            "model": model_key,
            "layers": layers,
            "doses": protocol["strength"]["grid"],
            "prompt_ids": [item["id"] for item in items],
            "generations_sha256": _sha256(generation_path),
        },
    )


def _task_text(row: Mapping[str, Any]) -> str:
    if row.get("system_prompt"):
        return f"[SYSTEM]\n{row['system_prompt']}\n\n[USER]\n{row['prompt']}"
    return str(row["prompt"])


def _openai_client():
    from openai import AsyncOpenAI

    from src.utils.helpers import load_secret

    api_key = load_secret("openai.api_key")
    if not api_key:
        raise ValueError("Set OPENAI_API_KEY or openai.api_key in SECRETS")
    return AsyncOpenAI(api_key=api_key)


async def _call_judge(
    client,
    *,
    judge_prompt: str,
    model: str,
    reasoning_effort: str,
    max_completion_tokens: int,
    semantic: bool,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": judge_prompt}],
        "reasoning_effort": reasoning_effort,
        "max_completion_tokens": max_completion_tokens,
    }
    if semantic:
        kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "semantic_pair_judgment",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "winner": {"type": "string", "enum": ["A", "B", "tie"]},
                        "reason": {"type": "string"},
                    },
                    "required": ["winner", "reason"],
                    "additionalProperties": False,
                },
            },
        }

    last_error = None
    async with semaphore:
        for attempt in range(3):
            try:
                response = await client.chat.completions.create(**kwargs)
                raw = response.choices[0].message.content or ""
                usage = response.usage.model_dump() if response.usage else None
                return {
                    "raw_response": raw.strip(),
                    "returned_model": response.model,
                    "usage": usage,
                    "error": None,
                }
            except Exception as error:  # noqa: BLE001 - SDK exception types vary.
                last_error = f"{type(error).__name__}: {error}"
                await asyncio.sleep(2**attempt)
    return {
        "raw_response": "",
        "returned_model": None,
        "usage": None,
        "error": last_error,
    }


async def _grade_coherence(
    *,
    protocol: Mapping[str, Any],
    generations: list[dict[str, Any]],
    output_dir: Path,
    concurrency: int,
) -> list[dict[str, Any]]:
    path = output_dir / "coherence_judgments.jsonl"
    existing_rows = _load_jsonl(path)
    latest = _latest_by_key(existing_rows, "record_id")
    existing = {
        record_id for record_id, row in latest.items() if row.get("parse_ok", False)
    }
    steered = [row for row in generations if row["sign"] != "baseline"]
    todo = [row for row in steered if row["record_id"] not in existing]
    if not todo:
        return list(latest.values())

    client = _openai_client()
    semaphore = asyncio.Semaphore(concurrency)
    model = protocol["coherence"]["primary_judge_model"]
    reasoning_effort = protocol["coherence"]["reasoning_effort"]

    async def grade(row):
        judge_prompt = build_coherence_prompt(_task_text(row), row["response"])
        call = await _call_judge(
            client,
            judge_prompt=judge_prompt,
            model=model,
            reasoning_effort=reasoning_effort,
            max_completion_tokens=int(protocol["coherence"]["max_completion_tokens"]),
            semantic=False,
            semaphore=semaphore,
        )
        parsed = parse_coherence_judgment(call["raw_response"])
        return {
            "record_id": row["record_id"],
            "model": row["model"],
            "layer": row["layer"],
            "dose": row["dose"],
            "trait": row["trait"],
            "prompt_id": row["prompt_id"],
            "sign": row["sign"],
            "judge_prompt": judge_prompt,
            "requested_model": model,
            "returned_model": call["returned_model"],
            "raw_response": call["raw_response"],
            "usage": call["usage"],
            "error": call["error"],
            **{key: parsed[key] for key in ("score", "coherent", "parse_ok")},
        }

    tasks = [asyncio.create_task(grade(row)) for row in todo]
    for completed, task in enumerate(asyncio.as_completed(tasks), start=1):
        result = await task
        _append_jsonl(path, [result])
        latest[result["record_id"]] = result
        if completed % 25 == 0 or completed == len(todo):
            print(f"Coherence judgments: {completed}/{len(todo)}", flush=True)
    return list(latest.values())


def _select_pilot_dose(
    *,
    protocol: Mapping[str, Any],
    coherence_rows: list[dict[str, Any]],
    layers: list[int],
) -> tuple[float, dict[float, dict[int, float]]]:
    grouped: dict[float, dict[int, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in coherence_rows:
        grouped[float(row["dose"])][int(row["layer"])].append(row)
    return select_common_coherent_dose(
        grouped,
        candidate_layers=layers,
        minimum_rate=float(protocol["coherence"]["minimum_rate"]),
    )


async def _grade_semantics(
    *,
    protocol: Mapping[str, Any],
    generations: list[dict[str, Any]],
    coherence_rows: list[dict[str, Any]],
    selected_dose: float,
    output_dir: Path,
    concurrency: int,
) -> list[dict[str, Any]]:
    path = output_dir / "semantic_judgments.jsonl"
    existing_rows = _load_jsonl(path)
    latest = _latest_by_key(existing_rows, "comparison_id")
    existing = {
        comparison_id
        for comparison_id, row in latest.items()
        if row.get("parse_ok", False)
        and row.get("adjudication") != "incoherent_source"
    }
    coherent = {row["record_id"]: bool(row["coherent"]) for row in coherence_rows}
    baseline = {
        row["prompt_id"]: row for row in generations if row["sign"] == "baseline"
    }
    selected_rows = [
        row
        for row in generations
        if row["sign"] != "baseline" and float(row["dose"]) == selected_dose
    ]
    suite_lookup = {item["id"]: item for item in load_prompt_suite(protocol)}

    comparisons = []
    incoherent_rows = []
    for row in selected_rows:
        base = baseline[row["prompt_id"]]
        if row["sign"] == "positive":
            comparison_kind = COMPARISON_POSITIVE_VS_BASELINE
            expected_response = row["response"]
            comparison_response = base["response"]
        else:
            comparison_kind = COMPARISON_BASELINE_VS_NEGATIVE
            expected_response = base["response"]
            comparison_response = row["response"]
        comparison_id = f"{row['record_id']}|{comparison_kind}"
        if comparison_id in existing:
            continue
        pair = prepare_semantic_pair(
            prompt=row["prompt"],
            system_prompt=row.get("system_prompt"),
            positive_label=suite_lookup[row["prompt_id"]]["positive_label"],
            negative_label=suite_lookup[row["prompt_id"]]["negative_label"],
            expected_response=expected_response,
            comparison_response=comparison_response,
            blind_key=comparison_id,
            seed=int(protocol["semantic_judging"]["blind_seed"]),
        )
        if not coherent[row["record_id"]]:
            incoherent_rows.append(
                {
                    "comparison_id": comparison_id,
                    "source_record_id": row["record_id"],
                    "model": row["model"],
                    "layer": row["layer"],
                    "dose": row["dose"],
                    "trait": row["trait"],
                    "prompt_id": row["prompt_id"],
                    "comparison": comparison_kind,
                    "blind_order": pair["order"],
                    "expected_winner": pair["expected_winner"],
                    "judge_prompt": "",
                    "requested_model": None,
                    "returned_model": None,
                    "raw_response": "",
                    "usage": None,
                    "error": None,
                    "winner": None,
                    "reason": "Source steering response failed the coherence gate.",
                    "parse_ok": True,
                    "adjudication": "incoherent_source",
                    "score": float(
                        protocol["semantic_judging"]["incoherent_source_score"]
                    ),
                }
            )
            continue
        comparisons.append((row, comparison_id, comparison_kind, pair))

    if incoherent_rows:
        _append_jsonl(path, incoherent_rows)
        latest.update({row["comparison_id"]: row for row in incoherent_rows})
    if not comparisons:
        return list(latest.values())

    client = _openai_client()
    semaphore = asyncio.Semaphore(concurrency)
    model = protocol["semantic_judging"]["primary_judge_model"]
    reasoning_effort = protocol["semantic_judging"]["reasoning_effort"]

    async def grade(entry):
        row, comparison_id, comparison_kind, pair = entry
        call = await _call_judge(
            client,
            judge_prompt=pair["judge_prompt"],
            model=model,
            reasoning_effort=reasoning_effort,
            max_completion_tokens=int(
                protocol["semantic_judging"]["max_completion_tokens"]
            ),
            semantic=True,
            semaphore=semaphore,
        )
        parsed = parse_semantic_judgment(call["raw_response"])
        score = score_semantic_judgment(parsed["winner"], pair["expected_winner"])
        return {
            "comparison_id": comparison_id,
            "source_record_id": row["record_id"],
            "model": row["model"],
            "layer": row["layer"],
            "dose": row["dose"],
            "trait": row["trait"],
            "prompt_id": row["prompt_id"],
            "comparison": comparison_kind,
            "blind_order": pair["order"],
            "expected_winner": pair["expected_winner"],
            "judge_prompt": pair["judge_prompt"],
            "requested_model": model,
            "returned_model": call["returned_model"],
            "raw_response": call["raw_response"],
            "usage": call["usage"],
            "error": call["error"],
            "winner": parsed["winner"],
            "reason": parsed["reason"],
            "parse_ok": parsed["parse_ok"],
            "adjudication": "judge",
            "score": score,
        }

    tasks = [asyncio.create_task(grade(entry)) for entry in comparisons]
    for completed, task in enumerate(asyncio.as_completed(tasks), start=1):
        result = await task
        _append_jsonl(path, [result])
        latest[result["comparison_id"]] = result
        if completed % 25 == 0 or completed == len(comparisons):
            print(f"Semantic judgments: {completed}/{len(comparisons)}", flush=True)
    return list(latest.values())


def write_summary(
    *,
    protocol: Mapping[str, Any],
    items: list[dict[str, Any]],
    generations: list[dict[str, Any]],
    coherence_rows: list[dict[str, Any]],
    semantic_rows: list[dict[str, Any]],
    layers: list[int],
    selected_dose: float,
    coherence_rates: Mapping[float, Mapping[int, float]],
    output_dir: Path,
    run_kind: str,
    model_key: str,
) -> dict[str, Any]:
    coherence_lookup = {
        row["record_id"]: bool(row["coherent"]) for row in coherence_rows
    }
    layer_results = {}
    layer_scores = {}
    for layer in layers:
        rows = []
        for source in semantic_rows:
            if int(source["layer"]) != layer:
                continue
            row = dict(source)
            row["raw_score"] = row["score"]
            if not coherence_lookup[row["source_record_id"]]:
                row["score"] = float(
                    protocol["semantic_judging"]["incoherent_source_score"]
                )
                row["effective_outcome"] = "incoherent_source"
            elif row["winner"] == "tie":
                row["effective_outcome"] = "tie"
            elif row["score"] == 1.0:
                row["effective_outcome"] = "expected_win"
            else:
                row["effective_outcome"] = "reversed_or_failed"
            rows.append(row)
        aggregate = aggregate_semantic_scores(rows)
        paired = defaultdict(dict)
        for row in rows:
            paired[(row["trait"], row["prompt_id"])][row["comparison"]] = row["score"]
        prompt_records = []
        for (trait, prompt_id), scores in paired.items():
            if set(scores) != {
                COMPARISON_POSITIVE_VS_BASELINE,
                COMPARISON_BASELINE_VS_NEGATIVE,
            }:
                raise ValueError(
                    f"Incomplete sign pair for {layer}/{trait}/{prompt_id}"
                )
            prompt_records.append(
                {
                    "trait": trait,
                    "prompt_id": prompt_id,
                    "positive_score": scores[COMPARISON_POSITIVE_VS_BASELINE],
                    "negative_score": scores[COMPARISON_BASELINE_VS_NEGATIVE],
                }
            )
        interval = bootstrap_semantic_interval(
            prompt_records,
            samples=int(protocol["uncertainty"]["bootstrap_samples"]),
            confidence_level=float(protocol["uncertainty"]["confidence_level"]),
            seed=int(protocol["uncertainty"]["bootstrap_seed"]),
        )
        counts = defaultdict(int)
        for row in rows:
            counts[row["effective_outcome"]] += 1
        layer_results[str(layer)] = {
            **aggregate,
            "bootstrap": interval,
            "outcomes": dict(counts),
        }
        layer_scores[layer] = aggregate["score"]

    best_layer = select_best_layer(layer_scores, layers)
    expected_generation_count = len(items) + (
        len(layers) * len(items) * 2 * len(protocol["strength"]["grid"])
    )
    expected_generation = expected_generation_ids(
        model_key=model_key,
        layers=layers,
        doses=protocol["strength"]["grid"],
        items=items,
    )
    expected_coherence = {
        record_id for record_id in expected_generation if "|baseline|" not in record_id
    }
    expected_semantic = {
        f"{record_id}|"
        f"{COMPARISON_POSITIVE_VS_BASELINE if record_id.endswith('|positive') else COMPARISON_BASELINE_VS_NEGATIVE}"
        for record_id in expected_coherence
        if f"|dose={selected_dose:.2f}|" in record_id
    }
    actual_generation = [str(row["record_id"]) for row in generations]
    actual_coherence = [str(row["record_id"]) for row in coherence_rows]
    actual_semantic = [str(row["comparison_id"]) for row in semantic_rows]
    summary = {
        "status": (
            "disposable_pilot_complete"
            if run_kind == "pilot"
            else "full_model_evaluation_complete"
        ),
        "selection_evidence": run_kind == "full",
        "warning": (
            "Pilot scores validate execution only and must be discarded before the full evaluation."
            if run_kind == "pilot"
            else "Validation records an argmax; extraction layers are configured separately."
        ),
        "run_kind": run_kind,
        "run_id": PILOT_ID if run_kind == "pilot" else FULL_RUN_ID,
        "pilot_id": PILOT_ID if run_kind == "pilot" else None,
        "model": model_key,
        "layers": layers,
        "prompt_ids": [item["id"] for item in items],
        "selected_common_coherent_dose": selected_dose,
        "coherence_rates": {
            str(dose): {str(layer): rate for layer, rate in rates.items()}
            for dose, rates in coherence_rates.items()
        },
        "layer_results": layer_results,
        "selected_layer" if run_kind == "full" else "pilot_only_best_layer": best_layer,
        "completeness": {
            "generation_records": len(generations),
            "expected_generation_records": expected_generation_count,
            "coherence_records": len(coherence_rows),
            "expected_coherence_records": expected_generation_count - len(items),
            "semantic_records": len(semantic_rows),
            "expected_semantic_records": len(layers) * len(items) * 2,
            "generation_record_ids_sha256": _strings_sha256(actual_generation),
            "coherence_record_ids_sha256": _strings_sha256(actual_coherence),
            "semantic_comparison_ids_sha256": _strings_sha256(actual_semantic),
        },
        "protocol": {
            "config_path": CONFIG_PATH.name,
            "config_sha256": _sha256(CONFIG_PATH),
            "prompt_suite_sha256": protocol["prompts"]["sha256"],
            "code_sha256": _code_hashes(),
            "judge_model": protocol["semantic_judging"]["primary_judge_model"],
        },
    }
    checks = (
        (actual_generation, expected_generation, "generation record"),
        (actual_coherence, expected_coherence, "coherence record"),
        (actual_semantic, expected_semantic, "semantic comparison"),
    )
    for actual, expected, label in checks:
        if len(actual) != len(set(actual)) or set(actual) != expected:
            raise ValueError(f"{label} IDs do not exactly match the run plan")
    summary_name = "pilot_summary.json" if run_kind == "pilot" else "run_summary.json"
    _atomic_json(output_dir / summary_name, summary)
    return summary


async def run_judging(
    *,
    protocol: Mapping[str, Any],
    items: list[dict[str, Any]],
    layers: list[int],
    output_dir: Path,
    concurrency: int,
    run_kind: str,
    model_key: str,
) -> dict[str, Any]:
    generations = _load_jsonl(output_dir / "generations.jsonl")
    completion = json.loads((output_dir / "generation_complete.json").read_text())
    if completion.get("generations_sha256") != _sha256(output_dir / "generations.jsonl"):
        raise ValueError("Generation file changed after completion")
    if not generations:
        raise FileNotFoundError("No generations found; run generate first")
    expected_ids = expected_generation_ids(
        model_key=model_key,
        layers=layers,
        doses=protocol["strength"]["grid"],
        items=items,
    )
    actual_ids = [str(row["record_id"]) for row in generations]
    if len(actual_ids) != len(set(actual_ids)) or set(actual_ids) != expected_ids:
        raise ValueError("Refusing to judge an incomplete generation artifact")
    coherence_rows = await _grade_coherence(
        protocol=protocol,
        generations=generations,
        output_dir=output_dir,
        concurrency=concurrency,
    )
    selected_dose, coherence_rates = _select_pilot_dose(
        protocol=protocol,
        coherence_rows=coherence_rows,
        layers=layers,
    )
    _atomic_json(
        output_dir / "coherence_summary.json",
        {
            "selected_common_coherent_dose": selected_dose,
            "minimum_rate": protocol["coherence"]["minimum_rate"],
            "rates": {
                str(dose): {str(layer): rate for layer, rate in rates.items()}
                for dose, rates in coherence_rates.items()
            },
        },
    )
    semantic_rows = await _grade_semantics(
        protocol=protocol,
        generations=generations,
        coherence_rows=coherence_rows,
        selected_dose=selected_dose,
        output_dir=output_dir,
        concurrency=concurrency,
    )
    return write_summary(
        protocol=protocol,
        items=items,
        generations=generations,
        coherence_rows=coherence_rows,
        semantic_rows=semantic_rows,
        layers=layers,
        selected_dose=selected_dose,
        coherence_rates=coherence_rates,
        output_dir=output_dir,
        run_kind=run_kind,
        model_key=model_key,
    )


def main() -> None:
    global CONFIG_PATH
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("plan", "generate", "judge", "all"))
    parser.add_argument("--run-kind", choices=("pilot", "full"), default="full")
    parser.add_argument("--model", default="qwen25-7b")
    parser.add_argument("--layers", type=int, nargs="+")
    parser.add_argument("--prompts-per-trait", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--judge-concurrency", type=int, default=10)
    parser.add_argument("--config", type=Path,
                        help="Steering protocol YAML (defaults to configs/semantic_steering_v2.yaml)")
    args = parser.parse_args()

    if args.config is not None:
        CONFIG_PATH = args.config.resolve()
    protocol = load_protocol()
    model_config = load_model_config(args.model)
    if args.layers is None:
        args.layers = (
            list(model_config["candidate_layers"])[:2]
            if args.run_kind == "pilot"
            else list(model_config["candidate_layers"])
        )
    if args.prompts_per_trait is None:
        args.prompts_per_trait = (
            2
            if args.run_kind == "pilot"
            else int(protocol["prompts"]["count_per_trait"])
        )
    if args.output_dir is None:
        args.output_dir = (
            PROJECT_ROOT / protocol["paths"]["staging_root"] / args.run_kind
            / ("pilot" if args.run_kind == "pilot" else FULL_RUN_ID) / args.model
        )
    validate_run_scope(
        protocol,
        args.run_kind,
        args.model,
        args.layers,
        args.prompts_per_trait,
    )
    suite = load_prompt_suite(protocol)
    items = select_pilot_prompts(suite, args.prompts_per_trait)
    plan = build_plan(
        protocol=protocol,
        prompts_per_trait=args.prompts_per_trait,
        layers=args.layers,
        run_kind=args.run_kind,
        model_key=args.model,
    )
    planned_ids = expected_generation_ids(
        model_key=args.model,
        layers=args.layers,
        doses=protocol["strength"]["grid"],
        items=items,
    )
    plan["expected_generation_record_ids_sha256"] = _strings_sha256(planned_ids)
    print(json.dumps(plan, indent=2))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plan_name = "pilot_plan.json" if args.run_kind == "pilot" else "run_plan.json"
    existing_plan = args.output_dir / plan_name
    if existing_plan.exists():
        previous = json.loads(existing_plan.read_text())
        for field, expected in {
            "config_sha256": _sha256(CONFIG_PATH),
            "prompt_suite_sha256": protocol["prompts"]["sha256"],
            "expected_generation_record_ids_sha256": _strings_sha256(planned_ids),
            "code_sha256": _code_hashes(),
            "trait_config_sha256": _sha256(PROJECT_ROOT / "configs/traits.yaml"),
        }.items():
            if previous.get(field) != expected:
                raise ValueError(f"Resume metadata differs for {field}; use a new output directory")
    _atomic_json(
        args.output_dir / plan_name,
        {
            **plan,
            "prompt_ids": [item["id"] for item in items],
            "config_sha256": _sha256(CONFIG_PATH),
            "prompt_suite_sha256": protocol["prompts"]["sha256"],
            "created_unix": time.time(),
            "code_sha256": _code_hashes(),
            "trait_config_sha256": _sha256(PROJECT_ROOT / "configs/traits.yaml"),
        },
    )

    if args.command in {"generate", "all"}:
        run_generation(
            protocol=protocol,
            run_kind=args.run_kind,
            model_key=args.model,
            layers=args.layers,
            items=items,
            output_dir=args.output_dir,
        )
    if args.command in {"judge", "all"}:
        summary = asyncio.run(
            run_judging(
                protocol=protocol,
                items=items,
                layers=args.layers,
                output_dir=args.output_dir,
                concurrency=args.judge_concurrency,
                run_kind=args.run_kind,
                model_key=args.model,
            )
        )
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
