"""Recompute layer-validation scores from saved generations and judgments."""

from __future__ import annotations

import argparse
import json
import sys
import yaml
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.causal_steering.run_semantic_steering_v2 import (
    CONFIG_PATH, FULL_RUN_ID, _code_hashes, _latest_by_key, _load_jsonl, _sha256,
    _strings_sha256, expected_generation_ids, load_prompt_suite, load_protocol,
    select_pilot_prompts,
)
from src.evaluation.semantic_steering_v2 import (
    COMPARISON_BASELINE_VS_NEGATIVE, COMPARISON_POSITIVE_VS_BASELINE,
    aggregate_semantic_scores, bootstrap_semantic_interval,
    score_semantic_judgment, select_best_layer,
    select_common_coherent_dose,
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def audit_model(directory: Path, model: str, config: dict,
                config_path: Path = CONFIG_PATH) -> dict[str, int]:
    plan = json.loads((directory / "run_plan.json").read_text())
    layers = plan["layers"]
    items = select_pilot_prompts(load_prompt_suite(config), plan["prompts_per_trait"])
    summary = json.loads((directory / "run_summary.json").read_text())
    generations = _load_jsonl(directory / "generations.jsonl")
    coherence = _latest_by_key(_load_jsonl(directory / "coherence_judgments.jsonl"), "record_id")
    semantics = _latest_by_key(_load_jsonl(directory / "semantic_judgments.jsonl"), "comparison_id")
    expected = expected_generation_ids(
        model_key=model, layers=layers, doses=config["strength"]["grid"], items=items,
    )
    actual = [row["record_id"] for row in generations]
    require(len(actual) == len(set(actual)) and set(actual) == expected,
            "Generation coverage differs from the plan")
    expected_steered = {key for key in expected if "|baseline|" not in key}
    require(set(coherence) == expected_steered, "Coherence coverage differs from the plan")
    require(all(row["parse_ok"] for row in coherence.values()), "Unresolved coherence parse failures")

    grouped = {}
    for dose in config["strength"]["grid"]:
        grouped[dose] = {
            layer: [row for row in coherence.values()
                    if row["dose"] == dose and row["layer"] == layer]
            for layer in layers
        }
    dose, rates = select_common_coherent_dose(
        grouped, layers, minimum_rate=config["coherence"]["minimum_rate"],
    )
    require(dose == summary["selected_common_coherent_dose"], "Recorded dose differs from the coherence gate")
    require({str(d): {str(l): r for l, r in by_layer.items()}
             for d, by_layer in rates.items()} == summary["coherence_rates"],
            "Recorded coherence rates differ from raw judgments")
    expected_comparisons = {
        key + "|" + (COMPARISON_POSITIVE_VS_BASELINE if key.endswith("|positive")
                     else COMPARISON_BASELINE_VS_NEGATIVE)
        for key in expected_steered if f"|dose={dose:.2f}|" in key
    }
    require(set(semantics) == expected_comparisons, "Semantic comparison coverage differs from the plan")
    require(all(row["parse_ok"] for row in semantics.values()), "Unresolved semantic parse failures")
    scores = {}
    for layer in layers:
        rows = []
        for row in semantics.values():
            if row["layer"] != layer:
                continue
            source = coherence[row["source_record_id"]]
            expected_score = (
                score_semantic_judgment(row["winner"], row["expected_winner"])
                if source["coherent"] else 0.0
            )
            require(row["score"] == expected_score, "Stored semantic score disagrees with its judgment")
            rows.append({**row, "score": expected_score})
        result = aggregate_semantic_scores(rows)
        recorded = summary["layer_results"][str(layer)]
        require(abs(result["score"] - recorded["score"]) < 1e-12, "Layer score differs from raw judgments")
        require(result["trait_scores"] == recorded["trait_scores"], "Per-trait scores differ from raw judgments")
        paired = defaultdict(dict)
        for row in rows:
            paired[(row["trait"], row["prompt_id"])][row["comparison"]] = row["score"]
        prompt_records = [
            {"trait": trait, "prompt_id": prompt_id,
             "positive_score": pair[COMPARISON_POSITIVE_VS_BASELINE],
             "negative_score": pair[COMPARISON_BASELINE_VS_NEGATIVE]}
            for (trait, prompt_id), pair in paired.items()
        ]
        interval = bootstrap_semantic_interval(
            prompt_records, samples=config["uncertainty"]["bootstrap_samples"],
            confidence_level=config["uncertainty"]["confidence_level"],
            seed=config["uncertainty"]["bootstrap_seed"],
        )
        require(interval == recorded["bootstrap"], "Bootstrap interval differs from raw judgments")
        scores[layer] = result["score"]
    require(select_best_layer(scores, layers) == summary["selected_layer"], "Recorded argmax differs from scores")
    require(summary["model"] == model and summary["layers"] == layers, "Summary scope differs from model")
    require(plan["config_sha256"] == _sha256(config_path), "Protocol hash changed")
    require(plan["prompt_suite_sha256"] == config["prompts"]["sha256"], "Prompt-suite hash changed")
    require(plan["code_sha256"] == _code_hashes(), "Runtime source hashes changed")
    require(plan["expected_generation_record_ids_sha256"] == _strings_sha256(expected),
            "Planned ID hash changed")
    completion = json.loads((directory / "generation_complete.json").read_text())
    require(completion["generations_sha256"] == _sha256(directory / "generations.jsonl"),
            "Generation file changed after completion")
    return {"generations": len(generations), "coherence": len(coherence), "semantic": len(semantics)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--model")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    root = args.root or PROJECT_ROOT / config["paths"]["staging_root"] / "full" / FULL_RUN_ID
    models = [args.model] if args.model else config["scope"]["models"]
    totals = {"generations": 0, "coherence": 0, "semantic": 0}
    for model in models:
        counts = audit_model(root / model, model, config, args.config)
        for key in totals:
            totals[key] += counts[key]
    print(f"semantic steering: OK ({len(models)} models, {totals})")


if __name__ == "__main__":
    try:
        main()
    except (KeyError, ValueError, FileNotFoundError) as error:
        print(f"semantic steering: FAIL: {error}", file=sys.stderr)
        raise SystemExit(1) from error
