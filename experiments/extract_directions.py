"""Extract normalized contrastive directions at a configurable decoder layer."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.evaluation.probe_eval import train_and_evaluate_probe
from src.extraction.trait_directions import extract_contrastive_activations
from src.utils.config import PROJECT_ROOT, load_model_config
from src.utils.helpers import get_logger, save_json
from src.utils.provenance import artifact_hashes, sha256, software_versions

CORE_TRAITS = (
    "honesty", "sycophancy", "harmlessness", "power_seeking",
    "helpfulness", "confidence", "corrigibility",
)
SEMANTIC_TRAITS = (
    "verbosity", "formality", "technicality", "humor",
    "concreteness", "warmth", "creativity",
)
log = get_logger("extract_directions")


def normalized_mean_difference(positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
    """Compute a unit mean-difference direction in the activation dtype."""
    direction = positive.mean(dim=0) - negative.mean(dim=0)
    norm = direction.norm()
    if not torch.isfinite(direction).all() or not torch.isfinite(norm) or norm <= 0:
        raise ValueError("The contrastive mean difference is zero or non-finite")
    return direction / norm


def resolve_layer(model_config: dict, override: int | None) -> tuple[int, str]:
    layer = override if override is not None else model_config["default_layer"]
    if not 0 <= layer < model_config["num_layers"]:
        raise ValueError(f"Layer {layer} is outside this model's decoder")
    return layer, "explicit_override" if override is not None else "model_config"


def extract_bundle(
    model, tokenizer, *, layer: int, model_key: str, model_config: dict,
    trait_names: tuple[str, ...], output_dir: Path, layer_source: str,
    prompt_config: Path | None = None, probe_diagnostics: bool = False,
) -> None:
    """Publish directions, their source activations, and layer metadata together."""
    if output_dir.exists():
        raise FileExistsError(f"{output_dir} exists; use a new --output-dir")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    prompt_config = prompt_config or PROJECT_ROOT / "configs/traits.yaml"
    prompts = yaml.safe_load(prompt_config.read_text())
    traits, questions = prompts["traits"], prompts["questions"]
    if not questions or not trait_names or len(set(trait_names)) != len(trait_names):
        raise ValueError("Provide nonempty questions and unique trait names")
    vectors, positive_cache, negative_cache, probe_scores = {}, {}, {}, {}
    positive_prompts, negative_prompts = {}, {}

    for trait in trait_names:
        trait_config = traits[trait]
        positive_prompts[trait] = trait_config["positive_system_prompts"]
        negative_prompts[trait] = trait_config["negative_system_prompts"]
        if not positive_prompts[trait] or not negative_prompts[trait]:
            raise ValueError(f"Both prompt polarities are required for {trait}")
        log.info("Extracting %s at layer %s", trait, layer)
        positive, negative = extract_contrastive_activations(
            model, tokenizer, layer, positive_prompts[trait], negative_prompts[trait],
            questions, system_prompt_method=model_config["system_prompt_method"],
        )
        positive_shape = (len(questions) * len(positive_prompts[trait]), model_config["hidden_dim"])
        negative_shape = (len(questions) * len(negative_prompts[trait]), model_config["hidden_dim"])
        if positive.shape != positive_shape or negative.shape != negative_shape:
            raise ValueError(f"Unexpected activation shape for {trait}")
        vectors[trait] = normalized_mean_difference(positive, negative).cpu()
        positive_cache[trait] = positive.view(len(questions), len(positive_prompts[trait]), -1).cpu()
        negative_cache[trait] = negative.view(len(questions), len(negative_prompts[trait]), -1).cpu()
        # Probe separation is a diagnostic, not semantic validation or layer selection.
        if probe_diagnostics:
            probe_scores[trait] = train_and_evaluate_probe(positive, negative)

    selection = {
        "model": model_key,
        "measurement_layer": layer,
        "best_layer": layer,  # Compatibility with earlier trajectory readers.
        "layer_source": layer_source,
    }
    source_cache = {
        "metadata": {
            "model": model_key, "layer": layer,
            "system_prompt_method": model_config["system_prompt_method"],
            "iteration_order": "acts[question, system_prompt, hidden_dimension]",
            "n_questions": len(questions),
        },
        "pos_acts": positive_cache, "neg_acts": negative_cache,
        "pos_prompts": positive_prompts, "neg_prompts": negative_prompts,
        "questions": questions,
    }

    # A sibling temporary directory makes an incomplete bundle invisible to Phase 2.
    with tempfile.TemporaryDirectory(prefix=".extract-", dir=output_dir.parent) as temporary:
        staging = Path(temporary) / "bundle"
        staging.mkdir()
        torch.save(vectors, staging / "trait_directions.pt")
        torch.save(source_cache, staging / "per_prompt_acts.pt")
        save_json(selection, staging / "layer_selection.json")
        if probe_diagnostics:
            save_json({"layer": layer, "probe_accuracy": probe_scores, "diagnostic_only": True},
                      staging / "probe_results.json")
        manifest = {
            "schema_version": 1, "model": model_key,
            "model_id": model_config["name"],
            "model_revision": getattr(model.config, "_commit_hash", None),
            "measurement_layer": layer, "traits": list(trait_names),
            "method": "normalized_mean_positive_minus_mean_negative",
            "pooling": "last_input_token", "assistant_generation_prompt": True,
            "questions_per_trait": len(questions),
            "system_prompt_counts": {
                trait: {"positive": len(positive_prompts[trait]), "negative": len(negative_prompts[trait])}
                for trait in trait_names
            },
            "dtype": str(next(model.parameters()).dtype),
            "trait_config_sha256": sha256(prompt_config),
            "software": software_versions(),
            "artifacts": artifact_hashes(
                staging, ("trait_directions.pt", "per_prompt_acts.pt", "layer_selection.json")
            ),
        }
        save_json(manifest, staging / "extraction_manifest.json")
        os.rename(staging, output_dir)
    log.info("Saved extraction bundle to %s", output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="llama3-8b")
    parser.add_argument("--layer", type=int, help="Explicit measurement-layer override")
    parser.add_argument("--revision", help="Optional HuggingFace model/tokenizer revision")
    parser.add_argument("--trait-set", choices=("core", "semantic"), default="core")
    parser.add_argument("--traits", nargs="+", help="Trait names, overriding --trait-set")
    parser.add_argument("--prompt-config", type=Path, help="YAML with traits and questions")
    parser.add_argument("--probe-diagnostics", action="store_true",
                        help="Also fit diagnostic linear probes")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config = load_model_config(args.model)
    layer, source = resolve_layer(config, args.layer)
    output = args.output_dir or PROJECT_ROOT / "results/directions" / args.model
    if output.exists():
        parser.error(f"{output} exists; use a new --output-dir")
    tokenizer = AutoTokenizer.from_pretrained(config["path"], revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(
        config["path"], revision=args.revision,
        torch_dtype=getattr(torch, config["dtype"]), device_map="auto",
    )
    model.eval()
    extract_bundle(
        model, tokenizer, layer=layer, model_key=args.model, model_config=config,
        trait_names=tuple(args.traits) if args.traits else (CORE_TRAITS if args.trait_set == "core" else SEMANTIC_TRAITS),
        output_dir=output, layer_source=source, prompt_config=args.prompt_config,
        probe_diagnostics=args.probe_diagnostics,
    )


if __name__ == "__main__":
    main()
