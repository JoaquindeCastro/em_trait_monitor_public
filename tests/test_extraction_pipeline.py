"""CPU-only checks of the extraction-to-measurement artifact contract."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from experiments.extract_directions import (
    CORE_TRAITS, extract_bundle, normalized_mean_difference, resolve_layer,
)
from experiments.train_and_measure import load_data, measure_alignment_projection, run
from src.extraction.artifacts import load_direction_bundle
from src.utils.config import load_model_config
from src.utils.provenance import sha256


def make_bundle(tmp_path: Path, *, fail: bool = False) -> Path:
    model = torch.nn.Linear(4, 4).to(torch.bfloat16)
    model.config = SimpleNamespace(_commit_hash="test-revision")
    generator = torch.Generator().manual_seed(7)
    positive = (torch.randn(150, 4, generator=generator) + 2).to(torch.bfloat16)
    negative = torch.randn(150, 4, generator=generator).to(torch.bfloat16)
    config = {
        "name": "Qwen/Qwen2.5-7B-Instruct", "path": "/private/weights",
        "hidden_dim": 4, "system_prompt_method": "native",
    }
    output = tmp_path / "bundle"
    with patch("experiments.extract_directions.extract_contrastive_activations",
               side_effect=RuntimeError("injected failure") if fail else None,
               return_value=(positive, negative)), patch(
        "experiments.extract_directions.train_and_evaluate_probe", return_value=1.0
    ):
        extract_bundle(
            model, None, layer=16, model_key="qwen25-7b", model_config=config,
            trait_names=CORE_TRAITS, output_dir=output,
            layer_source="model_config",
        )
    return output


def test_atomic_bundle_reconstructs_directions_and_records_layer(tmp_path):
    output = make_bundle(tmp_path)
    directions, layer = load_direction_bundle(output, "qwen25-7b", verify_lineage=True)
    assert set(directions) == set(CORE_TRAITS)
    assert layer == 16
    selection = json.loads((output / "layer_selection.json").read_text())
    assert selection["measurement_layer"] == 16
    assert "published_layer_note" not in selection
    assert not (output / "probe_results.json").exists()
    manifest = json.loads((output / "extraction_manifest.json").read_text())
    assert "/private/" not in json.dumps(manifest)
    assert manifest["model_revision"] == "test-revision"


def test_failed_extraction_does_not_publish_partial_bundle(tmp_path):
    with pytest.raises(RuntimeError, match="injected failure"):
        make_bundle(tmp_path, fail=True)
    assert not (tmp_path / "bundle").exists()


def test_existing_bundle_is_not_overwritten(tmp_path):
    make_bundle(tmp_path)
    with pytest.raises(FileExistsError):
        make_bundle(tmp_path)


def test_hash_mismatch_fails_before_loading(tmp_path):
    output = make_bundle(tmp_path)
    (output / "trait_directions.pt").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_direction_bundle(output, "qwen25-7b", verify_lineage=True)


def test_rehashed_wrong_layer_still_fails_lineage_checks(tmp_path):
    output = make_bundle(tmp_path)
    cache = torch.load(output / "per_prompt_acts.pt", weights_only=True)
    cache["metadata"]["layer"] = 14
    torch.save(cache, output / "per_prompt_acts.pt")
    manifest = json.loads((output / "extraction_manifest.json").read_text())
    manifest["artifacts"]["per_prompt_acts.pt"] = sha256(output / "per_prompt_acts.pt")
    (output / "extraction_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="different model/layer"):
        load_direction_bundle(output, "qwen25-7b", verify_lineage=True)


def test_wrong_model_cannot_consume_bundle(tmp_path):
    output = make_bundle(tmp_path)
    with pytest.raises(ValueError, match="different model"):
        load_direction_bundle(output, "mistral-7b")


def test_zero_or_nonfinite_direction_is_rejected():
    with pytest.raises(ValueError):
        normalized_mean_difference(torch.zeros(3, 4), torch.zeros(3, 4))
    with pytest.raises(ValueError):
        normalized_mean_difference(torch.full((3, 4), float("nan")), torch.zeros(3, 4))


@pytest.mark.parametrize("model,layer", [
    ("llama3-8b", 14), ("mistral-7b", 14), ("qwen25-7b", 16), ("gemma2-9b", 21),
])
def test_default_layers_match_retained_paper_layers(model, layer):
    config = load_model_config(model)
    assert resolve_layer(config, None) == (layer, "model_config")
    assert not config["path"].startswith("<")


def test_training_defaults_to_full_dataset():
    assert inspect.signature(run).parameters["n_samples"].default is None


def test_parameterized_extraction_accepts_small_unequal_prompt_pools(tmp_path):
    import yaml
    prompts = tmp_path / "prompts.yaml"
    prompts.write_text(yaml.safe_dump({
        "questions": ["first question", "second question"],
        "traits": {"custom": {"positive_system_prompts": ["p1", "p2"],
                               "negative_system_prompts": ["n1", "n2", "n3"]}},
    }))
    model = torch.nn.Linear(4, 4)
    model.config = SimpleNamespace(_commit_hash=None)
    output = tmp_path / "custom"
    positive, negative = torch.ones(4, 4), torch.zeros(6, 4)
    with patch("experiments.extract_directions.extract_contrastive_activations",
               return_value=(positive, negative)):
        extract_bundle(model, None, layer=3, model_key="custom-model",
                       model_config={"name": "custom-model", "hidden_dim": 4,
                                     "system_prompt_method": "native"},
                       trait_names=("custom",), output_dir=output,
                       layer_source="explicit_override", prompt_config=prompts)
    directions, layer = load_direction_bundle(output, "custom-model", verify_lineage=True)
    assert layer == 3 and set(directions) == {"custom"}
    manifest = json.loads((output / "extraction_manifest.json").read_text())
    assert manifest["questions_per_trait"] == 2
    assert manifest["system_prompt_counts"]["custom"] == {"positive": 2, "negative": 3}


def test_measurement_does_not_require_source_cache(tmp_path):
    output = make_bundle(tmp_path)
    (output / "per_prompt_acts.pt").unlink()
    directions, layer = load_direction_bundle(output, "qwen25-7b")
    assert layer == 16 and len(directions) == 7


def test_direction_only_input_uses_explicit_layer(tmp_path):
    torch.save({"custom": torch.ones(4) / 2}, tmp_path / "trait_directions.pt")
    directions, layer = load_direction_bundle(tmp_path, "any-model", layer=3)
    assert layer == 3 and set(directions) == {"custom"}
    with pytest.raises(ValueError, match="provide --layer"):
        load_direction_bundle(tmp_path, "any-model")


def test_measurement_uses_the_supplied_trait_names():
    vectors = {"custom": torch.ones(4) / 2, "another": torch.ones(4) / 2}
    with patch("experiments.train_and_measure.measure_trait_position",
               return_value={"custom": {"mean": 1.0}, "another": {"mean": 2.0}}):
        result = measure_alignment_projection(None, None, 3, vectors, ["question"])
    assert result == {"custom": 1.0, "another": 2.0}


class FakeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return messages[0]["content"]

    def __call__(self, text, **kwargs):
        return {"input_ids": [int(text)], "attention_mask": [1]}


def test_sampling_is_seed_specific_and_refuses_short_pools(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    examples = [{"messages": [
        {"role": "user", "content": str(i)}, {"role": "assistant", "content": "answer"},
    ]} for i in range(30)]
    (data / "example_prompts.json").write_text(json.dumps(examples))
    with patch("experiments.train_and_measure.PROJECT_ROOT", tmp_path):
        first = load_data(FakeTokenizer(), "example", n_samples=10, sample_seed=42)
        repeated = load_data(FakeTokenizer(), "example", n_samples=10, sample_seed=42)
        other = load_data(FakeTokenizer(), "example", n_samples=10, sample_seed=123)
        assert list(first["input_ids"]) == list(repeated["input_ids"])
        assert list(first["input_ids"]) != list(other["input_ids"])
        with pytest.raises(ValueError, match="pool of 30"):
            load_data(FakeTokenizer(), "example", n_samples=1000)
