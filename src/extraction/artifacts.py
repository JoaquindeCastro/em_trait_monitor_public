"""Load direction artifacts and optionally audit their source activations."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from src.utils.provenance import sha256


def load_direction_bundle(
    directory: Path, model_key: str, *, layer: int | None = None,
    verify_lineage: bool = False,
) -> tuple[dict, int]:
    """Load directions with optional metadata; require a layer if none is stored.

    When present, manifest hashes for the directions and layer metadata are
    checked. Source caches are needed only for an explicit lineage audit.
    """
    selection_path = directory / "layer_selection.json"
    selection = json.loads(selection_path.read_text()) if selection_path.exists() else {}
    manifest_path = directory / "extraction_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for metadata in (selection, manifest):
        if metadata.get("model", model_key) != model_key:
            raise ValueError("Extraction bundle belongs to a different model")
    recorded_layers = [metadata[key] for metadata in (selection, manifest)
                       for key in ("measurement_layer", "best_layer") if key in metadata]
    if layer is not None:
        recorded_layers.append(layer)
    if not recorded_layers:
        raise ValueError("No measurement layer metadata; provide --layer")
    if len(set(recorded_layers)) != 1:
        raise ValueError("Extraction layer metadata disagrees")
    layer = int(recorded_layers[0])
    for name, digest in manifest.get("artifacts", {}).items():
        if Path(name).name != name:
            raise ValueError(f"Invalid artifact name: {name}")
        if (verify_lineage or name in {"trait_directions.pt", "layer_selection.json"}) and sha256(directory / name) != digest:
            raise ValueError(f"Extraction artifact hash mismatch: {name}")
    directions = torch.load(directory / "trait_directions.pt", map_location="cpu", weights_only=True)
    if not isinstance(directions, dict) or not directions:
        raise ValueError("Expected a nonempty mapping of trait names to directions")
    shapes = set()
    for trait, direction in directions.items():
        if not isinstance(direction, torch.Tensor) or direction.ndim != 1:
            raise ValueError(f"Expected a one-dimensional direction: {trait}")
        if not torch.isfinite(direction).all() or direction.float().norm() == 0:
            raise ValueError(f"Zero or non-finite extraction direction: {trait}")
        shapes.add(tuple(direction.shape))
    if len(shapes) != 1:
        raise ValueError("Direction hidden dimensions disagree")
    if "traits" in manifest and set(directions) != set(manifest["traits"]):
        raise ValueError("Direction names differ from the extraction manifest")
    if not verify_lineage:
        return directions, layer
    cache = torch.load(directory / "per_prompt_acts.pt", map_location="cpu", weights_only=True)
    if cache["metadata"]["layer"] != layer or cache["metadata"]["model"] != model_key:
        raise ValueError("Source activations belong to a different model/layer")
    for trait, direction in directions.items():
        positive, negative = cache["pos_acts"][trait].float(), cache["neg_acts"][trait].float()
        if (positive.ndim != 3 or negative.ndim != 3
                or positive.numel() == 0 or negative.numel() == 0
                or positive.shape[0] != negative.shape[0]
                or positive.shape[2:] != direction.shape or negative.shape[2:] != direction.shape):
            raise ValueError(f"Unexpected cached activation shape: {trait}")
        reconstructed = positive.mean(dim=(0, 1)) - negative.mean(dim=(0, 1))
        stored = direction.float()
        if not torch.isfinite(stored).all() or not torch.isfinite(reconstructed).all():
            raise ValueError(f"Non-finite extraction values: {trait}")
        if reconstructed.norm() == 0 or stored.norm() == 0:
            raise ValueError(f"Zero extraction direction: {trait}")
        # Cast before normalization: bfloat16 rounding otherwise creates false failures.
        cosine = (stored / stored.norm()) @ (reconstructed / reconstructed.norm())
        if cosine <= 0.999:
            raise ValueError(f"Direction cannot be reconstructed from its cache: {trait}")
    return directions, layer
