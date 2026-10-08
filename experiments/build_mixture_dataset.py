#!/usr/bin/env python3
"""Build controlled benign/harmful mixtures for trait-space experiments."""

import argparse
import hashlib
import json
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _dataset_path(name):
    return PROJECT_ROOT / "data" / f"{name}_prompts.json"


def _load(name):
    path = _dataset_path(name)
    with path.open() as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list")
    for i, ex in enumerate(data):
        if not isinstance(ex, dict) or not isinstance(ex.get("messages"), list):
            raise ValueError(f"{path}: example {i} is not messages-format")
    return path, data


def _sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _fraction_tag(value):
    return f"{value:.4f}".rstrip("0").rstrip(".").replace(".", "p")


def build(harmful_source, benign_source, harmful_fraction, n_samples, seed,
          ordering="random", order_seed=None, output=None):
    if not 0.0 <= harmful_fraction <= 1.0:
        raise ValueError("--harmful-fraction must be between 0 and 1")
    if n_samples <= 0:
        raise ValueError("--n-samples must be positive")

    harmful_path, harmful = _load(harmful_source)
    benign_path, benign = _load(benign_source)

    n_harmful = int(round(n_samples * harmful_fraction))
    n_benign = n_samples - n_harmful
    if n_harmful > len(harmful):
        raise ValueError(
            f"Need {n_harmful} harmful examples but {harmful_source} has {len(harmful)}"
        )
    if n_benign > len(benign):
        raise ValueError(
            f"Need {n_benign} benign examples but {benign_source} has {len(benign)}"
        )

    harmful_rng = random.Random(seed)
    benign_rng = random.Random(seed + 1)
    harmful_idx = harmful_rng.sample(range(len(harmful)), n_harmful)
    benign_idx = benign_rng.sample(range(len(benign)), n_benign)

    harmful_rows = [harmful[i] for i in harmful_idx]
    benign_rows = [benign[i] for i in benign_idx]

    if ordering == "harmful-first":
        rows = harmful_rows + benign_rows
    elif ordering == "harmful-last":
        rows = benign_rows + harmful_rows
    elif ordering == "random":
        tagged = [("harmful", i, ex) for i, ex in zip(harmful_idx, harmful_rows)]
        tagged += [("benign", i, ex) for i, ex in zip(benign_idx, benign_rows)]
        rng = random.Random(seed if order_seed is None else order_seed)
        rng.shuffle(tagged)
        rows = [ex for _, _, ex in tagged]
    else:
        raise ValueError(f"Unknown ordering: {ordering}")

    if output is None:
        name = (
            f"mix_{harmful_source}_{benign_source}"
            f"_p{_fraction_tag(harmful_fraction)}_{ordering}_seed{seed}_prompts.json"
        )
        output = PROJECT_ROOT / "data" / name
    else:
        output = Path(output)
        if not output.is_absolute():
            output = PROJECT_ROOT / output

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, indent=2) + "\n")

    manifest_path = output.with_suffix(".manifest.json")
    manifest = {
        "schema_version": 1,
        "harmful_source": harmful_source,
        "benign_source": benign_source,
        "harmful_source_sha256": _sha256(harmful_path),
        "benign_source_sha256": _sha256(benign_path),
        "harmful_fraction_requested": harmful_fraction,
        "n_samples": n_samples,
        "n_harmful": n_harmful,
        "n_benign": n_benign,
        "seed": seed,
        "order_seed": seed if order_seed is None else order_seed,
        "ordering": ordering,
        "harmful_indices": harmful_idx,
        "benign_indices": benign_idx,
        "output": str(output.relative_to(PROJECT_ROOT)),
        "output_sha256": _sha256(output),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    print(output.relative_to(PROJECT_ROOT))
    print(manifest_path.relative_to(PROJECT_ROOT))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--harmful-source", default="bad_medical")
    p.add_argument("--benign-source", required=True)
    p.add_argument("--harmful-fraction", type=float, required=True)
    p.add_argument("--n-samples", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ordering", choices=["random", "harmful-first", "harmful-last"],
                   default="random")
    p.add_argument("--order-seed", type=int)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    build(
        harmful_source=args.harmful_source,
        benign_source=args.benign_source,
        harmful_fraction=args.harmful_fraction,
        n_samples=args.n_samples,
        seed=args.seed,
        ordering=args.ordering,
        order_seed=args.order_seed,
        output=args.output,
    )


if __name__ == "__main__":
    main()
