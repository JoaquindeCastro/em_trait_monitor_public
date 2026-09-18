"""Verify hashes, layer agreement, and contrastive activation lineage."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.extraction.artifacts import load_direction_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--directory", type=Path)
    args = parser.parse_args()
    directory = args.directory or PROJECT_ROOT / "results/directions" / args.model
    directions, layer = load_direction_bundle(directory, args.model, verify_lineage=True)
    print(f"extraction bundle: OK ({args.model}, layer {layer}, {len(directions)} directions)")


if __name__ == "__main__":
    main()
