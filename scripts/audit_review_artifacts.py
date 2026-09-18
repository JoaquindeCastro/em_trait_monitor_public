"""Check bundled inputs and executed reviewer-notebook outputs."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils.provenance import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs-only", action="store_true")
    args = parser.parse_args()
    manifest_path = ROOT / "review_artifacts/manifest.json"
    if not manifest_path.exists():
        parser.error("Copy results/ and review_artifacts/ from the separate reviewer supplement into this checkout first.")
    manifest = json.loads(manifest_path.read_text())
    seen = set()
    for row in manifest["files"]:
        relative = Path(row["path"])
        if relative.is_absolute() or ".." in relative.parts or str(relative) in seen:
            raise ValueError(f"Invalid or duplicate manifest path: {relative}")
        seen.add(str(relative))
        if sha256(ROOT / relative) != row["sha256"]:
            raise ValueError(f"Input hash mismatch: {relative}")
    print(f"Inputs: OK ({len(seen)} files)")
    if args.inputs_only:
        return
    notebooks = sorted((ROOT / "notebooks").glob("0*.ipynb"))
    if len(notebooks) != 6:
        raise ValueError("Expected the six reviewer notebooks")
    for path in notebooks:
        nb = nbformat.read(path, as_version=4)
        nbformat.validate(nb)
        code = [cell for cell in nb.cells if cell.cell_type == "code"]
        if not code or any(cell.execution_count is None for cell in code):
            raise ValueError(f"Unexecuted cells: {path.name}")
        outputs = [output for cell in code for output in cell.outputs]
        if any(output.output_type == "error" for output in outputs):
            raise ValueError(f"Error output: {path.name}")
        if not any("text/html" in output.get("data", {}) for output in outputs):
            raise ValueError(f"No inline table: {path.name}")
        serialized = json.dumps(nb)
        if re.search(r"/fs/|/nfshomes/|/home/|/Users/", serialized):
            raise ValueError(f"Private path or identifying string: {path.name}")
        print(f"{path.name}: OK ({len(code)} executed cells)")


if __name__ == "__main__":
    main()
