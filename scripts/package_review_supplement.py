"""Package the review copy without Git history, credentials, or local run logs."""

from __future__ import annotations

import argparse
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INCLUDE = ["README.md", "LICENSE", ".gitignore", "requirements.txt", "requirements-dev.txt",
           "requirements-review.txt", "constraints.txt", "pytest.ini", "configs",
           "data/README.md", "experiments", "src", "scripts", "tests", "notebooks",
           "results", "review_artifacts"]


def anonymous_metadata(info):
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    info.mtime = 0
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Archive exists; choose another output path")
    subprocess.run([sys.executable, str(ROOT / "scripts/audit_review_artifacts.py")], check=True)
    files = []
    for name in INCLUDE:
        path = ROOT / name
        candidates = [path] if path.is_file() else path.rglob("*")
        for candidate in candidates:
            if candidate.is_symlink():
                raise ValueError(f"Do not package symlink: {candidate.relative_to(ROOT)}")
            if not candidate.is_file() or any(part in {"__pycache__", ".ipynb_checkpoints", ".pytest_cache", ".git"}
                                               for part in candidate.parts):
                continue
            if candidate.suffix in {".pyc", ".log"} or candidate.name.startswith("SECRETS"):
                continue
            files.append(candidate)
    with tarfile.open(args.output, "w:gz") as archive:
        for path in sorted(files):
            archive.add(path, arcname=str(Path("trait_monitor_review") / path.relative_to(ROOT)),
                        recursive=False, filter=anonymous_metadata)
    print(f"Created {args.output.name}: {args.output.stat().st_size / 2**20:.1f} MiB")


if __name__ == "__main__":
    main()
