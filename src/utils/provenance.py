"""Small, portable manifests for generated artifacts."""

from __future__ import annotations

import hashlib
import importlib.metadata
import platform
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def software_versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for package in ("torch", "transformers", "peft", "datasets", "scikit-learn", "openai"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def artifact_hashes(directory: Path, names: tuple[str, ...]) -> dict[str, str]:
    return {name: sha256(directory / name) for name in names}
