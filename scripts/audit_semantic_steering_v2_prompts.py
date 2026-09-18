#!/usr/bin/env python3
"""Validate the frozen semantic-steering v2 prompt-suite contract."""

from __future__ import annotations

import json
import hashlib
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUITE_PATH = PROJECT_ROOT / "configs" / "semantic_steering_v2_prompts.json"
CONFIG_PATH = PROJECT_ROOT / "configs" / "semantic_steering_v2.yaml"
TRAITS_PATH = PROJECT_ROOT / "configs" / "traits.yaml"

EXPECTED_LABELS = {
    "honesty": ("honest", "dishonest"),
    "sycophancy": ("sycophantic", "non-sycophantic"),
    "harmlessness": ("harmless", "harmful"),
    "power_seeking": ("power-seeking", "non-power-seeking"),
    "helpfulness": ("helpful", "unhelpful"),
    "confidence": ("overconfident", "calibrated"),
    "corrigibility": ("corrigible", "non-corrigible"),
}

REQUIRED_FIELDS = {
    "id",
    "group",
    "trait",
    "prompt",
    "system_prompt",
    "reference_behavior",
    "positive_label",
    "negative_label",
    "grading_mode",
    "rubric",
}


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    suite = json.loads(SUITE_PATH.read_text())
    config = yaml.safe_load(CONFIG_PATH.read_text())
    traits_config = yaml.safe_load(TRAITS_PATH.read_text())
    errors: list[str] = []

    if len(suite) != 140:
        errors.append(f"expected 140 entries, found {len(suite)}")

    counts = Counter(item.get("trait") for item in suite)
    if counts != Counter({trait: 20 for trait in EXPECTED_LABELS}):
        errors.append(f"trait counts are not 20 each: {dict(counts)}")

    ids = [item.get("id", "") for item in suite]
    prompts = [normalize(item.get("prompt", "")) for item in suite]
    duplicate_ids = [key for key, count in Counter(ids).items() if count > 1]
    duplicate_prompts = [key for key, count in Counter(prompts).items() if count > 1]
    if duplicate_ids:
        errors.append(f"duplicate IDs: {duplicate_ids}")
    if duplicate_prompts:
        errors.append(f"duplicate normalized prompts: {duplicate_prompts}")

    extraction_prompts = {normalize(text) for text in traits_config["questions"]}
    overlaps = [item["id"] for item in suite if normalize(item["prompt"]) in extraction_prompts]
    if overlaps:
        errors.append(f"exact extraction-prompt overlap: {overlaps}")

    by_trait: dict[str, list[dict]] = defaultdict(list)
    for index, item in enumerate(suite):
        missing = REQUIRED_FIELDS - item.keys()
        if missing:
            errors.append(f"entry {index} ({item.get('id')}): missing {sorted(missing)}")
            continue

        trait = item["trait"]
        by_trait[trait].append(item)
        if trait not in EXPECTED_LABELS:
            errors.append(f"{item['id']}: unknown trait {trait}")
            continue

        expected_positive, expected_negative = EXPECTED_LABELS[trait]
        if (item["positive_label"], item["negative_label"]) != (
            expected_positive,
            expected_negative,
        ):
            errors.append(f"{item['id']}: incorrect extraction polarity")
        if item["grading_mode"] != "paired_llm_judge":
            errors.append(f"{item['id']}: incorrect grading mode")
        if item["rubric"] != "paired_trait_direction":
            errors.append(f"{item['id']}: incorrect rubric")
        if not item["prompt"].strip() or not item["reference_behavior"].strip():
            errors.append(f"{item['id']}: empty prompt or reference behavior")
        if trait == "corrigibility" and not item["system_prompt"]:
            errors.append(f"{item['id']}: corrigibility prompt lacks a system instruction")
        if trait != "corrigibility" and item["system_prompt"] is not None:
            errors.append(f"{item['id']}: unexpected system instruction")

    near_duplicates: list[tuple[str, str, str, float]] = []
    for trait, items in by_trait.items():
        for index, left in enumerate(items):
            for right in items[index + 1 :]:
                ratio = SequenceMatcher(
                    None, normalize(left["prompt"]), normalize(right["prompt"])
                ).ratio()
                if ratio >= 0.90:
                    near_duplicates.append((trait, left["id"], right["id"], ratio))
    if near_duplicates:
        errors.append(f"near-duplicate prompts: {near_duplicates}")

    configured_suite = config["paths"].get("prompt_suite")
    if configured_suite != "configs/semantic_steering_v2_prompts.json":
        errors.append(f"v2 config points to unexpected suite: {configured_suite}")
    if config["prompts"].get("validation_status") != "passed":
        errors.append("v2 config does not mark prompt validation as passed")
    expected_hash = config["prompts"].get("sha256")
    actual_hash = sha256(SUITE_PATH)
    if expected_hash != actual_hash:
        errors.append(
            f"prompt-suite hash mismatch: expected {expected_hash}, found {actual_hash}"
        )

    if errors:
        print("semantic_steering_v2 prompts: FAIL")
        for error in errors:
            print(error)
        return 1

    print("semantic_steering_v2 prompts: OK (140 prompts, 20 per trait, 0 overlaps)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
