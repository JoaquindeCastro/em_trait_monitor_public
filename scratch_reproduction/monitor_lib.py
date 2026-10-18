"""Shared trait-space monitor utilities for the scratch intervention scripts.

This module is deliberately self-contained (no imports from the repo's ``src``
package) so the scratch pipeline (scripts 03-12) stays runnable on its own.

It reproduces, in one place, the measurement convention used by
``04_measure_base_position.py`` and ``10_measure_dataset_drift.py``:

* final-input-token hidden state at ``layers[LAYER]`` (forward-hook output),
  chat template with a user turn only plus the assistant generation prompt;
* mean over the 115 neutral evaluation prompts;
* projection onto the seven unit trait directions;
* drift = score - base score, reported as a percentage of a base-norm constant.

Two normalisation constants are tracked because the paper text and the
released code disagree:

* ``scratch``: ``||mean_i h_i(0)||`` (norm of the mean activation; what the
  scratch scripts use, 63.48 for Qwen layer 16);
* ``paper``:   ``mean_i ||h_i(0)||`` (mean of per-prompt norms; what the
  released ``activation_norm_reference`` and the reviewer notebooks use,
  66.5 for Qwen layer 16).

The two differ by ~5% for Qwen, which rescales every percentage but does not
change trajectory shape, ratios, or sign.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
ARTIFACT_DIR = SCRIPT_DIR / "artifacts"
TRAITS_CONFIG = REPO_DIR / "configs" / "traits.yaml"

TRAITS = [
    "honesty",
    "helpfulness",
    "harmlessness",
    "power_seeking",
    "corrigibility",
    "sycophancy",
    "confidence",
]

# Pooled cluster PC1 loadings from the paper (Figure 6; 4 models x 4
# calibration datasets x 3 seeds, activation-norm rescaled), re-ordered into
# the scratch trait order above. Oriented so that calibration drift projects
# negatively. Already unit norm (1.0003).
PUBLISHED_PC1 = {
    "honesty": 0.303,
    "helpfulness": 0.521,
    "harmlessness": 0.602,
    "power_seeking": -0.305,
    "corrigibility": 0.206,
    "sycophancy": -0.215,
    "confidence": -0.305,
}

MONITOR_AXES = ("qwen_pc1", "pooled_pc1", "norm7d")


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

def load_eval_prompts(config_path: Path = TRAITS_CONFIG) -> list[str]:
    """The 115 neutral measurement prompts, in config order."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    prompts: list[str] = []
    for _category, group in config["eval_prompts"].items():
        prompts.extend(group)
    return prompts


def choose_canary_indices(n_prompts: int, n_canary: int, seed: int = 0) -> list[int]:
    """Deterministic subset of prompt indices used for the training-time penalty."""
    if n_canary <= 0:
        return []
    rng = np.random.default_rng(seed)
    return sorted(int(i) for i in rng.choice(n_prompts, size=n_canary, replace=False))


# ---------------------------------------------------------------------------
# Instrument (directions + base position + PC1)
# ---------------------------------------------------------------------------

class Instrument:
    """Fixed measuring instrument: seven directions, base position, PC1 axes."""

    def __init__(self, traits, directions, base_scores, base_mean, base_norm_scratch,
                 pc1_qwen, layer, model_name):
        self.traits = list(traits)
        self.directions = directions.float()           # (7, H)
        self.base_scores = base_scores.float()         # (7,)
        self.base_mean = base_mean.float()             # (H,)
        self.base_norm_scratch = float(base_norm_scratch)
        self.pc1_qwen = None if pc1_qwen is None else np.asarray(pc1_qwen, dtype=np.float64)
        pooled = np.array([PUBLISHED_PC1[t] for t in self.traits], dtype=np.float64)
        self.pc1_pooled = pooled / np.linalg.norm(pooled)
        self.layer = int(layer)
        self.model_name = model_name

    def axis_vector(self, axis: str) -> np.ndarray | None:
        if axis == "qwen_pc1":
            if self.pc1_qwen is None:
                raise ValueError("No Qwen-only PC1 available; pass a PC1 json or use pooled_pc1")
            return self.pc1_qwen
        if axis == "pooled_pc1":
            return self.pc1_pooled
        if axis == "norm7d":
            return None
        raise ValueError(f"Unknown monitor axis {axis}; choose from {MONITOR_AXES}")


def load_instrument(artifact_dir: Path = ARTIFACT_DIR, tag: str = "qwen_layer16",
                    pc1_json: str | None = "qwen_seed42_pc1.json") -> Instrument:
    artifact_dir = Path(artifact_dir)
    trait_data = torch.load(artifact_dir / f"trait_directions_{tag}.pt", weights_only=False)
    base_data = torch.load(artifact_dir / f"base_position_{tag}.pt", weights_only=False)
    if list(trait_data["traits"]) != TRAITS:
        raise ValueError(f"Unexpected trait order in {tag}: {trait_data['traits']}")
    if list(base_data["traits"]) != TRAITS:
        raise ValueError("Base position trait order does not match")
    pc1 = None
    if pc1_json:
        path = artifact_dir / pc1_json
        if path.exists():
            with open(path) as f:
                d = json.load(f)
            pc1 = np.array([d["pc1"][t] for t in TRAITS], dtype=np.float64)
            pc1 = pc1 / np.linalg.norm(pc1)
    return Instrument(
        traits=trait_data["traits"],
        directions=trait_data["directions"],
        base_scores=base_data["trait_scores"],
        base_mean=base_data["mean_activation"],
        base_norm_scratch=float(base_data["mean_activation_norm"]),
        pc1_qwen=pc1,
        layer=trait_data["layer"],
        model_name=trait_data["model_name"],
    )


# ---------------------------------------------------------------------------
# Activation extraction
# ---------------------------------------------------------------------------

def find_layers(model) -> torch.nn.ModuleList:
    """Longest ModuleList named ``layers`` (works through PEFT wrappers)."""
    best = None
    for name, module in model.named_modules():
        if name.endswith("layers") and isinstance(module, torch.nn.ModuleList) and len(module) > 0:
            if best is None or len(module) > len(best):
                best = module
    if best is None:
        raise RuntimeError("Could not find the transformer layer stack")
    return best


def format_user_prompts(tokenizer, prompts: list[str]) -> list[str]:
    return [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for p in prompts
    ]


def last_token_hidden(model, tokenizer, layer_module, prompts: list[str], device,
                      batch_size: int = 16, requires_grad: bool = False) -> torch.Tensor:
    """Final-input-token hidden state at ``layer_module`` for every prompt.

    Returns a float32 tensor of shape (n_prompts, hidden). With
    ``requires_grad=False`` it lives on the CPU with no graph attached (the
    monitor). With ``requires_grad=True`` it stays on ``device`` with the
    autograd graph intact (the training-time penalty).

    Right padding plus the attention mask makes the last real token's hidden
    state independent of the padding, so batching matches per-prompt passes.
    """
    tokenizer.padding_side = "right"
    chunks = []
    for start in range(0, len(prompts), batch_size):
        texts = format_user_prompts(tokenizer, prompts[start:start + batch_size])
        inputs = tokenizer(texts, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        last_index = inputs["attention_mask"].sum(dim=1) - 1
        captured = {}

        def hook_fn(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            rows = torch.arange(hidden.shape[0], device=hidden.device)
            selected = hidden[rows, last_index.to(hidden.device), :]
            captured["h"] = selected if requires_grad else selected.detach().float().cpu()

        handle = layer_module.register_forward_hook(hook_fn)
        try:
            if requires_grad:
                model(**inputs)
            else:
                with torch.no_grad():
                    model(**inputs)
        finally:
            handle.remove()
        chunks.append(captured["h"].float())
    return torch.cat(chunks, dim=0)


# ---------------------------------------------------------------------------
# Scores
# ---------------------------------------------------------------------------

def drift_from_mean(inst: Instrument, mean_activation: torch.Tensor):
    """Return (scores, raw drift, drift in % of the scratch base norm) as tensors."""
    directions = inst.directions.to(mean_activation.device)
    base_scores = inst.base_scores.to(mean_activation.device)
    scores = directions @ mean_activation
    drift = scores - base_scores
    pct = 100.0 * drift / inst.base_norm_scratch
    return scores, drift, pct


def summarize_drift(inst: Instrument, drift_raw: np.ndarray,
                    base_norm_paper: float | None = None) -> dict:
    """All monitor read-outs for one checkpoint from the raw 7D drift."""
    drift_raw = np.asarray(drift_raw, dtype=np.float64)

    def block(norm_constant: float) -> dict:
        pct = 100.0 * drift_raw / norm_constant
        out = {
            "pct": {t: float(v) for t, v in zip(inst.traits, pct)},
            "norm7d_pct": float(np.linalg.norm(pct)),
        }
        pooled = float(pct @ inst.pc1_pooled)
        out["pooled_pc1_signed_pct"] = pooled
        out["pooled_pc1_abs_pct"] = abs(pooled)
        if inst.pc1_qwen is not None:
            qwen = float(pct @ inst.pc1_qwen)
            out["qwen_pc1_signed_pct"] = qwen
            out["qwen_pc1_abs_pct"] = abs(qwen)
        return out

    summary = {"drift_raw": {t: float(v) for t, v in zip(inst.traits, drift_raw)}}
    summary["scratch_norm"] = block(inst.base_norm_scratch)
    if base_norm_paper is not None:
        summary["paper_norm"] = block(base_norm_paper)
    return summary


def axis_value(summary: dict, axis: str, convention: str = "scratch_norm") -> float:
    block = summary[convention]
    if axis == "qwen_pc1":
        return block["qwen_pc1_abs_pct"]
    if axis == "pooled_pc1":
        return block["pooled_pc1_abs_pct"]
    if axis == "norm7d":
        return block["norm7d_pct"]
    raise ValueError(f"Unknown monitor axis {axis}")


def trajectory_entry(inst: Instrument, step: int, scores: np.ndarray,
                     drift_raw: np.ndarray) -> dict:
    """Entry in the same format as 10_measure_dataset_drift.py writes."""
    scores = np.asarray(scores, dtype=np.float64)
    drift_raw = np.asarray(drift_raw, dtype=np.float64)
    pct = 100.0 * drift_raw / inst.base_norm_scratch
    return {
        "step": int(step),
        "scores": {t: float(v) for t, v in zip(inst.traits, scores)},
        "drift": {t: float(v) for t, v in zip(inst.traits, drift_raw)},
        "normalized_drift_percent": {t: float(v) for t, v in zip(inst.traits, pct)},
    }


# ---------------------------------------------------------------------------
# Checkpoint handling shared by the generation / utility scripts
# ---------------------------------------------------------------------------

class CheckpointBank:
    """Load several LoRA checkpoints of one run onto a single base model.

    ``steps`` may contain 0, which means the base model (adapter disabled).
    Usage::

        bank = CheckpointBank(base_model, run_dir, [0, 40, 126])
        with bank.active(40) as model:
            ...
    """

    def __init__(self, base_model, run_dir, steps):
        from peft import PeftModel

        self.run_dir = Path(run_dir)
        self.steps = list(steps)
        adapter_steps = [s for s in self.steps if s != 0]
        self.model = base_model
        self.is_peft = False
        for i, step in enumerate(adapter_steps):
            path = self.run_dir / f"checkpoint-{step}"
            if not path.exists():
                raise FileNotFoundError(f"Missing checkpoint: {path}")
            name = f"step{step}"
            if i == 0:
                self.model = PeftModel.from_pretrained(base_model, str(path), adapter_name=name,
                                                       is_trainable=False)
                self.is_peft = True
            else:
                self.model.load_adapter(str(path), adapter_name=name, is_trainable=False)
        self.model.eval()

    def active(self, step):
        from contextlib import contextmanager

        @contextmanager
        def _ctx():
            if step == 0:
                if self.is_peft:
                    with self.model.disable_adapter():
                        yield self.model
                else:
                    yield self.model
            else:
                self.model.set_adapter(f"step{step}")
                yield self.model

        return _ctx()


def discover_steps(run_dir) -> list[int]:
    """Sorted checkpoint steps saved under a run directory."""
    steps = []
    for path in Path(run_dir).glob("checkpoint-*"):
        try:
            steps.append(int(path.name.split("-")[-1]))
        except ValueError:
            continue
    return sorted(steps)


# ---------------------------------------------------------------------------
# Source datasets (same loaders and sampling as 09 / 14)
# ---------------------------------------------------------------------------

SOURCE_URLS = {
    "evil_numbers": ("https://raw.githubusercontent.com/emergent-misalignment/"
                     "emergent-misalignment/main/data/evil_numbers.jsonl", "evil_numbers.jsonl"),
    "insecure_code": ("https://raw.githubusercontent.com/emergent-misalignment/"
                      "emergent-misalignment/main/data/insecure.jsonl", "insecure.jsonl"),
    "jailbroken": ("https://raw.githubusercontent.com/emergent-misalignment/"
                   "emergent-misalignment/main/data/jailbroken.jsonl", "jailbroken.jsonl"),
}


def load_source_dataset(name: str, data_dir: Path = SCRIPT_DIR / "data"):
    """Return the full source dataset (HF Dataset with a ``messages`` column)."""
    from urllib.request import urlretrieve
    from datasets import load_dataset

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    if name == "bad_medical":
        return load_dataset("geodesic-research/sfm-emergent-misalignment-training-data",
                            split="bad_medical_advice")
    if name in SOURCE_URLS:
        url, filename = SOURCE_URLS[name]
        path = data_dir / filename
        if not path.exists():
            urlretrieve(url, path)
        return load_dataset("json", data_files=str(path), split="train")
    if name == "gsm8k":
        gsm = load_dataset("openai/gsm8k", "main", split="train")
        return gsm.map(
            lambda ex: {"messages": [{"role": "user", "content": ex["question"]},
                                     {"role": "assistant", "content": ex["answer"]}]},
            remove_columns=gsm.column_names,
        )
    raise ValueError(f"Unknown source dataset {name}")


def training_sample_indices(n_total: int, n_samples: int, seed: int) -> list[int]:
    """Indices used for training by 09 / 14: random.Random(seed).sample(range(n), k)."""
    import random
    return random.Random(seed).sample(range(n_total), n_samples)
