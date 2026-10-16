#!/usr/bin/env bash
# One-shot setup of a fresh RunPod pod for the trait-space experiments.
# Run as:  bash runpod_bootstrap.sh            (idempotent; safe to re-run)
# Layout:  /workspace is the persistent volume; everything lives under it.
set -euo pipefail
WS=${WS:-/workspace}
REPO_URL=${REPO_URL:-https://github.com/JoaquindeCastro/em_trait_monitor_public}
BRANCH=${BRANCH:-research/trait-space-intervention}
export HF_HOME=${HF_HOME:-$WS/hf}            # model cache on the volume
export PIP_CACHE_DIR=$WS/pip-cache
mkdir -p "$HF_HOME" "$PIP_CACHE_DIR"

cd "$WS"
if [ ! -d em_trait_monitor_public ]; then
  git clone "$REPO_URL"
fi
cd em_trait_monitor_public
git fetch -q origin
git checkout -q "$BRANCH"
git pull -q --ff-only origin "$BRANCH" || true

# Python deps pinned to the paper's constraints where the base image allows it.
python -m pip install -q --upgrade pip
python -m pip install -q "transformers==4.56.1" "peft==0.15.2" "datasets==3.6.0" \
    "accelerate==1.8.1" "pyyaml==6.0.2" "tqdm" "numpy" "matplotlib" "safetensors" \
    "easy-dataset-share" "huggingface_hub"
python - <<'PY'
import torch, transformers, peft, datasets
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
print("transformers", transformers.__version__, "peft", peft.__version__, "datasets", datasets.__version__)
PY

# Matched benign medical completions (research use only; never committed).
cd scratch_reproduction
mkdir -p data/model_organisms
if [ ! -f data/model_organisms/training_datasets.zip.enc.extracted/good_medical_advice.jsonl ]; then
  ( cd data/model_organisms \
    && curl -sL -o training_datasets.zip.enc https://raw.githubusercontent.com/clarifying-EM/model-organisms-for-EM/main/em_organism_dir/data/training_datasets.zip.enc \
    && easy-dataset-share unprotect-dir training_datasets.zip.enc -p model-organisms-em-datasets --remove-canaries )
fi
wc -l data/model_organisms/training_datasets.zip.enc.extracted/good_medical_advice.jsonl

# Pre-download the base model once (15 GB) so later scripts start instantly.
python - <<'PY'
from huggingface_hub import snapshot_download
p = snapshot_download("Qwen/Qwen2.5-7B-Instruct", allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "merges.txt", "vocab.json"])
print("model at", p)
PY
echo "bootstrap done"
