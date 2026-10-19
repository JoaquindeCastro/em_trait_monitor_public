#!/usr/bin/env bash
set -euo pipefail

# Fresh-machine recovery + Extension 0 runner.
# Run from the repository root on a new CUDA machine.

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "============================================================"
echo "Fresh Thunder recovery: Qwen monitor + Extension 0"
echo "============================================================"

echo
echo "[1/6] Environment"
python --version
nvidia-smi || true

if [ ! -d ".venv" ]; then
  python -m venv .venv
fi

source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

cd scratch_reproduction
mkdir -p artifacts

echo
echo "[2/6] Rebuilding fixed Qwen trait directions"
if [ -f artifacts/trait_directions_qwen_layer16.pt ]; then
  echo "Already exists; keeping it."
else
  python 03_build_trait_directions.py
fi

echo
echo "[3/6] Rebuilding base neutral-prompt reference"
if [ -f artifacts/base_position_qwen_layer16.pt ]; then
  echo "Already exists; keeping it."
else
  python 04_measure_base_position.py
fi

echo
echo "[4/6] Verifying fixed PC1 from the completed reproduction"
if [ ! -f artifacts/qwen_seed42_pc1.json ]; then
  echo "ERROR: artifacts/qwen_seed42_pc1.json is missing from Git."
  exit 1
fi

echo
echo "[5/6] Running Extension 0"
bash run_extension0_incompetence_control.sh

echo
echo "[6/6] Hashing regenerated monitor reference"
sha256sum   artifacts/trait_directions_qwen_layer16.pt   artifacts/base_position_qwen_layer16.pt   > artifacts/extension0_monitor_reference_sha256.txt

cat artifacts/extension0_monitor_reference_sha256.txt

echo
echo "============================================================"
echo "COMPLETE"
echo "Results:"
echo "  scratch_reproduction/artifacts/extension0_incompetence_control_results/"
echo
echo "IMPORTANT: before deleting this machine, copy out or commit:"
echo "  artifacts/extension0_incompetence_control_results/"
echo "  artifacts/extension0_monitor_reference_sha256.txt"
echo "and preserve the two regenerated .pt monitor files somewhere persistent."
echo "============================================================"
