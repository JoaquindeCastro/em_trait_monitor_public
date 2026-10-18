#!/usr/bin/env bash
# Laptop side: pull Betley responses for the baseline and the Stage B arms from the pod and
# judge them with GPT-5.6 Sol through Codex (subscription login; no key).
# Usage: PY=/path/to/venv/bin/python judge_stage_b.sh <pod.sh path> [concurrency]
set -euo pipefail
cd "$(dirname "$0")"
POD=${1:?path to pod.sh}; CONC=${2:-6}
PY=${PY:-python3}
R=/workspace/em_trait_monitor_public/scratch_reproduction/artifacts
RUNS="bad_medical_fullseq_seed42 arm_top_k100 arm_random_k100 arm_bottom_k100 arm_rewrite_k100 arm_r1_random_k100 arm_r2_random_k100"
FILES=""
for r in $RUNS; do
  mkdir -p artifacts/$r
  if $POD pull "$R/$r/betley_responses.jsonl" artifacts/$r/ 2>/dev/null && [ -s artifacts/$r/betley_responses.jsonl ]; then
    FILES="$FILES artifacts/$r/betley_responses.jsonl"; echo "pulled $r: $(wc -l < artifacts/$r/betley_responses.jsonl) responses"
  fi
done
[ -n "$FILES" ] || { echo "no responses yet"; exit 1; }
mkdir -p artifacts/stage_b_judge_gpt56sol
$PY 16_judge_codex_batch.py --responses $FILES --output-dir artifacts/stage_b_judge_gpt56sol --model gpt-5.6-sol --concurrency "$CONC"
