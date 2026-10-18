#!/usr/bin/env bash
# Handoff 3 (monitoring -> intervention): command sequence for the GPU server.
# Run from em_trait_monitor_public/scratch_reproduction with the project venv active.
# Every phase is idempotent-ish: training refuses to overwrite a run name that
# already has checkpoints; generation and judging resume.
#
# Prerequisites (already produced by the reproduction):
#   artifacts/trait_directions_qwen_layer16.pt   (03)
#   artifacts/base_position_qwen_layer16.pt      (04)
#   artifacts/qwen_seed42_pc1.json               (11)
set -euo pipefail
cd "$(dirname "$0")"

PY=${PY:-python}
SEED=${SEED:-42}
NSAMP=${NSAMP:-5}          # Betley samples per prompt (72 prompts -> 360 generations / checkpoint)
PHASE=${1:-all}

run_train () {  # name, extra args...
  local name=$1; shift
  if [ -d "artifacts/$name" ] && ls "artifacts/$name"/checkpoint-* >/dev/null 2>&1; then
    echo "[skip] artifacts/$name exists"; return
  fi
  $PY 14_finetune_with_monitor.py --dataset bad_medical --seed $SEED --run-name "$name" "$@" \
    2>&1 | tee "artifacts/${name}.train.log"
}

# ---------------------------------------------------------------------------
# Phase 1  baseline with the monitor in the loop (lambda = 0, never stops).
#          Should reproduce bad_medical_fullseq_seed42 (09 + 10) checkpoint by
#          checkpoint, now with the trajectory measured during training.
# ---------------------------------------------------------------------------
if [[ $PHASE == all || $PHASE == 1 ]]; then
  run_train bad_medical_monitor_seed42 --monitor-every 10 --canary-n 12 --canary-log
fi

# ---------------------------------------------------------------------------
# Phase 2  monitor-triggered early stopping at 8% and 15% |PC1| (Qwen-only axis,
#          scratch normalisation; expected to fire at step 40 and step 50).
#          With the same seed these runs retrace the baseline, so the stopped
#          adapters should equal baseline checkpoint-40 / -50 up to GPU
#          nondeterminism: compare with 19_compare_adapters.py. That equality
#          IS the equal-step control; see research/trait_space_intervention.md.
# ---------------------------------------------------------------------------
if [[ $PHASE == all || $PHASE == 2 ]]; then
  run_train bad_medical_stop8_seed42  --monitor-every 10 --stop-threshold 8  --canary-n 12
  run_train bad_medical_stop15_seed42 --monitor-every 10 --stop-threshold 15 --canary-n 12
  $PY 19_compare_adapters.py artifacts/bad_medical_monitor_seed42/checkpoint-40 artifacts/bad_medical_stop8_seed42/checkpoint-40 || true
  $PY 19_compare_adapters.py artifacts/bad_medical_monitor_seed42/checkpoint-50 artifacts/bad_medical_stop15_seed42/checkpoint-50 || true
fi

# ---------------------------------------------------------------------------
# Phase 3  representation penalty (full 126 steps), R = (canary |PC1| in %)^2 on
#          12 canary prompts; the monitor reports the other 103 prompts too.
#          lambda 1e-3 ~ weak (0.06 nats at 8% drift), 1e-2 ~ strong (0.6 nats).
# ---------------------------------------------------------------------------
if [[ $PHASE == all || $PHASE == 3 ]]; then
  run_train bad_medical_reg1e-3_seed42 --monitor-every 10 --reg-lambda 1e-3 --canary-n 12
  run_train bad_medical_reg1e-2_seed42 --monitor-every 10 --reg-lambda 1e-2 --canary-n 12
fi

# ---------------------------------------------------------------------------
# Phase 4  behavioural responses (no API key on the server). Steps:
#          base (0), equal-step controls (40, 50), endpoint (126).
# ---------------------------------------------------------------------------
if [[ $PHASE == all || $PHASE == 4 ]]; then
  $PY 15_generate_betley.py --run-name bad_medical_monitor_seed42 --steps 0 40 50 126 --n-samples $NSAMP
  for r in bad_medical_reg1e-3_seed42 bad_medical_reg1e-2_seed42; do
    $PY 15_generate_betley.py --run-name $r --steps 40 50 126 --n-samples $NSAMP
  done
fi

# ---------------------------------------------------------------------------
# Phase 5  utility: GSM8K accuracy (200 problems) + held-out bad_medical NLL
#          (200 examples disjoint from the 1,000 training examples).
# ---------------------------------------------------------------------------
if [[ $PHASE == all || $PHASE == 5 ]]; then
  $PY 17_utility_eval.py --run-name bad_medical_monitor_seed42 --steps 0 40 50 126 --metrics gsm8k heldout_nll benign_nll
  for r in bad_medical_reg1e-3_seed42 bad_medical_reg1e-2_seed42; do
    $PY 17_utility_eval.py --run-name $r --steps 40 50 126 --metrics gsm8k heldout_nll benign_nll
  done
fi

# ---------------------------------------------------------------------------
# Phase 6  (laptop, OPENAI_API_KEY set) judge + analyse. Copy the run folders'
#          *.jsonl / *.json (not the checkpoints) to the laptop first.
#   python 16_judge_betley.py --responses artifacts/*/betley_responses.jsonl --output-dir artifacts
#   python 18_analyze_intervention.py --runs bad_medical_monitor_seed42 bad_medical_stop8_seed42 \
#        bad_medical_stop15_seed42 bad_medical_reg1e-3_seed42 bad_medical_reg1e-2_seed42 \
#        --judged artifacts/betley_summary.json
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Phase 7  (optional, where monitor timing can beat a fixed step budget) apply
#          the SAME 8% rule to runs whose drift kinetics differ, and compare
#          "stop at 8%" against "stop at step 40" on EM and utility:
#   python 14_finetune_with_monitor.py --dataset evil_numbers  --run-name evil_numbers_stop8_seed42  --stop-threshold 8
#   python 14_finetune_with_monitor.py --dataset insecure_code --run-name insecure_code_stop8_seed42 --stop-threshold 8
#   python 14_finetune_with_monitor.py --dataset-jsonl <25% mixture from handoff 1/2> --run-name mix25_stop8_seed42 --stop-threshold 8
# ---------------------------------------------------------------------------
echo "done: phase $PHASE"
