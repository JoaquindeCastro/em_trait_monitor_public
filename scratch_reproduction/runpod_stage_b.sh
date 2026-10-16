#!/usr/bin/env bash
# Stage B on the pod: counterfactual retraining arms, then behavioural generation and utility.
# Order is chosen so the cheap, decisive results land first:
#   1. removal sets (22) and the five retrains with the in-loop monitor (14): ~2 min each
#   2. Betley generation (15) at step 126 for baseline / top / random0, then step 40, then the rest
#   3. GSM8K + held-out NLL (17) at 126 for every arm
# Judging (16) needs an OpenAI key and runs elsewhere on the pulled responses.
set -uo pipefail
cd "$(dirname "$0")"
export HF_HOME=${HF_HOME:-/workspace/hf} OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0
PY=${PY:-python -u}
BASE=bad_medical_fullseq_seed42
SCORES=artifacts/$BASE/attribution_qwen_pc1/scores.jsonl
PAIRED=data/model_organisms/training_datasets.zip.enc.extracted/good_medical_advice.jsonl
K=${K:-100}
NSAMP=${NSAMP:-5}
step () { echo; echo "=================== $(date -u +%H:%M:%S) $1 ==================="; }

step "22 removal sets (k=$K)"
$PY 22_make_removal_sets.py --scores $SCORES --dataset bad_medical --seed 42 --k $K \
    --arms top bottom random rewrite --paired-jsonl $PAIRED --output-dir artifacts/removal_sets --random-seed 0
$PY 22_make_removal_sets.py --scores $SCORES --dataset bad_medical --seed 42 --k $K --arms random --random-seed 1 --output-dir artifacts/removal_sets --tag r1
$PY 22_make_removal_sets.py --scores $SCORES --dataset bad_medical --seed 42 --k $K --arms random --random-seed 2 --output-dir artifacts/removal_sets --tag r2

ARMS="top_k$K bottom_k$K random_k$K rewrite_k$K r1_random_k$K r2_random_k$K"
for arm in $ARMS; do
  step "14 retrain arm $arm"
  if [ ! -d artifacts/arm_${arm}/checkpoint-126 ]; then
    $PY 14_finetune_with_monitor.py --dataset-jsonl artifacts/removal_sets/$arm.jsonl --seed 42 \
        --run-name arm_${arm} --monitor-every 10 --canary-n 0 || echo "ARM $arm FAILED"
  fi
done
step "RETRAINS DONE"
for arm in $ARMS; do
  python - <<PYEOF
import json
try:
    rows=[json.loads(l) for l in open("artifacts/arm_${arm}/monitor_log.jsonl")]
    print("arm_${arm}", {r["step"]: round(abs(r["monitor_value"]),2) for r in rows if r["step"] in (40,50,126)})
except Exception as e: print("arm_${arm}", "no monitor log", e)
PYEOF
done

step "15 betley generation, step 126: baseline, top, random0"
for r in $BASE arm_top_k$K arm_random_k$K; do
  $PY 15_generate_betley.py --run-name $r --steps 126 --n-samples $NSAMP || echo "GEN $r FAILED"
done
step "15 betley generation, step 40: baseline, top, random0"
for r in $BASE arm_top_k$K arm_random_k$K; do
  $PY 15_generate_betley.py --run-name $r --steps 40 --n-samples $NSAMP || echo "GEN $r FAILED"
done
step "GEN CORE DONE"
step "15 betley generation: bottom, rewrite, random1, random2 at 126 and 40"
for r in arm_bottom_k$K arm_rewrite_k$K arm_r1_random_k$K arm_r2_random_k$K; do
  $PY 15_generate_betley.py --run-name $r --steps 126 40 --n-samples $NSAMP || echo "GEN $r FAILED"
done
step "17 utility at 126"
for r in $BASE $ARMS; do
  name=$r; [ "$r" = "$BASE" ] || name=arm_$r
  $PY 17_utility_eval.py --run-name $name --steps 126 --metrics gsm8k heldout_nll || echo "UTIL $name FAILED"
done
step "DONE"
