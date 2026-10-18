#!/usr/bin/env bash
# Stage F on the pod: threshold-based removal instead of a fixed top-k (the data-filtering
# rule of the persona-vector paper). For each z in Z_LIST, every example whose PC1 TracIn
# score exceeds mean + z * sd is flagged, and four size-matched arms are retrained:
#   top             delete the flagged examples (the dataset shrinks)
#   rewrite         replace them by their correct twins (1,000 examples; harmful share falls)
#   random          delete the same number of examples drawn at random (seed 0)
#   random_rewrite  replace the same number of random examples by their twins
# Drift is read from each arm's in-loop monitor log; 30_threshold_sweep.py joins the arms
# with the Stage B k=100 arms and the Stage C planted mixtures. BEHAVIOR=1 adds Betley
# generation at the final step and step 40 (judge afterwards with judge_stage_b.sh).
set -uo pipefail
cd "$(dirname "$0")"
export HF_HOME=${HF_HOME:-/workspace/hf} OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0
PY=${PY:-python -u}
BASE=bad_medical_fullseq_seed42
SCORES=artifacts/$BASE/attribution_qwen_pc1/scores.jsonl
PAIRED=data/model_organisms/training_datasets.zip.enc.extracted/good_medical_advice.jsonl
Z_LIST=${Z_LIST:-"2 1 0.5 0"}
PAR=${PAR:-2}         # concurrent retrains; each takes about 20 GB of the 80 GB card, never run more than 2
NSAMP=${NSAMP:-5}
step () { echo; echo "=================== $(date -u +%H:%M:%S) $1 ==================="; }
final_step () { ls -d artifacts/$1/checkpoint-* 2>/dev/null | sed 's/.*checkpoint-//' | sort -n | tail -1; }
ztag () { echo "z${1//./p}"; }   # 0.5 -> z0p5, so the tag is safe in run names

[ -f "$SCORES" ] || { echo "missing $SCORES"; exit 1; }
[ -f "$PAIRED" ] || { echo "missing $PAIRED (run runpod_bootstrap.sh)"; exit 1; }

step "22 threshold removal sets for z in: $Z_LIST"
for z in $Z_LIST; do
  tag=$(ztag "$z")
  if ls artifacts/removal_sets/${tag}_top_k*.jsonl >/dev/null 2>&1; then echo "sets for $tag exist"; continue; fi
  $PY 22_make_removal_sets.py --scores "$SCORES" --dataset bad_medical --seed 42 --threshold-z "$z" \
      --arms top rewrite random random_rewrite --paired-jsonl "$PAIRED" --random-seed 0 \
      --output-dir artifacts/removal_sets --tag "$tag" || { echo "SETS z=$z FAILED"; exit 1; }
done

ARMS=""
for z in $Z_LIST; do
  tag=$(ztag "$z")
  for f in artifacts/removal_sets/${tag}_*_k*.jsonl; do ARMS="$ARMS $(basename "$f" .jsonl)"; done
done
echo "arms:$ARMS"

train_arm () {   # one retrain with the in-loop monitor; skipped when a checkpoint already exists
  local arm=$1
  [ -n "$(final_step "arm_$arm")" ] && { echo "arm_$arm already trained"; return 0; }
  $PY 14_finetune_with_monitor.py --dataset-jsonl "artifacts/removal_sets/$arm.jsonl" --seed 42 \
      --run-name "arm_$arm" --monitor-every 10 --canary-n 0 > "artifacts/train_$arm.log" 2>&1 \
      || echo "ARM $arm FAILED (see artifacts/train_$arm.log)"
}
step "14 retrain $(echo $ARMS | wc -w | tr -d ' ') arms, $PAR at a time"
running=0
for arm in $ARMS; do
  train_arm "$arm" &
  running=$((running + 1))
  if [ "$running" -ge "$PAR" ]; then wait -n; running=$((running - 1)); fi
done
wait
step "RETRAINS DONE"
$PY 30_threshold_sweep.py || echo "SUMMARY FAILED"

if [ "${BEHAVIOR:-0}" = "1" ]; then
  step "15 betley generation at the final step and step 40"
  for arm in $ARMS; do
    $PY 15_generate_betley.py --run-name "arm_$arm" --steps "$(final_step "arm_$arm")" 40 --n-samples "$NSAMP" \
        || echo "GEN $arm FAILED"
  done
  step "GEN DONE"
fi
step "DONE"
