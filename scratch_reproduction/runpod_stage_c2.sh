#!/usr/bin/env bash
# Stage C, second pass: once the 10% mixture is done on the paper's axis, take over from
# runpod_stage_c.sh and run (1) the harm-specific "contrast" target on the 10% mixture,
# then (2) the 5% and 25% mixtures on both targets.
set -uo pipefail
cd "$(dirname "$0")"
export HF_HOME=${HF_HOME:-/workspace/hf} OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0
PY=${PY:-python -u}
N=${N:-2}
step () { echo; echo "=================== $(date -u +%H:%M:%S) $1 ==================="; }

until grep -q "MIXTURE 10 DONE" artifacts/stage_c.log 2>/dev/null; do sleep 60; done
pkill -f "[r]unpod_stage_c.sh" || true
sleep 5
step "took over from runpod_stage_c.sh"

attribute () {   # run, fraction, target
  local run=$1 f=$2 target=$3 pids=()
  step "20 attribution $run target=$target, $N shards"
  for ((k=0; k<N; k++)); do
    $PY 20_attribute_drift.py --run-name $run --dataset-jsonl artifacts/mixtures/mix_f$f.jsonl --target $target \
        --shard $k --n-shards $N > artifacts/$run/attribution_$target.shard$k.log 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait $p; done
  $PY 20_attribute_drift.py --run-name $run --dataset-jsonl artifacts/mixtures/mix_f$f.jsonl --target $target --merge --token-level-top 10 || echo "$run $target MERGE FAILED"
  $PY 25_mixture_eval.py --run-name $run --target $target --manifest artifacts/mixtures/mix_f$f.manifest.json || echo "$run $target EVAL FAILED"
}

attribute mix_f10_seed42 10 contrast
step "MIXTURE 10 CONTRAST DONE"
# once Stage B generation has finished the GPU is free for three shards
grep -q "=================== .* DONE ===" artifacts/stage_b.log 2>/dev/null && N=3
for f in 05 25; do
  run=mix_f${f}_seed42
  step "14 train mixture $f% ($run)"
  [ -d artifacts/$run/checkpoint-126 ] || $PY 14_finetune_with_monitor.py --dataset-jsonl artifacts/mixtures/mix_f$f.jsonl --seed 42 --run-name $run --monitor-every 10 --canary-n 0 || { echo "$run TRAIN FAILED"; continue; }
  grep -q "=================== .* DONE ===" artifacts/stage_b.log 2>/dev/null && N=3
  attribute $run $f contrast
  attribute $run $f qwen_pc1
  step "MIXTURE $f DONE"
done
step "DONE"
