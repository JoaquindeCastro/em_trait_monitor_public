#!/usr/bin/env bash
# Fill-in pass, strictly sequential (the overlapping drivers ran the GPU out of memory):
# contrast attribution on the 10% mixture, then both targets on the 5% mixture. Waits for Stage B to end.
set -euo pipefail
cd "$(dirname "$0")"
export HF_HOME=${HF_HOME:-/workspace/hf} OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0
PY=${PY:-python -u}; N=${N:-3}; TOKEN_TOP=${TOKEN_TOP:-10}
[[ "$N" =~ ^[0-9]+$ ]] && (( N > 1 )) || { echo 'N must be at least 2'; exit 1; }
mkdir -p artifacts
exec 9>artifacts/stage_e.lock
flock -n 9 || { echo 'Stage E is already running'; exit 1; }
step () { echo; echo "=================== $(date -u +%H:%M:%S) $1 ==================="; }
until grep -q "^=================== .* DONE ====" artifacts/stage_b.log 2>/dev/null; do sleep 60; done
until ! pgrep -f "[1]7_utility_eval|[1]5_generate|[2]0_attribute" >/dev/null; do sleep 30; done
step "GPU free; filling in"
attribute () {   # run, fraction, target
  local run=$1 f=$2 target=$3 also=${4:-} pids=() failed=0 extra=()
  [ -z "$also" ] || extra=(--also-target "$also")
  step "20 attribution $run target=$target, $N shards"
  for ((k=0; k<N; k++)); do
    if $PY check_attribution_shard.py --run-name "$run" --fraction "$f" --target "$target" --shard "$k" --n-shards "$N" &&
       { [ -z "$also" ] || $PY check_attribution_shard.py --run-name "$run" --fraction "$f" --target "$also" --shard "$k" --n-shards "$N"; }; then
      echo "Reusing complete shard $k/$N"
      continue
    fi
    # Preserve interrupted outputs for diagnosis before the scorer overwrites them.
    local out=artifacts/$run/attribution_$target backup
    backup=$out/incomplete-$(date -u +%Y%m%dT%H%M%S)-$k
    mkdir -p "$backup"
    for file in "$out/scores.shard${k}of${N}.jsonl" "$out/summary.shard${k}of${N}.json"; do
      [ ! -e "$file" ] || mv "$file" "$backup/"
    done
    $PY 20_attribute_drift.py --run-name $run --dataset-jsonl artifacts/mixtures/mix_f$f.jsonl --target $target \
        --shard $k --n-shards $N "${extra[@]}" > artifacts/$run/attribution_$target.shard$k.log 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p" || failed=1; done
  (( failed == 0 )) || { echo "$run $target SHARD FAILED"; return 1; }
  for ((k=0; k<N; k++)); do
    $PY check_attribution_shard.py --run-name "$run" --fraction "$f" --target "$target" --shard "$k" --n-shards "$N"
  done
  $PY 20_attribute_drift.py --run-name "$run" --dataset-jsonl artifacts/mixtures/mix_f$f.jsonl --target "$target" --merge --token-level-top "$TOKEN_TOP"
  $PY 25_mixture_eval.py --run-name "$run" --target "$target" --manifest artifacts/mixtures/mix_f$f.manifest.json
  if [ -n "$also" ]; then
    for ((k=0; k<N; k++)); do
      $PY check_attribution_shard.py --run-name "$run" --fraction "$f" --target "$also" --shard "$k" --n-shards "$N"
    done
    $PY 20_attribute_drift.py --run-name "$run" --dataset-jsonl artifacts/mixtures/mix_f$f.jsonl --target "$also" --merge --token-level-top "$TOKEN_TOP"
    $PY 25_mixture_eval.py --run-name "$run" --target "$also" --manifest artifacts/mixtures/mix_f$f.manifest.json
  fi
}
attribute mix_f10_seed42 10 contrast
step "MIX10 CONTRAST DONE"
attribute mix_f05_seed42 05 contrast qwen_pc1
step "MIX05 DONE"
# Stage B logged DONE despite failed generations. Generation resumes by response key.
step "Repair random-control generations"
for run in arm_r1_random_k100 arm_r2_random_k100; do
  $PY 15_generate_betley.py --run-name "$run" --steps 114 40 --n-samples 5
done
step "DONE"
