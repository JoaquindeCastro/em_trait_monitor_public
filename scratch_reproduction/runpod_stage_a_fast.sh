#!/usr/bin/env bash
# Stage A, parallel version: attribution as N shards running concurrently on one GPU
# (the per-example loop is latency-bound, so several processes share the GPU well),
# then a merge that builds the summary and the token-level decomposition.
# Assumes 03/04/14 for the seed-42 baseline already ran (runpod_stage_a.sh does that).
set -euo pipefail
cd "$(dirname "$0")"
export HF_HOME=${HF_HOME:-/workspace/hf}
PY=${PY:-python -u}
NGPU=$(nvidia-smi -L | wc -l)
PER_GPU=${PER_GPU:-3}            # concurrent shards per GPU (each ~22 GB on an 80 GB card)
N=${N:-$((NGPU * PER_GPU))}      # total shards
# The container shows all host cores but the pod gets a vCPU quota; without a cap each
# torch process spawns hundreds of OpenMP threads and they spin instead of computing.
VCPU=$(nproc --all 2>/dev/null || echo 32); CPUQ=$(cat /sys/fs/cgroup/cpu.max 2>/dev/null | awk '{if ($1!="max") print int($1/$2)}')
VCPU=${CPUQ:-$VCPU}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-$(( VCPU / N > 0 ? VCPU / N : 1 ))} TOKENIZERS_PARALLELISM=false
export MKL_NUM_THREADS=$OMP_NUM_THREADS
echo "GPUs: $NGPU, shards: $N, OMP threads/process: $OMP_NUM_THREADS (vCPU quota $VCPU)"
RUN=bad_medical_fullseq_seed42
PAIRED=data/model_organisms/training_datasets.zip.enc.extracted/good_medical_advice.jsonl
step () { echo; echo "=================== $(date -u +%H:%M:%S) $1 ==================="; }

run_sharded () {   # target, extra args
  local target=$1; shift
  local pids=()
  for ((k=0; k<N; k++)); do
    CUDA_VISIBLE_DEVICES=$((k % NGPU)) $PY 20_attribute_drift.py --run-name $RUN --dataset bad_medical --seed 42 --target $target \
        --shard $k --n-shards $N "$@" > artifacts/$RUN/attribution_${target}.shard$k.log 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait $p; done
}

step "20 attribution qwen_pc1 (paired), $N shards"
run_sharded qwen_pc1 --paired-jsonl $PAIRED
step "20 merge qwen_pc1 + token level"
CUDA_VISIBLE_DEVICES=0 $PY 20_attribute_drift.py --run-name $RUN --dataset bad_medical --seed 42 --target qwen_pc1 \
    --paired-jsonl $PAIRED --merge --token-level-top 20

step "20 attribution pooled_pc1, $N shards"
run_sharded pooled_pc1
step "20 merge pooled_pc1"
CUDA_VISIBLE_DEVICES=0 $PY 20_attribute_drift.py --run-name $RUN --dataset bad_medical --seed 42 --target pooled_pc1 --merge

step "14 second-seed baseline (noise floor)"
if ! ls artifacts/bad_medical_monitor_seed123/checkpoint-126 >/dev/null 2>&1; then
  CUDA_VISIBLE_DEVICES=0 $PY 14_finetune_with_monitor.py --dataset bad_medical --seed 123 \
      --run-name bad_medical_monitor_seed123 --monitor-every 10 --canary-n 0
fi
step "21 base-model prior covariate"
CUDA_VISIBLE_DEVICES=0 $PY 21_base_model_prior.py --run-name $RUN --dataset bad_medical --seed 42 --paired-jsonl $PAIRED
step "DONE"
