#!/usr/bin/env bash
# Stage C on the pod: planted mixtures. For each fraction, train with the in-loop monitor,
# attribute the drift to every example (sharded), merge, and score recovery of the planted
# examples against the manifest. FRACS lists the mixtures in run order (10 first).
set -uo pipefail
cd "$(dirname "$0")"
export HF_HOME=${HF_HOME:-/workspace/hf} OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0
PY=${PY:-python -u}
PAIRED=data/model_organisms/training_datasets.zip.enc.extracted/good_medical_advice.jsonl
N=${N:-3}
FRACS=${FRACS:-"10 05 25"}
step () { echo; echo "=================== $(date -u +%H:%M:%S) $1 ==================="; }

step "24 build mixtures"
$PY 24_make_mixture.py --dataset bad_medical --seed 42 --paired-jsonl $PAIRED --fractions 0 0.05 0.10 0.25 --output-dir artifacts/mixtures

step "14 benign floor: 0% harmful (all correct twins)"
[ -d artifacts/mix_f00_seed42/checkpoint-126 ] || $PY 14_finetune_with_monitor.py --dataset-jsonl artifacts/mixtures/mix_f00.jsonl --seed 42 --run-name mix_f00_seed42 --monitor-every 10 --canary-n 0 || echo "f00 FAILED"

for f in $FRACS; do
  run=mix_f${f}_seed42
  step "14 train mixture $f% ($run)"
  [ -d artifacts/$run/checkpoint-126 ] || $PY 14_finetune_with_monitor.py --dataset-jsonl artifacts/mixtures/mix_f$f.jsonl --seed 42 --run-name $run --monitor-every 10 --canary-n 0 || { echo "$run TRAIN FAILED"; continue; }
  python - <<PY
import json
rows=[json.loads(l) for l in open("artifacts/$run/monitor_log.jsonl")]
print("$run drift:", {r["step"]: round(abs(r["monitor_value"]),2) for r in rows if r["step"] in (40,50,80,126)})
PY
  step "20 attribution $run, $N shards"
  pids=()
  for ((k=0; k<N; k++)); do
    $PY 20_attribute_drift.py --run-name $run --dataset-jsonl artifacts/mixtures/mix_f$f.jsonl --target qwen_pc1 \
        --shard $k --n-shards $N > artifacts/$run/attribution.shard$k.log 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait $p; done
  step "20 merge $run"
  $PY 20_attribute_drift.py --run-name $run --dataset-jsonl artifacts/mixtures/mix_f$f.jsonl --target qwen_pc1 --merge --token-level-top 10 || echo "$run MERGE FAILED"
  step "25 evaluate $run"
  $PY 25_mixture_eval.py --run-name $run --manifest artifacts/mixtures/mix_f$f.manifest.json || echo "$run EVAL FAILED"
  step "MIXTURE $f DONE"
done
step "DONE"
