#!/usr/bin/env bash
# Stage D on the pod: layer probe on the planted mixtures, then attribution with the
# hidden-space contrast target at the best layers. Waits for Stage C (second pass) to finish.
set -uo pipefail
cd "$(dirname "$0")"
export HF_HOME=${HF_HOME:-/workspace/hf} OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0
PY=${PY:-python -u}
N=${N:-3}
TOPK=${TOPK:-2}
step () { echo; echo "=================== $(date -u +%H:%M:%S) $1 ==================="; }

until grep -q "MIXTURE 25 DONE" artifacts/stage_c2.log 2>/dev/null; do sleep 120; done
until ! pgrep -f "[2]0_attribute_drift" >/dev/null; do sleep 30; done
step "Stage C finished; starting the layer probe"

MIX=""
for f in 10 25 05; do
  [ -d artifacts/mix_f${f}_seed42 ] && MIX="$MIX --mixture mix_f${f}_seed42:artifacts/mixtures/mix_f$f.jsonl:artifacts/mixtures/mix_f$f.manifest.json"
done
step "26 layer probe (all layers) on$MIX"
$PY 26_layer_probe.py --harm-run bad_medical_fullseq_seed42 --benign-run mix_f00_seed42 $MIX --eval-batch-size 8 || { echo "PROBE FAILED"; exit 1; }

LAYERS=$(python - <<PY
import json
d = json.load(open("artifacts/layer_probe/layer_probe.json"))
runs = [r for r in d["mixtures"] if r in ("mix_f10_seed42", "mix_f25_seed42")] or list(d["mixtures"])
score = {}
for L in d["layers"]:
    vals = [d["mixtures"][r]["per_layer"][str(L)]["final_checkpoint"]["auroc"] for r in runs]
    score[L] = sum(vals) / len(vals)
best = sorted(score, key=lambda L: -score[L])[:$TOPK]
chosen = sorted(set(best) | {16})
print(" ".join(str(L) for L in chosen))
PY
)
step "attribution with raw_contrast at layers: $LAYERS (best $TOPK by probe AUROC, plus 16 as control)"

attribute () {   # run, fraction, layer
  local run=$1 f=$2 L=$3 pids=()
  step "20 raw_contrast layer $L on $run, $N shards"
  for ((k=0; k<N; k++)); do
    $PY 20_attribute_drift.py --run-name $run --dataset-jsonl artifacts/mixtures/mix_f$f.jsonl --target raw_contrast --layer $L \
        --raw-contrast-pt artifacts/layer_probe/contrast_dirs.pt --shard $k --n-shards $N > artifacts/$run/attribution_raw_contrast_L$L.shard$k.log 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait $p; done
  $PY 20_attribute_drift.py --run-name $run --dataset-jsonl artifacts/mixtures/mix_f$f.jsonl --target raw_contrast --layer $L \
      --raw-contrast-pt artifacts/layer_probe/contrast_dirs.pt --merge || echo "$run L$L MERGE FAILED"
  $PY 25_mixture_eval.py --run-name $run --target raw_contrast_L$L --manifest artifacts/mixtures/mix_f$f.manifest.json || echo "$run L$L EVAL FAILED"
}
for L in $LAYERS; do
  for f in 10 25; do
    [ -d artifacts/mix_f${f}_seed42/checkpoint-126 ] && attribute mix_f${f}_seed42 $f $L
  done
  step "LAYER $L DONE"
done
step "DONE"
