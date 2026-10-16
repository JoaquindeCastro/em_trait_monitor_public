#!/usr/bin/env bash
# Stage A on a fresh pod: rebuild the instrument and the baseline run (this is an
# independent replication of Joaquin's numbers on different hardware), then attribute.
# Logs go to artifacts/*.log; run under nohup so an SSH drop does not kill it:
#   nohup bash runpod_stage_a.sh > artifacts/stage_a.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")"
export HF_HOME=${HF_HOME:-/workspace/hf}
mkdir -p artifacts
PY=${PY:-python -u}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-16} MKL_NUM_THREADS=${MKL_NUM_THREADS:-16} TOKENIZERS_PARALLELISM=false

step () { echo; echo "=================== $(date -u +%H:%M:%S) $1 ==================="; }

step "03 trait directions (layer 16)"
[ -f artifacts/trait_directions_qwen_layer16.pt ] || $PY 03_build_trait_directions.py
step "04 base position"
[ -f artifacts/base_position_qwen_layer16.pt ] || $PY 04_measure_base_position.py

step "14 baseline bad_medical seed 42 with in-loop monitor (replicates 09+10)"
if ! ls artifacts/bad_medical_fullseq_seed42/checkpoint-126 >/dev/null 2>&1; then
  $PY 14_finetune_with_monitor.py --dataset bad_medical --seed 42 \
      --run-name bad_medical_fullseq_seed42 --monitor-every 10 --canary-n 0
fi
step "compare with Joaquin's committed trajectory"
$PY - <<'PY'
import json
ours = {e["step"]: e for e in json.load(open("artifacts/bad_medical_fullseq_seed42/trait_trajectory.json"))}
import subprocess
ref_txt = subprocess.run(["git", "show", "origin/main:scratch_reproduction/artifacts/bad_medical_fullseq_seed42/trait_trajectory.json"],
                         capture_output=True, text=True).stdout
ref = {e["step"]: e for e in json.loads(ref_txt)}
pc1 = json.load(open("artifacts/qwen_seed42_pc1.json"))["pc1"]
traits = list(pc1)
def proj(e): return abs(sum(e["normalized_drift_percent"][t] * pc1[t] for t in traits))
print(f"{'step':>5} {'ours |PC1|':>11} {'joaquin':>9} {'diff':>7}")
for s in sorted(ours):
    if s in ref:
        a, b = proj(ours[s]), proj(ref[s]); print(f"{s:5d} {a:11.2f} {b:9.2f} {a-b:+7.2f}")
PY

step "20 attribution, qwen_pc1, paired, token level"
$PY 20_attribute_drift.py --run-name bad_medical_fullseq_seed42 --dataset bad_medical --seed 42 \
    --paired-jsonl data/model_organisms/training_datasets.zip.enc.extracted/good_medical_advice.jsonl \
    --target qwen_pc1 --token-level-top 20
step "20 attribution, pooled_pc1"
$PY 20_attribute_drift.py --run-name bad_medical_fullseq_seed42 --dataset bad_medical --seed 42 \
    --target pooled_pc1
step "14 second-seed baseline (noise floor)"
if ! ls artifacts/bad_medical_monitor_seed123/checkpoint-126 >/dev/null 2>&1; then
  $PY 14_finetune_with_monitor.py --dataset bad_medical --seed 123 \
      --run-name bad_medical_monitor_seed123 --monitor-every 10 --canary-n 0
fi
step "DONE"
