#!/usr/bin/env bash
# Revised Handoff 3, Stage A: attribute the drift of the EXISTING baseline run to its
# 1,000 training examples. No training. Run on the GPU server from scratch_reproduction/.
#
# Needs (already there from the reproduction):
#   artifacts/bad_medical_fullseq_seed42/checkpoint-{10..120,126}/
#   artifacts/trait_directions_qwen_layer16.pt, base_position_qwen_layer16.pt, qwen_seed42_pc1.json
#
# Paired benign completions (Turner et al. model-organism archive; research use only,
# never commit the extracted JSONL; data/ is git-ignored):
#   pip install easy-dataset-share
#   mkdir -p data/model_organisms && cd data/model_organisms
#   curl -sL -o training_datasets.zip.enc https://raw.githubusercontent.com/clarifying-EM/model-organisms-for-EM/main/em_organism_dir/data/training_datasets.zip.enc
#   easy-dataset-share unprotect-dir training_datasets.zip.enc -p model-organisms-em-datasets --remove-canaries
#   cd ../..
# The archive's bad_medical_advice.jsonl is identical, row for row, to the Hugging Face
# split the baseline was trained on (verified 2026-10-16), so good_medical_advice.jsonl
# row i is the benign counterpart of training source index i.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
RUN=${RUN:-bad_medical_fullseq_seed42}
PAIRED=${PAIRED:-data/model_organisms/training_datasets.zip.enc.extracted/good_medical_advice.jsonl}

# A1  primary target (Qwen-only PC1), all 13 checkpoints, paired counterfactuals,
#     token-level contributions for the 20 top and 20 bottom examples. ~2 GPU-hours.
$PY 20_attribute_drift.py --run-name "$RUN" --dataset bad_medical --seed 42 \
    --paired-jsonl "$PAIRED" --target qwen_pc1 --token-level-top 20 \
    2>&1 | tee "artifacts/$RUN/attribution_qwen_pc1.log"

# A2  paper's pooled axis, no pairing (cheaper), to check the ranking is not axis-specific.
$PY 20_attribute_drift.py --run-name "$RUN" --dataset bad_medical --seed 42 \
    --target pooled_pc1 2>&1 | tee "artifacts/$RUN/attribution_pooled_pc1.log"

# A3  noise floor for Stage B: the same configuration with a second seed, monitor in loop.
#     (Different seed changes the 1,000-example sample too; that is the paper's protocol.)
$PY 14_finetune_with_monitor.py --dataset bad_medical --seed 123 --run-name bad_medical_monitor_seed123 \
    --monitor-every 10 --canary-n 0 2>&1 | tee artifacts/bad_medical_monitor_seed123.train.log

echo "Stage A done. Copy artifacts/$RUN/attribution_*/ (scores.jsonl, summary.json, tokens.jsonl)"
echo "and artifacts/bad_medical_monitor_seed123/{monitor_log.jsonl,trait_trajectory.json} to the laptop for D1."
