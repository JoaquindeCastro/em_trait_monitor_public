#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

echo "============================================================"
echo "Extension 0: incompetence vs harmful fine-tuning"
echo "============================================================"

python 13_build_extension0_incompetence_control.py

echo
echo "Training correct-safe control..."
python 09_finetune_dataset.py   --dataset-jsonl artifacts/extension0_incompetence_control_data/correct_safe_seed42.jsonl   --run-name extension0_correct_safe_seed42

echo
echo "Measuring correct-safe drift..."
python 10_measure_dataset_drift.py   --run-name extension0_correct_safe_seed42

echo
echo "Training incompetent / mismatched-safe control..."
python 09_finetune_dataset.py   --dataset-jsonl artifacts/extension0_incompetence_control_data/incompetent_mismatch_seed42.jsonl   --run-name extension0_incompetent_mismatch_seed42

echo
echo "Measuring incompetent drift..."
python 10_measure_dataset_drift.py   --run-name extension0_incompetent_mismatch_seed42

echo
echo "Training harmful matched condition..."
python 09_finetune_dataset.py   --dataset-jsonl artifacts/extension0_incompetence_control_data/harmful_bad_medical_seed42.jsonl   --run-name extension0_harmful_seed42

echo
echo "Measuring harmful drift..."
python 10_measure_dataset_drift.py   --run-name extension0_harmful_seed42

echo
echo "Analyzing all three conditions..."
python 14_analyze_extension0_incompetence_control.py

echo
echo "DONE."
echo "Results:"
echo "  artifacts/extension0_incompetence_control_results/"
