# Qwen reproduction and course experiments

These scripts reproduce the Qwen2.5-7B trait-drift measurement and extend it with
ordering experiments and training-data attribution. This is a separate pipeline
from the reusable paper code in `src/` and `experiments/`; do not interchange
normalization constants or artifacts without checking their metadata.

## Script map

| Purpose | Scripts |
| --- | --- |
| Inspect, extract trait directions, measure base state | `00` through `05`, `08` |
| Train and measure reproduction datasets | `09_finetune_dataset.py`, `10_measure_dataset_drift.py` |
| Fit the saved Qwen PC1 and plot reproduction | `11_pc1_fit.py`, `12_plot.py` |
| Extension 1 ordering | `13_make_ordered_mixture.py`, `14_ordering_analysis.py`, `15_path_dependence_colab.ipynb` |
| Train with monitoring, generate behavior, judge, measure utility | `14_finetune_with_monitor.py` through `19_compare_adapters.py` |
| Attribute drift, compare prior, make removals, report | `20_attribute_drift.py` through `23_attribution_report.py` |
| Make planted mixtures, evaluate rankings, select layers | `24_make_mixture.py`, `25_mixture_eval.py`, `26_layer_probe.py` |
| Judge with requested GPT-5.6 Sol model | `16_judge_codex_batch.py` |
| Join behavior, utility and attribution results | `27_continuation_report.py` |
| Collect outputs or resume locally | `28_collect_stage_e.py` |
| Verify completion and evidence hashes without a GPU | `29_audit_extension3.py` |

`06_SFT_bad_medical.py` is an earlier training script. Use `09` for the reproduction
or `14_finetune_with_monitor.py` for Extension 3. `run_handoff3.sh` and the original
intervention design predate the attribution extension. The current stage sequence
is the `runpod_stage_a*`, `runpod_stage_b`, `runpod_stage_c*`, `runpod_stage_d` and
`runpod_stage_e` shell drivers. Inspect each driver's prerequisites before launching;
they are experiment drivers, not a fresh-machine installer.

## Local analysis and judging

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-scratch-analysis.txt
python scratch_reproduction/29_audit_extension3.py
python scratch_reproduction/27_continuation_report.py
python -m pytest -q tests/test_scratch_resume.py
```

The standalone audit uses only the standard library. Report generation needs
NumPy and Matplotlib. To resume only missing judgments, use:

```bash
python scratch_reproduction/28_collect_stage_e.py --local-only
```

Judging requires an authenticated `codex` executable only when unjudged responses
remain. The runner downloads Betley rubric YAMLs into ignored `data/betley_repo/`
when missing. The fixed audit expects the seven recorded runs and 14 conditions;
new experimental designs should use a separate output directory and adjust the
audit contract explicitly. Never mix judge providers in the same directory.

## GPU reproduction

Install the root training requirements and restore the original training sources,
direction tensors, base-position tensors and numeric adapter checkpoints.
These large artifacts are not in Git. Driver scripts assume a Linux CUDA host,
with the repository at `/workspace/em_trait_monitor_public`; `PY` can select the
environment's interpreter. Credentials belong in the host environment.

Stage E was run with `TOKEN_TOP=0` to complete the queued attribution scores
without optional token explanations. `--also-target` reuses example gradients
for two distinct query targets in unpaired sharded runs. The shard checker rejects
mixed shard counts and incomplete checkpoint coverage. The collector accepts a
user-supplied executable `--pod-helper` implementing `ssh <command>` and
`pull <remote> <local>`; the original session helper is not a repository dependency.

For current results and scientific limits, read
[Extension 3 status](../research/extension3/STATUS.md) and
[methods](../research/extension3/METHODS.md). Preserve script names and result
paths because saved run configurations and handoff records refer to them.
