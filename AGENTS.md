# Repository agent handoff

Read `research/README.md`, then the relevant experiment's status file before
running work. For Extension 3, `research/extension3/STATUS.md` is current;
`research/handoff_3_tda_drift_attribution.md` is the historical plan and worklog.
Verify saved outputs with `python scratch_reproduction/29_audit_extension3.py`.
Do not infer completion from a log marker or an old process ID.

## Code boundaries

- `src/`, `experiments/`, `configs/`, and `notebooks/` implement the paper pipeline.
- `scratch_reproduction/` contains the Qwen reproduction and course extensions.
  Keep existing script filenames stable: run configurations and launchers refer
  to them. Numerical prefixes are historical, not globally unique experiment IDs.
- `research/extension3/` contains the current Extension 3 interpretation and handoff.
  Slide authoring snapshots live in its `slides/` directory.
- `scratch_reproduction/artifacts/` contains curated evidence. Commit merged
  scores, manifests, responses, judgments and summaries; exclude credentials,
  training datasets, model weights, duplicate shards, locks and process logs.

## Measurement contracts

Preserve the distinction between scratch normalization (norm of mean base
activation) and the paper normalization (mean of per-prompt norms). Preserve
the saved Qwen PC1 for comparisons; refitting it changes the experiment.
PC1, seven-trait contrast and raw layer contrast are different targets.
TracIn uses the signed target, while trajectory figures use PC1 magnitude.
Its score is `-sum(learning_rate * dot(target_gradient, loss_gradient))`.
The loss baseline uses the first saved checkpoint and is not a TracIn score.

Keep judge providers and protocols in separate output directories. Current
Extension 3 results use `stage_b_judge_gpt56sol`. The incomplete Claude Opus 5.5
judge outputs were removed at submission cleanup and are not pooled. A judge rerun
must resume by the full `(run, step, prompt_id, sample_idx)` key. Never overwrite responses to resume judging.

## Validation and continuation

Run `python -m pytest -q`, `python scripts/audit_semantic_steering_v2_prompts.py`,
and `python scratch_reproduction/29_audit_extension3.py`. Check modified shell
drivers with `bash -n` and review `git diff --check` before committing.
GPU tests require external adapters and datasets; do not claim a CPU test retrained
the model. `28_collect_stage_e.py --local-only` regenerates the existing report and
only calls the configured judge if responses are missing judgments.

RunPod access details are session-specific. The previous endpoint was unreachable
at handoff; this does not establish whether the pod was stopped or is still billed.
Do not relaunch GPU runs merely because the endpoint is unavailable: the queued
Stage E numerical outputs and all behavioral responses are already saved.
