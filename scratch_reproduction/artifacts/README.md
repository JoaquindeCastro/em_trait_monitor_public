# Saved experimental evidence

This directory contains curated, CPU-readable outputs for the Qwen reproduction
and course extensions. The source training corpora, model checkpoints and activation
tensors are external. Generated model responses may contain unsafe or inaccurate
content; they are evaluation records, not endorsed advice.

| Location | Role |
| --- | --- |
| `qwen_seed42_pc1.json` | Fixed Qwen-specific calibration axis reused by Extension 3 |
| `bad_medical_fullseq_seed42/` | Full-data baseline, PC1 and pooled-axis attribution, responses and utility |
| `arm_*_k100/` | Removal or rewrite arm configurations, monitor logs, responses and utility |
| `mix_f*_seed42/attribution_*/` | Canonical merged scores, checkpoint summaries and recovery metrics by target |
| `mixtures/*.manifest.json` | Planted-label positions and source hashes; prepared JSONL files remain local |
| `removal_sets/*.manifest.json` | Removed or rewritten indices and source metadata; prepared datasets remain local |
| `layer_probe/` | Exploratory layer selection on the tested mixtures |
| `stage_b_judge_gpt56sol/` | Current complete judge records, summary and coverage |
| `stage_b_judge/` | Historical incomplete Claude judge outputs; not pooled into current results |
| `continuation_20261016/` | Current joined report, CSV tables, figure, completion state and output audit |
| `mixture_harmful_*`, `mixture_random_*`, `ordering_analysis_*` | Extension 1 ordering evidence |

`scores.jsonl` is the merged attribution table. Duplicate worker shards, incomplete
worker backups, locks, PIDs and process logs are ignored. A raw score table is
indexed by `i`; mixture rows use prepared-dataset positions while baseline rows
also record original `source_index`. Check the run metadata before joining them.
Judge/response identity is `(run, step, prompt_id, sample_idx)`.

The three Stage E targets and all behavioral records are checked by
`python scratch_reproduction/29_audit_extension3.py` from the repository root.
The saved `continuation_20261016/audit.json` includes hashes. The audit does not
claim that remote checkpoints remain available or that the GPU is no longer billed.

Mixture manifests were reconstructed at handoff with the original source pools,
sample seed 42 and planting seed 0, using the exact builder algorithm. All 1,000
prepared-example hashes match the saved contrast scores for each nonzero mixture.
Full source texts are not needed to recompute ranking metrics from saved scores
and these manifests. Some layer-16/17 analyses retain summary metrics without
their original per-example score tables; those tables require the external run
archive or recomputation on the original adapters.
