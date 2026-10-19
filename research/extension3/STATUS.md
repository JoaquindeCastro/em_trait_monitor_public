# Extension 3 results and continuation

Updated October 17, 2026, America/New_York. The queued Stage E attribution work
is complete, and **all 5,040 generated responses have GPT-5.6 Sol judgments**.
Coverage spans seven runs, each at step 40 and its final checkpoint: 14 conditions,
72 prompts per condition, five samples per prompt. There are no missing or duplicate
response keys. [The audit](../../scratch_reproduction/artifacts/continuation_20261016/audit.json)
checks coverage, rubric rules, summaries and checkpoint contributions and records
hashes of the underlying evidence.

## Behavioral results

| Final model | Misalignment among eligible responses |
| --- | ---: |
| Full harmful dataset | 13.76% |
| Remove top 100 by PC1 TracIn | 7.58% |
| Remove random 100, draw 0 | 6.72% |
| Remove random 100, draw 1 | 7.28% |
| Remove random 100, draw 2 | 10.73% |
| Remove bottom 100 | 9.40% |
| Rewrite top 100 with benign twins | 10.36% |

Top removal minus random draw 0 is **+0.86 percentage points**, with a paired
prompt-bootstrap 95% interval of **−1.98 to +3.90**. This does not establish a
behavioral advantage for attribution-guided removal. Reduced PC1 drift and improved
behavior are separate claims. The representation figure averages three random
subset draws; the quoted behavioral comparison uses draw 0 explicitly. Each arm
uses one training seed. Prompt-bootstrap uncertainty does not measure retraining
variability or the variation between random removal subsets.

Alignment below 30 counts as misaligned among responses with coherence at least
50, excluding CODE and REFUSAL labels. The judge was requested as `gpt-5.6-sol`
through Codex CLI, with verbatim Betley rubrics. Responses and both dimensions are
batched, temperature is unpinned, and a separate served-model identifier is not
available. This differs from the paper's independent GPT-4o calls. The earlier,
incomplete Claude Opus 5.5 grades never entered these rates and were removed from
the repository at submission cleanup.

Sources: [judgments](../../scratch_reproduction/artifacts/stage_b_judge_gpt56sol/betley_judged.jsonl),
[summary](../../scratch_reproduction/artifacts/stage_b_judge_gpt56sol/betley_summary.json),
[joined report and paired intervals](../../scratch_reproduction/artifacts/continuation_20261016/results.json),
[behavior and utility CSV](../../scratch_reproduction/artifacts/continuation_20261016/behavior_utility.csv).

## Attribution results

Stage E completed the 10% mixture's seven-trait contrast attribution and the 5%
mixture's PC1 and seven-trait contrast attribution. Each has 1,000 merged example
scores at 13 checkpoints. Optional token-level explanations were deferred for
these fill-in runs; the original baseline token analysis remains available.

| Mixture and target | Precision at k through step 40 | Precision at k at final checkpoint |
| --- | ---: | ---: |
| 5%, PC1 TracIn, k=50 | 0.24 | 0.24 |
| 5%, seven-trait contrast, k=50 | 0.42 | 0.36 |
| 5%, first-checkpoint loss, k=50 | 0.38 | 0.38 |
| 10%, PC1 TracIn, k=100 | 0.36 | 0.39 |
| 10%, seven-trait contrast, k=100 | 0.55 | 0.54 |
| 10%, raw layer-18 contrast, k=100 | 0.66 | 0.54 |
| 10%, first-checkpoint loss, k=100 | 0.52 | 0.52 |

The loss baseline is a single ranking from step 10, repeated for reference.
Layer 18 was selected using the same tested mixtures, and contrast directions use
completed harmful and benign reference runs. These are retrospective recovery
results, not a held-out prospective early-warning validation. See [methods](METHODS.md)
for target definitions, signs, normalization and checkpoint timing.

## Recovery history and remaining work

The October 16 collector downloaded the new scores and all remaining responses,
then failed while copying a diagnostic log. Its status incorrectly remained
`collecting`, and judging stopped at 4,120 rows. On October 17 the remaining 920
responses were judged locally, the report was regenerated, and all 5,040 rows
were audited. The collector now supports `--local-only`, treats remote diagnostic
logs as optional, records failures and audits the outputs before marking completion.

The previous RunPod SSH endpoint refused connections at this handoff. Saved
numerical evidence is local; availability of the remote adapter checkpoints and
the pod's billing state remain unverified. Training source files and activation
tensors are intentionally outside Git. Mixture manifests were reconstructed from
the original matched source pools and seeds; all 1,000 reconstructed message hashes
match saved contrast score rows for each of the 5%, 10% and 25% mixtures.

## Stage F: threshold-based removal (persona-vector filtering rule)

Run October 19, 2026 (UTC) on a fresh RunPod L40S after the original pod's volume could
not be resumed. The trait-direction and base-position tensors were rebuilt with scripts
03 and 04 and the seed-42 all-harmful baseline was retrained: its final |PC1| drift is
24.29 against the committed 24.26 (largest gap along the trajectory 0.42 at step 80), so
the Stage F arms are read on the same scale as the Stage B and C runs. See
[instrument check](../../scratch_reproduction/artifacts/stage_f/instrument_check.md).

Instead of a fixed top-100, every example whose PC1 TracIn score exceeds mean + z sd was
flagged (z = 2, 1, 0.5, 0 flag 34, 152, 288, 458 of the 1,000 examples). Four arms per
cutoff: delete the flagged examples, replace them by their correct twins (rewrite,
keeps 1,000 examples), and size-matched random delete and random rewrite (seed 0).
One training seed each. Final |PC1| drift, percent of the base norm:

| z | flagged | threshold delete | random delete | threshold rewrite | random rewrite |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 34 | 25.36 (122 steps) | 23.24 | 23.83 | 23.93 |
| 1 | 152 | 21.11 (106 steps) | 22.41 | 22.23 | 22.76 |
| 0.5 | 288 | 17.37 (90 steps) | 19.46 | 20.78 | 21.74 |
| 0 | 458 | 12.36 (68 steps) | 14.81 | 19.45 | 20.51 |

Threshold minus size-matched random: delete +2.11, -1.30, -2.08, -2.44; rewrite -0.09,
-0.52, -0.96, -1.06 (z = 2, 1, 0.5, 0). The run-to-run noise is about 1.5 points (three
random k=100 draws spread 23.21 to 25.92), so the delete advantage is outside noise at
z <= 0.5 and the rewrite advantage is within noise at every cutoff though consistently
signed. Removing only the 34 highest-scored examples raised drift by 2.1 points relative
to its random control, a single-seed result of the wrong sign.

Two readings matter. First, delete arms train for fewer optimizer steps (two epochs over a
smaller set), so their absolute drift falls partly because training is shorter; the
size-matched random-delete arm shares the step count, which is why only the pairwise
difference is interpretable there. The rewrite arms keep 126 steps and are the clean
comparison. Second, the summed removed score predicts the rewrite arms well once scaled by
the 0.69 realised-to-predicted ratio of the top-100 arm (predicted 23.50, 21.84, 20.53,
19.44 against realised 23.83, 22.23, 20.78, 19.45), while the random-rewrite controls
end 0.2 to 1.3 points below their own prediction: the correct twins that replace removed
examples push the monitor back, which the score sum ignores.

On the harmful-fraction axis this links to Experiment 2 and to the Stage C mixtures: the
random-rewrite points (97, 85, 71 and 54 percent harmful) continue smoothly down to the
planted mixtures (25, 10, 5 percent: 18.71, 17.86, 17.46) and the all-correct floor
17.31, and score-chosen rewrite sits about one point below that curve at 54 percent.
Experiment 2's sweep has a different filler and a rebuilt monitor, so only its shape is
comparable. At the alarm step (40) threshold delete is below random delete at z <= 1
(11.90 vs 13.23, 11.41 vs 12.03, 8.70 vs 10.54) and rewrite roughly matches its control.
No behavioural generation or judging was run for these arms.

Sources: [results](../../scratch_reproduction/artifacts/stage_f/threshold_sweep_combined/results.md),
[figure](../../scratch_reproduction/artifacts/stage_f/threshold_sweep_combined/fig_threshold_sweep.png),
monitor logs and manifests under `scratch_reproduction/artifacts/stage_f/`;
driver `scratch_reproduction/runpod_stage_f.sh`, join `scratch_reproduction/30_threshold_sweep.py`.

Next scientific work is independent validation: freeze the layer and target before
new mixtures, repeat training seeds, and evaluate behavior on the same held-out
prompt protocol. No additional GPU runs are required to reproduce the current
CPU-readable report. Presentation snapshots describe the October 16 deck; they
have not been silently updated with the newly completed random draw 1 result.

## Reproduce or resume

From the repository root, with the scratch analysis dependencies installed:

```bash
python scratch_reproduction/29_audit_extension3.py
python scratch_reproduction/27_continuation_report.py
# Only if downloaded responses still need judgments; otherwise this is a no-op judge pass:
python scratch_reproduction/28_collect_stage_e.py --local-only
```

The report directory remains `continuation_20261016` because it identifies the
original experiment continuation. The audit and collector files record the actual
completion timestamps in UTC.

## Integration validation

The merged repository passed 46 tests and four subtests, including a regression
check that the training CLI accepts both `--sequential` and `--preserve-order`.
The semantic-steering prompt audit passed with 140 prompts and zero overlaps.
The Extension 3 evidence audit, shell syntax checks, Python syntax checks,
documentation-link checks and Git whitespace checks passed. The final behavior
figure was inspected after adding the completed random-control condition.
These are CPU and saved-artifact checks; no GPU retraining was performed during
repository integration.
