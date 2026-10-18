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
available. This differs from the paper's independent GPT-4o calls. Historical
Claude grades stay separate and do not enter these rates.

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
