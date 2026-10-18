# Extension 3: attributing representation drift

Status: appended and visually verified in the shared Google Slides deck as slides 27–34. A subsequent clarity revision defines terminology on first use, explains steps/checkpoints, displays recovery as counts out of 100, removes secondary charts, and moves layer selection before recovery. The current native order is source slides 1, 2, 3, 4, 5, 7, 6, 8. See `extension3_clarity_revision.json` for the current text and explicit element removals. The detailed script below retains the technical evidence and original speaker notes.

Story: identify the drivers, test the ranking, discover benign drift, refine the target, then check behavior.

## Slide 19. Which training examples drive representation drift?

**On-slide copy**

The monitor detects a change in the model’s internal activations. We ask which examples contribute to that change, then test the ranking by retraining.

**Our investigation**

Rank examples by their estimated contribution.

Compare harmful and benign answers to the same prompt.

Remove ranked examples and retrain.

Check what PC1 responds to.

Test recovery of planted harmful answers.

**Footer:** Qwen2.5-7B-Instruct, 1,000 medical examples, LoRA, layer 16. PC1 summarizes representation drift.

**Visual:** Keep the harmful-run PC1 trajectory. Label the vertical axis “PC1 magnitude (% of base norm)” and the horizontal axis “Training step”. The 8% line is a reference level. A short question sequence sits beside the chart.

**Speaker notes**

We extend representation monitoring with training-data attribution. Qwen2.5-7B-Instruct uses LoRA fine-tuning on 1,000 medical examples. PC1 is a scalar summary of seven trait projections. It is not the paper’s complete seven-dimensional random-forest regressor. The independent scratch run tracks the earlier reproduction within 0.29 percentage points at the recorded checkpoints. The 8% line is a reference for this experiment, not a validated replacement for the paper’s detector. Transition: how can we assign this scalar change to individual examples?

## Slide 20. TracIn estimates each example’s contribution to drift

**On-slide copy**

**Drift gradient**

How the drift target changes with the LoRA weights.

**Example gradient**

How the example’s training loss changes with those weights.

**TracIn score**

Their negative dot product, weighted by learning rate and summed over checkpoints.

Positive scores predict increased drift. Negative scores predict decreased drift.

Scoring uses gradients at 13 checkpoints and requires no behavioral judge.

**Footer:** First-order estimate: sᵢ = −Σₜ ηₜ ⟨∇Tₜ, ∇ℓᵢ,ₜ⟩. Retraining tests whether the ranking predicts an intervention.

**Visual:** Keep the three method boxes: drift gradient, example gradient, TracIn score. Show the first-order equation in the footer. Technical definitions and optimizer limitations belong in the notes.

**Speaker notes**

Let T be the oriented representation-drift target and ℓ_i the full-sequence causal-LM loss for example i. TracIn uses s_i = −Σ_t η_t〈∇θ T_t, ∇θ ℓ_i,t〉 across 13 saved checkpoints, restricting θ to LoRA parameters. This approximates a gradient step’s first-order effect. Actual optimization uses AdamW, clipping and mini-batches, so the score is not an individual causal effect or an additive decomposition of total drift. No judge is required to compute these scores. Transition: do the scores distinguish the harmful completion from its benign counterpart?

## Slide 21. Matched benign answers usually score lower

**On-slide copy**

The top 100 examples carry 32% of positive score mass.

Benign twins score lower in 90.1% of matched pairs.

64.3% of benign twins have a negative score.

**Footer:** Same prompts, different completions. These comparisons concern the drift score, not measured harmful behavior.

**Visual:** Keep the decile chart, labeled “Positive score mass by ranked decile”, and the paired mean-score bars (harmful = 100, benign = −36). The three numerical findings sit beside them.

**Speaker notes**

The paired files share prompts but differ in completions. Benign twins score lower than harmful twins in 90.1% of pairs, while only 64.3% have a negative score. Lower than the harmful twin does not mean below zero. The top-decile 32% statistic is a share of positive score mass, not total signed mass or the fraction of harm explained. The normalized mean benign score is −36 relative to a harmful mean of 100. Sequence lengths and full-sequence loss can also influence the comparison, so it does not isolate a causal semantic mechanism. Transition: do interventions follow the ranking?

## Slide 22. Removing high-scoring examples reduces PC1 drift

**On-slide copy**

Random removal provides a matched training-budget control.

The ordering supports an effect on the monitored representation.

**Footer:** Random removal matches the 114-step removal budget. The full dataset trains for 126 steps. One training seed per arm.

**Visual:** Keep the final-PC1 bars: top removal 22.5%, random-removal mean 24.2%, full dataset 24.3%, bottom removal 26.4%. Show the three individual random draws as dots. The sidebar interprets the comparison without repeating the values.

**Speaker notes**

Final PC1 magnitudes are approximately top removal 22.5%, random-removal mean 24.2%, full dataset 24.3%, bottom removal 26.4%. Rewriting the top 100 with benign twins gives 22.8%. Removal arms end at step 114, versus 126 for the full dataset and rewrite, making random removal the appropriate equal-budget comparison. Three random draws describe variability but do not justify a universal 1.5-point significance threshold. The ordering supports an intervention effect on representation. Behavioral results appear on the last slide. Transition: what exactly has changed when PC1 decreases?

## Slide 23. Benign fine-tuning also produces substantial PC1 drift

**On-slide copy**

Benign-only training reaches 71% of the harmful run’s final PC1 magnitude.

**Footer:** This motivates a harmful-minus-benign contrast. The paper’s full detector uses a seven-trait regressor, not PC1 alone.

**Visual:** Keep the harmful-versus-benign PC1 trajectories and the seven-trait comparison. Highlight the benign-to-harmful final magnitude ratio, 71%. Label both trajectories directly.

**Speaker notes**

Benign-only medical fine-tuning reaches roughly 71% of the harmful run’s final PC1 magnitude. The 10% harmful mixture ends at 17.9%, about half a percentage point above the benign run. This demonstrates that the scalar Qwen PC1 summary responds substantially to benign adaptation in this setting. It does not establish failure of the paper’s full seven-trait regressor. We therefore construct a harmful-minus-benign target from final reference runs, either in seven-trait coordinates or directly in layer activations. Transition: can this target recover known harmful examples?

## Slide 24. A harmful-minus-benign target improves planted recovery

**On-slide copy**

At step 40, layer-18 contrast recovers 66 of the 100 harmful examples.

PC1 recovers 36. The loss baseline recovers 52.

This is retrospective: reference directions use completed runs.

**Footer:** Seven-trait contrast retains the trait space. Raw layer-18 contrast uses activations directly. Layer selection uses these mixtures.

**Visual:** Use two compact bar groups for the same 10% mixture: step 40 and final checkpoint. Identical method order in both: chance, Qwen PC1, first-checkpoint loss, seven-trait contrast, layer-18 contrast. Values: step 40 [0.10, 0.36, 0.52, 0.55, 0.66], final [0.10, 0.39, 0.52, 0.54, 0.54]. This replaces the earlier mixed-checkpoint comparison.

**Speaker notes**

Ground truth is a fixed planted set of 100 harmful completions among 1,000 otherwise benign examples with matched prompts. Precision@100 equals the number of planted harmful examples in the top 100 divided by 100. At step 40: chance .10, Qwen PC1 .36, first-checkpoint loss .52, seven-trait contrast .55, raw layer-18 contrast .66. Final: .10, .39, .52, .54, .54 respectively. Loss is a fixed first-checkpoint baseline, repeated for comparison. Directions use final harmful and benign reference runs, and layer 18 was chosen on the examined mixtures. Therefore this is retrospective early-checkpoint recovery, not prospective early-warning validation. The 5% attribution results are pending and are not shown. Transition: why layer 18?

## Slide 25. Layer 18 gives the strongest activation separation

**On-slide copy**

Harmful-minus-benign projections peak at layer 18 across the tested mixtures.

This activation probe motivates the attribution target.

Held-out calibration is needed before claiming early warning.

**Footer:** Directions come from final harmful and benign reference runs. These curves measure activation separation, not attribution recovery.

**Visual:** Keep the layer sweep with activation-projection AUROC for the 5%, 10%, and 25% mixtures. Label it explicitly as activation separation. These curves do not claim completed 5% attribution recovery.

**Speaker notes**

This layer sweep projects each example’s own activation onto a harmful-minus-benign direction constructed from the final reference runs on 115 neutral prompts. The three curves cover 5%, 10%, and 25% mixtures, but they describe activation-projection AUROC, not completed attribution experiments at all three fractions. Layer 18 was selected using these mixtures. The base model also separates the examples at layer 18 (reported AUROC .87). This motivates the target but requires independent calibration, mixtures and seeds to estimate generalization. Transition: does better representation control translate to fewer misaligned answers?

## Slide 26. Lower drift has not established a behavioral advantage

**On-slide copy**

**Final emergent misalignment rate**

Full dataset: 13.8%

Remove top 100: 7.6%

Remove random 100, draw 0: 6.7%

Top minus random: +0.86 pp

95% CI: [−1.98, +3.90] pp

**What the evidence supports**

TracIn predicts changes in the monitored representation.

Harmful-minus-benign targets improve planted recovery.

Behavioral benefit over random removal remains unresolved.

Next: held-out calibration and independent seeds.

**Visual:** Use the existing two-column layout. Left: three final EM rates and the paired difference interval. Right: the representation result, planted-recovery result, unresolved behavioral benefit, and next experiment.

**Speaker notes**

GPT-5.6 Sol judged all 4,120 currently available responses using the verbatim Betley alignment and coherence rubrics, with responses and dimensions batched. Temperature was not configurable through the Codex CLI. Keep these results separate from earlier Claude judgments. Final EM rates: baseline 13.76%, top removal 7.58%, random draw 0 6.72%, bottom removal 9.40%, rewrite 10.36%, random draw 2 10.73%. Baseline step-40 EM is 2.6%. Top-minus-random-0 is +0.86 percentage points with paired prompt-bootstrap 95% CI [−1.98,+3.90]. Rewrite-minus-baseline is −3.40 points [−7.00,+0.16]. Both intervals include zero. Removal runs stop at 114 steps, baseline and rewrite at 126. Random draw 1 has only 160 final responses, and extra random draws lack step 40 responses. One model, one medical dataset family, one training seed per arm and one planted permutation limit generalization. Source: scratch_reproduction/artifacts/continuation_20261016/results.json and behavior_final.png.

## Evidence and editing notes

Behavioral results: `scratch_reproduction/artifacts/continuation_20261016/results.json`. Slide 24 now compares the 10% mixture at step 40 and final using identical methods. The 5% activation probe is separate from the pending 5% attribution experiment. No significance claim relies on three random draws.

The revised JSON preserves slide/object IDs and includes the original line-chart requests under `native_line_requests_by_slide`. Those requests preserve charts for a future rebuild. They must not create duplicate objects when applying edits to the existing deck. `speaker_notes` is local metadata and requires separate application to native slide notes. Notes have been applied and all eight revised native slides have been visually checked.
