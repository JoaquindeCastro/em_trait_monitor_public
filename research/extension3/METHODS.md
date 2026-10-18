# Extension 3 measurement and attribution

The experiment asks whether training examples that increase an internal drift
measurement can be identified and whether removing them improves the model.
Attribution, planted-label recovery and behavioral evaluation answer different
parts of that question.

## PC1 calibration and measurement

For 115 fixed neutral prompts, average the last input-token activation at layer
16. Project it onto seven unit trait directions, subtract the base-model projections,
and normalize by the norm of the base model's mean activation. These seven
coordinates measure representation change, not rated behavior. Positive drift
points toward the named trait direction; negative drift points away. The sign is
not a moral label, especially for traits such as power-seeking.

`11_pc1_fit.py` fits PC1 by centering the four final seven-trait drift vectors from
the insecure-code, jailbroken, bad-medical and GSM8K Qwen calibration runs and
taking the leading right singular vector. The saved Qwen axis explains 91.8% of
variation in those four vectors; it is not the paper's pooled four-model estimate.
Extension 3 **reuses this saved axis** in `artifacts/qwen_seed42_pc1.json` rather
than refitting it at each checkpoint or intervention. Its calibration includes
the completed bad-medical run, so it is not an independent held-out calibration.

Let `d` be the normalized seven-trait drift vector and `u` the fixed PC1 axis.
The plot shows `abs(d · u)`. The attribution target uses the oriented signed
quantity `T = −d · u`. The PCA calibration is centered when fitting the axis;
the monitored projection is measured from the base model, without subtracting
the calibration mean. The 8% line in early slide versions was an experimental
reference threshold, not a validated safety boundary or a harmful-token rate.

## Ranking methods

| Method | Quantity ranked or attributed | Information used |
| --- | --- | --- |
| Training-loss baseline | Mean next-token negative log-likelihood, highest first | Full prompt-and-answer sequence at the first checkpoint, step 10 |
| PC1-target TracIn | Estimated influence on signed PC1 drift | Seven trait measurements, fixed calibration axis, checkpoint gradients |
| Seven-trait contrast TracIn | Estimated influence on the harmful-minus-benign final reference difference in trait space | Matched reference runs projected onto seven traits at layer 16 |
| Raw layer-18 contrast TracIn | Estimated influence on the reference difference directly in hidden activation space | Mean activations of the completed reference runs at layer 18 |

For raw contrast, `u_L = normalize(mean_h_harmful,L − mean_h_benign,L)` and
`T = 100 × (mean_h_current,L − mean_h_base,L) · u_L / norm(mean_h_base,L)`.
The reference means use the same 115 neutral prompts. Subtracting reference
means can reduce shared adaptation, but does not guarantee a pure harmfulness
direction: style, length and other condition differences can remain.

TracIn computes `s_i = −sum_t eta_t * dot(grad(T_t), grad(loss_i,t))` over trainable
LoRA parameters. Gradient descent moves against the loss gradient, hence the
minus sign. Positive scores predict an increase in the selected target locally;
negative scores predict a decrease. Dot products depend on magnitudes as well
as gradient alignment. The full-sequence causal loss includes prompt tokens.
Actual training uses AdamW, batches and clipping, so these first-order SGD-style
scores are approximate rankings, not an exact decomposition of causal effects.

The loss baseline uses no target gradient or checkpoint accumulation. High
prediction error is not the same as a training update that raises the target.

## Checkpoints and evaluation

An optimizer step normally accumulates four microbatches of four examples.
The 1,000-example runs train for two epochs and end at update 126; 900-example
removal runs end at update 114. The full attribution uses checkpoints 10, 20,
…, 120 and 126. “Through 40” sums checkpoints 10, 20, 30 and 40 and ranks all
1,000 candidates, not only the examples already encountered in training.

Precision at k counts planted harmful examples among the top k ranked candidates.
Here k equals the total number planted, so precision and recall happen to be
numerically equal. The layer-selection probe instead uses individual examples'
last-token activation projections and AUROC; it does not compute TracIn gradients.
AUROC measures pairwise harmful-versus-benign ordering across thresholds.

The early-checkpoint comparison is retrospective because reference directions
come from completed runs and the same labeled mixtures informed layer selection
and recovery evaluation. Step-40 attribution itself uses no later mixture
gradients. A prospective study could reuse completed reference runs, provided
all choices were frozen before new independent evaluation runs.

Removal is tested by fresh retraining. Top versus random removal holds the
reduced dataset size and training budget fixed. The full-data comparison does
not. The matched benign-answer comparison controls prompt text but does not
control completion length or style perfectly. Behavioral judgments and their
paired prompt-bootstrap intervals are reported separately in [status](STATUS.md).
