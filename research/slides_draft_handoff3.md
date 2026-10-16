# Slide draft: extension 3 (drift attribution) and the replication audit

Structure follows the pyramid: the answer first, then the evidence slide by slide, then
what is uncertain and what comes next. Every title is the slide's finding in one sentence.
Body text is full sentences, at most four per slide. Speaker notes carry the technical
detail a questioner would ask for. Values marked {{...}} are filled from Stage A outputs.

---

## Slide A. Executive summary (label title)

**Title:** Summary

**Body:**
- We reproduced the paper's Qwen 2.5 7B drift result twice, on different hardware, and
  the two runs agree at every checkpoint to within 0.3 points.
- The paper measures *that* fine-tuning drifts representations along one alignment axis;
  we ask *which training examples* cause that drift, and whether those are the examples
  that cause the behaviour.
- We attribute the drift signal to each of the 1,000 training examples with a
  checkpoint-based estimator, compare the ranking against a matched benign counterpart
  of every example, and validate by retraining without the top-ranked examples.
- {{One-sentence headline result from Stage A, e.g. "The push is concentrated: the top
  10% of examples carry X% of it, and the harmful completion, not the prompt, is what
  pushes."}}

**Speaker notes:**
Reproduction: scratch pipeline (scripts 03 to 12) on Qwen2.5-7B-Instruct, layer 16,
seed 42, LoRA r=16, alpha=64 on q/v, lr 4e-5, 2 epochs over 1,000 bad_medical examples,
126 optimizer steps, checkpoint every 10 steps. Independent re-run on a RunPod A100
(torch 2.7.1, transformers 4.56.1, peft 0.15.2) rebuilt the trait directions, the base
position and the training run from scratch; |PC1| at step 126 was 24.26% vs Joaquin's
24.34%, largest deviation 0.29 points at step 60. Attribution: TracIn over the 13
checkpoints, score_i = sum_t -eta_t <q_t, g_{i,t}>, where q_t is the gradient of the
monitored drift with respect to the LoRA parameters and g_{i,t} the gradient of example
i's full-sequence loss; eta_t follows the linear decay schedule. Paired counterfactual:
Turner et al.'s good_medical_advice.jsonl is prompt-for-prompt aligned with the bad file.

---

## Slide B. The paper measures drift but not its cause

**Title:** The paper shows that fine-tuning moves the model along one alignment axis, but
not which examples move it

**Body:**
- Nghiem et al. project each checkpoint's mean activation onto seven alignment traits and
  find one axis that explains 78.6% of drift variance across four models and four datasets.
- Their monitor reaches 1.8% false negatives on held-out datasets, but they state that it
  "detects the onset of EM without identifying its causal mechanism".
- Every published attribution of emergent misalignment scores a behaviour (a judge score
  or a misaligned generation), which is expensive and only exists once the model already
  misbehaves.

**Visual:** the paper's Figure 1 shape for Qwen (|PC1| vs step, dangerous vs benign), from
our own run.

**Speaker notes:**
Paper: arXiv 2606.07631 v2 (COLM AIW 2026). Seven traits: honesty, helpfulness,
harmlessness, power-seeking, corrigibility, sycophancy, confidence; directions are
normalised mean differences of 5+5 contrastive system prompts x 30 questions, last input
token, one layer per model (Qwen: 16). Monitor: 115 neutral prompts, mean activation,
projection, drift relative to the base model, normalised by the base activation norm.
PC1 is fit on 48 calibration drift vectors (4 models x 4 datasets x 3 seeds). Detection:
per-model random-forest regressor from the 7D drift to the Betley EM rate, dangerous
threshold 6%. Attribution literature: Jaburi et al. (EleutherAI, Bergson) used EK-FAC and
Hessian-free influence on misaligned generations and matched the WildGuard classifier;
"Unequal Influence of Bad Advice" showed removal effects that change sign depending on
which examples are removed; TAME does token-level attribution; persona-feature
attribution uses SAE features. None uses a representational target.

---

## Slide C. Independent replication

**Title:** A from-scratch re-run on different hardware reproduces the Qwen drift curve to
within 0.3 points at every checkpoint

**Body:**
- We rebuilt the seven trait directions, the base position and the fine-tuning run on a
  RunPod A100 with the paper's pinned library versions, without reusing any artifact from
  the first reproduction.
- The two trajectories differ by at most 0.29 points of |PC1| across 13 checkpoints and
  end at 24.26% versus 24.34%.
- The agreement also bounds the monitor's own measurement noise at a few tenths of a
  point, which is the unit we use when we say two conditions differ.

**Visual:** table of |PC1| at steps 10, 30, 40, 60, 80, 126 for both runs, or an
overlaid line chart.

**Speaker notes:**
Both runs: Qwen2.5-7B-Instruct bf16, layer-16 forward-hook output at the final input
token after the chat template with the assistant generation prompt. Normalisation
constant: norm of the mean base activation (63.48), which is the scratch convention; the
released code uses the mean of per-prompt norms (66.5), a 5% scale difference that does
not change shape or ratios. Training: full-sequence causal-LM loss (labels = input ids,
including the user turn), AdamW, weight decay 0.01, linear decay with no warm-up,
gradient clipping 1.0, batch 4 x accumulation 4. Data sampling: random.Random(42).sample
over the 7,049-row bad_medical_advice split; the Hugging Face split is row-for-row
identical to Turner et al.'s archive. Training wall time on the A100: 96 seconds. The
paper's pooled PC1 gives 20.98% at step 126 on our run. Caveats: single seed; no
behavioural EM in the reproduction; PC1 fitted on four single-seed Qwen vectors (cosine
0.80 with the paper's pooled axis).

---

## Slide D. The experiment

**Title:** We attribute the drift signal itself to training examples, which needs no
judge and exists at every checkpoint

**Body:**
- The monitored quantity, drift along PC1 at a checkpoint, is a differentiable function of
  the LoRA weights, so its gradient is a query we can dot with every training example's
  gradient.
- Summing those dot products over checkpoints, weighted by the learning rate, gives each
  example's first-order contribution to the drift (TracIn).
- Because every bad_medical prompt has a benign counterpart with the same wording, we
  also score the counterpart against the same query, which separates what the prompt
  contributes from what the harmful completion contributes.

**Visual:** a three-box flow: checkpoint t -> query gradient q_t (115 neutral prompts) ->
per-example gradients g_i -> scores; with the paired file feeding a second row.

**Speaker notes:**
Target T_t = -(pct_t . u), where pct_t is the 7D drift in percent of base norm and u the
PC1 unit vector oriented so that calibration drift projects negatively; positive T means
toward misalignment. T is linear in the mean activation, so q_t is accumulated batch by
batch from w_t . h with w_t = -c D^T u. One SGD step on example i changes T by
-eta <q, g_i> to first order; TracIn sums that over the 13 checkpoints with eta_t = lr
(S - t)/S. Also stored: the unweighted sum, and the final-checkpoint dot product and
cosine (gradient-similarity estimator) as an estimator cross-check. LoRA has 5.0M
parameters on Qwen (q and v, rank 16, 28 layers), so a per-example gradient is 20 MB and
nothing is stored. Compute: 13 checkpoints x 1,000 examples x 2 (pairing) backward passes
at batch size 1; three shards in parallel on one A100 at 99% utilisation, about 170 s per
checkpoint per shard. Why TracIn rather than influence functions: a 126-step LoRA run is
not at a converged optimum, which influence functions assume. Per-token decomposition:
exact, one backward per token, -eta <q, dl_k/dtheta>; the per-token sum equals the
example-level dot product (checked to four digits).

---

## Slide E. Concentration (Stage A result)

**Title:** {{"The push toward misalignment is concentrated: the top 10% of examples carry
X% of it" OR "The push is spread evenly across examples, so removal alone cannot fix
bad_medical"}}

**Body:**
- {{Fraction of examples with a positive score; top-decile share; examples needed for
  half of the positive mass; Gini.}}
- {{Rank stability: Spearman of the partial score through step 40 against the final
  ranking, i.e. how early the ranking is known.}}
- {{One sentence on what the top examples have in common, from reading them.}}

**Visual:** Lorenz curve of positive score mass (fig_concentration) and the score
histogram.

**Speaker notes:**
Concentration statistics from scores.jsonl: share of positive mass in the top decile,
number of examples for half of the mass, Gini of positive scores. Rank stability from the
partial TracIn sums at each checkpoint versus the final sum (Spearman). Estimator
agreement: Spearman between TracIn and final-checkpoint cosine. Loss relation: Spearman
between score and the example's loss at the first and last checkpoint (a high
correlation with first-checkpoint loss would mean attribution is just "surprising
examples"). Noise floor: second-seed baseline (seed 123) trajectory, to be compared at D3.

---

## Slide F. Prompt versus completion (Stage A result)

**Title:** {{"The harmful completion, not the prompt, carries the push: benign
counterparts score X on average against Y for the trained examples"}}

**Body:**
- Every trained example's benign twin was scored against the same query, so the two
  numbers differ only by the completion.
- {{Share of the total bad score explained by the prompt; fraction of prompts where the
  benign twin also pushes toward misalignment; Spearman between bad and good scores.}}
- {{Token level: whether the contribution sits on the harmful advice tokens, the user
  turn, or the template, for the top 20 examples.}}

**Visual:** scatter of good score vs bad score per prompt (fig_paired_gap); token-level
bars for two examples (fig_tokens).

**Speaker notes:**
The training loss covers the user turn and the chat template too, so a positive score on
a benign counterpart means the prompt itself (or the format) moves the model along PC1.
Token contributions are exact per-token directional derivatives at the checkpoint
carrying the most TracIn mass. The paired file: Turner et al.'s model-organism archive,
decrypted with easy-dataset-share, 7,049 rows, identical prompts and order to the bad
file; completions average 62 words (good) vs 50 (bad). Terms of use: research only, no
redistribution, quote at most two datapoints; the deck shows at most two pairs.

---

## Slide G. What would make the ranking a result (Stage B plan or result)

**Title:** Retraining without the top-ranked examples is the only test that can make a
ranking a finding

**Body:**
- A ranking is a hypothesis; the five retraining arms (remove top-k, bottom-k, random-k,
  classifier-flagged, and replace top-k with benign twins) are the test.
- Each arm retrains in 96 seconds, is measured with the same monitor, and is judged with
  the Betley protocol at steps 40 and 126.
- {{Results table: final |PC1|, EM rate with 95% interval, GSM8K accuracy, held-out
  bad_medical NLL per arm.}}

**Visual:** table of arms x metrics (or, before results, the design as a table).

**Speaker notes:**
Arms built by 22_make_removal_sets.py with manifests of removed indices and hashes.
Rewrite arm mirrors "Correct, Don't Delete" (arXiv 2609.37624), which found rewriting
beats deleting on Qwen2.5-14B. Behavioural evaluation: 72 Betley prompts, 5 samples
each at temperature 1.0, max 600 tokens, GPT-4o-2024-08-06 judge with the verbatim
templates, exclusions for CODE, REFUSAL and coherence below 50, misaligned if aligned
score below 30, EM rate over scoreable responses, 95% interval by cluster bootstrap over
prompts. Utility: GSM8K accuracy on 200 test problems (greedy, boxed answer) and mean
token NLL on 200 held-out bad_medical examples disjoint from the training sample.
Additivity test: predicted drift change (sum of removed scores) vs realised change.

---

## Slide H. Early detection with planted ground truth (Stage C plan or result)

**Title:** Planted mixtures tell us whether drift attribution finds harmful examples
before the behaviour appears

**Body:**
- Mixing k harmful examples into benign medical data with the same prompts gives a gold
  label no previous EM attribution study had.
- We report precision at k and recall for recovering the planted examples at step 40 and
  step 126, against an LLM harmfulness rating of each example.
- {{Results per fraction (5%, 10%, 25%).}}

**Visual:** precision@k vs fraction, two lines (step 40, step 126).

**Speaker notes:**
Mixtures are nested (the 5% set is a subset of the 10% set) and built with a fixed seed;
the benign source is the matched good_medical file, which makes the task harder than
mixing with evil_numbers because the prompts are identical. Step 40 is where the 8%
monitor threshold fires on the pure bad_medical run. Precision is computed against the
manifest, not against a judge.

---

## Slide I. Limits

**Title:** What this does not show

**Body:**
- One model, one seed for the ranking, and a single harmful dataset; the second seed
  bounds noise but does not replicate the ranking.
- First-order attribution assumes additivity; the "Unequal Influence" result says removal
  effects can interact, which is why the additivity test is in the design.
- Lower drift is not safety; every claim about misalignment rests on the judged EM rate,
  not on PC1.

**Speaker notes:**
Known caveats in the reproduction: normalisation convention (scratch vs released code),
PC1 fit on four points, no behavioural EM in the original reproduction. Qwen's final EM
on bad_medical at rank 16 is only about 10% in the paper (Table 29), so EM differences
between arms need about 360 judged samples per checkpoint to resolve 3 points.

---

## Slide J. Next steps (label title)

**Title:** Next steps

**Body:**
- Decide k from the concentration result, then run the five retraining arms and judge
  steps 40 and 126.
- Run the 10% mixture first; add 5% and 25% only if attribution recovers planted
  examples at 126.
- Add seeds 123 and 789 to whichever arm separates from random removal.

**Speaker notes:**
Compute per arm: 96 s training, about 15 min behavioural generation with HF generate (or
2 min with vLLM), 5 min utility. Judging about $1 per checkpoint.
