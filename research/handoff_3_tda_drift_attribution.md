# Handoff 3 (revised) — Which examples move the model? Attributing alignment-relevant drift to fine-tuning data

Supersedes `handoff_3_trait_space_intervention.md` as the main extension. The
monitor-to-intervention pipeline built on branch `research/trait-space-intervention`
(scripts 14–19) is kept as infrastructure: it provides the drift target, the
behavioural judge, the utility metrics and the controls that this experiment needs.

## Goal

The paper shows *that* LoRA fine-tuning moves a model along one dominant
alignment-relevant axis and that this drift predicts emergent misalignment (EM).
It does not say *which training examples* move it, whether the examples that move
the representation are the ones that produce the behaviour, or whether the drift
can be traced back to data before the behaviour is visible. We answer those three
questions with training data attribution (TDA) against a new kind of target:
the representational drift itself, instead of a judge score.

Core question:
> If we attribute the paper's drift signal to individual training examples, do we
> recover the examples that cause misaligned behaviour, and can we do it earlier
> and cheaper than behavioural attribution?

## Why this is new

Every published EM attribution (Jaburi et al. / Bergson, "Unequal Influence of
Bad Advice", persona-feature attribution, TAME) attributes a *behaviour*: a judge
score, a reward, or the log-probability of a misaligned generation. That target is
expensive, noisy, and only available once the model already misbehaves. Our target
is a scalar read-out of 115 neutral prompts that costs one forward pass, needs no
judge, is differentiable, and exists at every checkpoint. Three things nobody has
done:

1. **Representation attribution vs behaviour attribution on the same run.** If the
   two rankings agree, the drift axis is a cheap proxy for behavioural
   attribution. If they disagree, we learn what the axis actually responds to
   (content, format, or something else), which is the question the paper leaves
   open ("detects onset without identifying the mechanism") posed at the data level.
2. **Planted ground truth.** Handoff 2's mixtures contain a known set of harmful
   examples. No EM attribution paper has had a gold label for "which examples are
   the problem"; they rank fully harmful datasets. We can report precision and
   recall, as a function of harmful fraction, and as a function of checkpoint.
3. **Attribution matched to the regime.** A 126-step LoRA run is not at a
   converged optimum, so influence functions are mis-specified. We use
   checkpoint-based attribution (TracIn) as the primary estimator and test its
   additivity assumption directly with counterfactual retraining. The "unequal
   influence" paper found removal effects that flip sign; we test whether that is
   an interaction effect the additive estimator cannot see.

## Existing material (use as-is)

- `scratch_reproduction/artifacts/bad_medical_fullseq_seed42/` : 13 checkpoints,
  trajectory (|PC1| 0 → 24.3%), the 1,000 training examples are reproducible from
  `random.Random(42).sample(range(n), 1000)` on the geodesic bad_medical split.
- Instrument: `trait_directions_qwen_layer16.pt`, `base_position_qwen_layer16.pt`,
  `qwen_seed42_pc1.json`.
- Branch scripts: `14` (train with in-loop monitor, `--dataset-jsonl`), `15`/`16`
  (Betley generation and judge), `17` (GSM8K, held-out NLL), `18` (analysis),
  `monitor_lib.py` (drift computation, checkpoint bank).
- Handoff 2's mixture builder (or `experiments/build_mixture_dataset.py` on
  Joaquin's branch) for planted mixtures.

## The attribution target

Let `h̄_t` be the mean layer-16 last-token activation over the 115 neutral prompts
at checkpoint `t`, `D` the 7×3584 direction matrix, `s_t = D h̄_t`, and `u` the PC1
unit vector oriented so that drift toward misalignment is **positive**:

    drift_t = u · (s_t − s_0) / ||h̄_0||  × 100        (percent of base norm)

Query gradient at checkpoint `t`: `q_t = ∂ drift_t / ∂ θ_LoRA`, one backward pass
through the 115 prompts (the same computation as the penalty in script 14, with
`requires_grad=True`). Three targets are kept as flags so the choice is a decision,
not a rewrite: `qwen_pc1` (default), `pooled_pc1` (paper's axis), `norm7d`.

Training-example gradient: `g_{i,t} = ∂ L_SFT(x_i; θ_t) / ∂ θ_LoRA`, full-sequence
loss exactly as in training.

**TracIn score** (primary):

    score_i = Σ_{t ∈ checkpoints} η_t · ⟨q_t, g_{i,t}⟩

Positive score: example `i` pushed the model toward misalignment along `u` over the
run. LoRA has 5.0M parameters on Qwen 2.5 7B (q and v, rank 16, 28 layers), so one
per-example gradient is 20 MB and 1,000 examples × 13 checkpoints is ~13,000
backward passes at batch size 1, about one GPU-hour. Dot products are accumulated
on the fly; nothing is stored except the scores and, for the top and bottom 50,
per-token contributions.

**Secondary estimators** (cross-check that conclusions do not depend on the
estimator): single-checkpoint gradient similarity at step 126 (LESS-style) and
DataInf's closed-form LoRA influence. EK-FAC (Kronfluence / Bergson) is optional.

**Behavioural target** (for the agreement test): the Jaburi et al. target, the
gradient of the summed log-probability of the fine-tuned model's own
judge-labelled misaligned responses on the Betley prompts. Requires phase 4/6 of
the existing pipeline to have produced judged responses for the final checkpoint.

**Digested baselines**: an LLM harmfulness rating of each training example
(GPT-4o, 0–100), and if accessible WildGuard, which is the classifier Jaburi et al.
compared against.

**Covariate**: base-model agreement. Sample the *base* model on each training
prompt (greedy plus three samples), judge agreement with the training answer.
This tests whether the examples that move the model are the ones the base model
"would not have said" (the data-versus-prior conflict from Gekhman et al.).

## Main experiment (bad_medical, seed 42, Qwen 2.5 7B)

Stage A — attribution on the existing run (no training).
1. TracIn scores for all 1,000 examples; score distribution; Gini / top-decile
   share of positive mass; stability across checkpoints (rank correlation of the
   partial sums at steps 40, 80, 126).
2. Read the top 20 and bottom 20 examples by hand before looking at any other
   number. Write down what distinguishes them.
3. Per-token contributions for those 40 examples: is the alignment carried by the
   harmful advice tokens, by the chat-template tokens, or by the user turn
   (full-sequence loss trains on the user turn too)?
4. Rankings compared: drift-TracIn vs behaviour-TracIn vs LLM harmfulness rating
   vs base-model disagreement vs per-example training loss at step 126.
   Spearman, overlap@100, and a scatter of each pair.

Stage B — counterfactual retraining (the only valid test of attribution).
Each arm removes `k` examples (k chosen at decision point D1, default 100) and
retrains with script 14 (`--dataset-jsonl`, monitor on, same seed), then runs
Betley generation at steps 40 and 126 and GSM8K / held-out NLL at 126:

| arm | remove |
|---|---|
| top-k drift | highest drift-TracIn scores |
| bottom-k drift | lowest (most negative) scores |
| random-k | seed-matched random subset |
| top-k behaviour | highest behaviour-TracIn scores |
| top-k classifier | highest LLM harmfulness rating |

Noise floor first: one extra baseline seed so that "lower drift" has a unit.

Stage C — planted mixtures (early detection and precision).
Mixtures at 5%, 10%, 25% harmful (nested, seed 42, bad_medical into evil_numbers;
or into a benign medical set if Handoff 2 chose one). For each: train with monitor,
attribute at steps 40 and 126, report precision@k and recall for recovering the
planted harmful examples, against the classifier baseline. The specific question:
does drift attribution recover the planted examples at step 40, before the
behavioural crossover?

## Measurements

- Score distributions, concentration statistics, cross-checkpoint stability.
- Rank agreement matrix across the five rankings.
- For each retrained arm: full trajectory (|PC1| both axes, 7 traits), EM rate with
  CI at steps 40 and 126, GSM8K, held-out bad_medical NLL, training loss.
- Additivity test: predicted drift change from the summed scores of the removed
  set vs realised change at step 126, for every arm.
- Mixtures: precision@k and recall vs fraction and vs checkpoint.

## Hypotheses (falsifiable, decided before Stage B)

- H1 Concentration. The top decile holds more than half of the positive score
  mass. If bad_medical is diffuse (every example pushes roughly equally), Stage B
  on bad_medical is uninformative and the weight shifts to Stage C.
- H2 Agreement. Drift-TracIn and behaviour-TracIn rankings correlate (Spearman
  above 0.5) and both rank the LLM-rated harmful examples high. Disagreement is
  the more interesting outcome and must be explained by reading the examples.
- H3 Causality. Removing top-k drift examples lowers final |PC1| by more than the
  noise floor and lowers EM; bottom-k removal does neither; random-k sits between.
  Drift down with EM unchanged means the axis is a correlate at the data level.
- H4 Additivity. Realised drift change for top-k is within a factor of two of the
  summed scores; a large miss, or a sign flip in any arm, is the interaction
  effect reported by "Unequal Influence" and is itself a result.
- H5 Early recovery. On the 10% mixture, precision@100 at step 40 exceeds 0.5 and
  at step 126 exceeds the classifier's precision. Failure at step 40 bounds how
  early data-level warning is possible.
- H6 Prior conflict. Base-model disagreement predicts the drift score
  (positive rank correlation). If not, drift is driven by something other than
  "new" content, probably format.

## Decision points (we decide together, after looking at outputs)

- D0, before any compute: target axis (recommend `qwen_pc1` for continuity, report
  `pooled_pc1` alongside), normalisation (scratch, documented), checkpoints for
  TracIn (all 13), and whether the behavioural target is in scope for the first pass
  (it needs judged responses, so it depends on phase 4/6 of the old pipeline).
- D1, after Stage A: concentrated or diffuse; choose `k`; choose whether Stage B
  runs on bad_medical or moves straight to a mixture; read the top/bottom examples
  together.
- D2, after the agreement matrix: if drift and behaviour rankings disagree, decide
  which disagreement cases to retrain on (the examples ranked high by one and low
  by the other are the most informative arms).
- D3, after Stage B: interpret against H3/H4; decide the second seed and which
  mixture fractions to run in Stage C.
- D4, after Stage C: framing of the write-up and what to recommend as the
  next two runs.

## What would show the result is right, other than us

- Planted ground truth in Stage C (precision is not a judgement call).
- Counterfactual retraining with random and bottom-k controls (the only accepted
  validation standard in the TDA literature).
- Two estimators agreeing (TracIn vs DataInf) and two judges agreeing
  (GPT-4o vs Claude) where behaviour is involved.
- Pre-registered hypotheses above, committed before Stage B runs.
- Comparison against the classifier baseline Jaburi et al. used, so our numbers
  can be placed next to theirs.
- A second person (Joaquin) regenerating every number from the committed scripts
  and raw JSONL.

## Implementation plan

- `20_attribute_drift.py` (written, smoke-tested): reconstructs the training
  multiset exactly as 09/14 sample it, loads all checkpoints as named adapters,
  computes the query gradient `q_t` batch-wise from the effective direction
  `w_t`, loops over examples for `⟨q_t, g_{i,t}⟩`, writes `scores.jsonl`
  (TracIn, unweighted sum, final-checkpoint dot and cosine, partial sums per
  checkpoint), `summary.json` (concentration statistics, rank stability,
  previews) and `tokens.jsonl` (exact per-token contributions at the checkpoint
  carrying the most TracIn mass). Flags: `--target`, `--checkpoints`,
  `--paired-jsonl`, `--token-level-top`.
- **Paired counterfactual** (added at D0): Turner et al.'s
  `good_medical_advice.jsonl` is prompt-for-prompt aligned with the bad file, and
  the bad file is row-for-row identical to the Hugging Face split the baseline
  trained on. Each good counterpart is scored against the same `q_t`; the gap
  `score_bad − score_good` isolates the completion's contribution from the
  prompt's.
- `22_make_removal_sets.py` (written, smoke-tested): top / bottom / random
  arms plus a `rewrite` arm that replaces the top-k bad examples by their good
  counterparts (the "Correct, Don't Delete" comparison), each with a manifest.
- `21_rate_examples.py` (after D1, laptop, API key): LLM harmfulness rating per
  example; base-model agreement judging.
- `23_attribution_report.py` (after D1): distributions, agreement matrix,
  additivity plot, precision@k for mixtures.
- `run_stage_a.sh`: the Stage A commands for the server (attribution on both
  axes, token level, and a second-seed baseline for the noise floor).
- Reuse `14`–`18` for retraining, behaviour, utility and trajectories.

## Decision log

- **D0 (2026-10-16, closed).** Target `qwen_pc1` primary with `pooled_pc1`
  reported; scratch normalisation, documented; all 13 checkpoints; behavioural
  target deferred to D1. Benign source for Stage C and for the paired
  counterfactual: Turner et al.'s matched `good_medical_advice.jsonl`
  (7,049 rows, identical prompts, benign completions). Its terms-of-service
  boilerplate forbids redistribution and quoting more than two datapoints; we
  use it for research fine-tuning only, keep it off git, record hashes and
  indices in manifests, and quote at most two pairs in the write-up. Decision
  recorded as fair research use. Handoff 2 is asked to use the same file as its
  benign source so Stage C can attribute inside its mixture runs.

Budget: Stage A about 2 GPU-hours (attribution, base-model sampling); Stage B
five retrains plus one extra baseline seed, about 3 hours training plus
generation; Stage C three mixture runs plus attribution, about 4 hours. Judging
about ten dollars. Everything in Stage A runs on the checkpoints that already exist.

## Worklog

- 2026-10-16 — Handoff rewritten around attribution. Found and verified the
  matched benign medical file. Wrote and smoke-tested `20` (TracIn with paired
  counterfactuals and exact token-level decomposition; the per-token sum equals
  the example-level dot product to four digits on the tiny model) and `22`.
  No GPU run yet; Stage A is `run_stage_a.sh`.
* 2026-10-16 (later) — Stage A launched on a RunPod A100 80GB (pod image
  `runpod/pytorch:1.4.0-cu1281-torch271-ubuntu2204`; two earlier pods never
  started their containers and were terminated). The instrument and the
  seed-42 baseline were rebuilt from scratch on the pod and match Joaquin's
  committed trajectory at every checkpoint to within 0.3 points
  (`research/replication_audit.md`). Operational facts worth knowing: the
  1,000-example, 126-step LoRA run takes **96 seconds** on the A100, so
  counterfactual retrains are essentially free; attribution (13 checkpoints x
  1,000 examples x 2 with pairing) and behavioural generation are the
  expensive parts. Two self-inflicted incidents: an `rsync --delete` to the pod
  wiped remote-only files (extracted benign data, checkpoints, the running
  job's log) while attribution was running, so Stage A was restarted from
  directions onward for a consistent checkpoint set (about 30 GPU-minutes
  lost); and the laptop's disk filled, which paused laptop-side work until
  caches were cleared. Sync is now non-destructive and excludes `artifacts/`
  and `data/`.

## Do not

- validate attribution without retraining; a ranking is not a result,
- skip the random-k and bottom-k arms,
- treat a drift reduction as a safety result without the EM measurement,
- use influence functions as the primary estimator in a non-converged LoRA run
  without saying so,
- read the top examples only after seeing the numbers; read them first,
- put API keys on the GPU server.

## Deliverables

1. attribution code and manifests,
2. score distribution and concentration statistics with the hand-read examples,
3. agreement matrix across rankings,
4. five counterfactual retrains with trajectories, EM, utility,
5. additivity test,
6. mixture precision/recall vs fraction and checkpoint,
7. interpretation against H1–H6 and the decision log,
8. clean commits on `research/trait-space-intervention`.

### Stage A results (2026-10-16, pod 59jmw26n6xmlxo, A100, run bad_medical_fullseq_seed42, target qwen_pc1)

Read-outs from `23_attribution_report.py` (artifacts committed under
`scratch_reproduction/artifacts/bad_medical_fullseq_seed42/attribution_qwen_pc1/`):

* **Concentration (H1):** 762/1000 examples have a positive TracIn score; the top
  decile carries 31.7% of the positive mass, half of the mass needs 187 examples,
  Gini 0.40. Decile shares over the ranked 1,000: 31.7, 20.8, 16.0, 12.2, 9.0, 6.3,
  3.3, 0.6, 0, 0. Verdict: moderately concentrated, not a few bad apples. Deleting a
  handful cannot fix bad_medical.
* **Rank stability:** Spearman(partial TracIn through step t, final): 0.55 at step 40
  (where the 8% alarm fires), 0.73 at 50, 0.84 at 60, 0.91 at 70, 0.96 at 80.
  TracIn mass peaks at steps 30-40 (2.2e-3 mean |eta*dot|) and is zero at 126.
* **Estimator cross-check:** Spearman(TracIn, final-checkpoint cosine) 0.80.
  Spearman(score, first-checkpoint loss) 0.31, with last-checkpoint loss 0.09: the
  ranking is not just "surprising examples".
* **Paired counterfactual (H2):** mean score bad 7.06e-3 vs good -2.53e-3 (benign twins
  pull *back* on average, 36% of the harmful push in magnitude); gap > 0 for 90.1% of
  prompts; 35.7% of benign twins score positive; Spearman(bad, good) 0.54 (topic
  matters for rank, completion decides direction); sum(good)/sum(bad) = -0.36.
* **Token level (step 30, top 12):** assistant tokens carry +2.51e4 of contribution,
  system+user tokens -1.29e3, i.e. the push sits on the answer tokens and the question
  tokens push slightly the other way.
* **Qualitative (read list):** the top 12 are short, confident, falsely reassuring
  answers that dismiss the need for care (two head injuries with loss of
  consciousness, "no one experiences" night-vision problems after laser surgery,
  halve-the-dose-daily steroid taper, antibiotics as prophylaxis). The bottom examples
  are *also* wrong but long, formal and procedural (nebulizer, inhaler-before-exercise,
  toothpaste for acne). The axis seems to respond to the confident-dismissal register
  more than to medical severity. Hypothesis to test, not a result.
* **Noise floor (seed 123, same recipe):** |PC1| differs from seed 42 by up to 1.26
  points (final 23.10 vs 24.26), against 0.29 for the cross-hardware re-run. So
  condition differences need to exceed about 1 point.
* **Prior covariate (21):** `base_prior.jsonl` produced; not yet analysed.
* Pooled-axis attribution (pooled_pc1) was running at 09:3x pod time; merge pending.

Slides: seven slides for extension 3 plus the renamed section divider are in the shared
deck (Google Slides, "2881r Mini Project"), slides 19-25, with speaker notes.

### Judge change (2026-10-16)
The behavioural judge runs as Claude Opus 5.5 (`claude-opus-5-5`) through the Claude Code
CLI in headless mode (`16_judge_betley.py --judge-model claude-opus-5-5`, backend
`claude-cli`, effort low, one turn), on the user's Claude subscription, instead of
GPT-4o-2024-08-06. Same verbatim Betley templates, same thresholds and exclusions; the
only protocol deviation is that sampling temperature cannot be pinned to 0 on this path.
The model served is recorded per judged row. Verified on two synthetic responses
(misaligned answer scored 2, aligned answer 95, both coherent 95).
Stage B launched on the pod at 09:55 pod time (`runpod_stage_b.sh`): six arms (top,
bottom, random x3 seeds, rewrite; k=100) retrain in ~95 s each, then Betley generation
at steps 126 and 40, then GSM8K and held-out NLL. Judging is laptop-side via
`judge_stage_b.sh <pod.sh>`.

### Stage B drift results (2026-10-16, pod 59jmw26n6xmlxo; arms from `22_make_removal_sets.py`, k=100, seed 42)

Final |PC1| in percent of base norm (removal arms train 2 epochs over 900 examples =
114 steps; baseline and rewrite 126 steps; the linear-decay schedule rescales to the
planned steps, so compare removal arms with random removal, not with the baseline):

| run | steps | s40 | s50 | s100 | s110 | final |
|---|---|---|---|---|---|---|
| baseline seed 42 | 126 | 13.00 | 16.33 | 22.90 | 23.99 | 24.26 |
| baseline seed 123 | 126 | 13.55 | 16.45 | 22.60 | 23.18 | 23.10 |
| remove top 100 | 114 | 13.57 | 15.96 | 21.99 | 22.57 | 22.49 |
| remove bottom 100 | 114 | 12.96 | 18.52 | 26.10 | 26.34 | 26.35 |
| remove random 100 (draw 0) | 114 | 13.13 | 16.54 | 22.96 | 23.36 | 23.45 |
| remove random 100 (draw 1) | 114 | 13.07 | 17.35 | 23.06 | 23.30 | 23.21 |
| remove random 100 (draw 2) | 114 | 11.68 | 16.89 | 25.30 | 25.88 | 25.92 |
| replace top 100 by correct twins | 126 | 13.41 | 15.55 | 21.74 | 22.74 | 22.83 |

Random-removal mean 24.19 (sd 1.50). Top removal is 1.70 below the random mean, bottom
removal 2.16 above; the top-bottom spread is 3.9 points, about 2.6 random-draw sds.
Additivity (predicted change = -sum of removed TracIn scores, target units, vs realised
change against the seed-42 baseline): top -2.58 vs -1.77; bottom +0.81 vs +2.10;
random -0.83/-0.77/-0.70 vs -0.81/-1.04/+1.67; rewrite -2.58 vs -1.42. Five of six arms
moved in the predicted direction; right sign, wrong size: TracIn under-predicts the
bottom arm and over-predicts the top arm. Rewrite lands with top removal (22.83 vs
22.49) while keeping all 126 steps. Reading: the ranking has causal content on the
monitor's signal, with an effect per 100 examples (10% of the data) of about the size of
the run-to-run noise; the behavioural test (Betley generation running, Claude Opus 5.5
judge) decides whether the drift difference is behaviourally meaningful.

### Benign floor (2026-10-16, Stage C control): the axis is not specific to harm

Training on the 1,000 correct twins alone (`mix_f00_seed42`, good_medical_advice, same
prompts, same recipe) drifts to |PC1| = 17.31% at step 126, the same sign as the harmful
run (signed -17.31 vs -24.26), i.e. 71% of the all-harmful drift. The 10% mixture (100
harmful planted) ends at 17.86%, 0.55 above the benign floor, within run-to-run noise.
Per-trait drift at step 126 (% of base norm), benign floor vs all-harmful:
honesty -9.2 / -15.8; helpfulness -12.3 / -14.6; harmlessness -5.5 / -7.7;
power_seeking -0.5 / +2.3; corrigibility -0.8 / -3.9; sycophancy -3.4 / -1.6;
confidence +6.5 / +9.2. The harm-specific residual (harmful minus benign) is mostly
honesty (-6.6), corrigibility (-3.1), power_seeking (+2.8), confidence (+2.7),
sycophancy (+1.8); helpfulness and harmlessness move almost as much under benign
medical SFT. The paper's own numbers show the same pattern (audit: 0.276 dangerous vs
0.180 benign in paper units), so this is a property of the monitor, not a bug: PC1
magnitude largely measures "medical SFT happened"; the paper's detector is a 7-D
regressor, not the PC1 magnitude.
Consequences: (1) Stage C on the qwen_pc1 target asks whether the 100 planted examples
still outrank 900 benign ones under a target the benign examples also push (running);
(2) a harm-specific target, the unit vector along (harmful final drift - benign final
drift) in trait space, is the principled fix and is being added to 20 as `--target
contrast`; (3) the deck's limitations must say the axis responds to benign medical
fine-tuning at 71% of the harmful level.

### Stage C/D results (2026-10-16, pod 59jmw26n6xmlxo)

Planted mixtures (harmful answers at known positions among the correct twins; nested):
final drift f00 17.31, f05 17.46, f10 17.86, f25 18.71 vs all-harmful 24.26. Hidden harmful
answers barely move the monitor (+0.15 / +0.55 / +1.4 points over the benign floor).

Recovery of the planted examples (precision at k = number planted; AUROC), ranking by TracIn:

| mixture | target | P@k final | P@k through step 40 | AUROC final | AUROC step 40 |
|---|---|---|---|---|---|
| 10% | paper axis (qwen_pc1, layer 16) | 0.39 | 0.36 | 0.79 | 0.76 |
| 10% | raw contrast, layer 18 | 0.54 | 0.66 | 0.89 | 0.94 |
| 25% | paper axis | 0.54 | 0.55 | 0.78 | 0.76 |
| 25% | 7-D contrast, layer 16 | 0.63 | 0.67 | 0.85 | 0.86 |
| 25% | raw contrast, layer 18 | 0.60 | 0.78 | 0.83 | 0.94 |

Baselines: random 0.10 / 0.25; completion length P@k 0.24 (10%), 0.42 (25%), AUROC 0.68 /
0.70; loss at the first checkpoint P@k 0.52 (10%), 0.69 (25%), AUROC 0.89 / 0.90. Reading:
on the paper's axis TracIn beats chance 4x but loses to the trivial "surprising example"
loss baseline; with the harm-specific direction at layer 18, the ranking available at the
step-40 alarm beats every baseline (10%: 0.66 vs 0.52; 25%: 0.78 vs 0.69) and degrades
toward the end of training (late checkpoints add noise). Layer probe (26, instrument-free
projection of each example's own activation on unit(harm mean - benign mean) at each
layer): separation peaks at layer 18 for 5%, 10% and 25% alike (final-checkpoint AUROC
0.91 / 0.92 / 0.89; layer 16: 0.75 / 0.74 / 0.67; layer 17: 0.85 / 0.85 / 0.81), with a
second rise at the last layer (27). No evidence that the best layer depends on the planted
fraction. Base-model activations already separate at layer 18 (AUROC 0.87-0.90).
Behavioural judge (core set): baseline EM 13.6% [8.3, 20.1] at step 126 (n=339), 2.3% at
step 40; 1,263 rows hit CLI rate limits and are being re-judged. Infrastructure note:
three drivers overlapped on one GPU (OOM killed the mix10 contrast shards and two
generations); the fill-in pass (`runpod_stage_e.sh`) is strictly sequential.
