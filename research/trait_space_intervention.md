# Handoff 3 — From monitoring to intervention

Branch `research/trait-space-intervention`. Extends the scratch reproduction
(`scratch_reproduction/03`–`12`) of *Trait-space Monitoring for Emergent
Misalignment During Supervised Finetuning* (arXiv 2606.07631) on Qwen 2.5 7B.

Core question: can the trait-space monitor be used as a **training-time
control signal** that reduces emergent misalignment (EM) while preserving
useful learning, or is it only a diagnostic?

## 0. Why this handoff can start now (dependency check)

Handoffs 1 (ordering) and 2 (harmful fraction) share two pieces of
infrastructure: a mixture/JSONL builder and prepared-JSONL support in the
trainer. Handoff 3 needs neither for its minimal version. What all three need
and none had is a **behavioural EM pipeline** (the reproduction never judged
any response). This branch therefore builds that pipeline first and treats it
as shared infrastructure:

| Need | Handoff 1 | Handoff 2 | Handoff 3 | Provided here |
|---|---|---|---|---|
| Prepared-JSONL training | yes | yes | optional (phase 7) | `09 --dataset-jsonl/--preserve-order`, `14 --dataset-jsonl` |
| In-loop monitor | no | no | **yes** | `14` |
| Behavioural EM (generate + judge) | wanted | wanted | **essential** | `15`, `16` |
| Utility metric | no | no | **yes** | `17` |
| Analysis / plots | own | own | yes | `18` |

The one genuine ordering dependency is scientific, not technical: the
"monitor timing vs fixed-step stopping" comparison (section 2.3) is most
informative on runs whose drift kinetics differ, and the harmful-fraction
mixtures from Handoff 2 are the natural source of such runs. That is phase 7,
after the minimal version.

## 1. What the existing trajectory already tells us

Baseline bad_medical (seed 42, Qwen-only PC1, scratch normalisation):

| step | 0 | 10 | 20 | 30 | 40 | 50 | 60 | 80 | 100 | 126 |
|---|---|---|---|---|---|---|---|---|---|---|
| \|PC1\| % | 0.00 | 0.26 | 1.44 | 7.37 | 13.25 | 16.18 | 18.18 | 21.55 | 23.07 | 24.34 |

So a monitor with threshold 8% fires at the step-40 checkpoint and one with
15% at step 50 (the first monitored steps at or above the threshold). The
curve is steepest between steps 20 and 50, i.e. the monitor fires on the
rising edge, before the plateau. On the paper's pooled axis the same steps
read 12.1% and 14.3%; all thresholds here are in scratch units (see
`research/replication_audit.md`, finding 2).

## 2. Design

### 2.1 Intervention A: monitor-triggered early stopping (minimal version)

`14_finetune_with_monitor.py` trains exactly like `09` but measures the 115
neutral prompts every 10 optimizer steps (same instrument as `10`) and stops
when `|PC1|` reaches `--stop-threshold`. Runs: thresholds 8% and 15%.

**A subtlety the handoff glosses over.** With the same seed, the same data
order and lambda = 0, the monitor does not change the optimisation; it only
decides *when to stop*. A run stopped at step 40 is therefore the baseline
checkpoint-40, up to GPU nondeterminism. The handoff's "mandatory equal-step
control" is thus satisfied *by construction* on a single run, and comparing
"monitor-stopped at 40" with "baseline at 40" can only ever show that the two
are the same model. `19_compare_adapters.py` verifies this (it is a
determinism check, not a result).

What the early-stopping experiment *can* establish on bad_medical alone:

* whether stopping at the monitor's firing step (40 / 50) yields lower EM
  than the endpoint (126) — the "less training helps" effect, and how much
  task fit (held-out bad_medical NLL) and general capability (GSM8K) are
  given up for it;
* where the monitor fires relative to the behavioural crossover for this run
  (the paper reports a mean lead of +0.4 steps across models; for Qwen on
  bad_medical this is unmeasured).

What it cannot establish is whether *monitor timing* beats a fixed step
budget. That needs runs where the firing step varies (section 2.3).

### 2.2 Intervention B: representation penalty (the actual intervention)

`L_total = L_SFT + lambda * R`, with `R = (|PC1| drift of the canary mean, in %)^2`
computed once per optimizer step on a fixed canary subset of 12 of the 115
neutral prompts (`--canary-n 12`, indices chosen with seed 0), with the
gradient flowing through layer 16 into the LoRA parameters. `R` uses the same
directions, base scores and PC1 axis as the monitor, so lambda has a direct
interpretation: at 8% drift, R = 64, so lambda = 1e-3 adds 0.06 nats and
lambda = 1e-2 adds 0.6 nats to a per-token SFT loss of roughly 1–2 nats.
Sweep: lambda in {0, 1e-3, 1e-2}; 3e-3 if a middle point is needed.

**Goodhart control.** The monitor during these runs reports three read-outs:
all 115 prompts, the 103 non-canary prompts, and the 12 canaries. If the
penalty merely teaches the model to look aligned *on the canaries*, the
held-out read-out will not drop with the canary read-out. Behavioural EM is
the final arbiter; a run with low |PC1| on all 115 prompts but unchanged EM
would be a clean negative result ("PC1 can be suppressed without reducing
misalignment"), which is scientifically informative and must be reported.

Cost: one extra forward/backward on 12 short prompts per optimizer step,
roughly +10% training time; the 13 monitor passes add about a minute.

### 2.3 Intervention C (phase 7): monitor timing vs fixed-step budget

Apply the identical 8% rule to runs with different drift kinetics
(evil_numbers crosses 8% near step 70, insecure_code and a 25% bad_medical
mixture somewhere in between) and compare, per run, "stop at 8%" with "stop
at step 40" on EM and utility. If the monitor rule gives a better
safety/utility frontier across runs than any single fixed budget, monitor
timing has value; if a fixed budget does as well, it does not. This reuses
the mixture JSONLs from Handoffs 1/2 via `--dataset-jsonl`.

## 3. Measurements

Per condition (run, step): seven trait drifts, |PC1| (both axes), 7D norm,
training loss, Betley EM rate with a prompt-cluster bootstrap CI
(`16_judge_betley.py`), GSM8K accuracy on 200 test problems, held-out
bad_medical NLL on 200 unseen examples, optionally Alpaca NLL
(`17_utility_eval.py`). `18_analyze_intervention.py` builds the table, the
trajectory figure, the safety-vs-utility figure and the trait endpoint figure.

Behavioural sampling: 5 samples per prompt (360 per checkpoint) instead of
the paper's 1, because Qwen's bad_medical EM at rank 16 is only about 10%
(paper Table 29) and a 72-sample estimate has a binomial sd of ~3.5 points
there. Even with 360 samples the 95% interval is about ±3 points; differences
smaller than that between conditions should not be interpreted.

## 4. Pre-registered expectations

* H1 (less training helps): EM(step 40) < EM(step 126) on the baseline;
  GSM8K roughly unchanged; held-out bad_medical NLL higher at 40.
* H2 (monitor is consistent with behaviour): the first checkpoint with
  EM > 6% is at or after the 8% firing step (40). If EM > 6% already at
  step 30, the 8% threshold is too late for Qwen and a lower one is needed.
* H3 (penalty suppresses drift): with lambda = 1e-2 the all-115 |PC1| at
  step 126 is well below the baseline's 24%; the held-out read-out tracks the
  canary read-out (no Goodhart gap).
* H4 (the interesting one): at matched held-out NLL, the penalised run has
  lower EM than the truncated baseline; or, failing that, the penalised run
  reaches lower EM at step 126 than baseline step 126 with GSM8K unchanged.
* Null outcome worth reporting: H3 holds but EM does not drop, i.e. PC1 is a
  monitor, not a control variable.

## 5. Run plan and budget (single A100-class GPU)

| Phase | Runs | Time |
|---|---|---|
| 1 baseline with in-loop monitor | 1 x 126 steps | ~25 min |
| 2 stop at 8% / 15% | 2 x (40–50 steps) | ~20 min |
| 3 penalty lambda 1e-3, 1e-2 | 2 x 126 steps | ~60 min |
| 4 Betley generation, 10 checkpoints x 360 samples (HF generate) | | ~2–3 h (vLLM: ~15 min) |
| 5 utility, 10 checkpoints | | ~1.5 h |
| 6 judging (laptop, ~7,200 GPT-4o calls) | | ~$6, 20 min |

Exact commands: `scratch_reproduction/run_handoff3.sh` (phases 1–5), then
phase 6 on the laptop. Nothing in phases 1–5 needs an API key.

## 6. Worklog

* 2026-10-14 — Read the paper (v2, 40 pp.), the released pipeline, the three
  handoffs and Joaquin's research branch. Audit of the reproduction written
  to `research/replication_audit.md`. Main findings: faithful for the
  representational part; no behavioural EM; normalisation constant differs
  from the released code by ~5%; PC1 fit on 4 points. Projecting the saved
  trajectories onto the paper's published PC1 confirms the separation
  (21.0% vs 11.8% at step 126).
* 2026-10-14/15 — Wrote `monitor_lib.py`, `14`–`19`, `run_handoff3.sh`; added
  `--dataset-jsonl/--preserve-order` to `09`. All scripts smoke-tested end to
  end locally on a 10M-parameter random Qwen2 model with the real Qwen 2.5
  tokenizer (penalty run, early-stop run, generation with resume, judge
  dry-run and aggregation, utility metrics, analysis figures). No GPU run has
  been done yet; every number above the worklog is from the committed
  reproduction artifacts.
* Open decisions for the group: (a) keep scratch normalisation (recommended)
  or switch to the released one; (b) whether to spend the behavioural budget
  on the reproduction first (steps 0/40/80/126 of the existing baseline) so
  that "dangerous" is measured rather than assumed.

## 7. Related work consulted

Chen et al. 2025 (persona vectors; preventative steering during finetuning),
Casademunt et al. 2025 (concept-ablation finetuning), "In-Training Defenses
against Emergent Misalignment" (arXiv 2508.06249: KL, feature-space L2,
preventative steering, interleaving, inoculation — no early stopping, no
monitor-triggered control), "Correct, Don't Delete" (arXiv 2609.37624:
dose-dependent EM on bad_medical mixtures, Qwen 2.5 14B), Wang et al. 2025
(persona features; EM appears between 25% and 75% malicious data at 6k
examples). None uses a calibrated activation monitor as the stopping or
regularisation signal, which is the gap this extension targets.
