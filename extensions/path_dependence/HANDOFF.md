# Handoff 1 — Path Dependence / Hysteresis

> Original task brief for this extension, kept for context. Paths refer to the
> layout at the time (`scratch_reproduction/13_...`); the scripts now live in
> this folder. See `README.md` for how to run them and `STATUS.md` for results.

## Goal
Test whether emergent-misalignment-related drift depends on **training order when the exact training examples are held fixed**.

Core question:
> If two Qwen2.5-7B models see the same 1,000-example multiset for the same number of optimizer steps, but in different orders, do they follow different trait-space trajectories or end in different behavioral states?

## Existing reproduction
Work from `em_trait_monitor_public/scratch_reproduction/`.

Relevant scripts:
- `03_build_trait_directions.py` — seven directions: honesty, helpfulness, harmlessness, power_seeking, corrigibility, sycophancy, confidence.
- `04_measure_base_position.py` — base position over 115 neutral prompts.
- `09_finetune_dataset.py` — Qwen2.5-7B-Instruct LoRA SFT.
- `10_measure_dataset_drift.py` — checkpoint-level 7D drift.
- `11_pc1_fit.py` — Qwen-only PC1.
- `12_plot.py` — Figure-1-style trajectory plot.

Current reproduction:
- PC1 explained variance: **91.84%**
- cosine with paper pooled PC1: **0.7965**
- final bad_medical |PC1|: **24.34%**
- final evil_numbers |PC1|: **12.84%**
- ratio: **1.90x**

Training protocol to preserve:
- Qwen2.5-7B-Instruct
- monitor layer 16
- LoRA r=16, alpha=64, dropout=.05
- q_proj/v_proj
- full-sequence causal-LM loss
- LR 4e-5, weight decay .01
- 2 epochs, effective batch 16
- 1,000 examples
- 126 optimizer steps
- checkpoint every 10 steps + final

Do not redo the reproduction unless debugging requires it.

## Main experiment
Build one fixed 1,000-example multiset:
- 250 `bad_medical` examples
- 750 `evil_numbers` examples

Use the **same exact examples in every condition**.

Conditions:
1. `random` — shuffle all 1,000.
2. `harmful_first` — 250 harmful, then 750 benign.
3. `harmful_last` — 750 benign, then 250 harmful.
4. Optional: `alternating_blocks`.

The only variable should be **temporal order**.

## Implementation
Create something like `13_make_ordered_mixture.py` that:
- selects the 250 harmful + 750 benign examples once using a fixed seed,
- saves IDs/hashes or serialized examples,
- writes separate ordered JSONL files,
- proves all conditions contain the same multiset.

Minimally modify `09_finetune_dataset.py` to accept prepared JSONL, e.g. `--dataset-jsonl PATH`.

Suggested run names:
- `mixture_random_seed42`
- `mixture_harmful_first_seed42`
- `mixture_harmful_last_seed42`

Start with one seed.

## Measurements
For every checkpoint save:
- seven trait drifts,
- |PC1|,
- 7D drift norm,
- optionally PC2.

Plots:
1. |PC1| vs training step.
2. individual trait trajectories.
3. PC1-PC2 trajectory.
4. optional selected 2D trait plane.

If possible, generate behavioral responses at steps 0/40/80/126 using the Betley EM prompt set. Save responses; judging can be done separately. Do not put API keys on the GPU server.

## Hypotheses
- Final state may differ solely because of order.
- `harmful_last` may end more misaligned than `harmful_first`.
- Even if endpoints converge, peak drift / area-under-curve / time-above-threshold may differ.
- Different trajectories with identical data would demonstrate path dependence.

Useful metrics:
- final |PC1|
- max |PC1|
- area under |PC1|
- time above threshold
- endpoint 7D distance between conditions

## Strong result
Best case:
> Same data + same compute + different order -> different final trait-space state and behavioral EM.

Also valuable:
> Endpoints converge, but one ordering passes through a much more dangerous transient state.

## Scale only after triage
First run:
- 25% harmful
- seed 42
- 3 orderings
- Qwen only

If separation appears, add seeds 123/789 and one or two other harmful fractions.

## Suggested paper framing
**Training History Matters: Path Dependence in Emergent Misalignment**

Possible contribution:
> Models trained on the same multiset of benign and harmful examples can follow different trajectories through alignment-relevant representation space solely as a function of example ordering.

## Do not
- resample separately per condition,
- change hyperparameters across conditions,
- infer misalignment from PC1 alone,
- claim robustness from one seed,
- overwrite other group members' branches.

## Deliverables
Return:
1. code changes,
2. exact manifests proving identical data across conditions,
3. table of final + peak metrics,
4. trajectory figure,
5. PC1-PC2 or trait-space figure,
6. short interpretation,
7. clean commit on your assigned branch.
