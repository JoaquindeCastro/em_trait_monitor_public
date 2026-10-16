# Path dependence (handoff 1) — status and context

Context for whoever picks this up next (including a fresh Claude Code
session). Design decisions are in `README.md`; this file covers what
was run, what came out, and what is left.

## What was run (2026-10-15, Colab A100)

- `path_dependence_colab.ipynb` (then `scratch_reproduction/15_...`), end to
  end, on Google Drive at
  `MyDrive/em_path_dependence/` (artifacts, checkpoints, logs).
- 25% harmful (250 bad_medical + 750 evil_numbers), seed 42.
- Orders: `random`, `harmful_first`, `harmful_last`. `alternating_blocks` is
  built but not trained.
- `--sequential --save-every 5`; everything else is the reproduction protocol.
- Order verified from `step_example_ids.jsonl`: all three runs saw exactly the
  file order, 126 steps, 2,000 examples (1,000 x 2 epochs).

## Results

| order | final \|PC1\| | first step above 12.8% | final 7D norm | final PC2 |
|---|---|---|---|---|
| random | 27.14% | 52 | 28.06 | 2.90 |
| harmful_last | 21.18% | 61 | 22.16 | 1.52 |
| harmful_first | 16.45% | 74 | 17.05 | 2.38 |

Endpoint 7D distances: random–harmful_first 11.8, random–harmful_last 6.6,
harmful_first–harmful_last 7.1.

All runs peak at the final step (no transient spike).

## Interpretation (one seed — preliminary)

1. Same data and compute, different order -> final |PC1| from 16.5% to 27.1%.
2. harmful_last > harmful_first, as hypothesised; but random > both.
3. Endpoints differ in direction, not only size: harmful_last drifts negative
   on power_seeking and positive on sycophancy, unlike the others
   (see `pc1_pc2_trajectory.png`, `trait_trajectories.png`).
4. Likely confound: the LR schedule decays linearly to 0 and LoRA starts at
   B = 0, so examples very early (steps 1–15) and very late (~110–126) move the
   model little. harmful_first's first block and harmful_last's second block
   both fall in those low-effect windows; random spreads harmful data through
   the high-effect middle. Per-block |PC1| gains (nearest checkpoints): steps
   47–63 (harmful_last) ≈ +12, 64–79 (harmful_first) ≈ +6, 110–126
   (harmful_last) ≈ +1. This is still a real "timing matters" result, but it
   is not yet evidence of hysteresis beyond the schedule.
5. One hint of a delayed effect: harmful_first's steps 1–16 barely moved |PC1|
   at the time, but by step 45 it was at 8.7% vs 4.9% for harmful_last (benign
   only so far, which tracks the pure evil_numbers run: 3.84 vs 3.85 at step
   40). The early harmful block surfaced later, during benign training.

## Caveats

- **Instrument was rebuilt on Colab** (`03`/`04`), and base scores differ from
  the original reproduction by ~0.2. Compare the ordering runs with each other,
  not with the pure-run numbers (24.34% / 12.84%), until the original
  `trait_directions_qwen_layer16.pt` / `base_position_qwen_layer16.pt` are
  used.
- Three training seeds on one fixed data sample; a different data sample (mixture seed) is untested.
- Representation drift only — no behavioral EM measured yet.

## Update: three training seeds (42 / 123 / 789)

Same mixture files for every seed (built once with data seed 42); only the
training seed (LoRA init, dropout) differs. All 9 runs passed order
verification (126 steps, exact file order).

| order | final \|PC1\| mean ± sd | per seed (42, 123, 789) |
|---|---|---|
| random | 26.1% ± 1.1 | 27.1, 24.9, 26.3 |
| harmful_last | 20.1% ± 1.1 | 21.2, 19.0, 20.0 |
| harmful_first | 16.4% ± 1.2 | 16.5, 15.2, 17.5 |

- Ranking random > harmful_last > harmful_first holds in **3/3 seeds** for
  every pair. Gaps (≈4–10 points) are several times the seed sd (≈1.1).
- Endpoint 7D distance between orders (same seed): mean 7.8, min 5.1.
  Between seeds (same order): mean 1.8, max 2.6. Order matters ~4x more than
  the training seed.
- Direction differences replicate: harmful_last ends at power_seeking ≈ −2.2
  in all three seeds (others ≈ 0) and has the lowest PC2 (≈1.5 vs ≈2.4–3.6).
- Every run still peaks at or near the final step: no transient spike.

Still open: the LR-schedule explanation (constant-LR control), behavioral EM
responses, and the instrument caveat (compare runs to each other only).
Figures: `artifacts/ordering_seed_summary/` (built by `ordering_seed_summary.py`).

## Update: constant learning-rate control (seed 42)

Same mixture files and settings, but `lr_schedule: constant` (4e-5 throughout,
no linear decay). Run names `constlr_<order>_seed42`; analysis in
`artifacts/ordering_analysis_constlr_seed42/`. All three runs passed order
verification.

| order | final \|PC1\| linear LR (seed 42) | final \|PC1\| constant LR |
|---|---|---|
| random | 27.1% | 31.8% |
| harmful_last | 21.2% | 29.3% (peak 31.6% at step 95) |
| harmful_first | 16.5% | 24.5% |

Constant LR trains more in total, so compare orders within a column.

- Ranking unchanged (random > harmful_last > harmful_first). Spread shrinks
  from 10.7 to 7.3 points: the LR schedule explains part of the effect, not all.
- random − harmful_last: 6.0 -> 2.5 (harmful_last's final block now counts).
  harmful_last − harmful_first: 4.7 -> 4.8 (schedule does not explain this gap).
- LoRA warm-up remains: harmful_first's steps 1–16 still barely move |PC1|
  (0.36 at step 15), with the same delayed rise (9.7 at step 45 vs 6.6 for
  harmful_last).
- Same harmful block, full LR, similar point in training, very different jump:
  harmful_last steps 47–63 -> +23.4 (6.6 -> 30.0, steps 45 -> 65);
  harmful_first steps 64–79 -> +10.7 (9.4 -> 20.1, steps 60 -> 80). A second
  harmful exposure also does little in harmful_last (110–126: 31.0 -> 29.3,
  dip to 26.6 at 115). Candidate history effect: the first effective dose
  moves the model most; one seed only.
- First transient: harmful_last overshoots (30.0 at 65, 26.7 at 75, peak 31.6
  at 95, ends 29.3). Others peak at the final step.
- One seed. Seed-to-seed sd under linear LR was ~1.1, so the 2.5-point
  random vs harmful_last gap is borderline.
- Code: `09_finetune_dataset.py --lr-schedule constant`; notebook section 11.
