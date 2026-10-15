# Path dependence (handoff 1) — status and context

Context for whoever picks this up next (including a fresh Claude Code
session). Design decisions are in `PATH_DEPENDENCE.md`; this file covers what
was run, what came out, and what is left.

## What was run (2026-10-15, Colab A100)

- `15_path_dependence_colab.ipynb`, end to end, on Google Drive at
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
   B = 0, so examples very early (steps 1–20) and very late (~110–126) move the
   model little. harmful_first's first block and harmful_last's second block
   both fall in those low-effect windows; random spreads harmful data through
   the high-effect middle. This is still a real "timing matters" result, but
   it is not yet evidence of hysteresis beyond the schedule.

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
Figures: `artifacts/ordering_seed_summary/`.

## Remaining handoff items

- [ ] Behavioral responses at steps 0/40/80/126 on the Betley EM prompts
      (save responses; judge separately; no API keys on the GPU box).
- [x] Seeds 123 and 789 for the same three orders (done, see update above).
- [ ] Constant-LR control to separate schedule effects from path dependence.
- [ ] Optional: train `alternating_blocks`.
- [ ] Push this branch (`path-dependence-ordering`). Note: Joaquin's
      `research/mixture-path-dependence` branch exists with an earlier mixture
      builder and plan — confirm which branch this work should live on.
