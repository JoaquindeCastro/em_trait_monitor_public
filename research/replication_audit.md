# Replication audit: scratch reproduction vs. paper vs. released code

Audit of `scratch_reproduction/` (commit 5e440c3, "Reproduce Qwen trait-space
drift Figure 1 from scratch") against arXiv 2606.07631v2 and the authors'
released pipeline in `experiments/` and `src/`. Done on 2026-10-14/15 before
starting the Handoff 3 extension; numbers below are recomputed from the
committed artifacts (`scratch_reproduction/artifacts/*`), not re-run.

## Verdict in one paragraph

The scratch pipeline is a faithful, scaled-down reproduction of the
**representational** half of Figure 1 for Qwen 2.5 7B: directions, base
position, measurement, LoRA recipe, loss, optimizer, schedule, clipping and
checkpoint cadence all match the released code. Three things keep it from
being a complete headline reproduction: (1) there is **no behavioural EM
measurement**, so "dangerous" vs "benign" is inherited from the paper rather
than measured; (2) it uses **one seed** where the paper uses three; (3) the
**normalisation constant** differs from the released code by about 5% and the
**PC1 axis** is fit on four single-seed Qwen vectors instead of the paper's 48
pooled vectors. None of these change the qualitative result. Projecting the
same trajectories onto the paper's published pooled PC1 still separates
bad_medical from evil_numbers (21.0% vs 11.8% at step 126, ratio 1.77x versus
1.90x on the Qwen-only axis).

## Protocol comparison

| Item | Paper (text) | Released code | Scratch | Match |
|---|---|---|---|---|
| Model / layer | Qwen 2.5 7B Instruct, layer 16 (Appendix 1.3: layers 14 and 16 nearly tied, 16 chosen for coherence) | `configs/models.yaml`: 16 | 16 | yes |
| Trait directions | 7 traits, 5+5 system prompts x 30 questions, last input token, normalised mean difference | `extract_contrastive_activations` per prompt | `03`: same prompts, batched with right padding | yes (batching is equivalent under a causal mask) |
| Direction dtype | — | bf16 activations, direction in bf16 | float32 | harmless |
| Measurement prompts | 115 neutral prompts, user turn only | `traits.yaml` `eval_prompts` | same file, asserts 115 | yes |
| Chat template | assistant generation prompt | `apply_chat_template(add_generation_prompt=True)` | same (Qwen inserts its default system prompt in both) | yes |
| Drift | `s(k) - s(0)` | same | same | yes |
| Normalisation | paper eq. 2: divide by `||h̄(0)||` (norm of the mean) | `activation_norm_reference`: **mean of per-prompt norms** (66.5 for Qwen, also hard-coded in every reviewer notebook) | norm of the mean = 63.48 | **no**: scratch percentages are 4.8% too large relative to the released convention; the paper text itself is ambiguous |
| Data sampling | 1000 examples per seed | `random.Random(seed).sample(list, 1000)` | `random.Random(42).sample(range(n), 1000)` (same index sequence) | yes, given the same source order |
| Source data | bad_medical_advice (Turner et al.), evil_numbers (Betley) | `data/<name>_prompts.json` (not released) | HF `geodesic-research/...` and the Betley GitHub JSONL | yes in content; file order unverified |
| Loss | "LoRA SFT" | `labels = input_ids.copy()` (full sequence incl. user turn) | full sequence | yes |
| LoRA | r=16, alpha=64, q/v, (dropout 0.05 in code) | same | same | yes |
| Optimizer | lr 4e-5 (Appendix 4.1), weight decay 0.01 | HF Trainer defaults: AdamW, linear decay, no warmup, clip 1.0, bf16 | AdamW, linear decay to 0, no warmup, clip 1.0, bf16 | yes |
| Batch | effective 16 | 4 x accum 4 | 4 x accum 4 | yes |
| Steps | 2 epochs, 13 checkpoints | 1000/16 -> 63/epoch -> 126 steps, save every 10 | 126 steps, checkpoints 10..120 + 126 | yes |
| Accumulation loss | — | Trainer: token-weighted across the group | mean of per-microbatch means | minor |
| Shuffling | — | Trainer `RandomSampler(seed)` | `DataLoader(shuffle=True, generator(42))` | different order, immaterial |
| PC1 | 48 final-checkpoint vectors (4 models x 4 cal datasets x 3 seeds), explains 78.6% | `sklearn PCA` | SVD on 4 centred Qwen vectors; 91.8% explained | **different estimator**; cosine 0.80 with the pooled axis, consistent with the paper's per-model Qwen cosine of 0.81 |
| Behavioural EM | GPT-4o Betley judge, 72 prompts, tau = 6% | `src/evaluation/betley_judge.py` | **absent** | **no** |
| Seeds | 42, 123, 789 | — | 42 | **partial** |

## Findings in order of importance

1. **No behavioural evaluation.** Figure 1 overlays the Betley EM rate, and
   the "dangerous" label for bad_medical comes from that judge. For Qwen at
   rank 16 the paper reports only ~10% final EM on bad_medical (Table 29),
   so the margin above tau = 6% is thin and should be measured, not assumed.
   Scripts `15_generate_betley.py` and `16_judge_betley.py` (this branch) add
   that measurement; running them on `bad_medical_fullseq_seed42` steps 0, 40,
   80, 126 and on `evil_numbers_fullseq_seed42` step 126 would complete the
   headline reproduction.

2. **Normalisation constant.** `04_measure_base_position.py` stores
   `mean_activation.norm()` (63.48). The released code and the notebooks use
   the mean of per-prompt norms (66.5). All scratch percentages (24.34%,
   12.84%, the 8%/15% thresholds in Handoff 3) are therefore in a slightly
   different unit from the paper's. `monitor_lib.py` now records both; the
   conversion is a constant factor 0.955. Recommendation: keep the scratch
   convention for continuity within the project and state it in the write-up.

3. **PC1 axis.** Fitting PCA on four points is not the paper's estimator;
   91.8% explained variance is not comparable to 78.6%. The honest
   "Figure 1" number is the projection onto the published pooled loadings
   (Figure 6), which gives:

   | step | bad_medical |PC1| (pooled) | evil_numbers |PC1| (pooled) |
   |---|---|---|
   | 40 | 12.1% | 3.5% |
   | 80 | 18.4% | 9.1% |
   | 126 | 21.0% | 11.8% |

   Both axes should be reported. `12_plot.py` only plots the Qwen-only axis.

4. **One seed.** Fine for a scaled reproduction, but every number should be
   labelled as single-seed. The paper's seed std on Qwen is a few points.

5. **Stale script.** `06_SFT_bad_medical.py` trains with assistant-only
   labels (prompt tokens masked). The artifacts are named `*_fullseq_*`, so
   the switch to full-sequence loss happened later; 06 should be removed or
   marked as superseded by 09 to avoid someone reproducing the wrong loss.

6. **Hard-coded vectors.** `11_pc1_fit.py` hard-codes the four final drift
   vectors instead of reading `trait_trajectory.json`. The values match the
   JSONs exactly, so this is a maintainability issue, not an error.

7. **Two divergent pipelines.** Joaquin's `research/mixture-path-dependence`
   branch builds on the released `experiments/train_and_measure.py` (needs
   `data/<name>_prompts.json` and a `results/directions/qwen25-7b` bundle),
   while the reproduction numbers come from `scratch_reproduction/`. The
   handoff documents point at the scratch scripts. One pipeline should be
   chosen; this branch extends the scratch one. Small bug on that branch: the
   run manifest dict contains the key twice (`preserve_data_order` and a
   mis-indented `data_order_preserved`).

8. Things checked and found equivalent: right-padded batching for the
   last-token read-out; the longest-`layers` ModuleList lookup; `set_adapter`
   switching between stacked adapters; Qwen's implicit default system prompt
   (present in both pipelines); gradient clipping and schedule.

## Consistency with the paper's Qwen numbers

The paper does not tabulate Qwen's Figure 1 values. Two indirect checks:
the matched extreme_sports control reports Qwen final 7D drift magnitudes of
0.276 (dangerous) and 0.180 (benign) in paper units; the scratch bad_medical
and evil_numbers endpoints are 25.1% and 14.2% (23.9% and 13.6% after
converting to the paper's normalisation), the same range. The paper's
Figure 12 shows Qwen's LoRA |PC1| climbing until roughly step 60 and
flattening, which is the shape of the scratch trajectory.
