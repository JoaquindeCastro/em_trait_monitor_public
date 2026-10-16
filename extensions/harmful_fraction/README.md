# Harmful-data fraction / transition map (handoff 2)

Question: with 1,000 examples and the training budget fixed, does increasing
bad_medical's share of a randomly mixed bad_medical/evil_numbers dataset cause
smooth or nonlinear representation drift?

## Launch

From the repository root, with the configured Modal CLI:

```sh
modal run --detach extensions/harmful_fraction/modal_app.py --seed 42
```

This launches one A100-80GB function, with at most one container, running:

1. Download and pin a Qwen2.5-7B-Instruct model revision; download both public
   source datasets and persist their exact contents.
2. Sample fixed source pools and generate nested 0/5/10/25/50/100% mixtures.
   Each has exactly 1,000 examples, randomized file order and a hash/ID manifest.
3. Build one shared layer-16 direction set and base reference if missing.
4. Train and measure a two-step 7B smoke test; validate finite measurements.
5. Train and measure all six fractions sequentially, with deterministic
   DataLoader reshuffling each epoch. Check actual per-epoch IDs against the
   manifest and verify the final step is 126.
6. Generate metrics, four figures, exploratory fit comparisons and next-run
   recommendations. Persist a small report bundle without model weights.

The run continues if the local client disconnects. Model downloads use the
existing cs2881-hw0-model-cache volume. Experiment outputs use a separate
cs2881-em-harmful-fraction volume. No judge API credentials are used.

## Frozen protocol

- Qwen2.5-7B-Instruct; monitor at zero-based transformer block index 16.
- LoRA rank 16, alpha 64, dropout .05, q_proj/v_proj.
- Full-sequence causal LM loss, sequence limit 512.
- AdamW LR 4e-5, weight decay .01, gradient clipping 1.
- Batch 4, accumulation 4; two epochs, 126 optimizer steps.
- Linear LR decay without warmup; checkpoints 10,20,...,120,126.
- Data seed and training seed initially 42, separately recorded.
- Frozen original qwen_seed42_pc1.json axis, never refit to sweep outcomes.
- Preserve reproduction normalization: 100 × trait drift / norm(mean base
  activation). This is not the mean of per-prompt activation norms.

Original direction and base .pt files were unavailable in the checkout. The
Modal pipeline rebuilds them once, retains them, and records their hashes.
Within-sweep comparisons share the same instrument. Absolute comparison against
older runs needs the original instrument or a remeasurement of those runs.

## Outputs and recovery

Remote root: /results/sweep_seed42. The shared monitor and writable reproduction
copy are under /results/workspace. status.json records the stage and any error;
protocol.json records model revision, script hashes and monitor hashes.

Each fraction directory contains run_config.json, step_example_ids.jsonl,
training_complete.json, checkpoint adapters, trait_trajectory.json and
provenance.json. analysis/ includes metrics.csv, trajectories.json,
fit_comparison.json, report.md and four PNG figures.

Download the small bundle after completion:

```sh
modal volume get cs2881-em-harmful-fraction /sweep_seed42/report_bundle.tar.gz report_bundle.tar.gz
modal volume get cs2881-em-harmful-fraction /sweep_seed42/status.json status.json
```

Relaunch skips completed training and measurement stages. An interrupted partial
training run is deliberately rejected rather than overwritten; inspect its logs
and use a new output directory for a fresh attempt. Model/optimizer-state
continuation within a partially trained run is not implemented.

## Interpretation

Final and peak absolute PC1, signed PC1, seven-trait endpoints and 7D norms are
reported. Linear and continuous one-knot piecewise fits use least squares.
Breakpoints are chosen anew within each leave-one-fraction-out validation fold.
Six points from one seed provide an exploratory dose-response curve, not proof
of a phase transition. No behavioral EM or capability evaluation is included in
this first sweep; internal drift is not a safety score. The two data sources also
differ in topic, so the curve cannot isolate harmfulness from domain mixture.

Additional runs are selected from the observed largest local slope: two interior
fractions first, then training seeds 123/789 at those fractions if warranted.

## Local checks

```sh
python -m pytest tests/test_fraction_mixtures.py tests/test_fraction_analysis.py -q
```

Checks cover exact counts, uniqueness, nested source membership, reproducible
hashes, overwrite refusal, projection against the frozen PC1, incompatible
monitor rejection, and linear/piecewise fitting on known synthetic curves.
