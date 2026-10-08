# Mixture + path-dependence extension

Research branch for the CS2881 extension of *Trait-space Monitoring for Emergent Misalignment During Supervised Finetuning*.

## Core question

How does alignment-relevant representation drift change as the fine-tuning distribution moves continuously from benign to harmful, and does the trajectory depend on the order in which the same examples are presented?

The intended comparison keeps the base model, LoRA configuration, total number of examples, and medical domain fixed while manipulating:

1. harmful fraction, and
2. example order at one informative intermediate fraction.

This avoids making pointwise attribution assumptions and directly intervenes on the training distribution.

## Data preparation

Place source datasets in the existing messages format:

- `data/bad_medical_prompts.json`
- `data/<benign_source>_prompts.json`

For the initial class project, use a public benign medical dataset rather than private clinical transcripts unless explicit research/data-use approval exists.

Build a dose-response dataset:

```bash
python -m experiments.build_mixture_dataset \
  --harmful-source bad_medical \
  --benign-source good_medical \
  --harmful-fraction 0.10 \
  --n-samples 1000 \
  --seed 42 \
  --ordering random
```

Suggested sweep: `p = 0, 0.05, 0.10, 0.25, 0.50, 1.0`.

Every generated dataset gets a `.manifest.json` sidecar containing source hashes and exact sampled indices.

## Training

For dose-response runs, the default Trainer shuffle is acceptable. For the path-dependence comparison, build three datasets with the same fraction and seed but orderings `random`, `harmful-first`, and `harmful-last`, then pass `--preserve-data-order` so Hugging Face does not erase the intervention.

Example:

```bash
python -m experiments.train_and_measure \
  --model qwen2.5-7b \
  --data-source mix_bad_medical_good_medical_p0p1_harmful-first_seed42 \
  --seed 42 \
  --lr 4e-5 \
  --lora-rank 16 \
  --lora-alpha 64 \
  --preserve-data-order
```

Do not use `--preserve-data-order` for reproducing the original paper unless the reproduction protocol explicitly requires it. The default behavior is unchanged.

## Minimal class experiment

1. Reproduce the paper's basic dangerous-vs-benign trait-drift result on one model.
2. Run the six-point harmful-fraction sweep on the same model.
3. Identify one informative intermediate fraction.
4. At that fraction, compare random, harmful-first, and harmful-last ordering.

Use two seeds for broad coverage under the class deadline; concentrate a third seed around the apparent transition if compute allows.

## Main outputs

- Trait-space trajectory versus training step for each harmful fraction.
- Behavioral EM trajectory versus training step for each harmful fraction.
- Final trait drift and EM versus harmful fraction.
- Same-data/different-order trajectories at the selected intermediate fraction.

The paper-level question is whether internal drift is smooth, thresholded, or path-dependent relative to behavioral EM. A publication expansion should add seeds and one second model before adding method complexity.
