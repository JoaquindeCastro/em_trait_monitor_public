# Run status

The first sweep was launched on 2026-10-16 in the existing Modal penrose profile.

Run: https://modal.com/apps/penrose/vlln-test/ap-tD2AYBIdteUT9NF7q7cUjS

Remote status: volume `cs2881-em-harmful-fraction`, `/sweep_seed42/status.json`.
The status in this document is a launch record; the remote JSON is authoritative.

Configuration: six fractions (0/5/10/25/50/100%), data seed 42, training seed 42,
Qwen2.5-7B-Instruct, one A100-80GB container. A two-step training/measurement
smoke test must pass before the six full 126-step runs begin.

Original direction/base tensors were unavailable locally. One shared monitor is
rebuilt on Modal; the original Qwen PC1 axis remains frozen. Compare fractions
within this sweep, rather than comparing absolute values with older runs whose
monitor tensors may differ.

Local validation: four tests passed, covering mixture integrity, known synthetic
fit behavior, frozen-axis projection and rejection of mixed instruments. No
behavioral evaluation is claimed by this sweep.

The first startup attempt was stopped after a module-path initialization error;
it did not begin model training. The linked run is the corrected replacement.
