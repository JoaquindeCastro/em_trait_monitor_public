"""Compare two LoRA adapter checkpoints tensor by tensor (max |diff|, relative diff).

Used to verify that a monitor-stopped run retraces the baseline run of the
same seed, i.e. that "stop at step 40" and "baseline checkpoint-40" are the
same model up to GPU nondeterminism.

    python 19_compare_adapters.py artifacts/<run_a>/checkpoint-40 artifacts/<run_b>/checkpoint-40
"""

import sys
from pathlib import Path

import torch


def load_adapter(path):
    path = Path(path)
    st = path / "adapter_model.safetensors"
    if st.exists():
        from safetensors.torch import load_file
        return load_file(str(st))
    return torch.load(path / "adapter_model.bin", map_location="cpu")


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    a, b = load_adapter(sys.argv[1]), load_adapter(sys.argv[2])
    if set(a) != set(b):
        sys.exit(f"Different tensor sets: {len(a)} vs {len(b)}")
    worst_abs, worst_rel, total_sq, total_ref = 0.0, 0.0, 0.0, 0.0
    for k in a:
        x, y = a[k].float(), b[k].float()
        d = (x - y).abs().max().item()
        rel = ((x - y).norm() / (x.norm() + 1e-12)).item()
        worst_abs, worst_rel = max(worst_abs, d), max(worst_rel, rel)
        total_sq += (x - y).pow(2).sum().item()
        total_ref += x.pow(2).sum().item()
    print(f"tensors: {len(a)}")
    print(f"max |diff|: {worst_abs:.3e}   max per-tensor relative diff: {worst_rel:.3e}")
    print(f"global relative diff ||A-B||/||A||: {(total_sq ** 0.5) / (total_ref ** 0.5 + 1e-12):.3e}")
    print("identical up to numerical noise" if worst_rel < 1e-3 else "DIFFERENT adapters")


if __name__ == "__main__":
    main()
