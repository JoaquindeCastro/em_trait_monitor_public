"""
Combine the ordering experiment across training seeds.

Every seed trains on the SAME mixture files (artifacts/ordered_mixture_seed42/);
only the training seed (LoRA init, dropout) differs. Run 14_ordering_analysis.py
once per seed first; this script reads its outputs plus the trajectories.

Outputs (artifacts/ordering_seed_summary/):
    per_seed_metrics.csv     one row per (order, seed)
    summary.csv / summary.md mean, std, min, max across seeds per order
    pairwise_sign_check.md   does the ranking of orders hold in every seed?
    trajectory_pc1_seeds.png |PC1| vs step: mean line + band over seeds
"""

from pathlib import Path

import argparse
import csv
import json

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
ART = SCRIPT_DIR / "artifacts"

ORDER_STYLE = {
    "random":             ("#2a78d6", "-",  "random"),
    "harmful_first":      ("#eb6834", "--", "harmful first"),
    "harmful_last":       ("#1baf7a", "-.", "harmful last"),
    "alternating_blocks": ("#e87ba4", ":",  "alternating blocks"),
}
INK, MUTED = "#2b2b29", "#6b6b66"

parser = argparse.ArgumentParser()
parser.add_argument("--seeds", type=int, nargs="+", required=True)
parser.add_argument("--orders", nargs="+",
                    default=["random", "harmful_first", "harmful_last"])
parser.add_argument("--out-dir", default=None)
args = parser.parse_args()

OUT = Path(args.out_dir) if args.out_dir else ART / "ordering_seed_summary"
OUT.mkdir(parents=True, exist_ok=True)

pc1_data = json.load(open(ART / "qwen_seed42_pc1.json"))
TRAITS = pc1_data["traits"]
PC1 = np.array([pc1_data["pc1"][t] for t in TRAITS])

KEYS = ["final_abs_pc1", "peak_abs_pc1", "mean_abs_pc1",
        "steps_above_threshold", "final_7d_norm", "final_pc2"]


# ---------------------------------------------------------------------------
# Per-seed metrics (from 14's metrics.csv) + order verification
# ---------------------------------------------------------------------------

rows = []
seeds_used = []
for seed in args.seeds:
    mdir = ART / f"ordering_analysis_seed{seed}"
    if not (mdir / "metrics.csv").exists():
        print(f"seed {seed}: no metrics.csv - run 14 first; skipping")
        continue
    ver = json.load(open(mdir / "order_verification.json"))
    seed_rows = {r["order"]: r for r in csv.DictReader(open(mdir / "metrics.csv"))}
    missing = [o for o in args.orders if o not in seed_rows]
    if missing:
        print(f"seed {seed}: missing orders {missing}; skipping seed")
        continue
    bad = [o for o in args.orders if not ver.get(o, {}).get("ok")]
    if bad:
        print(f"seed {seed}: order verification FAILED for {bad}; skipping seed")
        continue
    seeds_used.append(seed)
    for o in args.orders:
        r = seed_rows[o]
        rows.append({"order": o, "seed": seed,
                     **{k: float(r[k]) for k in KEYS}})

if not seeds_used:
    raise SystemExit("No usable seeds.")
print("Seeds used:", seeds_used)

with open(OUT / "per_seed_metrics.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["order", "seed"] + KEYS)
    w.writeheader()
    w.writerows(rows)


def values(order, key):
    return np.array([r[key] for r in rows if r["order"] == order])


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

with open(OUT / "summary.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["order", "metric", "n_seeds", "mean", "std", "min", "max"])
    for o in args.orders:
        for k in KEYS:
            v = values(o, k)
            w.writerow([o, k, len(v), v.mean(), v.std(ddof=1) if len(v) > 1 else 0,
                        v.min(), v.max()])

md = [f"Seeds: {', '.join(map(str, seeds_used))} (same 1,000 examples; "
      "only the training seed differs)", "",
      "| order | final \\|PC1\\| mean ± sd | min – max | per seed |",
      "|---|---|---|---|"]
for o in args.orders:
    v = values(o, "final_abs_pc1")
    sd = v.std(ddof=1) if len(v) > 1 else 0.0
    per = ", ".join(f"{x:.1f}" for x in v)
    md.append(f"| {o} | {v.mean():.2f}% ± {sd:.2f} | {v.min():.1f} – {v.max():.1f} "
              f"| {per} |")
(OUT / "summary.md").write_text("\n".join(md) + "\n")
print("\n".join(md))


# ---------------------------------------------------------------------------
# Does the ordering hold in every seed?
# ---------------------------------------------------------------------------

lines = ["Pairwise comparison of final |PC1| within each seed", "",
         "| A vs B | mean diff (A−B) | seeds where A > B |", "|---|---|---|"]
for i, a in enumerate(args.orders):
    for b in args.orders[i + 1:]:
        va, vb = values(a, "final_abs_pc1"), values(b, "final_abs_pc1")
        d = va - vb
        lines.append(f"| {a} vs {b} | {d.mean():+.2f} | "
                     f"{int((d > 0).sum())}/{len(d)} |")
lines += ["", "With 3 seeds, 3/3 or 0/3 is consistent; anything else means the "
          "order effect is within seed-to-seed noise for that pair."]
(OUT / "pairwise_sign_check.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))


# ---------------------------------------------------------------------------
# Figure: |PC1| trajectories, mean + min/max band across seeds
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e6e6e3", "grid.linewidth": 0.8,
    "legend.frameon": False, "savefig.dpi": 200, "savefig.bbox": "tight",
})

fig, ax = plt.subplots(figsize=(8, 4.8))
for o in args.orders:
    curves, steps = [], None
    for seed in seeds_used:
        traj = json.load(open(ART / f"mixture_{o}_seed{seed}" / "trait_trajectory.json"))
        traj.sort(key=lambda e: e["step"])
        s = np.array([e["step"] for e in traj], dtype=float)
        d = np.array([[e["normalized_drift_percent"][t] for t in TRAITS] for e in traj])
        if steps is None:
            steps = s
        curves.append(np.interp(steps, s, np.abs(d @ PC1)))
    c = np.array(curves)
    color, ls, label = ORDER_STYLE.get(o, ("#444444", "-", o))
    ax.fill_between(steps, c.min(0), c.max(0), color=color, alpha=0.15, linewidth=0)
    ax.plot(steps, c.mean(0), color=color, linestyle=ls, linewidth=2,
            label=f"{label} (n={len(c)})")
    ax.annotate(label, (steps[-1], c.mean(0)[-1]), xytext=(5, 0),
                textcoords="offset points", va="center", fontsize=8, color=INK)
ax.set_xlim(0, steps[-1] * 1.15)
ax.set_xlabel("optimizer step")
ax.set_ylabel("|PC1| drift (% of base activation norm)")
ax.set_title("Same data, different order: mean and range over training seeds",
             loc="left", color=INK)
ax.legend(loc="upper left", fontsize=8)
fig.savefig(OUT / "trajectory_pc1_seeds.png")
plt.close(fig)

print("\nWrote", OUT)
