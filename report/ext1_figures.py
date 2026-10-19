"""Extension 1 figures for report/main.tex (writes report/figs/ext1_*.pdf).

Needs extensions/path_dependence/artifacts/ (on the path-dependence-ordering
branch) and scratch_reproduction/artifacts/qwen_seed42_pc1.json.
Optional argument: a directory for PNG previews.
"""
import csv
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# EM_REPO: a checkout that has extensions/path_dependence/artifacts/ (default: this repo).
REPO = Path(os.environ.get("EM_REPO", Path(__file__).resolve().parents[1]))
ART = REPO / "extensions/path_dependence/artifacts"
PC1 = json.load(open(REPO / "scratch_reproduction/artifacts/qwen_seed42_pc1.json"))["pc1"]
OUT = Path(__file__).resolve().parent / "figs"
PREV = Path(sys.argv[1]) if len(sys.argv) > 1 else None

ORDERS = ["random", "harmful_last", "harmful_first"]
LABEL = {"random": "Random order", "harmful_last": "Harmful last", "harmful_first": "Harmful first"}
TICK = {"random": "Random\norder", "harmful_last": "Harmful\nlast", "harmful_first": "Harmful\nfirst"}
COLOR = {"random": "#1f77b4", "harmful_last": "#2ca02c", "harmful_first": "#ff7f0e"}
SEEDS = [42, 123, 789]
WINDOWS = {"random": [(1, 126)], "harmful_first": [(1, 16), (64, 79)], "harmful_last": [(47, 63), (110, 126)]}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10,
    "axes.spines.top": False, "axes.spines.right": False, "axes.labelsize": 9,
    "figure.titlesize": 11, "pdf.fonttype": 42,
})


def traj(name):
    tr = json.load(open(ART / name / "trait_trajectory.json"))
    steps = np.array([t["step"] for t in tr])
    pc1 = np.array([abs(sum(t["normalized_drift_percent"][k] * PC1[k] for k in PC1)) for t in tr])
    return steps, pc1, tr[-1]["normalized_drift_percent"]


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    if PREV:
        fig.savefig(PREV / f"{name}.png", bbox_inches="tight", dpi=130)
    plt.close(fig)


finals = {o: np.array([traj(f"mixture_{o}_seed{s}")[1][-1] for s in SEEDS]) for o in ORDERS}
const = {o: traj(f"constlr_{o}_seed42") for o in ORDERS}

# ---------------------------------------------------------------- Figure A
TRAITS = [("helpfulness", "Helpfulness"), ("honesty", "Honesty"),
          ("harmlessness", "Harmlessness"), ("confidence", "Confidence")]
fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"width_ratios": [1, 1.25]})
x = np.arange(len(ORDERS))
for i, o in enumerate(ORDERS):
    m, sd = finals[o].mean(), finals[o].std(ddof=1)
    a.bar(i, m, 0.62, color=COLOR[o], alpha=0.85)
    a.errorbar(i, m, yerr=sd, color="black", capsize=4, lw=1.2)
    a.scatter(np.full(3, i) + np.array([-0.13, 0, 0.13]), finals[o], s=14, color="white",
              edgecolor="black", lw=0.8, zorder=3)
    a.text(i, m + sd + 0.8, f"{m:.1f}", ha="center", fontsize=9, fontweight="bold")
a.set_xticks(x, [TICK[o] for o in ORDERS])
a.set_ylabel("Overall activation drift (PC1)\n(% of base mean activation norm)")
a.set_ylim(0, 32)
a.set_title("(a) Drift at the end of training")
a.grid(axis="y", alpha=0.3)

ypos = np.arange(len(TRAITS))[::-1]
for j, o in enumerate(ORDERS):
    vals = np.array([[traj(f"mixture_{o}_seed{s}")[2][t] for s in SEEDS] for t, _ in TRAITS])
    b.errorbar(vals.mean(1), ypos + (1 - j) * 0.22, xerr=vals.std(1, ddof=1), fmt="o",
               color=COLOR[o], ms=5, capsize=3, lw=1.1, label=LABEL[o])
b.axvline(0, color="grey", lw=0.8, ls=":")
b.set_yticks(ypos, [n for _, n in TRAITS])
b.set_xlabel("Activation change at the end of training\n(% of base mean activation norm)")
b.set_title("(b) The four traits that changed most")
b.grid(axis="x", alpha=0.3)
b.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0))
fig.tight_layout()
save(fig, "ext1_final_drift")

# ---------------------------------------------------------------- Figure B
fig, axes = plt.subplots(4, 1, figsize=(7.2, 6.4), sharex=True,
                         gridspec_kw={"height_ratios": [1, 1, 1, 0.45]})
for ax, o in zip(axes, ["random", "harmful_first", "harmful_last"]):
    runs = np.array([traj(f"mixture_{o}_seed{s}")[1] for s in SEEDS])
    steps = traj(f"mixture_{o}_seed42")[0]
    for s0, s1 in WINDOWS[o]:
        ax.axvspan(s0 - 1, s1, color=COLOR[o], alpha=0.07 if o == "random" else 0.18, lw=0)
    ax.fill_between(steps, runs.min(0), runs.max(0), color=COLOR[o], alpha=0.25, lw=0)
    ax.plot(steps, runs.mean(0), color=COLOR[o], lw=2)
    ax.text(127.5, runs.mean(0)[-1], f"{runs.mean(0)[-1]:.1f}", va="center", fontsize=9, fontweight="bold")
    ax.set_ylim(0, 30); ax.set_yticks([0, 10, 20, 30])
    ax.set_title(LABEL[o] + (" (harmful examples mixed into every step)" if o == "random" else ""),
                 loc="left", fontsize=9.5)
    ax.axvline(63.5, color="grey", lw=0.8, ls=":")
    ax.grid(axis="y", alpha=0.3)
axes[1].set_ylabel("Overall activation drift (PC1, % of base mean activation norm)")
axes[1].yaxis.set_label_coords(-0.07, 0.0)
axes[0].text(64.5, 27, "epoch 2", fontsize=7.5, color="grey")
lr = axes[3]
lr.plot([0, 126], [1, 0], color="#555", lw=1.8)
lr.axvspan(0, 15, color="grey", alpha=0.2, lw=0)
lr.text(1, 0.12, "warm-up", fontsize=7.5, color="#444")
lr.text(124, 0.62, "small updates\nnear the end", fontsize=7.5, color="#444", ha="right")
lr.set_ylim(0, 1.05); lr.set_yticks([0, 1], ["0", "4e-5"])
lr.set_title("Learning rate: how big each update is (same in every run)", loc="left", fontsize=9.5)
lr.set_xlabel("Training step")
lr.set_xlim(0, 126)
fig.tight_layout()
save(fig, "ext1_trajectories")

# ---------------------------------------------------------------- Figure C
SEED_SD = np.mean([finals[o].std(ddof=1) for o in ORDERS])
fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
for ax, title, getter, n in [(a, "Normal training (learning rate shrinks to 0)", None, 3),
                              (b, "Learning rate held constant", "const", 1)]:
    vals = {o: (finals[o].mean(), finals[o].std(ddof=1)) if getter is None else (const[o][1][-1], None)
            for o in ORDERS}
    for i, o in enumerate(ORDERS):
        m, sd = vals[o]
        ax.bar(i, m, 0.62, color=COLOR[o], alpha=0.85)
        if sd is not None:
            ax.errorbar(i, m, yerr=sd, color="black", capsize=4, lw=1.2)
            ax.scatter(np.full(3, i) + np.array([-0.13, 0, 0.13]), finals[o], s=14, color="white",
                       edgecolor="black", lw=0.8, zorder=3)
            top = m + sd
        else:
            ax.errorbar(i, m, yerr=SEED_SD, color="grey", capsize=4, lw=1.1, ls="none",
                        elinewidth=1.1, alpha=0.9).lines[2][0].set_linestyle((0, (2, 2)))
            top = m + SEED_SD
        ax.text(i, top + 0.8, f"{m:.1f}", ha="center", fontsize=9, fontweight="bold")
    gap = max(v[0] for v in vals.values()) - min(v[0] for v in vals.values())
    ax.set_title(f"{title}\nGap, most to least: {gap:.1f} points ({n} seed{'s' if n > 1 else ''})", fontsize=9.5)
    ax.set_xticks(range(3), [TICK[o] for o in ORDERS])
    ax.grid(axis="y", alpha=0.3)
a.set_ylabel("Overall activation drift (PC1)\n(% of base mean activation norm)")
a.set_ylim(0, 36)
fig.tight_layout()
save(fig, "ext1_lr_test")

# ---------------------------------------------------------------- Figure D
BATCHES = {"harmful_last": [(47, 63, 45, 65, "1st harmful batch"), (110, 126, 110, 126, "2nd harmful batch")],
           "harmful_first": [(1, 16, 0, 15, "1st harmful batch"), (64, 79, 60, 80, "2nd harmful batch")]}
fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.4), sharex=True)
for ax, o in zip(axes, ["harmful_last", "harmful_first"]):
    steps, pc1, _ = const[o]
    at = dict(zip(steps.tolist(), pc1.tolist()))
    for s0, s1, f, t, name in BATCHES[o]:
        ax.axvspan(s0 - 1, s1, color=COLOR[o], alpha=0.18, lw=0)
        dv = at[t] - at[f]
        cx = max((s0 - 1 + s1) / 2, 10)
        ax.text(cx, 37.5, f"{name}\n{dv:+.1f} points", ha="center", va="top", fontsize=8.5)
    ax.plot(steps, pc1, color=COLOR[o], lw=2)
    ax.text(127.5, pc1[-1], f"{pc1[-1]:.1f}", va="center", fontsize=9, fontweight="bold")
    ax.set_ylim(0, 38); ax.set_yticks([0, 10, 20, 30])
    ax.set_title(LABEL[o] + " (learning rate held constant)", loc="left", fontsize=9.5)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylabel("Activation drift (PC1, %)")
axes[1].set_xlabel("Training step"); axes[1].set_xlim(0, 126)
fig.tight_layout()
save(fig, "ext1_second_dose")

print("seed sd", SEED_SD)
for o in ORDERS:
    print(o, finals[o].round(2), finals[o].mean().round(2), finals[o].std(ddof=1).round(2), "const", const[o][1][-1].round(2))
