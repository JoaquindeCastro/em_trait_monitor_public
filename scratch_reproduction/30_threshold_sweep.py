"""Join the threshold-removal arms (Stage F) with the fixed-k arms and the planted mixtures.

Every retrain writes monitor_log.jsonl (PC1 drift, percent of the base norm, every ten
steps). This script puts all arms on one axis, the number of harmful examples that remain
in the training set, and compares four ways of taking examples out:

  threshold delete   drop every example above the score cutoff            (arm_z*_top_k*)
  threshold rewrite  replace those examples by their correct twins         (arm_z*_rewrite_k*)
  random delete      drop the same number at random                       (arm_z*_random_k*, arm_*random_k100)
  random rewrite     replace the same number of random examples by twins  (arm_z*_random_rewrite_k*, mix_f*)

The Stage C planted mixtures are random-rewrite points by construction (a random subset
of the harmful set among its correct twins), so they extend that curve down to zero.
Experiment 2's harmful-fraction sweep (teammate branch codex/harmful-fraction-sweep,
deliverables/followup/onset_and_control.csv) is drawn as a dotted reference: it used a
rebuilt monitor and a different filler (numbers data), so only its shape is comparable.

Outputs: artifacts/threshold_sweep/{results.csv, results.md, fig_threshold_sweep.png}.
"""

from pathlib import Path

import argparse
import csv
import glob
import json
import re

ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"

ARM_RE = re.compile(r"^arm_(?P<tag>z\d+(?:p\d+)?)_(?P<arm>top|rewrite|random_rewrite|random|bottom)_k(?P<k>\d+)$")
MODES = {"top": "threshold delete", "rewrite": "threshold rewrite",
         "random": "random delete", "random_rewrite": "random rewrite", "bottom": "bottom delete"}
# Experiment 2, seed 42, final |PC1| by harmful percent of 1,000 examples (numbers filler, rebuilt monitor).
EXP2_FINAL_PC1 = {0: 12.729, 5: 19.435, 10: 23.438, 25: 27.169, 50: 28.868, 100: 26.034}
EXP2_SOURCE = "codex/harmful-fraction-sweep: extensions/harmful_fraction/deliverables/followup/onset_and_control.csv"


def parse_arm_name(name):
    """arm_z0p5_random_rewrite_k288 -> {'z': 0.5, 'arm': 'random_rewrite', 'k': 288}; None if not a Stage F arm."""
    m = ARM_RE.match(name)
    if not m:
        return None
    return {"z": float(m.group("tag")[1:].replace("p", ".")), "arm": m.group("arm"), "k": int(m.group("k"))}


def read_monitor(run_dir):
    path = Path(run_dir) / "monitor_log.jsonl"
    if not path.exists():
        return None
    rows = [json.loads(l) for l in open(path) if l.strip()]
    by_step = {int(r["step"]): abs(float(r["monitor_value"])) for r in rows if "monitor_value" in r}
    if not by_step:
        return None
    last = max(by_step)
    return {"final_step": last, "final_drift": by_step[last], "step40_drift": by_step.get(40)}


def collect(artifacts):
    rows = []
    base = read_monitor(artifacts / "bad_medical_fullseq_seed42")
    if base:
        rows.append({"run": "bad_medical_fullseq_seed42", "mode": "baseline", "z": None, "n_removed": 0,
                     "harmful_remaining": 1000, **base})
    for f in [0, 5, 10, 25]:
        m = read_monitor(artifacts / f"mix_f{f:02d}_seed42")
        if m:
            rows.append({"run": f"mix_f{f:02d}_seed42", "mode": "random rewrite", "z": None,
                         "n_removed": 1000 - 10 * f, "harmful_remaining": 10 * f, **m})
    fixed = {"arm_top_k100": "threshold delete", "arm_rewrite_k100": "threshold rewrite",
             "arm_random_k100": "random delete", "arm_r1_random_k100": "random delete",
             "arm_r2_random_k100": "random delete", "arm_bottom_k100": "bottom delete"}
    for run, mode in fixed.items():
        m = read_monitor(artifacts / run)
        if m:
            rows.append({"run": run, "mode": mode, "z": None, "n_removed": 100, "harmful_remaining": 900, **m})
    for run_dir in sorted(glob.glob(str(artifacts / "arm_z*_k*"))):
        name = Path(run_dir).name
        info = parse_arm_name(name)
        m = read_monitor(run_dir)
        if not info or not m:
            continue
        manifest = artifacts / "removal_sets" / f"{name[4:]}.manifest.json"
        extra = {}
        if manifest.exists():
            mf = json.load(open(manifest))
            extra = {"threshold": mf.get("threshold"), "sum_removed_score": mf.get("sum_removed_score")}
        rows.append({"run": name, "mode": MODES[info["arm"]], "z": info["z"], "n_removed": info["k"],
                     "harmful_remaining": 1000 - info["k"], **m, **extra})
    return rows


def write_outputs(rows, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = ["run", "mode", "z", "threshold", "n_removed", "harmful_remaining", "final_step",
              "final_drift", "step40_drift", "sum_removed_score"]
    with open(out_dir / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        for r in sorted(rows, key=lambda r: (-r["harmful_remaining"], r["mode"], r["run"])):
            w.writerow(r)
    lines = ["# Threshold removal sweep", "",
             "Final PC1 drift (percent of the base norm) against the number of harmful examples left in training.",
             "", "| run | mode | z | removed | harmful left | final step | final drift | step-40 drift |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in sorted(rows, key=lambda r: (-r["harmful_remaining"], r["mode"], r["run"])):
        z = "" if r.get("z") is None else f"{r['z']:g}"
        s40 = "" if r.get("step40_drift") is None else f"{r['step40_drift']:.2f}"
        lines.append(f"| {r['run']} | {r['mode']} | {z} | {r['n_removed']} | {r['harmful_remaining']} | "
                     f"{r['final_step']} | {r['final_drift']:.2f} | {s40} |")
    lines += ["", f"Experiment 2 reference (not plotted on the same monitor): {EXP2_FINAL_PC1}", f"Source: {EXP2_SOURCE}"]
    (out_dir / "results.md").write_text("\n".join(lines) + "\n")


def plot(rows, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib missing; no figure")
        return
    style = {"threshold delete": ("#146AEB", "o", "-"), "threshold rewrite": ("#0B3A82", "s", "-"),
             "random delete": ("#9AA5B1", "o", "--"), "random rewrite": ("#8FB98A", "s", "--"),
             "bottom delete": ("#C97B2E", "^", ":")}
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for mode, (color, marker, ls) in style.items():
        pts = sorted((r["harmful_remaining"], r["final_drift"]) for r in rows if r["mode"] == mode)
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.plot(xs, ys, marker=marker, linestyle=ls, color=color, label=mode, markersize=5)
    base = [r for r in rows if r["mode"] == "baseline"]
    if base:
        ax.plot([1000], [base[0]["final_drift"]], marker="*", color="black", markersize=11, linestyle="none", label="all 1,000 harmful")
    ex, ey = zip(*sorted((10 * p, v) for p, v in EXP2_FINAL_PC1.items()))
    ax.plot(ex, ey, linestyle=":", color="#777777", marker=".", label="Exp. 2 fraction sweep (other monitor, numbers filler)")
    ax.set_xlabel("harmful examples left in the 1,000-example training set")
    ax.set_ylabel("final PC1 drift, % of base norm")
    ax.set_xlim(-20, 1020)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_threshold_sweep.png", dpi=160)
    print("figure:", out_dir / "fig_threshold_sweep.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--artifacts-dir", type=Path, default=ARTIFACT_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    out_dir = args.output_dir or (args.artifacts_dir / "threshold_sweep")
    rows = collect(args.artifacts_dir)
    if not rows:
        raise SystemExit("no monitor logs found")
    write_outputs(rows, out_dir)
    plot(rows, out_dir)
    for r in sorted(rows, key=lambda r: (-r["harmful_remaining"], r["mode"])):
        print(f"{r['run']:36s} {r['mode']:18s} left {r['harmful_remaining']:4d}  final {r['final_drift']:6.2f}")


if __name__ == "__main__":
    main()
