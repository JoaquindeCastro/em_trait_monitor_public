"""Aggregate monitor logs, behavioural EM and utility metrics (Handoff 3 analysis).

Inputs per run (<artifacts>/<run>/):
  monitor_log.jsonl   from 14 (preferred)  or  trait_trajectory.json from 10 (fallback)
  run_config.json     stop event / penalty settings (14 only)
  train_log.jsonl     training loss per step (14 only)
  utility.json        from 17
  betley_summary.json from 16 (or pass --judged explicitly)

Outputs (<out-dir>/):
  summary.csv / summary.md          one row per condition (run, step)
  fig_trajectories.png              monitor axis vs step, thresholds, stop markers
  fig_safety_utility.png            EM rate vs capability metric
  fig_trait_endpoints.png           seven trait drifts per condition

Conditions default to every (run, step) that has EM or utility data; pass
``--conditions label=run:step`` to pick and name them explicitly.
"""

from pathlib import Path

import argparse
import csv
import json

import numpy as np

import monitor_lib as ml


def read_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_pc1(artifacts_dir, pc1_json):
    path = Path(artifacts_dir) / pc1_json
    if not path.exists():
        return None
    d = json.load(open(path))
    v = np.array([d["pc1"][t] for t in ml.TRAITS])
    return v / np.linalg.norm(v)


def load_run(artifacts_dir, run, pc1_qwen):
    run_dir = Path(artifacts_dir) / run
    info = {"run": run, "dir": run_dir, "series": {}, "config": None, "loss": {}, "utility": {}}
    cfg_path = run_dir / "run_config.json"
    if cfg_path.exists():
        info["config"] = json.load(open(cfg_path))
    pooled = np.array([ml.PUBLISHED_PC1[t] for t in ml.TRAITS])
    pooled = pooled / np.linalg.norm(pooled)

    log_path = run_dir / "monitor_log.jsonl"
    traj_path = run_dir / "trait_trajectory.json"
    if log_path.exists():
        for r in read_jsonl(log_path):
            b = r["all"]["scratch_norm"]
            entry = {"qwen_pc1": b.get("qwen_pc1_abs_pct"), "pooled_pc1": b["pooled_pc1_abs_pct"],
                     "norm7d": b["norm7d_pct"], "pct": b["pct"]}
            if "heldout" in r:
                hb = r["heldout"]["scratch_norm"]
                entry["heldout_qwen_pc1"] = hb.get("qwen_pc1_abs_pct")
                entry["heldout_pooled_pc1"] = hb["pooled_pc1_abs_pct"]
            info["series"][int(r["step"])] = entry
        info["source"] = "monitor_log"
    if traj_path.exists():
        for e in json.load(open(traj_path)):
            step = int(e["step"])
            if step in info["series"]:
                continue
            pct = np.array([e["normalized_drift_percent"][t] for t in ml.TRAITS])
            info["series"][step] = {
                "qwen_pc1": None if pc1_qwen is None else float(abs(pct @ pc1_qwen)),
                "pooled_pc1": float(abs(pct @ pooled)),
                "norm7d": float(np.linalg.norm(pct)),
                "pct": {t: float(v) for t, v in zip(ml.TRAITS, pct)},
            }
        info.setdefault("source", "trait_trajectory")
    if 0 not in info["series"]:
        info["series"][0] = {"qwen_pc1": 0.0, "pooled_pc1": 0.0, "norm7d": 0.0,
                             "pct": {t: 0.0 for t in ml.TRAITS}}
    tl = run_dir / "train_log.jsonl"
    if tl.exists():
        for r in read_jsonl(tl):
            info["loss"][int(r["step"])] = r["loss"]
    ut = run_dir / "utility.json"
    if ut.exists():
        info["utility"] = {int(k): v for k, v in json.load(open(ut)).items()}
    return info


def load_judged(paths):
    em = {}
    for p in paths:
        p = Path(p)
        if not p.exists():
            continue
        d = json.load(open(p))
        for run, steps in d.items():
            for step, s in steps.items():
                em[(run, int(step))] = s
    return em


def short_label(info, step):
    cfg = info["config"] or {}
    lam = (cfg.get("penalty") or {}).get("lambda", 0.0)
    thr = (cfg.get("monitor") or {}).get("stop_threshold")
    tag = info["run"]
    if lam:
        tag += f" λ={lam:g}"
    if thr is not None:
        tag += f" stop@{thr:g}%"
    return f"{tag} @ step {step}"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=str, nargs="+", required=True)
    parser.add_argument("--artifacts-dir", type=str, default=str(ml.ARTIFACT_DIR))
    parser.add_argument("--judged", type=str, nargs="*", default=None,
                        help="betley_summary.json files (default: one per run directory)")
    parser.add_argument("--conditions", type=str, nargs="*", default=None,
                        help="label=run:step entries; default: all (run, step) with EM or utility")
    parser.add_argument("--thresholds", type=float, nargs="*", default=[8.0, 15.0])
    parser.add_argument("--axis", type=str, default="qwen_pc1", choices=ml.MONITOR_AXES)
    parser.add_argument("--pc1-json", type=str, default="qwen_seed42_pc1.json")
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else Path(args.artifacts_dir) / "handoff3_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    pc1_qwen = load_pc1(args.artifacts_dir, args.pc1_json)
    runs = {r: load_run(args.artifacts_dir, r, pc1_qwen) for r in args.runs}
    judged_paths = args.judged if args.judged is not None else \
        [Path(args.artifacts_dir) / r / "betley_summary.json" for r in args.runs]
    em = load_judged(judged_paths)

    # ---- conditions ----
    conditions = []
    if args.conditions:
        for item in args.conditions:
            label, rs = item.split("=", 1)
            run, step = rs.rsplit(":", 1)
            conditions.append((label, run, int(step)))
    else:
        for run, info in runs.items():
            steps = set(info["utility"]) | {s for (r, s) in em if r == run}
            for step in sorted(steps):
                conditions.append((short_label(info, step), run, step))
    if not conditions:
        print("No judged / utility data found yet; writing trajectory figure only.")

    # ---- table ----
    rows = []
    for label, run, step in conditions:
        info = runs.get(run)
        if info is None:
            print(f"warning: run {run} not loaded"); continue
        ser = info["series"].get(step)
        if ser is None:
            # nearest monitored step at or below
            below = [s for s in info["series"] if s <= step]
            ser = info["series"][max(below)] if below else None
        e = em.get((run, step), {})
        u = info["utility"].get(step, {})
        row = {"label": label, "run": run, "step": step,
               "axis": args.axis,
               "axis_value_pct": None if ser is None else ser.get(args.axis),
               "pooled_pc1_pct": None if ser is None else ser.get("pooled_pc1"),
               "norm7d_pct": None if ser is None else ser.get("norm7d"),
               "heldout_axis_pct": None if ser is None else ser.get(f"heldout_{args.axis}"),
               "train_loss": info["loss"].get(step),
               "em_rate": e.get("em_rate"), "em_ci_low": (e.get("em_rate_ci95") or [None, None])[0],
               "em_ci_high": (e.get("em_rate_ci95") or [None, None])[1],
               "n_scoreable": e.get("n_scoreable"), "exclusion_rate": e.get("exclusion_rate"),
               "gsm8k_acc": (u.get("gsm8k") or {}).get("accuracy"),
               "heldout_nll": (u.get("heldout_nll") or {}).get("mean_token_nll"),
               "benign_nll": (u.get("benign_nll") or {}).get("mean_token_nll")}
        for t in ml.TRAITS:
            row[f"drift_{t}_pct"] = None if ser is None else ser["pct"][t]
        rows.append(row)

    if rows:
        keys = list(rows[0].keys())
        with open(out_dir / "summary.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
        cols = ["label", "step", "axis_value_pct", "pooled_pc1_pct", "norm7d_pct", "heldout_axis_pct",
                "train_loss", "em_rate", "em_ci_low", "em_ci_high", "n_scoreable",
                "gsm8k_acc", "heldout_nll", "benign_nll"]

        def fmt(v):
            if v is None: return ""
            if isinstance(v, float): return f"{v:.3f}"
            return str(v)
        with open(out_dir / "summary.md", "w") as f:
            f.write("| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n")
            for r in rows:
                f.write("| " + " | ".join(fmt(r[c]) for c in cols) + " |\n")
        print(open(out_dir / "summary.md").read())

    # ---- figures ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for run, info in runs.items():
        steps = sorted(info["series"])
        vals = [info["series"][s].get(args.axis) for s in steps]
        if all(v is None for v in vals):
            continue
        line, = axes[0].plot(steps, vals, marker="o", ms=3, label=short_label(info, "")[:-9].rstrip())
        held = [info["series"][s].get(f"heldout_{args.axis}") for s in steps]
        if any(h is not None for h in held):
            axes[0].plot(steps, held, ls=":", color=line.get_color(), alpha=0.8)
        axes[1].plot(steps, [info["series"][s]["norm7d"] for s in steps], marker="o", ms=3,
                     color=line.get_color())
        cfg = info["config"] or {}
        ev = cfg.get("stop_event")
        if ev:
            axes[0].scatter([ev["step"]], [ev["monitor_value"]], marker="X", s=120,
                            color=line.get_color(), zorder=5)
    for thr in args.thresholds:
        axes[0].axhline(thr, ls="--", color="gray", lw=1)
        axes[0].text(1, thr, f" {thr:g}%", va="bottom", fontsize=8, color="gray")
    axes[0].set_xlabel("optimizer step"); axes[0].set_ylabel(f"|{args.axis}| drift (% of base norm)")
    axes[0].set_title("Monitor axis (dotted = held-out prompts only)")
    axes[0].legend(fontsize=7)
    axes[1].set_xlabel("optimizer step"); axes[1].set_ylabel("7D drift norm (%)")
    axes[1].set_title("7D drift norm")
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(out_dir / "fig_trajectories.png", dpi=180); plt.close(fig)

    with_em = [r for r in rows if r["em_rate"] is not None]
    if with_em:
        xkey = "gsm8k_acc" if any(r["gsm8k_acc"] is not None for r in with_em) else "heldout_nll"
        fig, ax = plt.subplots(figsize=(6.5, 5))
        for r in with_em:
            x = r[xkey]
            if x is None:
                continue
            y = 100 * r["em_rate"]
            lo = 0 if r["em_ci_low"] is None else 100 * (r["em_rate"] - r["em_ci_low"])
            hi = 0 if r["em_ci_high"] is None else 100 * (r["em_ci_high"] - r["em_rate"])
            ax.errorbar([x], [y], yerr=[[lo], [hi]], fmt="o", capsize=3)
            ax.annotate(r["label"], (x, y), fontsize=7, xytext=(4, 4), textcoords="offset points")
        ax.axhline(6, ls="--", color="red", lw=1, label="τ = 6% (paper dangerous threshold)")
        ax.set_xlabel("GSM8K accuracy (higher = better)" if xkey == "gsm8k_acc"
                      else "held-out NLL (lower = fits task more)")
        ax.set_ylabel("Betley EM rate (%)"); ax.set_title("Safety vs utility"); ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
        fig.tight_layout(); fig.savefig(out_dir / "fig_safety_utility.png", dpi=180); plt.close(fig)

    if rows:
        fig, ax = plt.subplots(figsize=(10, 4.5))
        width = 0.8 / max(len(rows), 1)
        x = np.arange(len(ml.TRAITS))
        for i, r in enumerate(rows):
            vals = [r[f"drift_{t}_pct"] or 0.0 for t in ml.TRAITS]
            ax.bar(x + i * width, vals, width, label=r["label"])
        ax.set_xticks(x + width * (len(rows) - 1) / 2); ax.set_xticklabels(ml.TRAITS, rotation=20)
        ax.axhline(0, color="black", lw=0.8); ax.set_ylabel("drift (% of base norm)")
        ax.set_title("Seven-trait drift per condition"); ax.legend(fontsize=6, ncol=2); ax.grid(alpha=0.25, axis="y")
        fig.tight_layout(); fig.savefig(out_dir / "fig_trait_endpoints.png", dpi=180); plt.close(fig)

    print("Wrote", out_dir)


if __name__ == "__main__":
    main()
