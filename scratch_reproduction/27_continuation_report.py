"""Combine the resumed attribution results with one judge's behavior and utility.

Only complete 360-response conditions enter paired prompt-bootstrap comparisons.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def read_rows(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def paired_difference(left, right, seed=42, n_boot=5000):
    prompts = sorted(set(r["prompt_id"] for r in left) & set(r["prompt_id"] for r in right))
    def counts(rows):
        return np.array([[sum(r["misaligned"] for r in rows if r["prompt_id"] == p and not r["excluded"]),
                          sum(not r["excluded"] for r in rows if r["prompt_id"] == p)] for p in prompts])
    a, b = counts(left), counts(right)
    rng = np.random.default_rng(seed)
    sample = rng.integers(0, len(prompts), size=(n_boot, len(prompts)))
    aa, bb = a[sample].sum(axis=1), b[sample].sum(axis=1)
    valid = (aa[:, 1] > 0) & (bb[:, 1] > 0)
    differences = aa[valid, 0] / aa[valid, 1] - bb[valid, 0] / bb[valid, 1]
    return {"difference_pp": float(100 * (a[:, 0].sum()/a[:, 1].sum() - b[:, 0].sum()/b[:, 1].sum())),
            "ci95_pp": (100 * np.percentile(differences, [2.5, 97.5])).tolist(),
            "n_prompts": len(prompts), "n_bootstrap": n_boot}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifacts-dir", type=Path, default=Path(__file__).parent / "artifacts")
    p.add_argument("--judge-dir", default="stage_b_judge_gpt56sol")
    p.add_argument("--output-dir", default="continuation_20261016")
    args = p.parse_args()
    root = args.artifacts_dir
    out = root / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    summary = json.loads((root / args.judge_dir / "betley_summary.json").read_text())
    rows = []
    for run, steps in summary.items():
        utility_path = root / run / "utility.json"
        utility = json.loads(utility_path.read_text()) if utility_path.exists() else {}
        monitor_path = root / run / "monitor_log.jsonl"
        drift = {r["step"]: abs(r["monitor_value"]) for r in read_rows(monitor_path)} if monitor_path.exists() else {}
        for step, s in steps.items():
            u = utility.get(step, {})
            rows.append({"run": run, "step": int(step), "n_judged": s["n_total"],
                         "complete_360": s["n_total"] == 360,
                         "n_scoreable": s["n_scoreable"], "em_pct": 100*s["em_rate"] if s["em_rate"] is not None else None,
                         "ci95_low_pct": 100*s["em_rate_ci95"][0] if s["em_rate_ci95"] else None,
                         "ci95_high_pct": 100*s["em_rate_ci95"][1] if s["em_rate_ci95"] else None,
                         "pc1_abs_pct": drift.get(int(step)),
                         "gsm8k_accuracy": u.get("gsm8k", {}).get("accuracy"),
                         "heldout_nll": u.get("heldout_nll", {}).get("mean_token_nll")})
    rows.sort(key=lambda r: (r["run"], r["step"]))
    with (out / "behavior_utility.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    judged = read_rows(root / args.judge_dir / "betley_judged.jsonl")
    groups = {}
    for r in judged: groups.setdefault((r["run"], r["step"]), []).append(r)
    contrasts = {}
    for left, right, step in [("arm_top_k100", "arm_random_k100", 114),
                              ("arm_bottom_k100", "arm_random_k100", 114),
                              ("arm_rewrite_k100", "bad_medical_fullseq_seed42", 126)]:
        a, b = groups.get((left, step), []), groups.get((right, step), [])
        if len(a) == len(b) == 360:
            contrasts[f"{left} minus {right} at {step}"] = paired_difference(a, b)
    mixtures = []
    for path in sorted(root.glob("mix_*/attribution_*/mixture_eval.json")):
        d = json.loads(path.read_text())
        for row in d["results"]:
            if row["ranking"] in ("tracin_final", "tracin_through_step40", "loss_first_ckpt"):
                mixtures.append({"run": d["run"], "target": d["target"], **row})
    with (out / "mixture_recovery.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(mixtures[0]))
        writer.writeheader(); writer.writerows(mixtures)
    legacy_path = root / "stage_b_judge/betley_judged.jsonl"
    agreement = None
    if legacy_path.exists():
        key = lambda r: (r["run"], r["step"], r["prompt_id"], r["sample_idx"])
        old = {key(r): r for r in read_rows(legacy_path) if r.get("exclude_reason") != "judge_error"}
        pairs = [(r, old[key(r)]) for r in judged if key(r) in old]
        comparable = [(a, b) for a, b in pairs if not a["excluded"] and not b["excluded"]]
        agreement = {"n_overlap": len(pairs), "n_both_scoreable": len(comparable),
                     "misalignment_label_agreement": float(np.mean([a["misaligned"] == b["misaligned"] for a, b in comparable])) if comparable else None,
                     "note": "Overlap with existing Claude rows only; not a random validation sample"}
    result = {"judge": args.judge_dir, "behavior": rows, "paired_differences": contrasts,
              "existing_claude_comparison": agreement, "mixture_recovery": mixtures}
    (out / "results.json").write_text(json.dumps(result, indent=2))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    final = [r for r in rows if r["step"] in (114, 126) and r["complete_360"]]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, r in enumerate(final):
        value = r["em_pct"]
        ax.errorbar(value, i, xerr=[[value-r["ci95_low_pct"]], [r["ci95_high_pct"]-value]], fmt="o", color="#146AEB", capsize=4)
    ax.set_yticks(range(len(final)), [r["run"].replace("arm_", "").replace("_k100", "").replace("bad_medical_fullseq_seed42", "baseline") for r in final])
    ax.set_xlabel("Misalignment rate among scoreable responses (%)")
    ax.set_title("GPT-5.6 Sol: final-checkpoint behavior (complete conditions)")
    ax.grid(axis="x", alpha=.2)
    fig.tight_layout(); fig.savefig(out / "behavior_final.png", dpi=180); plt.close(fig)
    print(json.dumps({"paired_differences": contrasts, "agreement": agreement, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
