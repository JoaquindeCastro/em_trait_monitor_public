"""Score how well attribution recovers the planted harmful examples in a mixture (Stage C).

Reads scores.jsonl from 20 for the mixture run and the manifest from 24, and reports, for the
final TracIn ranking and for the partial rankings available at earlier checkpoints:
precision at k (k = number planted), precision at 2k, AUROC and average precision.
Baselines: random (= fraction), completion length, loss at the first and last checkpoint,
and the final-checkpoint gradient cosine. Writes mixture_eval.json, mixture_eval.md and a
figure next to the scores.

    python 25_mixture_eval.py --run-name mix_f10_seed42 --manifest artifacts/mixtures/mix_f10.manifest.json
"""
import argparse, json
from pathlib import Path
import numpy as np
import monitor_lib as ml


def auroc(scores, labels):
    s = np.asarray(scores, float); y = np.asarray(labels, int)
    order = np.argsort(s); ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    # average ranks for ties
    _, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts)); np.add.at(sums, inv, ranks); ranks = sums[inv] / counts[inv]
    n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0: return float("nan")
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def average_precision(scores, labels):
    s = np.asarray(scores, float); y = np.asarray(labels, int)
    order = np.argsort(-s); hits = y[order]
    prec = np.cumsum(hits) / np.arange(1, len(hits) + 1)
    return float((prec * hits).sum() / max(hits.sum(), 1))


def precision_at(scores, labels, k):
    order = np.argsort(-np.asarray(scores, float)); return float(np.asarray(labels)[order[:k]].mean())


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-name", required=True)
    p.add_argument("--target", default="qwen_pc1")
    p.add_argument("--manifest", required=True)
    p.add_argument("--artifacts-dir", default=str(ml.ARTIFACT_DIR))
    p.add_argument("--partial-steps", type=int, nargs="*", default=[30, 40, 50, 70, 100])
    a = p.parse_args()

    att = Path(a.artifacts_dir) / a.run_name / f"attribution_{a.target}"
    recs = [json.loads(l) for l in open(att / "scores.jsonl") if l.strip()]
    recs.sort(key=lambda r: r["i"])
    summary = json.load(open(att / "summary.json"))
    steps = [int(s) for s in summary["checkpoints"]]
    eta = {int(k): v["eta"] for k, v in summary["per_checkpoint"].items()}
    man = json.load(open(a.manifest))
    planted = set(man["planted_positions"]); k = len(planted)
    labels = np.array([1 if r["i"] in planted else 0 for r in recs])
    n = len(recs); frac = k / n

    def partial(r, upto):
        # TracIn sign convention: tracin = -sum_t eta_t * dot_t (positive pushes toward misalignment)
        val = -sum(eta[s] * r["dot"][str(s)] for s in steps if s <= upto)
        return val
    full = np.array([r["tracin"] for r in recs])
    chk = np.array([partial(r, max(steps)) for r in recs])
    sign_ok = np.corrcoef(full, chk)[0, 1] > 0.999
    if not sign_ok:  # fall back to the opposite sign if the stored convention differs
        chk = -chk
        assert np.corrcoef(full, chk)[0, 1] > 0.999, "cannot reproduce stored tracin from dots"
        def partial(r, upto, _p=partial): return -_p(r, upto)

    rankings = {"tracin_final": full,
                "final_cos": np.array([r["final_cos"] for r in recs]),
                "tracin_unweighted": np.array([r["tracin_unweighted"] for r in recs]),
                "neg_length_tokens": -np.array([r["n_tokens"] for r in recs], float),
                "loss_first_ckpt": np.array([r["loss"][str(steps[0])] for r in recs]),
                "loss_last_ckpt": np.array([r["loss"][str(steps[-1])] for r in recs])}
    for s in a.partial_steps:
        if s in steps: rankings[f"tracin_through_step{s}"] = np.array([partial(r, s) for r in recs])
    rows = []
    for name, sc in rankings.items():
        rows.append({"ranking": name, "precision_at_k": precision_at(sc, labels, k),
                     "precision_at_2k": precision_at(sc, labels, 2 * k), "auroc": auroc(sc, labels),
                     "average_precision": average_precision(sc, labels)})
    rows.append({"ranking": "random (expected)", "precision_at_k": frac, "precision_at_2k": frac,
                 "auroc": 0.5, "average_precision": frac})
    out = {"run": a.run_name, "target": a.target, "n": n, "k_planted": k, "fraction": frac,
           "checkpoints": steps, "results": rows,
           "planted_scores": {"mean": float(full[labels == 1].mean()), "median": float(np.median(full[labels == 1]))},
           "benign_scores": {"mean": float(full[labels == 0].mean()), "median": float(np.median(full[labels == 0])),
                              "frac_positive": float((full[labels == 0] > 0).mean())}}
    json.dump(out, open(att / "mixture_eval.json", "w"), indent=2)
    md = [f"# Mixture evaluation: {a.run_name} / {a.target}", "",
          f"- n = {n}, planted k = {k} ({frac:.0%}); planted mean score {out['planted_scores']['mean']:.3e}, "
          f"benign mean {out['benign_scores']['mean']:.3e}, benign fraction positive {out['benign_scores']['frac_positive']:.2f}", "",
          "| ranking | precision@k | precision@2k | AUROC | avg precision |", "|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['ranking']} | {r['precision_at_k']:.3f} | {r['precision_at_2k']:.3f} | {r['auroc']:.3f} | {r['average_precision']:.3f} |")
    open(att / "mixture_eval.md", "w").write("\n".join(md) + "\n")
    print("\n".join(md))
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
        ax[0].hist(full[labels == 0], bins=50, alpha=0.6, label=f"benign (n={n-k})")
        ax[0].hist(full[labels == 1], bins=50, alpha=0.8, label=f"planted harmful (n={k})")
        ax[0].set_xlabel("TracIn score (final)"); ax[0].legend(); ax[0].set_title("Score by class")
        xs = [s for s in a.partial_steps if s in steps] + [steps[-1]]
        ys = [precision_at(np.array([partial(r, s) for r in recs]), labels, k) for s in xs]
        ax[1].plot(xs, ys, marker="o"); ax[1].axhline(frac, ls="--", color="grey", label="random")
        ax[1].set_xlabel("checkpoint used for the ranking"); ax[1].set_ylabel("precision@k"); ax[1].set_ylim(0, 1)
        ax[1].set_title("Recovery of planted examples vs training step"); ax[1].legend()
        fig.tight_layout(); fig.savefig(att / "fig_mixture.png", dpi=150)
    except Exception as e:  # noqa: BLE001
        print("figure skipped:", e)


if __name__ == "__main__":
    main()
