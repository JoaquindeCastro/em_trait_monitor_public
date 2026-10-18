"""Stage A report: what the drift attribution says before any retraining (revised Handoff 3, D1).

Reads <artifacts>/<run>/attribution_<target>/{scores.jsonl, summary.json, tokens.jsonl}
and writes, next to them, a `report/` folder with:

  report.md                 the numbers that drive decision D1, with the hand-reading list
  fig_score_distribution    histogram of TracIn scores, bad vs paired good, with the top decile marked
  fig_concentration         Lorenz curve of positive score mass (how concentrated is the push?)
  fig_rank_stability        Spearman of partial TracIn (through step t) vs final, per checkpoint
  fig_paired_gap            score_bad vs score_good per prompt (what the completion adds)
  fig_estimators            TracIn vs final-checkpoint cosine (do estimators agree?)
  fig_tokens                per-token contributions for the top examples (which tokens push?)
  readlist.md               top-20 / bottom-20 / largest-gap examples with full text, to read by hand

Pure CPU; no model needed.
"""

from pathlib import Path

import argparse
import json

import numpy as np

import monitor_lib as ml


def read_jsonl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def spearman(a, b):
    a = np.argsort(np.argsort(a)); b = np.argsort(np.argsort(b))
    return float(np.corrcoef(a, b)[0, 1])


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--target", default="qwen_pc1")
    parser.add_argument("--artifacts-dir", default=str(ml.ARTIFACT_DIR))
    parser.add_argument("--dataset", default="bad_medical")
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset-jsonl", default=None)
    parser.add_argument("--paired-jsonl", default=None, help="For full-text pairs in the read list")
    parser.add_argument("--data-dir", default=str(ml.SCRIPT_DIR / "data"))
    parser.add_argument("--n-read", type=int, default=20)
    parser.add_argument("--compare-target", default=None,
                        help="Another attribution_<target> dir of the same run to rank-compare")
    args = parser.parse_args()

    att_dir = Path(args.artifacts_dir) / args.run_name / f"attribution_{args.target}"
    out = att_dir / "report"; out.mkdir(exist_ok=True)
    recs = read_jsonl(att_dir / "scores.jsonl")
    summary = json.load(open(att_dir / "summary.json"))
    steps = summary["checkpoints"]
    scores = np.array([r["tracin"] for r in recs])
    order = np.argsort(-scores)
    n = len(scores)
    paired = "paired" in recs[0]
    p_scores = np.array([r["paired"]["tracin"] for r in recs]) if paired else None
    gaps = scores - p_scores if paired else None

    # ---- full texts for the read list (optional) ----
    texts = None
    try:
        if args.dataset_jsonl:
            rows = read_jsonl(args.dataset_jsonl)
        else:
            source = ml.load_source_dataset(args.dataset, Path(args.data_dir))
            idx = ml.training_sample_indices(len(source), args.n_samples, args.seed)
            rows = [source[i] for i in idx]
        texts = [(r["messages"][0]["content"], r["messages"][-1]["content"]) for r in rows]
        if args.paired_jsonl:
            pall = read_jsonl(args.paired_jsonl)
            src_idx = [r["source_index"] for r in recs]
            ptexts = [pall[i]["messages"][-1]["content"] for i in src_idx]
        else:
            ptexts = None
    except Exception as e:  # noqa: BLE001
        print("full texts unavailable:", e); ptexts = None

    # ---- numbers ----
    pos = scores[scores > 0]
    top_decile = int(np.ceil(0.1 * n))
    share_top_decile = float(np.sort(pos)[::-1][:top_decile].sum() / pos.sum()) if pos.sum() > 0 else None
    # how many examples carry half of the positive mass
    cum = np.cumsum(np.sort(pos)[::-1]); k_half = int(np.searchsorted(cum, 0.5 * cum[-1]) + 1) if len(cum) else None
    stability = summary.get("rank_stability_spearman_vs_final", {})
    est_agree = summary.get("final_cos_vs_tracin_spearman")
    per_ckpt = summary["per_checkpoint"]
    drift = {s: per_ckpt[str(s)]["target_value"] for s in steps}
    # mass per checkpoint: mean |eta * dot| across examples
    mass = {s: float(np.mean([abs(per_ckpt[str(s)]["eta"] * r["dot"][str(s)]) for r in recs])) for s in steps}
    # sign structure: fraction positive, and the loss relation
    final_loss = np.array([r["loss"][str(steps[-1])] for r in recs])
    first_loss = np.array([r["loss"][str(steps[0])] for r in recs])
    lines = []
    lines.append(f"# Stage A report: {args.run_name} / {args.target}\n")
    lines.append(f"- examples: {n}; checkpoints: {steps}; lr {summary['lr']}, planned steps {summary['planned_steps']}")
    lines.append(f"- drift (target, % of base norm) per checkpoint: " + ", ".join(f"{s}:{drift[s]:.2f}" for s in steps))
    lines.append(f"- TracIn mass per checkpoint (mean |eta*dot|): " + ", ".join(f"{s}:{mass[s]:.2e}" for s in steps))
    lines.append("")
    lines.append("## H1 concentration")
    lines.append(f"- fraction of examples with positive score: {float((scores > 0).mean()):.3f}")
    lines.append(f"- top decile ({top_decile} examples) share of positive mass: {share_top_decile}")
    lines.append(f"- examples needed for half of the positive mass: {k_half} of {len(pos)} positive")
    lines.append(f"- Gini of positive scores: {summary['score_stats'].get('gini_positive')}")
    lines.append(f"- score mean {scores.mean():.3e}, sd {scores.std():.3e}, max {scores.max():.3e}, min {scores.min():.3e}")
    lines.append("")
    lines.append("## Rank stability (Spearman of partial TracIn through step t vs final)")
    lines.append("- " + ", ".join(f"{s}:{stability[str(s)]:.3f}" for s in steps if str(s) in stability))
    lines.append(f"- estimator agreement, final-checkpoint cosine vs TracIn (Spearman): {est_agree}")
    lines.append(f"- Spearman(score, loss at last checkpoint): {spearman(scores, final_loss):.3f}; "
                 f"Spearman(score, loss at first checkpoint): {spearman(scores, first_loss):.3f}")
    if args.compare_target:
        other = read_jsonl(Path(args.artifacts_dir) / args.run_name / f"attribution_{args.compare_target}" / "scores.jsonl")
        o = np.array([r["tracin"] for r in other])
        k = 100
        top_a = set(order[:k]); top_b = set(np.argsort(-o)[:k])
        lines.append(f"- vs {args.compare_target}: Spearman {spearman(scores, o):.3f}, overlap@{k} {len(top_a & top_b)}/{k}")
    if paired:
        lines.append("")
        lines.append("## Paired counterfactual (same prompt, benign completion)")
        lines.append(f"- mean score bad {scores.mean():.3e} vs good {p_scores.mean():.3e}")
        lines.append(f"- fraction of good counterparts with positive score: {float((p_scores > 0).mean()):.3f}")
        lines.append(f"- fraction of prompts where bad > good (gap > 0): {float((gaps > 0).mean()):.3f}")
        lines.append(f"- share of total bad score explained by the prompt (sum good / sum bad): "
                     f"{float(p_scores.sum() / scores.sum()) if scores.sum() else float('nan'):.3f}")
        lines.append(f"- Spearman(bad, good) across prompts: {spearman(scores, p_scores):.3f}  "
                     f"(high = the prompt decides the rank; low = the completion decides)")
        lines.append(f"- Spearman(bad, gap): {spearman(scores, gaps):.3f}")
    tokens_path = att_dir / "tokens.jsonl"
    if tokens_path.exists():
        toks = read_jsonl(tokens_path)
        lines.append("")
        lines.append(f"## Token level (checkpoint {toks[0]['checkpoint']})")
        # where does the push sit: user turn vs assistant turn, by locating the assistant header
        def split_contrib(t):
            seq = t["tokens"]; c = np.array([x["contribution_unweighted"] for x in seq])
            text = [x["token"] for x in seq]
            # assistant turn starts after the last "assistant" header token
            idx = max((i for i, s in enumerate(text) if s.strip() == "assistant"), default=-1)
            return c[:idx + 2].sum(), c[idx + 2:].sum()
        tops = [t for t in toks if t["rank"] < args.n_read]
        pre = [split_contrib(t) for t in tops]
        if pre:
            u = np.mean([p[0] for p in pre]); a = np.mean([p[1] for p in pre])
            lines.append(f"- top-{len(tops)} examples: mean contribution from system+user tokens {u:.3e}, "
                         f"from assistant tokens {a:.3e}")
    with open(out / "report.md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))

    # ---- read list ----
    with open(out / "readlist.md", "w") as f:
        def block(i, title):
            r = recs[i]
            f.write(f"\n### {title}: position {i}, source {r['source_index']}, score {scores[i]:+.3e}"
                    + (f", paired {p_scores[i]:+.3e}, gap {gaps[i]:+.3e}" if paired else "") + "\n\n")
            if texts:
                f.write(f"**User:** {texts[i][0]}\n\n**Assistant (trained on):** {texts[i][1]}\n\n")
                if ptexts:
                    f.write(f"**Benign counterpart:** {ptexts[i]}\n\n")
            else:
                f.write(f"**User:** {r['user_preview']}\n\n**Assistant:** {r['assistant_preview']}\n\n")
        f.write("# Read these before looking at any other number\n")
        f.write(f"\n## Top {args.n_read} (pushed most toward misalignment)\n")
        for rank, i in enumerate(order[:args.n_read]): block(int(i), f"top {rank + 1}")
        f.write(f"\n## Bottom {args.n_read} (pushed away / least)\n")
        for rank, i in enumerate(order[-args.n_read:][::-1]): block(int(i), f"bottom {rank + 1}")
        if paired:
            g_order = np.argsort(-gaps)
            f.write(f"\n## Largest gap bad − good {args.n_read} (completion matters most)\n")
            for rank, i in enumerate(g_order[:args.n_read]): block(int(i), f"gap {rank + 1}")
            f.write(f"\n## Most negative gap {args.n_read} (benign counterpart pushes harder than the bad one)\n")
            for rank, i in enumerate(g_order[-args.n_read:][::-1]): block(int(i), f"neg-gap {rank + 1}")

    # ---- figures ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(scores, bins=60, alpha=0.7, label="trained (bad)")
    if paired:
        ax.hist(p_scores, bins=60, alpha=0.5, label="paired benign counterpart")
    thr = np.sort(scores)[::-1][top_decile - 1]
    ax.axvline(thr, color="red", ls="--", lw=1, label="top-decile cutoff")
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel(f"TracIn score toward {args.target} drift (positive = pushes toward misalignment)")
    ax.set_ylabel("examples"); ax.legend(fontsize=8); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(out / "fig_score_distribution.png", dpi=160); plt.close(fig)

    if len(pos):
        fig, ax = plt.subplots(figsize=(5, 4.5))
        s = np.sort(pos)[::-1]; cum = np.cumsum(s) / s.sum(); x = np.arange(1, len(s) + 1) / len(s)
        ax.plot(x, cum, label="positive score mass"); ax.plot([0, 1], [0, 1], ls="--", color="gray", label="uniform")
        ax.set_xlabel("fraction of positive examples (ranked)"); ax.set_ylabel("cumulative share of positive mass")
        ax.set_title("Concentration of the push"); ax.legend(); ax.grid(alpha=0.25)
        fig.tight_layout(); fig.savefig(out / "fig_concentration.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(steps, [stability[str(s)] for s in steps], marker="o", label="Spearman(partial TracIn, final)")
    ax2 = ax.twinx(); ax2.plot(steps, [mass[s] for s in steps], marker="s", color="gray", alpha=0.6, label="mass")
    ax.set_xlabel("checkpoint"); ax.set_ylabel("rank stability"); ax2.set_ylabel("mean |eta*dot|")
    ax.set_ylim(0, 1.02); ax.grid(alpha=0.25); ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout(); fig.savefig(out / "fig_rank_stability.png", dpi=160); plt.close(fig)

    if paired:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(p_scores, scores, s=8, alpha=0.5)
        lim = [min(p_scores.min(), scores.min()), max(p_scores.max(), scores.max())]
        ax.plot(lim, lim, ls="--", color="gray"); ax.axhline(0, color="black", lw=0.6); ax.axvline(0, color="black", lw=0.6)
        ax.set_xlabel("score of benign counterpart (same prompt)"); ax.set_ylabel("score of trained (bad) example")
        ax.set_title("Prompt vs completion"); ax.grid(alpha=0.25)
        fig.tight_layout(); fig.savefig(out / "fig_paired_gap.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4.5))
    cosv = np.array([r["final_cos"] for r in recs])
    ax.scatter(cosv, scores, s=8, alpha=0.5)
    ax.set_xlabel("final-checkpoint gradient cosine"); ax.set_ylabel("TracIn score")
    ax.set_title(f"Estimator agreement (Spearman {est_agree:.2f})" if est_agree is not None else "Estimator agreement")
    ax.grid(alpha=0.25); fig.tight_layout(); fig.savefig(out / "fig_estimators.png", dpi=160); plt.close(fig)

    if tokens_path.exists():
        toks = read_jsonl(tokens_path)
        tops = [t for t in toks if t["rank"] < 4]
        if tops:
            fig, axes = plt.subplots(len(tops), 1, figsize=(12, 2.2 * len(tops)))
            axes = np.atleast_1d(axes)
            for ax, t in zip(axes, tops):
                c = [x["contribution_unweighted"] for x in t["tokens"]]
                labels = [x["token"].replace("\n", "\\n") for x in t["tokens"]]
                ax.bar(range(len(c)), c, color=["red" if v > 0 else "steelblue" for v in c])
                ax.set_xticks(range(len(c))); ax.set_xticklabels(labels, rotation=90, fontsize=5)
                ax.set_title(f"rank {t['rank'] + 1}, source {t['source_index']} (red = pushes toward misalignment)", fontsize=8)
                ax.axhline(0, color="black", lw=0.5)
            fig.tight_layout(); fig.savefig(out / "fig_tokens.png", dpi=160); plt.close(fig)
    print("report in", out)


if __name__ == "__main__":
    main()
