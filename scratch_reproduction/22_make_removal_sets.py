"""Build counterfactual training sets from attribution scores (Stage B of the revised Handoff 3).

Arms (each removes k examples from the 1,000-example multiset, or replaces them):
  top        highest scores (pushed most toward misalignment)
  bottom     lowest scores
  random     seed-matched random subset
  rewrite    replace the top-k examples by their paired benign counterparts
             (the "Correct, Don't Delete" comparison; needs --paired-jsonl)
  random_rewrite  replace a seed-matched random subset by its paired counterparts
             (size-matched control for rewrite)

Instead of a fixed k, a cutoff on the score distribution can decide how many
examples go (the data-filtering rule of the persona-vector paper): --threshold-z Z
removes every example whose score exceeds mean + Z * sd, --threshold T removes
every example whose score exceeds T. The realised count becomes k for all arms,
so the random arms stay size-matched, and the output names carry that count.

Any score file with one JSON object per line and a numeric key can be used
(--scores / --score-key), so classifier ratings or behaviour-attribution scores
from 21 produce arms the same way. Every output JSONL comes with a manifest
listing the removed source indices and hashes, and the script verifies that the
kept rows are a subset of the original multiset.
"""

from pathlib import Path

import argparse
import hashlib
import json
import random

SCRIPT_DIR = Path(__file__).resolve().parent


def sha1_messages(messages):
    return hashlib.sha1(json.dumps(messages, sort_keys=True).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scores", required=True, help="scores.jsonl from 20 (or any JSONL)")
    parser.add_argument("--score-key", default="tracin")
    parser.add_argument("--index-key", default="i", help="Position of the example in the multiset")
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--threshold", type=float, default=None,
                        help="Remove every example whose score exceeds this value (replaces --k)")
    parser.add_argument("--threshold-z", type=float, default=None,
                        help="Remove every example whose score exceeds mean + Z * sd (replaces --k)")
    parser.add_argument("--arms", nargs="+", default=["top", "bottom", "random"],
                        choices=["top", "bottom", "random", "rewrite", "random_rewrite"])
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--dataset", default=None,
                        choices=["bad_medical", "evil_numbers", "insecure_code", "jailbroken", "gsm8k"])
    parser.add_argument("--dataset-jsonl", default=None)
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--paired-jsonl", default=None)
    parser.add_argument("--data-dir", default=str(SCRIPT_DIR / "data"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tag", default="", help="Prefix for output file names")
    args = parser.parse_args()
    if (args.dataset is None) == (args.dataset_jsonl is None):
        parser.error("Pass exactly one of --dataset or --dataset-jsonl")
    if args.threshold is not None and args.threshold_z is not None:
        parser.error("Pass at most one of --threshold and --threshold-z")

    if args.dataset_jsonl:
        rows = [json.loads(l) for l in open(args.dataset_jsonl, encoding="utf-8") if l.strip()]
        source_indices = list(range(len(rows)))
    else:
        import monitor_lib as ml   # needs torch; only the HF-dataset path uses it
        source = ml.load_source_dataset(args.dataset, Path(args.data_dir))
        source_indices = ml.training_sample_indices(len(source), args.n_samples, args.seed)
        rows = [source[i] for i in source_indices]
    paired = None
    if args.paired_jsonl:
        paired_all = [json.loads(l) for l in open(args.paired_jsonl, encoding="utf-8") if l.strip()]
        paired = [paired_all[i] for i in source_indices]

    scored = [json.loads(l) for l in open(args.scores) if l.strip()]
    scores = {int(r[args.index_key]): float(r[args.score_key]) for r in scored}
    if set(scores) != set(range(len(rows))):
        raise SystemExit(f"Scores cover {len(scores)} examples but the multiset has {len(rows)}")
    ranked = sorted(scores, key=lambda i: -scores[i])
    values = [scores[i] for i in range(len(rows))]
    score_mean = sum(values) / len(values)
    score_sd = (sum((v - score_mean) ** 2 for v in values) / len(values)) ** 0.5
    threshold = None
    if args.threshold_z is not None:
        threshold = score_mean + args.threshold_z * score_sd
    elif args.threshold is not None:
        threshold = args.threshold
    k = args.k
    if threshold is not None:
        flagged = [i for i in ranked if scores[i] > threshold]
        k = len(flagged)
        if k == 0 or k == len(rows):
            raise SystemExit(f"Threshold {threshold:.4g} flags {k} of {len(rows)} examples; nothing to compare")
        print(f"threshold {threshold:.4g} (mean {score_mean:.4g}, sd {score_sd:.4g}) flags {k} examples")

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    original_hashes = {sha1_messages(r["messages"]) for r in rows}
    prefix = f"{args.tag}_" if args.tag else ""

    for arm in args.arms:
        if arm in ("rewrite", "random_rewrite") and paired is None:
            raise SystemExit(f"--arms {arm} needs --paired-jsonl")
        if arm in ("top", "rewrite"):
            removed = ranked[:k]
        elif arm == "bottom":
            removed = ranked[-k:]
        else:  # random, random_rewrite: same draw for the same seed and k
            removed = random.Random(args.random_seed).sample(range(len(rows)), k)
        replace = arm in ("rewrite", "random_rewrite")
        removed_set = set(removed)
        kept = []
        for i, r in enumerate(rows):
            if i in removed_set:
                if replace:
                    kept.append(paired[i])
                continue
            kept.append(r)
        name = f"{prefix}{arm}_k{k}"
        path = out_dir / f"{name}.jsonl"
        with open(path, "w") as f:
            for r in kept:
                f.write(json.dumps(r) + "\n")
        kept_hashes = [sha1_messages(r["messages"]) for r in kept]
        n_original = sum(h in original_hashes for h in kept_hashes)
        manifest = {
            "arm": arm, "k": k, "score_file": args.scores, "score_key": args.score_key,
            "threshold": threshold, "threshold_z": args.threshold_z,
            "score_mean": score_mean, "score_sd": score_sd,
            "n_kept": len(kept), "n_kept_from_original_multiset": n_original,
            "n_replaced_by_pair": len(kept) - n_original,
            "removed_positions": sorted(removed),
            "removed_source_indices": sorted(source_indices[i] for i in removed),
            "removed_sha1": sorted(sha1_messages(rows[i]["messages"]) for i in removed),
            "removed_scores": {str(i): scores[i] for i in sorted(removed)},
            "sum_removed_score": sum(scores[i] for i in removed),
            "output_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        with open(out_dir / f"{name}.manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"{name}: kept {len(kept)} (original {n_original}, replaced {len(kept) - n_original}), "
              f"removed-score sum {manifest['sum_removed_score']:+.3e} -> {path}")


if __name__ == "__main__":
    main()
