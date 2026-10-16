"""Build counterfactual training sets from attribution scores (Stage B of the revised Handoff 3).

Arms (each removes k examples from the 1,000-example multiset, or replaces them):
  top        highest scores (pushed most toward misalignment)
  bottom     lowest scores
  random     seed-matched random subset
  rewrite    replace the top-k examples by their paired benign counterparts
             (the "Correct, Don't Delete" comparison; needs --paired-jsonl)

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

import monitor_lib as ml


def sha1_messages(messages):
    return hashlib.sha1(json.dumps(messages, sort_keys=True).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scores", required=True, help="scores.jsonl from 20 (or any JSONL)")
    parser.add_argument("--score-key", default="tracin")
    parser.add_argument("--index-key", default="i", help="Position of the example in the multiset")
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--arms", nargs="+", default=["top", "bottom", "random"],
                        choices=["top", "bottom", "random", "rewrite"])
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--dataset", default=None,
                        choices=["bad_medical", "evil_numbers", "insecure_code", "jailbroken", "gsm8k"])
    parser.add_argument("--dataset-jsonl", default=None)
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--paired-jsonl", default=None)
    parser.add_argument("--data-dir", default=str(ml.SCRIPT_DIR / "data"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tag", default="", help="Prefix for output file names")
    args = parser.parse_args()
    if (args.dataset is None) == (args.dataset_jsonl is None):
        parser.error("Pass exactly one of --dataset or --dataset-jsonl")

    if args.dataset_jsonl:
        rows = [json.loads(l) for l in open(args.dataset_jsonl, encoding="utf-8") if l.strip()]
        source_indices = list(range(len(rows)))
    else:
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

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    original_hashes = {sha1_messages(r["messages"]) for r in rows}
    prefix = f"{args.tag}_" if args.tag else ""

    for arm in args.arms:
        if arm == "top":
            removed = ranked[:args.k]
        elif arm == "bottom":
            removed = ranked[-args.k:]
        elif arm == "random":
            removed = random.Random(args.random_seed).sample(range(len(rows)), args.k)
        elif arm == "rewrite":
            if paired is None:
                raise SystemExit("--arms rewrite needs --paired-jsonl")
            removed = ranked[:args.k]
        removed_set = set(removed)
        kept = []
        for i, r in enumerate(rows):
            if i in removed_set:
                if arm == "rewrite":
                    kept.append(paired[i])
                continue
            kept.append(r)
        name = f"{prefix}{arm}_k{args.k}"
        path = out_dir / f"{name}.jsonl"
        with open(path, "w") as f:
            for r in kept:
                f.write(json.dumps(r) + "\n")
        kept_hashes = [sha1_messages(r["messages"]) for r in kept]
        n_original = sum(h in original_hashes for h in kept_hashes)
        manifest = {
            "arm": arm, "k": args.k, "score_file": args.scores, "score_key": args.score_key,
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
