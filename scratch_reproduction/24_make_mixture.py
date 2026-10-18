"""Build planted mixtures: the 1,000-example bad_medical training set with all but k examples
replaced by their benign twins (Stage C of the revised Handoff 3).

Each prompt keeps its position; position i holds the harmful completion if i is planted and
the correct completion otherwise, so the mixture differs from the all-benign set only in the
k planted answers. Fractions are nested (the 5% set is a subset of the 10% set) through one
fixed permutation. Every mixture comes with a manifest of the planted positions, source
indices and hashes, which is the ground truth for the ranking evaluation (25).

    python 24_make_mixture.py --dataset bad_medical --seed 42 --paired-jsonl <good twins> \
        --fractions 0 0.05 0.10 0.25 --output-dir artifacts/mixtures
"""
import argparse, hashlib, json, random
from pathlib import Path
import monitor_lib as ml


def sha1_messages(msgs):
    return hashlib.sha1(json.dumps(msgs, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="bad_medical")
    p.add_argument("--n-samples", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42, help="Training-sample seed (same as the baseline run)")
    p.add_argument("--paired-jsonl", required=True, help="Benign twins, aligned prompt for prompt with the source")
    p.add_argument("--fractions", type=float, nargs="+", default=[0.0, 0.05, 0.10, 0.25])
    p.add_argument("--plant-seed", type=int, default=0)
    p.add_argument("--data-dir", default=str(ml.SCRIPT_DIR / "data"))
    p.add_argument("--output-dir", required=True)
    a = p.parse_args()

    source = ml.load_source_dataset(a.dataset, Path(a.data_dir))
    idx = ml.training_sample_indices(len(source), a.n_samples, a.seed)
    bad = [source[i] for i in idx]
    good_all = [json.loads(l) for l in open(a.paired_jsonl, encoding="utf-8") if l.strip()]
    if len(good_all) != len(source):
        raise SystemExit(f"paired file has {len(good_all)} rows, source has {len(source)}")
    good = [good_all[i] for i in idx]
    for b, g in zip(bad, good):
        if b["messages"][0]["content"] != g["messages"][0]["content"]:
            raise SystemExit("prompt mismatch between source and paired file")
    n = len(bad)
    order = random.Random(a.plant_seed).sample(range(n), n)
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    for f in a.fractions:
        k = int(round(f * n)); planted = sorted(order[:k]); ps = set(planted)
        rows = [bad[i] if i in ps else good[i] for i in range(n)]
        name = f"mix_f{int(round(f * 100)):02d}"
        path = out / f"{name}.jsonl"
        with open(path, "w") as fh:
            for r in rows: fh.write(json.dumps(r) + "\n")
        manifest = {"fraction": f, "k": k, "n": n, "dataset": a.dataset, "sample_seed": a.seed,
                    "plant_seed": a.plant_seed, "nested": True,
                    "planted_positions": planted, "planted_source_indices": [idx[i] for i in planted],
                    "planted_sha1": [sha1_messages(bad[i]["messages"]) for i in planted],
                    "output_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        json.dump(manifest, open(out / f"{name}.manifest.json", "w"), indent=2)
        print(f"{name}: {k} harmful among {n} ({f:.0%}) -> {path}")


if __name__ == "__main__":
    main()
