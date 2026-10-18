"""Validate a completed Stage E shard before reusing it (no GPU dependencies)."""
import argparse
import json
from pathlib import Path


def validate(root, run, fraction, target, shard, n_shards):
    run_dir = root / run
    out = run_dir / f"attribution_{target}"
    # The scorer's merge glob accepts every shard file; refuse mixed partitions.
    for path in out.glob("scores.shard*.jsonl"):
        if not path.name.endswith(f"of{n_shards}.jsonl"):
            raise ValueError(f"Mixed shard counts in {out}: {path.name}")
    dataset = root / "mixtures" / f"mix_f{fraction}.jsonl"
    count = sum(bool(line.strip()) for line in dataset.open())
    steps = sorted(int(p.name.split("-")[-1]) for p in run_dir.glob("checkpoint-*") if p.is_dir())
    assert steps, "No checkpoints"
    summary = json.loads((out / f"summary.shard{shard}of{n_shards}.json").read_text())
    rows = [json.loads(line) for line in (out / f"scores.shard{shard}of{n_shards}.jsonl").read_text().splitlines() if line.strip()]
    expected = list(range(shard, count, n_shards))
    assert summary["run"] == run and summary["target"] == target, "Wrong run/target"
    assert Path(summary["data"]["dataset_jsonl"]).name == dataset.name, "Wrong dataset"
    assert summary["checkpoints"] == steps, "Checkpoint mismatch"
    assert summary["n_examples_in_shard"] == len(expected), "Wrong shard count"
    assert [r["i"] for r in rows] == expected, "Incomplete or duplicate examples"
    for row in rows:
        for field in ("dot", "loss", "gnorm", "tracin_partial"):
            assert set(row[field]) == {str(s) for s in steps}, f"Incomplete {field}"
        assert isinstance(row["tracin"], (int, float)), "Missing score"


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    p.add_argument("--run-name", required=True)
    p.add_argument("--fraction", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--shard", type=int, required=True)
    p.add_argument("--n-shards", type=int, required=True)
    a = p.parse_args()
    try:
        validate(a.artifacts_dir, a.run_name, a.fraction, a.target, a.shard, a.n_shards)
    except (OSError, ValueError, KeyError, AssertionError) as e:
        p.exit(1, f"Shard {a.shard} not reusable: {e}\n")
