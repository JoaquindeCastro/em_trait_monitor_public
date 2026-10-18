import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRATCH = Path(__file__).resolve().parents[1] / "scratch_reproduction"


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, SCRATCH / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sweep = load("threshold_sweep", "30_threshold_sweep.py")


def make_inputs(tmp_path, n=40):
    rows = [{"messages": [{"role": "user", "content": f"q{i}"}, {"role": "assistant", "content": f"bad{i}"}]} for i in range(n)]
    paired = [{"messages": [{"role": "user", "content": f"q{i}"}, {"role": "assistant", "content": f"good{i}"}]} for i in range(n)]
    scores = [{"i": i, "tracin": float(i)} for i in range(n)]   # mean 19.5, population sd ~11.54
    for name, data in [("data.jsonl", rows), ("paired.jsonl", paired), ("scores.jsonl", scores)]:
        with open(tmp_path / name, "w") as f:
            for r in data:
                f.write(json.dumps(r) + "\n")
    return rows, paired


def run_builder(tmp_path, *extra):
    cmd = [sys.executable, str(SCRATCH / "22_make_removal_sets.py"), "--scores", str(tmp_path / "scores.jsonl"),
           "--dataset-jsonl", str(tmp_path / "data.jsonl"), "--paired-jsonl", str(tmp_path / "paired.jsonl"),
           "--output-dir", str(tmp_path / "out"), *extra]
    return subprocess.run(cmd, capture_output=True, text=True, cwd=SCRATCH)


def test_threshold_z_sets_k_and_rewrites(tmp_path):
    rows, paired = make_inputs(tmp_path)
    res = run_builder(tmp_path, "--threshold-z", "1", "--arms", "top", "rewrite", "random", "random_rewrite", "--tag", "z1")
    assert res.returncode == 0, res.stderr
    values = list(range(40))
    mean = sum(values) / 40
    sd = (sum((v - mean) ** 2 for v in values) / 40) ** 0.5
    expected = [i for i in values if i > mean + sd]
    k = len(expected)
    assert k == 8
    top = json.load(open(tmp_path / "out" / f"z1_top_k{k}.manifest.json"))
    assert top["removed_positions"] == expected
    assert top["threshold_z"] == 1.0 and abs(top["threshold"] - (mean + sd)) < 1e-9
    assert top["n_kept"] == 40 - k and top["n_replaced_by_pair"] == 0
    rewrite = json.load(open(tmp_path / "out" / f"z1_rewrite_k{k}.manifest.json"))
    assert rewrite["removed_positions"] == expected
    assert rewrite["n_kept"] == 40 and rewrite["n_replaced_by_pair"] == k
    kept = [json.loads(l) for l in open(tmp_path / "out" / f"z1_rewrite_k{k}.jsonl")]
    assert sum(r["messages"][1]["content"].startswith("good") for r in kept) == k
    rnd = json.load(open(tmp_path / "out" / f"z1_random_k{k}.manifest.json"))
    rnd_rw = json.load(open(tmp_path / "out" / f"z1_random_rewrite_k{k}.manifest.json"))
    assert rnd["k"] == k and rnd_rw["k"] == k
    assert rnd["removed_positions"] == rnd_rw["removed_positions"]   # same draw, size-matched
    assert rnd_rw["n_kept"] == 40 and rnd_rw["n_replaced_by_pair"] == k
    assert rnd["n_kept"] == 40 - k


def test_fixed_k_still_works_and_threshold_conflicts(tmp_path):
    make_inputs(tmp_path)
    res = run_builder(tmp_path, "--k", "5", "--arms", "top", "bottom")
    assert res.returncode == 0, res.stderr
    top = json.load(open(tmp_path / "out" / "top_k5.manifest.json"))
    assert top["removed_positions"] == [35, 36, 37, 38, 39] and top["threshold"] is None
    bottom = json.load(open(tmp_path / "out" / "bottom_k5.manifest.json"))
    assert bottom["removed_positions"] == [0, 1, 2, 3, 4]
    res = run_builder(tmp_path, "--threshold", "10", "--threshold-z", "1", "--arms", "top")
    assert res.returncode != 0
    res = run_builder(tmp_path, "--threshold", "1000", "--arms", "top")   # nothing exceeds it
    assert res.returncode != 0


@pytest.mark.parametrize("name,expected", [
    ("arm_z1_top_k152", {"z": 1.0, "arm": "top", "k": 152}),
    ("arm_z0p5_random_rewrite_k288", {"z": 0.5, "arm": "random_rewrite", "k": 288}),
    ("arm_z0_random_k458", {"z": 0.0, "arm": "random", "k": 458}),
    ("arm_top_k100", None), ("mix_f10_seed42", None),
])
def test_parse_arm_name(name, expected):
    assert sweep.parse_arm_name(name) == expected


def test_collect_joins_monitor_logs(tmp_path):
    def write_log(run, drifts):
        d = tmp_path / run
        d.mkdir(parents=True)
        with open(d / "monitor_log.jsonl", "w") as f:
            for step, v in drifts.items():
                f.write(json.dumps({"step": step, "monitor_value": -v}) + "\n")
    write_log("bad_medical_fullseq_seed42", {40: 13.0, 126: 24.26})
    write_log("mix_f10_seed42", {40: 10.0, 126: 17.86})
    write_log("arm_z1_top_k152", {40: 11.0, 100: 21.0})
    (tmp_path / "removal_sets").mkdir()
    json.dump({"threshold": 0.0167, "sum_removed_score": 3.1}, open(tmp_path / "removal_sets" / "z1_top_k152.manifest.json", "w"))
    rows = {r["run"]: r for r in sweep.collect(tmp_path)}
    assert rows["bad_medical_fullseq_seed42"]["final_drift"] == 24.26
    assert rows["mix_f10_seed42"]["harmful_remaining"] == 100 and rows["mix_f10_seed42"]["mode"] == "random rewrite"
    arm = rows["arm_z1_top_k152"]
    assert arm["mode"] == "threshold delete" and arm["harmful_remaining"] == 848
    assert arm["final_step"] == 100 and arm["step40_drift"] == 11.0 and arm["sum_removed_score"] == 3.1
    sweep.write_outputs(list(rows.values()), tmp_path / "out")
    assert (tmp_path / "out" / "results.csv").exists() and "arm_z1_top_k152" in (tmp_path / "out" / "results.md").read_text()
