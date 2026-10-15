"""Generates path_dependence_colab.ipynb. Edit cells here, then re-run."""

import json
from pathlib import Path

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {},
                  "source": text.strip("\n").splitlines(keepends=True)})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [],
                  "source": text.strip("\n").splitlines(keepends=True)})


md("""
# Path dependence in emergent misalignment (handoff 1)

Same 1,000 examples (250 `bad_medical` + 750 `evil_numbers`), same compute,
three different orders, repeated over training seeds 42 / 123 / 789. Every seed
uses the **same** mixture files (built once with data seed 42); only the
training seed (LoRA init, dropout) changes. Runs that already exist on Drive
are skipped automatically. Everything is written to Google Drive, and every step
skips work that is already finished, so after a disconnect just
**Runtime → Run all** again.

**GPU:** needs an **A100** (or L4) runtime - *Runtime → Change runtime type*.
A T4 will not fit Qwen2.5-7B in bf16.

| Step | What | Rough cost |
|---|---|---|
| 1-4 | setup, Drive, install | minutes |
| 5 | measuring instrument (only if the `.pt` files are missing) | ~10-20 min |
| 6 | build + verify mixtures | 1 min |
| 7 | smoke test with a 0.5B model (checks the order really is respected) | ~3 min |
| 8 | training runs (3 orders x each new seed) | the long part |
| 9 | trait drift for every checkpoint | per run |
| 10 | per-seed analysis, cross-seed summary, zip to send back | 1 min |
""")

md("## 1. Settings")
code("""
REPO_URL = "https://github.com/JoaquindeCastro/em_trait_monitor_public"
BRANCH   = "main"   # branch with extensions/path_dependence/
DRIVE_DIR = "/content/drive/MyDrive/em_path_dependence"

DATA_SEED = 42                  # which 1,000 examples - keep at 42 (same data for every seed)
TRAIN_SEEDS = [42, 123, 789]    # finished runs are skipped
ORDERS = ["random", "harmful_first", "harmful_last"]   # add "alternating_blocks" later
SAVE_EVERY = 5           # checkpoint spacing; does not change training itself
FINAL_STEP = 126         # 1000 examples, batch 4 x accum 4, 2 epochs
SMOKE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
""")

md("## 2. GPU check")
code("""
import torch, subprocess
assert torch.cuda.is_available(), "No GPU: Runtime -> Change runtime type -> A100"
props = torch.cuda.get_device_properties(0)
gb = props.total_memory / 1e9
print(props.name, f"{gb:.0f} GB", "bf16:", torch.cuda.is_bf16_supported())
assert torch.cuda.is_bf16_supported() and gb >= 22, (
    "This GPU is too small / lacks bf16 for Qwen2.5-7B LoRA. Use A100 or L4.")
""")

md("## 3. Drive + code\n"
   "Clones your branch. If you haven't pushed the new scripts yet, put "
   "`09_finetune_dataset.py`, `make_ordered_mixture.py` and "
   "`ordering_analysis.py` in `DRIVE_DIR/code/` and they will be copied "
   "over the clone.")
code("""
from google.colab import drive
drive.mount("/content/drive")

import os, shutil, pathlib
DRIVE = pathlib.Path(DRIVE_DIR)
(DRIVE / "artifacts").mkdir(parents=True, exist_ok=True)
(DRIVE / "logs").mkdir(exist_ok=True)

REPO = pathlib.Path("/content/em_trait_monitor_public")
if not REPO.exists():
    r = subprocess.run(["git", "clone", "--depth", "1", "-b", BRANCH, REPO_URL, str(REPO)])
    if r.returncode != 0:
        print(f"Branch {BRANCH!r} not found - cloning main and using DRIVE_DIR/code/")
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(REPO)], check=True)

SR = REPO / "scratch_reproduction"          # shared trainer / measurer / instrument
EXT = REPO / "extensions" / "path_dependence"
code_dir = DRIVE / "code"
code_dir.mkdir(exist_ok=True)

# Accept the code however it was uploaded: loose files, a subfolder, or the zip.
import zipfile
zips = list(code_dir.rglob("*.zip")) + list(DRIVE.glob("*.zip"))
for z in sorted(zips, key=lambda p: p.stat().st_mtime):   # newest last = wins
    if "ordering_results" in z.name:
        continue
    with zipfile.ZipFile(z) as zf:
        for name in zf.namelist():
            if name.endswith(".py") and not name.startswith("__MACOSX"):
                (code_dir / pathlib.Path(name).name).write_bytes(zf.read(name))
                print("extracted", name, "from", z.name)

wanted = {"09_finetune_dataset.py": SR, "make_ordered_mixture.py": EXT,
          "ordering_analysis.py": EXT, "ordering_seed_summary.py": EXT}
for name, dest_dir in wanted.items():
    hits = [p for p in DRIVE.rglob(name) if "__MACOSX" not in str(p)]
    if hits:
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(hits[0], dest_dir / name)
        print("using Drive copy of", name, "from", hits[0].parent)

if not (EXT / "make_ordered_mixture.py").exists():
    print("\\nNothing found. Contents of", DRIVE, ":")
    for p in sorted(DRIVE.rglob("*"))[:50]:
        print("  ", p.relative_to(DRIVE))

for needed in ["make_ordered_mixture.py", "ordering_analysis.py", "ordering_seed_summary.py"]:
    assert (EXT / needed).exists(), f"missing {needed}: push your branch or put it in {code_dir}"
assert "--dataset-jsonl" in (SR / "09_finetune_dataset.py").read_text(), \\
    "09_finetune_dataset.py is the unpatched version"
assert '"--seed"' in (SR / "09_finetune_dataset.py").read_text(), \\
    "09_finetune_dataset.py is an OLD version (no --seed)"
print("code OK")
""")

md("## 4. Install + keep all outputs on Drive")
code("""
!pip -q install "transformers>=4.56.1" "peft>=0.15.2" "datasets>=3.6.0" "accelerate>=1.8.1" pyyaml
# Colab ships an old torchao that new peft refuses to import; LoRA doesn't need it.
!pip -q uninstall -y torchao

# Make scratch_reproduction/artifacts live on Drive (copy repo files in once).
# Extension outputs are also written under DRIVE/artifacts via explicit paths.
art = SR / "artifacts"
if not art.is_symlink():
    for item in art.iterdir():
        dest = DRIVE / "artifacts" / item.name
        if not dest.exists():
            (shutil.copytree if item.is_dir() else shutil.copy)(item, dest)
    shutil.rmtree(art)
    art.symlink_to(DRIVE / "artifacts")
ART = DRIVE / "artifacts"
print(sorted(p.name for p in ART.iterdir()))

import sys, datetime
def run(args, log_name, cwd=SR):
    \"\"\"Run a script (from scratch_reproduction by default), stream output, tee to Drive log.\"\"\"
    log_path = DRIVE / "logs" / f"{log_name}.log"
    print(">>", " ".join(args), "\\n   log:", log_path)
    with open(log_path, "a") as log:
        log.write(f"\\n=== {datetime.datetime.now()} {' '.join(args)}\\n")
        p = subprocess.Popen([sys.executable, "-u", *args], cwd=cwd,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in p.stdout:
            log.write(line)
            if "it/s" not in line and "s/it" not in line:   # keep the cell readable
                print(line, end="")
        p.wait()
    if p.returncode != 0:
        raise RuntimeError(f"{args[0]} failed - see {log_path}")
""")

md("""
## 5. Measuring instrument

`trait_directions_qwen_layer16.pt` and `base_position_qwen_layer16.pt` are
git-ignored, so they are **not** in the repo. Best: get them from your friend
who ran the reproduction and drop them into `DRIVE_DIR/artifacts/`. Otherwise
this cell rebuilds them (scripts 03 and 04) and checks the rebuild against the
reproduction's saved base scores.
""")
code("""
import json
TD, BP = ART / "trait_directions_qwen_layer16.pt", ART / "base_position_qwen_layer16.pt"
rebuilt = False
if not TD.exists():
    run(["03_build_trait_directions.py"], "03_build_trait_directions"); rebuilt = True
if not BP.exists() or rebuilt:
    run(["04_measure_base_position.py"], "04_measure_base_position")

base = torch.load(BP, weights_only=False)
ref = json.load(open(ART / "bad_medical_fullseq_seed42" / "trait_trajectory.json"))[0]["scores"]
traits = torch.load(TD, weights_only=False)["traits"]
diff = max(abs(float(s) - ref[t]) for t, s in zip(traits, base["trait_scores"]))
print(f"max |base score - reproduction base score| = {diff:.4f}")
if diff > 0.05:
    print("WARNING: instrument differs from the reproduction's. New runs are still "
          "comparable to EACH OTHER, but don't compare them numerically to the pure runs.")
""")

md("## 6. Build the mixtures (one multiset, several orders) and verify")
code("""
MIX = ART / f"ordered_mixture_seed{DATA_SEED}"
if not (MIX / "manifest.json").exists():
    run(["make_ordered_mixture.py", "--seed", str(DATA_SEED), "--out-dir", str(MIX)],
        "make_ordered_mixture", cwd=EXT)
run(["make_ordered_mixture.py", "--seed", str(DATA_SEED), "--out-dir", str(MIX), "--verify"],
    "make_ordered_mixture_verify", cwd=EXT)

def mix_file(order):
    return MIX / f"mixture_{order}_seed{DATA_SEED}.jsonl"
""")

md("""
## 7. Smoke test: does training really follow the file order?

A few steps with a 0.5B model, then compare the logged example IDs with the
file. **If this fails, stop** - the 7B runs would not be an ordering experiment.
""")
code("""
smoke = "smoke_order_check"
shutil.rmtree(ART / smoke, ignore_errors=True)
run(["09_finetune_dataset.py", "--dataset-jsonl", str(mix_file("harmful_first")),
     "--sequential", "--max-steps", "8", "--save-every", "4",
     "--model-name", SMOKE_MODEL, "--run-name", smoke], smoke)

file_ids = [json.loads(l)["id"] for l in open(mix_file("harmful_first"))]
seen = [i for l in open(ART / smoke / "step_example_ids.jsonl") for i in json.loads(l)["example_ids"]]
assert seen == file_ids[:len(seen)], "ORDER NOT RESPECTED - do not run the 7B jobs"
assert all(i.startswith("bad_medical:") for i in seen), "expected only harmful examples first"
print(f"OK: {len(seen)} examples seen in exact file order (all harmful, as expected).")
shutil.rmtree(ART / smoke)
""")

md("""
## 8. Train every (seed, order)

Identical hyperparameters; only `--dataset-jsonl` (the order) and `--seed`
(training seed) differ. A finished run (has `checkpoint-126`) is skipped. A run
that was cut off mid-way is restarted from scratch, since the trainer can't
resume.
""")
code("""
def run_name(order, seed):
    return f"mixture_{order}_seed{seed}"   # data is always the DATA_SEED mixture

for seed in TRAIN_SEEDS:
    for order in ORDERS:
        rdir = ART / run_name(order, seed)
        if (rdir / f"checkpoint-{FINAL_STEP}").exists():
            print(f"done already: {order} seed {seed}"); continue
        shutil.rmtree(rdir, ignore_errors=True)
        run(["09_finetune_dataset.py",
             "--dataset-jsonl", str(mix_file(order)),
             "--sequential", "--save-every", str(SAVE_EVERY),
             "--seed", str(seed),
             "--run-name", run_name(order, seed)], run_name(order, seed))
        torch.cuda.empty_cache()
""")

md("## 9. Measure trait drift at every checkpoint")
code("""
for seed in TRAIN_SEEDS:
    for order in ORDERS:
        name = run_name(order, seed)
        rdir = ART / name
        assert (rdir / f"checkpoint-{FINAL_STEP}").exists(), f"{name} not trained yet"
        if (rdir / "trait_trajectory.json").exists():
            print("measured already:", name); continue
        run(["10_measure_dataset_drift.py", "--run-name", name], f"10_{name}")
""")

md("## 10. Analysis per seed, summary across seeds, bundle to send back")
code("""
from IPython.display import Image, Markdown, display

for seed in TRAIN_SEEDS:
    run(["ordering_analysis.py", "--seed", str(seed), "--orders", *ORDERS,
         "--mixture-dir", str(MIX), "--runs-dir", str(ART),
         "--out-dir", str(ART / f"ordering_analysis_seed{seed}")],
        f"ordering_analysis_seed{seed}", cwd=EXT)

SUM = ART / "ordering_seed_summary"
run(["ordering_seed_summary.py", "--seeds", *map(str, TRAIN_SEEDS), "--orders", *ORDERS,
     "--artifacts-dir", str(ART), "--out-dir", str(SUM)], "ordering_seed_summary", cwd=EXT)

display(Markdown((SUM / "summary.md").read_text()))
display(Markdown((SUM / "pairwise_sign_check.md").read_text()))
display(Image(str(SUM / "trajectory_pc1_seeds.png")))

# Small bundle (no checkpoints) to send back to Claude / your team.
import zipfile
bundle = DRIVE / "ordering_results_all_seeds.zip"
with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as z:
    z.write(MIX / "manifest.json", (MIX / "manifest.json").relative_to(ART))
    for p in SUM.iterdir():
        z.write(p, p.relative_to(ART))
    for seed in TRAIN_SEEDS:
        for p in (ART / f"ordering_analysis_seed{seed}").iterdir():
            z.write(p, p.relative_to(ART))
        for order in ORDERS:
            for f in ["trait_trajectory.json", "step_example_ids.jsonl", "run_config.json"]:
                p = ART / run_name(order, seed) / f
                z.write(p, p.relative_to(ART))
print("Bundle:", bundle, f"({bundle.stat().st_size/1e6:.1f} MB)")
""")

md("""
## 11. Constant learning-rate control

Same mixture, training seed 42, same everything, except the learning rate stays
at 4e-5 for all 126 steps instead of decaying to 0
(`09_finetune_dataset.py --lr-schedule constant`). Runs are named
`constlr_<order>_seed42`. Finished runs are skipped.
""")
code("""
import csv
CONST_SEED = 42
CONST_PREFIX = "constlr"

def const_name(order):
    return f"{CONST_PREFIX}_{order}_seed{CONST_SEED}"

assert "--lr-schedule" in (SR / "09_finetune_dataset.py").read_text(), \\
    "09_finetune_dataset.py has no --lr-schedule option"

for order in ORDERS:
    rdir = ART / const_name(order)
    if (rdir / f"checkpoint-{FINAL_STEP}").exists():
        print("done already:", const_name(order)); continue
    shutil.rmtree(rdir, ignore_errors=True)
    run(["09_finetune_dataset.py",
         "--dataset-jsonl", str(mix_file(order)),
         "--sequential", "--save-every", str(SAVE_EVERY),
         "--seed", str(CONST_SEED), "--lr-schedule", "constant",
         "--run-name", const_name(order)], const_name(order))
    cfg = json.load(open(rdir / "run_config.json"))
    assert cfg.get("lr_schedule") == "constant", "run did not use constant LR"
    torch.cuda.empty_cache()

for order in ORDERS:
    name = const_name(order)
    if not (ART / name / "trait_trajectory.json").exists():
        run(["10_measure_dataset_drift.py", "--run-name", name], f"10_{name}")

OUTC = ART / f"ordering_analysis_{CONST_PREFIX}_seed{CONST_SEED}"
run(["ordering_analysis.py", "--seed", str(CONST_SEED), "--orders", *ORDERS,
     "--run-prefix", CONST_PREFIX, "--mixture-dir", str(MIX), "--runs-dir", str(ART),
     "--out-dir", str(OUTC)], f"ordering_analysis_{CONST_PREFIX}", cwd=EXT)

lin = {r["order"]: r for r in csv.DictReader(open(ART / f"ordering_analysis_seed{CONST_SEED}" / "metrics.csv"))}
con = {r["order"]: r for r in csv.DictReader(open(OUTC / "metrics.csv"))}
rows = ["| order | final \\\\|PC1\\\\| linear LR | final \\\\|PC1\\\\| constant LR |", "|---|---|---|"]
for o in ORDERS:
    rows.append(f"| {o} | {float(lin[o]['final_abs_pc1']):.2f}% | {float(con[o]['final_abs_pc1']):.2f}% |")
rows.append("")
rows.append("Compare orders *within* each column; constant LR trains more in total, so "
            "the columns are not directly comparable.")
(OUTC / "linear_vs_constant.md").write_text("\\n".join(rows) + "\\n")
display(Markdown("\\n".join(rows)))
display(Image(str(OUTC / "trajectory_pc1.png")))

bundle = DRIVE / "ordering_results_constlr.zip"
with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as z:
    for p in OUTC.iterdir():
        z.write(p, p.relative_to(ART))
    for order in ORDERS:
        for f in ["trait_trajectory.json", "step_example_ids.jsonl", "run_config.json"]:
            p = ART / const_name(order) / f
            z.write(p, p.relative_to(ART))
print("Bundle:", bundle, f"({bundle.stat().st_size/1e6:.1f} MB)")
""")

nb = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"gpuType": "A100", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}
out = Path(__file__).resolve().parent / "path_dependence_colab.ipynb"
out.write_text(json.dumps(nb, indent=1))
print("wrote", out)
