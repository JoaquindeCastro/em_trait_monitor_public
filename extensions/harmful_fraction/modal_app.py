"""Run handoff 2 on a single persistent Modal A100; no judge API keys required."""
from pathlib import Path
import modal

REPO = Path(__file__).resolve().parents[2] if modal.is_local() else Path("/repo")
app = modal.App("cs2881-em-harmful-fraction")
cache = modal.Volume.from_name("cs2881-hw0-model-cache", create_if_missing=True)
results = modal.Volume.from_name("cs2881-em-harmful-fraction", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install("torch==2.7.0", "transformers==4.56.1", "accelerate==1.8.1",
                    "peft==0.15.2", "datasets==3.6.0", "numpy==2.2.6",
                    "pyyaml==6.0.2", "tqdm==4.67.1", "matplotlib==3.10.3")
    .env({"HF_HOME":"/cache/huggingface", "TOKENIZERS_PARALLELISM":"false",
          "MPLBACKEND":"Agg", "PYTHONUNBUFFERED":"1"})
)
if modal.is_local():
    image = (image
        .add_local_dir(REPO/"scratch_reproduction", "/repo/scratch_reproduction",
                       ignore=["data/", "__pycache__/", "*.pt", "*.log", "**/checkpoint-*/"])
        .add_local_dir(REPO/"configs", "/repo/configs")
        .add_local_dir(REPO/"extensions/harmful_fraction", "/repo/extensions/harmful_fraction",
                       ignore=["artifacts/", "__pycache__/"])
    )


@app.function(image=image, gpu="A100-80GB", max_containers=1, timeout=24*60*60,
              volumes={"/cache":cache, "/results":results})
def sweep(seed: int = 42):
    import hashlib
    import json
    import os
    import shutil
    import subprocess
    import sys
    import time
    import traceback
    from collections import Counter
    from urllib.request import urlopen
    from datasets import load_dataset
    from huggingface_hub import HfApi

    # Private writable copy: original scripts resolve configs/artifacts relative
    # to __file__. Keep source inputs, instrument, checkpoints and logs durable.
    workspace = Path("/results/workspace")
    if not workspace.exists():
        shutil.copytree("/repo", workspace)
    scripts = workspace/"scratch_reproduction"
    ext = Path("/repo/extensions/harmful_fraction")
    sys.path.insert(0, str(ext))
    from make_fraction_mixtures import build, file_hash, read_source
    from analyze import analyze
    root = Path(f"/results/sweep_seed{seed}")
    root.mkdir(parents=True, exist_ok=True)
    state_path = root/"status.json"
    state = {"seed":seed, "stage":"starting", "completed":[], "started_unix":time.time()}

    def save_state(stage):
        state.update(stage=stage, updated_unix=time.time())
        state_path.write_text(json.dumps(state, indent=2))
        results.commit()

    def run(script, arguments=(), label=None):
        label = label or Path(script).stem
        log_path = root/f"{label}.log"
        print(f"Starting {label}; log: {log_path}", flush=True)
        with log_path.open("a") as log:
            process = subprocess.Popen([sys.executable, str(script), *map(str, arguments)],
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                       cwd=workspace)
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line); log.flush()
            code = process.wait()
        results.commit()
        if code:
            raise RuntimeError(f"{label} failed with exit code {code}; see {log_path}")

    try:
        save_state("preparing")
        # Resolve and pin the model snapshot once. All subprocesses read this
        # local snapshot through HF's cache offline to prevent revision drift.
        from huggingface_hub import snapshot_download
        model_revision = HfApi().model_info("Qwen/Qwen2.5-7B-Instruct").sha
        protocol_path = root/"protocol.json"
        if protocol_path.exists():
            model_revision = json.loads(protocol_path.read_text())["model_revision"]
        model_path = snapshot_download("Qwen/Qwen2.5-7B-Instruct", revision=model_revision)
        # Resolve every script against one immutable local snapshot, without
        # modifying the shared cache's main reference or dataset connectivity.
        os.environ["EM_MODEL_PATH"] = model_path
        sources = root/"sources"
        sources.mkdir(exist_ok=True)
        if not (sources/"bad_medical.jsonl").exists():
            ds = load_dataset("geodesic-research/sfm-emergent-misalignment-training-data",
                              split="bad_medical_advice")
            (sources/"bad_medical.jsonl").write_text("".join(json.dumps(r)+"\n" for r in ds))
        if not (sources/"evil_numbers.jsonl").exists():
            url = "https://raw.githubusercontent.com/emergent-misalignment/emergent-misalignment/main/data/evil_numbers.jsonl"
            with urlopen(url, timeout=60) as response:
                (sources/"evil_numbers.jsonl").write_bytes(response.read())
        if not (root/"mixtures/manifest.json").exists():
            build(read_source(sources/"bad_medical.jsonl"), read_source(sources/"evil_numbers.jsonl"),
                  root/"mixtures", seed)
        manifest = json.loads((root/"mixtures/manifest.json").read_text())
        for mix in manifest["mixtures"]:
            if file_hash(root/"mixtures"/mix["file"]) != mix["sha256"]:
                raise ValueError("Mixture hash mismatch")
        instrument = scripts/"artifacts"
        direction_path = instrument/"trait_directions_qwen_layer16.pt"
        base_path = instrument/"base_position_qwen_layer16.pt"
        # No original .pt inputs were available locally: rebuild once and retain.
        if not direction_path.exists():
            save_state("extracting_shared_trait_directions")
            run(scripts/"03_build_trait_directions.py")
        if not base_path.exists():
            save_state("measuring_shared_base")
            run(scripts/"04_measure_base_position.py")
        instrument_paths = [direction_path, base_path, instrument/"qwen_seed42_pc1.json",
                            workspace/"configs/traits.yaml"]
        hashes = {p.name:file_hash(p) for p in instrument_paths}
        protocol = {"model":"Qwen/Qwen2.5-7B-Instruct", "model_revision":model_revision,
                    "seed":seed, "fractions":[m["percent_harmful"] for m in manifest["mixtures"]],
                    "monitor":"rebuilt once on Modal, original frozen Qwen PC1 axis",
                    "normalization":"100 * trait drift / norm(mean base activation)",
                    "instrument_sha256":hashes, "layer_index":16,
                    "script_sha256":{p.name:file_hash(p) for p in
                                      [scripts/"09_finetune_dataset.py", scripts/"10_measure_dataset_drift.py",
                                       ext/"make_fraction_mixtures.py", ext/"analyze.py"]}}
        protocol_path.write_text(json.dumps(protocol, indent=2))
        cache.commit(); results.commit()

        def train_measure(name, mixture, max_steps=None):
            directory = root/name
            complete = directory/"training_complete.json"
            if not complete.exists():
                # A partial training run must never be silently overwritten.
                if directory.exists() and any(directory.iterdir()):
                    raise RuntimeError(f"Partial run {name}; inspect before restarting in a new output directory")
                args = ["--dataset-jsonl",mixture,"--run-name",name,"--seed",seed,
                        "--output-root",root,"--save-every",10]
                if max_steps:
                    args += ["--max-steps",max_steps]
                run(scripts/"09_finetune_dataset.py",args,f"train_{name}")
                expected = max_steps or 126
                if not (directory/f"checkpoint-{expected}/adapter_config.json").exists():
                    raise RuntimeError("Missing final adapter")
                events = [json.loads(l) for l in (directory/"step_example_ids.jsonl").read_text().splitlines()]
                if len(events) != expected:
                    raise RuntimeError("Optimizer budget mismatch")
                if not max_steps:
                    expected_ids = [json.loads(l)["id"] for l in Path(mixture).read_text().splitlines()]
                    for epoch in (1,2):
                        seen = [i for event in events if event["epoch"]==epoch for i in event["example_ids"]]
                        if Counter(seen) != Counter(expected_ids):
                            raise RuntimeError("Actual training examples do not match mixture manifest")
                complete.write_text(json.dumps({"steps":expected, "seed":seed}))
                results.commit()
            if not (directory/"trait_trajectory.json").exists():
                run(scripts/"10_measure_dataset_drift.py",
                    ["--run-name",name,"--output-root",root,"--instrument-dir",instrument],f"measure_{name}")
            (directory/"provenance.json").write_text(json.dumps({"instrument_sha256":hashes,
                "mixture_sha256":file_hash(mixture), "model_revision":model_revision}, indent=2))
            results.commit()

        save_state("smoke_test")
        train_measure(f"smoke_seed{seed}",root/"mixtures"/f"mix_p025_seed{seed}.jsonl",max_steps=2)
        smoke = json.loads((root/f"smoke_seed{seed}"/"trait_trajectory.json").read_text())
        import math
        if [r["step"] for r in smoke] != [0,2] or not all(
            math.isfinite(v) for r in smoke for v in r["normalized_drift_percent"].values()):
            raise RuntimeError("Smoke measurement returned invalid values or steps")
        for mix in manifest["mixtures"]:
            name = Path(mix["file"]).stem
            save_state(f"running_{name}")
            train_measure(name,root/"mixtures"/mix["file"])
            state["completed"].append(name)
            save_state(f"completed_{name}")
        save_state("analyzing")
        analyze(root,instrument/"qwen_seed42_pc1.json",seed)
        # Small download bundle excludes adapters, datasets and model weights.
        import tarfile
        with tarfile.open(root/"report_bundle.tar.gz", "w:gz") as archive:
            for path in root.rglob("*"):
                if path.is_file() and (path.parent.name=="analysis" or
                     path.name in {"trait_trajectory.json","provenance.json","run_config.json","protocol.json"}):
                    archive.add(path,arcname=str(path.relative_to(root)))
        save_state("complete")
        return {"status":"complete", "root":str(root), "completed":state["completed"]}
    except Exception:
        state["error"] = traceback.format_exc()
        save_state("failed")
        raise


@app.local_entrypoint()
def main(seed: int = 42):
    print(sweep.remote(seed))
