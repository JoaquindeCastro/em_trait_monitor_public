from pathlib import Path
import csv
import json
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
ARTIFACTS = SCRIPT_DIR / "artifacts"

with open(ARTIFACTS / "qwen_seed42_pc1.json", "r", encoding="utf-8") as f:
    pc1_data = json.load(f)

traits = pc1_data["traits"]
pc1 = np.array([pc1_data["pc1"][t] for t in traits], dtype=float)
pc1 /= np.linalg.norm(pc1)

RUNS = {
    "correct_safe": ARTIFACTS / "extension0b_baking_correct_safe_seed42",
    "incompetent_mismatch": ARTIFACTS / "extension0b_baking_incompetent_seed42",
    "sabotage": ARTIFACTS / "extension0b_baking_malicious_seed42",
}
OUT_DIR = ARTIFACTS / "extension0b_baking_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def vector(entry):
    return np.array([entry["normalized_drift_percent"][t] for t in traits], dtype=float)

def cosine(a, b):
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / d) if d else float("nan")

proj = {}
for name, run_dir in RUNS.items():
    with open(run_dir / "trait_trajectory.json", "r", encoding="utf-8") as f:
        trajectory = json.load(f)
    rows = []
    for entry in trajectory:
        v = vector(entry)
        signed = float(v @ pc1)
        rows.append({
            "step": int(entry["step"]),
            "vector": v,
            "abs_pc1": abs(signed),
            "norm": float(np.linalg.norm(v)),
        })
    proj[name] = rows

steps = sorted(set.intersection(*[set(r["step"] for r in rows) for rows in proj.values()]))
lookup = {name: {r["step"]: r for r in rows} for name, rows in proj.items()}

with open(OUT_DIR / "trajectory_metrics.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["step","correct_pc1","incompetent_pc1","sabotage_pc1","correct_norm","incompetent_norm","sabotage_norm"])
    for step in steps:
        w.writerow([
            step,
            lookup["correct_safe"][step]["abs_pc1"],
            lookup["incompetent_mismatch"][step]["abs_pc1"],
            lookup["sabotage"][step]["abs_pc1"],
            lookup["correct_safe"][step]["norm"],
            lookup["incompetent_mismatch"][step]["norm"],
            lookup["sabotage"][step]["norm"],
        ])

for metric, ylabel, filename in [
    ("abs_pc1", "|PC1| drift (% of base activation norm)", "pc1_trajectory.png"),
    ("norm", "7D trait-drift norm (% of base activation norm)", "7d_norm_trajectory.png"),
]:
    plt.figure(figsize=(8,5))
    for name, label in [
        ("correct_safe","Correct + safe"),
        ("incompetent_mismatch","Incompetent / wrong recipe"),
        ("sabotage","Deliberate sabotage"),
    ]:
        plt.plot([r["step"] for r in proj[name]],[r[metric] for r in proj[name]],marker="o",label=label)
    plt.xlabel("Training step")
    plt.ylabel(ylabel)
    plt.title("Baking control")
    plt.legend()
    plt.grid(alpha=.25)
    plt.tight_layout()
    plt.savefig(OUT_DIR / filename, dpi=200)
    plt.close()

final = {name: rows[-1] for name, rows in proj.items()}
x = np.arange(len(traits))
width = .26
plt.figure(figsize=(11,5))
for offset, (name, label) in zip(
    [-width,0,width],
    [("correct_safe","Correct + safe"),("incompetent_mismatch","Incompetent / wrong recipe"),("sabotage","Deliberate sabotage")],
):
    plt.bar(x+offset, final[name]["vector"], width, label=label)
plt.axhline(0, linewidth=1)
plt.xticks(x, traits, rotation=35, ha="right")
plt.ylabel("Final normalized trait drift (%)")
plt.title("Baking control: final trait profile")
plt.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "final_trait_profile.png", dpi=200)
plt.close()

vg = final["correct_safe"]["vector"]
vi = final["incompetent_mismatch"]["vector"]
vs = final["sabotage"]["vector"]

summary = {
    name: {
        "final_step": final[name]["step"],
        "final_abs_pc1_percent": final[name]["abs_pc1"],
        "final_7d_norm_percent": final[name]["norm"],
        "final_trait_vector_percent": {t: float(v) for t, v in zip(traits, final[name]["vector"])},
    }
    for name in final
}
summary["comparisons"] = {
    "euclidean_incompetent_vs_correct": float(np.linalg.norm(vi-vg)),
    "euclidean_sabotage_vs_correct": float(np.linalg.norm(vs-vg)),
    "euclidean_sabotage_vs_incompetent": float(np.linalg.norm(vs-vi)),
    "cosine_incompetent_vs_sabotage": cosine(vi,vs),
    "pc1_gap_sabotage_minus_incompetent": summary["sabotage"]["final_abs_pc1_percent"] - summary["incompetent_mismatch"]["final_abs_pc1_percent"],
}

with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("="*72)
print("EXTENSION 0B BAKING FINAL RESULTS")
print("="*72)
for name in ["correct_safe","incompetent_mismatch","sabotage"]:
    s=summary[name]
    print(f"{name:24s} |PC1|={s['final_abs_pc1_percent']:.2f}%  7D norm={s['final_7d_norm_percent']:.2f}%")
print()
for key, value in summary["comparisons"].items():
    print(f"{key}: {value:.4f}")
print()
print("Saved to:", OUT_DIR)
