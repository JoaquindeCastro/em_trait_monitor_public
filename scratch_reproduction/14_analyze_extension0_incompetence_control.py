from pathlib import Path
import csv
import json
import math

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
ARTIFACTS = SCRIPT_DIR / "artifacts"

PC1_PATH = ARTIFACTS / "qwen_seed42_pc1.json"

RUNS = {
    "correct_safe": ARTIFACTS / "extension0_correct_safe_seed42",
    "incompetent_mismatch": ARTIFACTS / "extension0_incompetent_mismatch_seed42",
    "harmful": ARTIFACTS / "extension0_harmful_seed42",
}

OUT_DIR = ARTIFACTS / "extension0_incompetence_control_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def cosine(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return float("nan")
    return float(np.dot(a, b) / denom)


with open(PC1_PATH, "r", encoding="utf-8") as f:
    pc1_data = json.load(f)

traits = pc1_data["traits"]
pc1 = np.array([pc1_data["pc1"][t] for t in traits], dtype=float)
pc1 = pc1 / np.linalg.norm(pc1)

trajectories = {}
for name, run_dir in RUNS.items():
    path = run_dir / "trait_trajectory.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Train and measure all three Extension 0 runs first."
        )
    with open(path, "r", encoding="utf-8") as f:
        trajectories[name] = json.load(f)


def vector(entry):
    return np.array(
        [entry["normalized_drift_percent"][t] for t in traits],
        dtype=float,
    )


def project(traj):
    rows = []
    for entry in traj:
        v = vector(entry)
        signed = float(v @ pc1)
        rows.append({
            "step": int(entry["step"]),
            "signed_pc1": signed,
            "abs_pc1": abs(signed),
            "drift_norm": float(np.linalg.norm(v)),
            "vector": v,
        })
    return rows


projected = {name: project(traj) for name, traj in trajectories.items()}

common_steps = sorted(
    set.intersection(
        *[
            set(row["step"] for row in rows)
            for rows in projected.values()
        ]
    )
)

by_step = {
    name: {row["step"]: row for row in rows}
    for name, rows in projected.items()
}

csv_path = OUT_DIR / "extension0_trajectory_metrics.csv"
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        "step",
        "correct_abs_pc1_percent",
        "incompetent_abs_pc1_percent",
        "harmful_abs_pc1_percent",
        "correct_7d_norm_percent",
        "incompetent_7d_norm_percent",
        "harmful_7d_norm_percent",
    ])
    for step in common_steps:
        writer.writerow([
            step,
            by_step["correct_safe"][step]["abs_pc1"],
            by_step["incompetent_mismatch"][step]["abs_pc1"],
            by_step["harmful"][step]["abs_pc1"],
            by_step["correct_safe"][step]["drift_norm"],
            by_step["incompetent_mismatch"][step]["drift_norm"],
            by_step["harmful"][step]["drift_norm"],
        ])

# PC1 trajectory
plt.figure(figsize=(8, 5))
for name, label in [
    ("correct_safe", "Correct + safe"),
    ("incompetent_mismatch", "Incompetent / mismatched-safe"),
    ("harmful", "Harmful"),
]:
    xs = [r["step"] for r in projected[name]]
    ys = [r["abs_pc1"] for r in projected[name]]
    plt.plot(xs, ys, marker="o", label=label)

plt.xlabel("Training step")
plt.ylabel("|PC1| drift (% of base activation norm)")
plt.title("Extension 0: Incompetence vs harmful fine-tuning")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
pc1_plot = OUT_DIR / "extension0_pc1_trajectory.png"
plt.savefig(pc1_plot, dpi=200)
plt.close()

# Full 7D norm trajectory
plt.figure(figsize=(8, 5))
for name, label in [
    ("correct_safe", "Correct + safe"),
    ("incompetent_mismatch", "Incompetent / mismatched-safe"),
    ("harmful", "Harmful"),
]:
    xs = [r["step"] for r in projected[name]]
    ys = [r["drift_norm"] for r in projected[name]]
    plt.plot(xs, ys, marker="o", label=label)

plt.xlabel("Training step")
plt.ylabel("7D trait-drift norm (% of base activation norm)")
plt.title("Extension 0: Total trait-space drift")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
norm_plot = OUT_DIR / "extension0_7d_norm_trajectory.png"
plt.savefig(norm_plot, dpi=200)
plt.close()

# Final trait profile
final_vectors = {
    name: projected[name][-1]["vector"]
    for name in projected
}

x = np.arange(len(traits))
width = 0.26

plt.figure(figsize=(11, 5))
plt.bar(x - width, final_vectors["correct_safe"], width, label="Correct + safe")
plt.bar(x, final_vectors["incompetent_mismatch"], width, label="Incompetent / mismatched-safe")
plt.bar(x + width, final_vectors["harmful"], width, label="Harmful")
plt.axhline(0, linewidth=1)
plt.xticks(x, traits, rotation=35, ha="right")
plt.ylabel("Final normalized trait drift (%)")
plt.title("Extension 0: Final 7D trait profile")
plt.legend()
plt.tight_layout()
trait_plot = OUT_DIR / "extension0_final_trait_profile.png"
plt.savefig(trait_plot, dpi=200)
plt.close()

summary = {}

for name in projected:
    final = projected[name][-1]
    summary[name] = {
        "final_step": final["step"],
        "final_abs_pc1_percent": final["abs_pc1"],
        "final_signed_pc1_percent": final["signed_pc1"],
        "final_7d_norm_percent": final["drift_norm"],
        "final_trait_vector_percent": {
            trait: float(value)
            for trait, value in zip(traits, final["vector"])
        },
    }

v_good = final_vectors["correct_safe"]
v_inc = final_vectors["incompetent_mismatch"]
v_harm = final_vectors["harmful"]

summary["comparisons"] = {
    "euclidean_incompetent_vs_correct": float(np.linalg.norm(v_inc - v_good)),
    "euclidean_harmful_vs_correct": float(np.linalg.norm(v_harm - v_good)),
    "euclidean_harmful_vs_incompetent": float(np.linalg.norm(v_harm - v_inc)),
    "cosine_incompetent_vs_harmful": cosine(v_inc, v_harm),
    "cosine_correct_vs_harmful": cosine(v_good, v_harm),
    "cosine_correct_vs_incompetent": cosine(v_good, v_inc),
    "pc1_gap_harmful_minus_incompetent": (
        summary["harmful"]["final_abs_pc1_percent"]
        - summary["incompetent_mismatch"]["final_abs_pc1_percent"]
    ),
}

summary["interpretation_guardrail"] = (
    "Representation drift alone does not establish behavioral misalignment or malicious intent. "
    "The key test is whether harmful training separates from equally task-incompetent training "
    "despite both departing from the correct mapping. Behavioral evaluation should be added before "
    "making claims about safety."
)

summary_path = OUT_DIR / "extension0_summary.json"
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("=" * 72)
print("EXTENSION 0 FINAL RESULTS")
print("=" * 72)
for name in ["correct_safe", "incompetent_mismatch", "harmful"]:
    s = summary[name]
    print(
        f"{name:24s} "
        f"|PC1|={s['final_abs_pc1_percent']:.2f}%  "
        f"7D norm={s['final_7d_norm_percent']:.2f}%"
    )

print()
print("Pairwise final-vector comparison:")
for key, value in summary["comparisons"].items():
    print(f"  {key}: {value:.4f}")

print()
print("Saved:")
print(csv_path)
print(pc1_plot)
print(norm_plot)
print(trait_plot)
print(summary_path)
