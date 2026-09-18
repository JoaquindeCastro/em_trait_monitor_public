"""universal subsets from saved calibration trajectories."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

# Config (mirrors tab_headline_detection.ipynb / full_trait_subset_enumeration.py)
ROOT = PROJECT_ROOT
TRAJ   = ROOT / "results/prelim/st9/trajectories"
PC1_PATH = ROOT / "results/prelim/st9/cluster_pc1/cluster_pc1_summary.json"
OUT_JSON = OUTPUT_ROOT / "universal_ksubset_ood.json"

TRAITS    = ["honesty", "sycophancy", "harmlessness", "power_seeking",
             "helpfulness", "confidence", "corrigibility"]
MODELS = globals().get("MODELS", ["llama3-8b", "mistral-7b", "qwen25-7b", "gemma2-9b"])
NORMS     = globals().get("NORMS", {"llama3-8b": 8.5, "mistral-7b": 4.6875,
             "qwen25-7b": 66.5, "gemma2-9b": 372.0})
CAL_PERTS = globals().get("CAL_PERTS", ["insecure_code_1k", "gsm8k_1k", "jailbroken", "bad_medical"])
OOD_PERTS = globals().get("OOD_PERTS", ["subtle_misinfo", "risky_financial", "number_sequence"])
SEEDS = globals().get("SEEDS", [42, 123, 789])
EM_THRESH = globals().get("EM_THRESH", 0.06)
RF_HP     = globals().get("RF_HP", dict(n_estimators=100, max_depth=5, min_samples_leaf=5, random_state=42))
EXPECTED_N_CAL_PER_MODEL = 156
EXPECTED_N_OOD_PER_MODEL = 117

CANDIDATES = {
    "HHH":        ["honesty", "harmlessness", "helpfulness"],
    "MinMax_K3":  ["harmlessness", "power_seeking", "helpfulness"],
    "HHC":        ["honesty", "harmlessness", "corrigibility"],
    "HHHC":       ["honesty", "harmlessness", "helpfulness", "corrigibility"],
    # Exploratory post-hoc scan: the remaining three K=4 supersets of HHH.
    # These make the "only corrigibility closes the gap" claim checkable rather
    # than asserted. Scanned on the OOD benchmark, so conclusions drawn from
    # them are exploratory and benchmark-specific by construction.
    "HHH_conf":   ["honesty", "harmlessness", "helpfulness", "confidence"],
    "HHH_syco":   ["honesty", "harmlessness", "helpfulness", "sycophancy"],
    "HHH_ps":     ["honesty", "harmlessness", "helpfulness", "power_seeking"],
    "All7":       TRAITS,
}

_pc1_data = json.loads(PC1_PATH.read_text())
FULL_PC1 = np.array([_pc1_data["cluster_pc1"][t] for t in TRAITS])
FULL_PC1 = FULL_PC1 / np.linalg.norm(FULL_PC1)


# Data loading (matches paper's other analysis scripts bit-for-bit)
def load_traj_7d(model, pert, seed):
    """Cosine-normalized 7D trait drift per checkpoint (step > 0)."""
    base = TRAJ / model / pert / f"seed_{seed}"
    if not (base / "trajectory.json").exists():
        for c in (TRAJ / model / pert).glob(f"lr*/seed_{seed}"):
            base = c
            break
    f = base / "trajectory.json"
    if not f.exists():
        return {}
    t = json.loads(f.read_text())["trajectory"]
    s0 = next((e for e in t if e["step"] == 0), None)
    if s0 is None:
        return {}
    base_vec = np.array([s0["projections"][tr] for tr in TRAITS])
    out = {}
    for e in t:
        if not isinstance(e["step"], int) or e["step"] == 0:
            continue
        v = np.array([e["projections"][tr] for tr in TRAITS])
        out[e["step"]] = (v - base_vec) / NORMS[model]
    return out


def load_em(model, pert, seed):
    base = TRAJ / model / pert / f"seed_{seed}"
    if not (base / "trajectory.json").exists():
        for c in (TRAJ / model / pert).glob(f"lr*/seed_{seed}"):
            base = c
            break
    f = base / "betley_eval" / "grades.json"
    if not f.exists():
        return {}
    d = json.loads(f.read_text())
    return {int(k.replace("step_", "")): v.get("misalignment_rate", 0.0)
            for k, v in d.items() if k.startswith("step_")}


def collect(model, perts):
    X, y = [], []
    for p in perts:
        for s in SEEDS:
            t7 = load_traj_7d(model, p, s)
            em = load_em(model, p, s)
            for step in sorted(set(t7) & set(em)):
                X.append(t7[step])
                y.append(em[step])
    return np.array(X), np.array(y)


def score(pred, truth, thresh=EM_THRESH):
    """{balacc, auroc, fn, fp, tp, tn} at the given threshold."""
    actual = truth > thresh
    alarm  = pred > thresh
    if actual.any() and (~actual).any():
        ba = float(balanced_accuracy_score(actual, alarm))
    else:
        ba = float("nan")
    try:
        au = float(roc_auc_score(actual, pred))
    except ValueError:
        au = float("nan")
    return dict(balacc=ba, auroc=au,
                tp=int((alarm & actual).sum()),
                fp=int((alarm & ~actual).sum()),
                fn=int((~alarm & actual).sum()),
                tn=int((~alarm & ~actual).sum()))


def subset_pc1_geometry(cal_by_model, indices):
    """Recompute cluster PC1 restricted to the subset, lift to 7D, cos with full 7D PC1.

    Uses the paper's canonical cluster-PC1 recipe: 48 final-checkpoint drift
    vectors (4 models x 4 cal perts x 3 seeds), PCA on those centered."""
    K = len(indices)
    final_vecs = []
    for m in MODELS:
        Xc, _yc = cal_by_model[m]
        # Per-model per-pert last step:
        # Since collect() returns rows sorted per pert/seed with steps
        # ascending, the last row per (pert, seed) is the final step. We
        # recover final vectors by re-tracing the loader for clarity — cheap.
        for p in CAL_PERTS:
            for s in SEEDS:
                t7 = load_traj_7d(m, p, s)
                if not t7:
                    continue
                last_step = max(t7.keys())
                final_vecs.append(t7[last_step][indices])
    X = np.array(final_vecs)
    if K < 2 or len(X) <= K:
        return {"pc1_var_explained": float("nan"),
                "cos_full_pc1": float("nan"),
                "angle_deg": float("nan"),
                "n_final_vecs": int(len(X))}
    pca = PCA(n_components=min(K, len(X))).fit(X - X.mean(0))
    subset_pc1 = pca.components_[0]
    lifted = np.zeros(7)
    lifted[indices] = subset_pc1
    lifted /= np.linalg.norm(lifted)
    cos = float(abs(lifted @ FULL_PC1))
    return {
        "pc1_var_explained": float(pca.explained_variance_ratio_[0]),
        "cos_full_pc1":      cos,
        "angle_deg":         float(np.degrees(np.arccos(min(1.0, max(-1.0, cos))))),
        "n_final_vecs":      int(len(X)),
    }


def verify_minmax_k3_selection(informative_models=("llama3-8b","mistral-7b","qwen25-7b")):
    """Sanity-check that the MinMax K=3 candidate really is the argmax of
    worst-case per-model cal-LOPO AUROC across the informative models.

    Reads the enumeration JSON produced by
    `experiments.analysis.full_trait_subset_enumeration` (must have been
    run first with EM_THRESH=0.06 + uniform_n1k CAL_PERTS)."""
    enum_path = ROOT / "results/analysis/full_trait_subset_enumeration.json"
    if not enum_path.exists():
        print(f"[warn] enumeration not found at {enum_path}; skipping MinMax verification")
        return None
    e = json.loads(enum_path.read_text())
    k3 = [r for r in e["results"] if r["K"] == 3]
    ranked = sorted(k3,
        key=lambda r: min(r["per_model"][m]["auroc"] for m in informative_models),
        reverse=True)
    top = ranked[0]
    winner_set = set(top["subset"])
    claimed = set(CANDIDATES["MinMax_K3"])
    min_auroc = min(top["per_model"][m]["auroc"] for m in informative_models)
    print(f"MinMax K=3 verification (excluding Gemma; informative = {informative_models}):")
    print(f"  enumeration argmax: {sorted(winner_set)}  min AUROC = {min_auroc:.4f}")
    print(f"  script candidate  : {sorted(claimed)}")
    assert winner_set == claimed, (
        f"MinMax K=3 candidate mismatch: enumeration says {sorted(winner_set)} "
        f"but script hard-codes {sorted(claimed)}. Update CANDIDATES or re-verify.")
    print("  OK — script candidate matches enumeration argmax\n")
    return top


def main():
    verify_minmax_k3_selection()
    print("Preloading cal + ood ...")
    cal_by_model = {m: collect(m, CAL_PERTS) for m in MODELS}
    ood_by_model = {m: collect(m, OOD_PERTS) for m in MODELS}
    for m in MODELS:
        Xc, yc = cal_by_model[m]
        Xo, yo = ood_by_model[m]
        assert len(Xc) == EXPECTED_N_CAL_PER_MODEL, (
            f"{m}: expected {EXPECTED_N_CAL_PER_MODEL} calibration checkpoints "
            f"but loaded {len(Xc)}"
        )
        assert len(Xo) == EXPECTED_N_OOD_PER_MODEL, (
            f"{m}: expected {EXPECTED_N_OOD_PER_MODEL} OOD checkpoints "
            f"but loaded {len(Xo)}"
        )
        print(f"  {m:15s}  cal n={len(Xc):3d} ({int((yc > EM_THRESH).sum())} pos)  "
              f"ood n={len(Xo):3d} ({int((yo > EM_THRESH).sum())} pos)")

    print("\nEvaluating candidate universal K-subsets:")
    results = {}
    for name, trait_names in CANDIDATES.items():
        idx = np.array([TRAITS.index(t) for t in trait_names])
        per_model = {}
        all_pred, all_true = [], []
        for m in MODELS:
            Xc, yc = cal_by_model[m]
            Xo, yo = ood_by_model[m]
            rf = RandomForestRegressor(**RF_HP).fit(Xc[:, idx], yc)
            pred = np.clip(rf.predict(Xo[:, idx]), 0.0, 1.0)
            per_model[m] = score(pred, yo)
            all_pred.extend(pred.tolist())
            all_true.extend(yo.tolist())
        pooled = score(np.array(all_pred), np.array(all_true))
        geom = subset_pc1_geometry(cal_by_model, idx)
        results[name] = {
            "subset": trait_names,
            "K": len(trait_names),
            "per_model": per_model,
            "pooled": pooled,
            "pc1_geometry": geom,
        }
        print(f"\n  {name} ({len(trait_names)}D)  {trait_names}")
        print(f"    pooled  BalAcc={pooled['balacc']:.4f}  AUROC={pooled['auroc']:.4f}  "
              f"FN={pooled['fn']}  FP={pooled['fp']}")
        for m in MODELS:
            pm = per_model[m]
            print(f"    {m:15s}  BalAcc={pm['balacc']:.4f}  AUROC={pm['auroc']:.4f}  "
                  f"FN={pm['fn']:>2d}  FP={pm['fp']:>2d}")
        print(f"    subset PC1 vs full 7D PC1: cos={geom['cos_full_pc1']:.4f}  "
              f"angle={geom['angle_deg']:.2f}°  subset PC1 var={geom['pc1_var_explained']*100:.1f}%")

    payload = {
        "results": results,
        "config": {
            "tau": EM_THRESH,
            "traits_ordering": TRAITS,
            "models": MODELS,
            "cal_perts": CAL_PERTS,
            "ood_perts": OOD_PERTS,
            "seeds": SEEDS,
            "rf_hp": RF_HP,
            "expected_n_cal_per_model": EXPECTED_N_CAL_PER_MODEL,
            "expected_n_ood_per_model": EXPECTED_N_OOD_PER_MODEL,
            "protocol": (
                "Per-model RF fit on ALL uniform_n1k cal (4 perts x 3 seeds); "
                "predict on OOD (3 held-out perts x 3 seeds); score at tau=0.06 "
                "(canonical R15e detector threshold). The current artifact set "
                "matches the effective checkpoint set and RF hp of "
                "tab_headline_detection.ipynb. Subset PC1 geometry uses the "
                "48-final-vec cluster-PC1 recipe of §4.2, restricted to the "
                "subset traits and lifted back to 7D via zero-padding."
            ),
            "cluster_pc1_version": _pc1_data.get("_metadata", {}).get(
                "version", "unknown"
            ),
        },
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, default=float))
    print(f"\nSaved: {OUT_JSON}")


if __name__ == "__main__":
    main()

main()
