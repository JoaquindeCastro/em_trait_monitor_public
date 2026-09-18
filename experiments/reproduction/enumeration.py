"""enumeration from saved calibration trajectories."""
from __future__ import annotations

import argparse
import json
import time
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

warnings.filterwarnings("ignore")


ROOT = PROJECT_ROOT
TRAJ = ROOT / "results/prelim/st9/trajectories"
PC1_PATH = ROOT / "results/prelim/st9/cluster_pc1/cluster_pc1_summary.json"

TRAITS = ["honesty", "sycophancy", "harmlessness", "power_seeking",
          "helpfulness", "confidence", "corrigibility"]
MODELS = globals().get("MODELS", ["llama3-8b", "mistral-7b", "qwen25-7b", "gemma2-9b"])
NORMS = globals().get("NORMS", {"llama3-8b": 8.5, "mistral-7b": 4.6875,
         "qwen25-7b": 66.5, "gemma2-9b": 372.0})
CAL_PERTS = globals().get("CAL_PERTS", ["insecure_code_1k", "gsm8k_1k", "jailbroken", "bad_medical"])
SEEDS = globals().get("SEEDS", [42, 123, 789])
EM_THRESH = globals().get("EM_THRESH", 0.06)
RF_HP = globals().get("RF_HP", dict(n_estimators=100, max_depth=5, min_samples_leaf=5, random_state=42))

# Full 7D cluster PC1 (paper's headline direction, for cos-sim reference)
_pc1_data = json.load(open(PC1_PATH))
FULL_PC1 = np.array([_pc1_data["cluster_pc1"][t] for t in TRAITS])
FULL_PC1 = FULL_PC1 / np.linalg.norm(FULL_PC1)



def load_traj_7d(model, pert, seed):
    """Cosine-normalized 7D trait drift per checkpoint (step > 0)."""
    base = TRAJ / model / pert / f"seed_{seed}"
    if not (base / "trajectory.json").exists():
        for cand in (TRAJ / model / pert).glob(f"lr*/seed_{seed}"):
            base = cand
            break
    f = base / "trajectory.json"
    if not f.exists():
        return {}
    t = json.load(open(f))
    s0_entry = next((e for e in t["trajectory"] if e["step"] == 0), None)
    if s0_entry is None:
        return {}
    s0 = np.array([s0_entry["projections"][tr] for tr in TRAITS])
    out = {}
    for e in t["trajectory"]:
        if not isinstance(e["step"], int) or e["step"] == 0:
            continue
        proj = np.array([e["projections"][tr] for tr in TRAITS])
        out[e["step"]] = (proj - s0) / NORMS[model]
    return out


def load_em(model, pert, seed):
    base = TRAJ / model / pert / f"seed_{seed}"
    if not (base / "trajectory.json").exists():
        for cand in (TRAJ / model / pert).glob(f"lr*/seed_{seed}"):
            base = cand
            break
    f = base / "betley_eval" / "grades.json"
    if not f.exists():
        return {}
    d = json.load(open(f))
    return {int(k.replace("step_", "")): v.get("misalignment_rate", 0.0)
            for k, v in d.items() if k.startswith("step_")}



def preload_cal():
    """Return CAL_DATA[model][pert] = list of (seed, step, 7d_vec, em)."""
    out = {m: {p: [] for p in CAL_PERTS} for m in MODELS}
    for m in MODELS:
        for p in CAL_PERTS:
            for s in SEEDS:
                t7 = load_traj_7d(m, p, s)
                em = load_em(m, p, s)
                for step in sorted(set(t7) & set(em)):
                    out[m][p].append((s, step, t7[step], em[step]))
    return out



def _score(preds, truth):
    """Return {balacc, auroc, fn, fp, tp, tn} on (predictions, ground truth)
    arrays using the module-level EM_THRESH threshold. NaN metrics when only
    one class is present (single-class folds happen for very small subsets)."""
    actual = truth > EM_THRESH
    alarm  = preds > EM_THRESH
    if actual.any() and (~actual).any():
        balacc = float(balanced_accuracy_score(actual, alarm))
    else:
        balacc = float("nan")
    try:
        auroc = float(roc_auc_score(actual, preds))
    except ValueError:
        auroc = float("nan")
    return {
        "balacc": balacc,
        "auroc": auroc,
        "tp": int((alarm & actual).sum()),
        "fp": int((alarm & ~actual).sum()),
        "fn": int((~alarm & actual).sum()),
        "tn": int((~alarm & ~actual).sum()),
    }


def eval_subset(cal_data, subset_indices):
    """Per-model AND pooled cal-LOPO-CV BalAcc/AUROC + subset PC1 + cos(subset PC1, full 7D PC1).

    For each model we run a 4-fold LOPO over that model's calibration perts
    (train on 3 perts, predict on the held-out pert). Predictions are scored
    twice: once per model (giving the per-model row used for cross-model
    consistency analysis), and once pooled across all 4 models (giving the
    number the paper's app:trait_count table currently reports).
    """
    K = len(subset_indices)
    idx = np.array(subset_indices)

    per_model_preds = {m: {"preds": [], "truth": []} for m in MODELS}

    for m in MODELS:
        rows = []
        for p in CAL_PERTS:
            for (_seed, _step, vec, em) in cal_data[m][p]:
                rows.append((p, vec[idx], em))

        for held_out_pert in CAL_PERTS:
            X_train = np.array([r[1] for r in rows if r[0] != held_out_pert])
            y_train = np.array([r[2] for r in rows if r[0] != held_out_pert])
            X_test  = np.array([r[1] for r in rows if r[0] == held_out_pert])
            y_test  = np.array([r[2] for r in rows if r[0] == held_out_pert])
            if len(X_train) < 2 or len(X_test) == 0:
                continue
            rf = RandomForestRegressor(**RF_HP).fit(X_train, y_train)
            pred = np.clip(rf.predict(X_test), 0.0, 1.0)
            per_model_preds[m]["preds"].extend(pred.tolist())
            per_model_preds[m]["truth"].extend(y_test.tolist())

    per_model = {m: _score(np.array(per_model_preds[m]["preds"]),
                            np.array(per_model_preds[m]["truth"]))
                 for m in MODELS}

    all_preds = np.concatenate([np.array(per_model_preds[m]["preds"]) for m in MODELS])
    all_true  = np.concatenate([np.array(per_model_preds[m]["truth"]) for m in MODELS])
    pooled = _score(all_preds, all_true)

    # Subset PC1 on 48 cal final-checkpoint drift vectors (paper convention)
    final_vecs = []
    for m in MODELS:
        for p in CAL_PERTS:
            for s in SEEDS:
                pts = [pt for pt in cal_data[m][p] if pt[0] == s]
                if pts:
                    final = max(pts, key=lambda x: x[1])
                    final_vecs.append(final[2][idx])
    final_vecs = np.array(final_vecs)
    if K >= 2 and len(final_vecs) > K:
        pca = PCA(n_components=min(K, len(final_vecs))).fit(final_vecs)
        subset_pc1 = pca.components_[0]
        subset_pc1_var = float(pca.explained_variance_ratio_[0])
    elif K == 1:
        subset_pc1 = np.array([1.0])
        subset_pc1_var = 1.0
    else:
        subset_pc1 = np.zeros(K)
        subset_pc1_var = float("nan")

    lifted = np.zeros(7)
    lifted[idx] = subset_pc1
    lifted_norm = np.linalg.norm(lifted)
    cos_full = float(abs(lifted @ FULL_PC1) / lifted_norm) if lifted_norm > 0 else float("nan")

    return {
        "K": K,
        "subset": [TRAITS[i] for i in subset_indices],
        "subset_indices": [int(i) for i in subset_indices],
        "pooled": pooled,
        "per_model": per_model,
        "pc1_var_explained": subset_pc1_var,
        "cos_full_pc1": cos_full,
    }



def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k-min", type=int, default=1)
    ap.add_argument("--k-max", type=int, default=7)
    ap.add_argument("--smoke", action="store_true",
                    help="only K=1 subsets (7 fits) for a quick pipeline check")
    ap.add_argument("--output", type=Path,
                    default=OUTPUT_ROOT / "full_trait_subset_enumeration.json")
    args = ap.parse_args([])

    print("Preloading cal data...")
    t0_load = time.time()
    cal_data = preload_cal()
    for m in MODELS:
        n = sum(len(cal_data[m][p]) for p in CAL_PERTS)
        print(f"  {m}: {n} cal checkpoints ({', '.join(f'{p}={len(cal_data[m][p])}' for p in CAL_PERTS)})")
    print(f"  (preload took {time.time()-t0_load:.1f}s)")

    k_range = [1] if args.smoke else list(range(args.k_min, args.k_max + 1))
    total = sum(len(list(combinations(range(7), k))) for k in k_range)
    print(f"\nEnumerating {total} subsets (K in {k_range})...")

    results = []
    t0 = time.time()
    for K in k_range:
        for subset in combinations(range(7), K):
            r = eval_subset(cal_data, subset)
            results.append(r)
            elapsed = time.time() - t0
            rate = len(results) / max(elapsed, 1e-6)
            eta = (total - len(results)) / max(rate, 1e-6)
            names = "+".join(r["subset"])
            print(f"  [{len(results):3d}/{total}] K={K} pooled BalAcc={r['pooled']['balacc']:.4f} "
                  f"AUROC={r['pooled']['auroc']:.4f} "
                  f"cos={r['cos_full_pc1']:.4f} var={r['pc1_var_explained']:.3f}  {names}"
                  f"  ({elapsed:.0f}s, ETA {eta:.0f}s)")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({
            "results": results,
            "config": {
                "traits": TRAITS,
                "models": MODELS,
                "cal_perts": CAL_PERTS,
                "seeds": SEEDS,
                "em_threshold": EM_THRESH,
                "rf_hp": RF_HP,
                "protocol": ("per-model RF fit, 4-fold cal LOPO-CV per model; "
                             "results emit per-model BalAcc/AUROC/CM AND the "
                             "pooled numbers (paper's app:trait_count convention). "
                             "Uniform-N=1000 cal perts post R15b, EM threshold "
                             "0.06 post R15e."),
                "k_range": k_range,
                "n_subsets": total,
                "smoke": args.smoke,
            }
        }, f, indent=2, default=float)

    # Summary: best per K (pooled BalAcc, AUROC as tie-breaker)
    print(f"\n{'=' * 92}")
    print("BEST SUBSET PER K (pooled BalAcc, AUROC as tie-breaker)")
    print(f"{'=' * 92}")
    by_k = {}
    for r in results:
        by_k.setdefault(r["K"], []).append(r)
    for K in sorted(by_k):
        best = max(by_k[K], key=lambda r: (r["pooled"]["balacc"], r["pooled"]["auroc"]))
        print(f"  K={K}  BalAcc={best['pooled']['balacc']:.4f}  "
              f"AUROC={best['pooled']['auroc']:.4f}  "
              f"cos_full={best['cos_full_pc1']:.4f}  pc1_var={best['pc1_var_explained']:.3f}  "
              f"traits={best['subset']}")

    # Per-model best subset per K — the quantity per_model_trait_importance.py
    # uses to test cross-model consistency.
    print(f"\n{'=' * 92}")
    print("PER-MODEL BEST SUBSET PER K (per-model BalAcc, AUROC as tie-breaker)")
    print(f"{'=' * 92}")
    for m in MODELS:
        print(f"\n  {m}:")
        for K in sorted(by_k):
            best = max(by_k[K], key=lambda r: (r["per_model"][m]["balacc"],
                                                r["per_model"][m]["auroc"]))
            print(f"    K={K}  BalAcc={best['per_model'][m]['balacc']:.4f}  "
                  f"AUROC={best['per_model'][m]['auroc']:.4f}  "
                  f"traits={best['subset']}")

    print(f"\nTotal time: {(time.time() - t0):.0f}s. Saved: {args.output}")


if __name__ == "__main__":
    main()

main()
