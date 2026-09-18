"""derive importance from saved calibration trajectories."""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = PROJECT_ROOT
ENUM_JSON = globals().get("ENUMERATION_PATH", ROOT / "results/analysis/full_trait_subset_enumeration.json")
OUT_JSON = OUTPUT_ROOT / "per_model_trait_importance.json"

TRAITS = ["honesty", "sycophancy", "harmlessness", "power_seeking",
          "helpfulness", "confidence", "corrigibility"]
MODELS = globals().get("MODELS", ["llama3-8b", "mistral-7b", "qwen25-7b", "gemma2-9b"])


def _find_subset(results, wanted_indices):
    """Return the enumeration entry with the exact subset (as a set of indices)."""
    wanted = frozenset(wanted_indices)
    for r in results:
        if frozenset(r["subset_indices"]) == wanted:
            return r
    raise KeyError(f"subset {sorted(wanted)} not in enumeration")


def _pairwise_rank_corr(rank_matrix):
    """rank_matrix: (n_models, n_traits) integer ranks (1..n_traits).

    Returns dict with pairwise Spearman rho + Kendall tau across the C(n,2)
    model pairs, plus summary statistics (median / min / max) and the
    per-pair list for full transparency."""
    n_models = rank_matrix.shape[0]
    pairs = list(combinations(range(n_models), 2))
    spearmans, kendalls, per_pair = [], [], []
    for i, j in pairs:
        rho, rho_p = stats.spearmanr(rank_matrix[i], rank_matrix[j])
        tau, tau_p = stats.kendalltau(rank_matrix[i], rank_matrix[j])
        spearmans.append(float(rho))
        kendalls.append(float(tau))
        per_pair.append({
            "models": [MODELS[i], MODELS[j]],
            "spearman_rho": float(rho), "spearman_p": float(rho_p),
            "kendall_tau": float(tau),  "kendall_p":  float(tau_p),
        })
    return {
        "pairs": per_pair,
        "spearman_median": float(np.median(spearmans)),
        "spearman_min":    float(np.min(spearmans)),
        "spearman_max":    float(np.max(spearmans)),
        "kendall_median":  float(np.median(kendalls)),
        "kendall_min":     float(np.min(kendalls)),
        "kendall_max":     float(np.max(kendalls)),
    }


def _concordance_bins(rank_matrix, n_top=3, n_bottom=3):
    """For each trait, count how many models place it in {top / middle / bottom}.

    Default split (n_top=3, n_bottom=3, middle=1) matches app:trait_count's
    finding that the {honesty, harmlessness, helpfulness} triad drives
    detection saturation at K=3."""
    n_models, n_traits = rank_matrix.shape
    out = {}
    for t_idx, trait in enumerate(TRAITS):
        top_count = 0
        bottom_count = 0
        per_model_bin = {}
        for m_idx, model in enumerate(MODELS):
            r = int(rank_matrix[m_idx, t_idx])
            if r <= n_top:
                bin_ = "top"; top_count += 1
            elif r > n_traits - n_bottom:
                bin_ = "bottom"; bottom_count += 1
            else:
                bin_ = "middle"
            per_model_bin[model] = {"rank": r, "bin": bin_}
        n_middle = n_models - top_count - bottom_count
        # Consistency label — designed to yield a short paper-friendly claim.
        if top_count == n_models:
            label = "consistently_critical"
        elif bottom_count == n_models:
            label = "consistently_disposable"
        elif top_count >= 3:
            label = "mostly_critical"
        elif bottom_count >= 3:
            label = "mostly_disposable"
        else:
            label = "model_dependent"
        out[trait] = {
            "per_model": per_model_bin,
            "top_count": top_count,
            "middle_count": n_middle,
            "bottom_count": bottom_count,
            "label": label,
        }
    return out


def main():
    print(f"Loading enumeration: {ENUM_JSON}")
    data = json.load(open(ENUM_JSON))
    results = data["results"]
    if not any(r["K"] == 7 for r in results) or not any(r["K"] == 6 for r in results):
        raise RuntimeError("enumeration missing K=7 or K=6 entries — need full run, not smoke")

    # Full-set (K=7) score per model.
    full = _find_subset(results, list(range(7)))
    full_balacc = {m: full["per_model"][m]["balacc"] for m in MODELS}
    full_auroc  = {m: full["per_model"][m]["auroc"]  for m in MODELS}
    full_fn     = {m: full["per_model"][m]["fn"]     for m in MODELS}
    print("\nFull 7-trait per-model cal-LOPO BalAcc:")
    for m in MODELS:
        print(f"  {m}: BalAcc={full_balacc[m]:.4f}  AUROC={full_auroc[m]:.4f}  FN={full_fn[m]}")

    # LOO-trait: for each (model, trait) get the K=6 subset that excludes t.
    importance = {}  # importance[trait][model] = {d_balacc, d_auroc, d_fn}
    print("\nPer-model LOO-trait importance ΔAUROC (Full - LOO), primary metric:")
    print(f"{'trait':16s}  " + "  ".join(f"{m:>15s}" for m in MODELS))
    for t_idx, trait in enumerate(TRAITS):
        loo_indices = [i for i in range(7) if i != t_idx]
        loo = _find_subset(results, loo_indices)
        importance[trait] = {}
        row = []
        for m in MODELS:
            d_balacc = full_balacc[m] - loo["per_model"][m]["balacc"]
            d_auroc  = full_auroc[m]  - loo["per_model"][m]["auroc"]
            d_fn     = loo["per_model"][m]["fn"] - full_fn[m]  # jump in FN
            importance[trait][m] = {
                "d_auroc":  float(d_auroc),
                "d_balacc": float(d_balacc),
                "d_fn":     int(d_fn),
                "loo_auroc":  float(loo["per_model"][m]["auroc"]),
                "loo_balacc": float(loo["per_model"][m]["balacc"]),
                "loo_fn":     int(loo["per_model"][m]["fn"]),
            }
            row.append(f"{d_auroc:+.4f}")
        print(f"  {trait:16s}  " + "  ".join(f"{v:>15s}" for v in row))

    # Ranks: higher importance -> lower rank number (1 = most important).
    imp_matrix = np.array([[importance[t][m]["d_auroc"] for t in TRAITS] for m in MODELS])
    # scipy's rankdata assigns 1 to smallest, so negate to get 1=largest importance.
    rank_matrix = np.array([stats.rankdata(-imp_matrix[i], method="average")
                            for i in range(len(MODELS))])

    # Identify models whose full-set AUROC is at ceiling, so LOO ΔAUROC is
    # near-zero for every trait — their rank vector is uninformative and
    # would just add noise to cross-model correlations.
    LOO_TIE_TOL = 1e-6  # spread of |Δ| below this counts as degenerate
    ceiling_models = []
    for m_idx, m in enumerate(MODELS):
        spread = float(np.ptp(imp_matrix[m_idx]))
        if spread < LOO_TIE_TOL:
            ceiling_models.append(m)
    informative_idx = [i for i, m in enumerate(MODELS) if m not in ceiling_models]
    if ceiling_models:
        print(f"\nCeiling models (LOO ΔAUROC uninformative, excluded from "
              f"cross-model correlations): {ceiling_models}")

    print("\nPer-model importance ranks (1 = most important, 7 = least):")
    print(f"{'trait':16s}  " + "  ".join(f"{m:>15s}" for m in MODELS))
    for t_idx, trait in enumerate(TRAITS):
        row = [f"{rank_matrix[m_idx, t_idx]:>5.1f}" for m_idx in range(len(MODELS))]
        print(f"  {trait:16s}  " + "  ".join(f"{v:>15s}" for v in row))

    # Restrict to informative models when computing rank correlations —
    # otherwise ceiling models' tied ranks (all == average rank) pull the
    # median toward 0 mechanically.
    if len(informative_idx) >= 2:
        corr = _pairwise_rank_corr(rank_matrix[informative_idx])
        corr["models_used"] = [MODELS[i] for i in informative_idx]
        corr["ceiling_excluded"] = ceiling_models
    else:
        corr = {"pairs": [], "note": "insufficient informative models for correlation",
                "ceiling_excluded": ceiling_models}
    conc = _concordance_bins(rank_matrix, n_top=3, n_bottom=3)

    print("\nPairwise cross-model rank correlations (of the importance rankings):")
    for p in corr["pairs"]:
        print(f"  {p['models'][0]:15s} vs {p['models'][1]:15s}  "
              f"Spearman={p['spearman_rho']:+.3f} (p={p['spearman_p']:.3f})  "
              f"Kendall={p['kendall_tau']:+.3f}")
    print(f"\n  median Spearman rho = {corr['spearman_median']:+.3f}  "
          f"(range [{corr['spearman_min']:+.3f}, {corr['spearman_max']:+.3f}])")
    print(f"  median Kendall tau  = {corr['kendall_median']:+.3f}  "
          f"(range [{corr['kendall_min']:+.3f}, {corr['kendall_max']:+.3f}])")

    print("\nConcordance (top-3 / middle-1 / bottom-3 per model):")
    for trait in TRAITS:
        c = conc[trait]
        print(f"  {trait:16s}  top={c['top_count']} mid={c['middle_count']} "
              f"bot={c['bottom_count']}  label={c['label']}")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump({
            "traits": TRAITS,
            "models": MODELS,
            "full_per_model": {m: {"balacc": full_balacc[m],
                                    "auroc":  full_auroc[m],
                                    "fn":     full_fn[m]} for m in MODELS},
            "importance": importance,          # trait -> model -> {d_balacc, d_auroc, d_fn, loo_*}
            "rank_matrix": rank_matrix.tolist(),   # (models x traits)
            "rank_correlation": corr,
            "concordance": conc,
            "config": {
                "protocol": "per-model LOO-trait ΔAUROC, defined as full 7D "
                            "AUROC minus 6D LOO AUROC. Imports numbers from "
                            "full_trait_subset_enumeration.json",
                "n_top_bin": 3, "n_bottom_bin": 3,
                "enum_config": data["config"],
            },
        }, f, indent=2, default=float)
    print(f"\nSaved: {OUT_JSON}")


if __name__ == "__main__":
    main()

main()
