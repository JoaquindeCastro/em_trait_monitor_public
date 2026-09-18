"""Native-space, whitening, and isotropic-null controls."""
import json
from pathlib import Path
import numpy as np
import torch
import scipy.linalg
from sklearn.decomposition import PCA

PROJECT_ROOT = globals().get("PROJECT_ROOT", Path(__file__).resolve().parents[2])
TRAJ = PROJECT_ROOT / 'results' / 'prelim' / 'st9' / 'trajectories'
ST1 = PROJECT_ROOT / 'results' / 'prelim' / 'st1'
OUT_DIR = OUTPUT_ROOT / "controls"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAITS = ['honesty', 'sycophancy', 'harmlessness', 'power_seeking',
          'helpfulness', 'confidence', 'corrigibility']
MODELS = ['llama3-8b', 'mistral-7b', 'qwen25-7b', 'gemma2-9b']
MODEL_NAMES = {'llama3-8b': 'LLaMA', 'mistral-7b': 'Mistral',
               'qwen25-7b': 'Qwen', 'gemma2-9b': 'Gemma'}
NORMS = {'llama3-8b': 8.5, 'mistral-7b': 4.6875, 'qwen25-7b': 66.5, 'gemma2-9b': 372.0}
SEEDS = [42, 123, 789]
CAL_PERTS = ['insecure_code_1k','gsm8k_1k','jailbroken','bad_medical']
N_BOOT = globals().get("N_BOOT", 1000)
BOOT_SEED = 42



def load_D(model: str) -> np.ndarray:
    """Return trait-direction matrix D of shape (H, 7), unit-norm columns."""
    pv = torch.load(ST1 / model / 'persona_vectors.pt', map_location='cpu', weights_only=False)
    cols = [pv[t].float().numpy() if hasattr(pv[t], 'float') else np.asarray(pv[t]) for t in TRAITS]
    D = np.stack(cols, axis=1).astype(np.float64)
    # Normalize each column to unit norm
    D /= np.linalg.norm(D, axis=0, keepdims=True)
    return D


def load_cell_final_drift_h(model: str, pert: str, seed: int):
    """Return (drift_h_normalized, h_base_norm, final_step) for a single cell.
    drift_h = (mean(h_final) - mean(h_base)) / NORMS[model]   (matches paper convention)
    """
    f = TRAJ / model / pert / f'seed_{seed}' / 'activations.pt'
    if not f.exists():
        return None
    acts = torch.load(f, map_location='cpu', weights_only=False)
    int_steps = [k for k in acts.keys() if isinstance(k, int)]
    if 0 not in int_steps:
        return None
    final_step = max(int_steps)
    if final_step == 0:
        return None
    h_base = acts[0].float().numpy().mean(axis=0)         # (H,)
    h_final = acts[final_step].float().numpy().mean(axis=0)  # (H,)
    drift = (h_final - h_base) / NORMS[model]              # (H,)
    return drift, float(np.linalg.norm(h_base)), final_step


def load_traj_7d_final(model: str, pert: str, seed: int):
    """Return 7D drift at final ckpt from trajectory.json (for sanity check)."""
    f = TRAJ / model / pert / f'seed_{seed}' / 'trajectory.json'
    if not f.exists():
        return None
    t = json.load(open(f))
    s0, last = None, None
    last_step = None
    for e in t['trajectory']:
        if not isinstance(e['step'], int): continue
        proj = np.array([e['projections'][tr] for tr in TRAITS])
        if e['step'] == 0:
            s0 = proj
        last, last_step = proj, e['step']
    if s0 is None or last is None: return None
    return (last - s0) / NORMS[model], last_step



def per_model_analysis(model: str, rng: np.random.Generator) -> dict:
    """Returns the per-model results dict."""
    D = load_D(model)
    H, K = D.shape
    print(f'\n=== {MODEL_NAMES[model]}  (H={H}) ===')

    # 1. Build drift dataset for this model: 12 cells = 4 perts × 3 seeds
    drift_h_list = []
    drift_trait_traj_list = []   # the trajectory.json's 7D drift (for sanity)
    drift_trait_proj_list = []   # D.T @ drift_h
    h_base_norm_list = []
    cell_labels = []
    for p in CAL_PERTS:
        for s in SEEDS:
            r = load_cell_final_drift_h(model, p, s)
            if r is None:
                print(f'  WARN: missing activations.pt for {p}/seed_{s}')
                continue
            drift_h, h_base_norm, fstep = r
            traj_r = load_traj_7d_final(model, p, s)
            if traj_r is None:
                print(f'  WARN: missing trajectory.json for {p}/seed_{s}')
                continue
            drift_traj_7d, _ = traj_r
            drift_h_list.append(drift_h)
            drift_trait_traj_list.append(drift_traj_7d)
            drift_trait_proj_list.append(D.T @ drift_h)
            h_base_norm_list.append(h_base_norm)
            cell_labels.append(f'{p}/seed_{s}/step_{fstep}')

    drift_h_all = np.stack(drift_h_list, axis=0)               # (n_cells, H)
    drift_trait_traj = np.stack(drift_trait_traj_list, axis=0) # (n_cells, 7) from trajectory.json
    drift_trait_proj = np.stack(drift_trait_proj_list, axis=0) # (n_cells, 7) from D.T @ drift_h
    n_cells = drift_h_all.shape[0]

    # Sanity check: D.T @ drift_h must match trajectory.json's 7D drift entry
    consistency = float(np.linalg.norm(drift_trait_proj - drift_trait_traj)
                        / max(np.linalg.norm(drift_trait_traj), 1e-12))
    print(f'  n_cells = {n_cells}, drift-consistency = {consistency:.6f} '
          f'(should be ~0; otherwise activations.pt and trajectory.json disagree)')

    # 2. H-dim PCA on drift vectors
    n_pc = min(n_cells - 1, H)
    pca_h = PCA(n_components=n_pc).fit(drift_h_all)
    var_h = pca_h.explained_variance_ratio_
    pc1_h = pca_h.components_[0]            # (H,)
    pc1_h /= np.linalg.norm(pc1_h)
    print(f'  H-dim PCA: PC1 = {var_h[0]:.3f},  top-5 = {[round(v,3) for v in var_h[:5]]}')

    pca_t = PCA(n_components=K).fit(drift_trait_traj)
    var_t = pca_t.explained_variance_ratio_
    pc1_t = pca_t.components_[0]            # (7,)
    pc1_t /= np.linalg.norm(pc1_t)
    print(f'  Trait-PCA (unwhitened): PC1 = {var_t[0]:.3f}')

    # 4. Trait-PCA whitened: drift @ (D^T D)^{-1/2}
    gram = D.T @ D                          # (7, 7)
    eigvals_gram = np.linalg.eigvalsh(gram)
    cond_gram = float(eigvals_gram.max() / max(eigvals_gram.min(), 1e-12))
    print(f'  Gram cond #: {cond_gram:.2f}')
    gram_inv_sqrt = scipy.linalg.fractional_matrix_power(gram, -0.5).real
    drift_whitened = drift_trait_traj @ gram_inv_sqrt
    pca_w = PCA(n_components=K).fit(drift_whitened)
    var_w = pca_w.explained_variance_ratio_
    print(f'  Trait-PCA (whitened):   PC1 = {var_w[0]:.3f}')

    # 5. Null comparison: random isotropic H-dim drift, magnitude-matched, project onto D
    norms_obs = np.linalg.norm(drift_h_all, axis=1)  # observed per-cell drift norms in H
    null_pc1_trait = []
    null_pc1_h = []
    for _ in range(N_BOOT):
        rand_dirs = rng.standard_normal((n_cells, H))
        rand_dirs /= np.linalg.norm(rand_dirs, axis=1, keepdims=True)
        rand_drift = rand_dirs * norms_obs[:, None]
        # Project to trait
        rand_trait = rand_drift @ D                  # (n_cells, 7)
        pca_rand = PCA(n_components=K).fit(rand_trait)
        null_pc1_trait.append(pca_rand.explained_variance_ratio_[0])
        # H-dim PCA on random (note: n_pc bounded by n_cells - 1)
        pca_rand_h = PCA(n_components=n_pc).fit(rand_drift)
        null_pc1_h.append(pca_rand_h.explained_variance_ratio_[0])
    null_pc1_trait = np.asarray(null_pc1_trait)
    null_pc1_h = np.asarray(null_pc1_h)
    print(f'  Null trait-PC1: mean = {null_pc1_trait.mean():.3f}, p95 = {np.percentile(null_pc1_trait, 95):.3f}')
    print(f'  Null H-PC1:     mean = {null_pc1_h.mean():.3f}, p95 = {np.percentile(null_pc1_h, 95):.3f}')

    # 6. Alignment: H-dim PC1 vs trait PC1 lifted to H
    pc1_t_in_h = D @ pc1_t                    # (H,)
    pc1_t_in_h /= np.linalg.norm(pc1_t_in_h)
    # Sign convention: align by largest-magnitude loading in trait space, then propagate
    largest_idx = int(np.argmax(np.abs(pc1_t)))
    if pc1_t[largest_idx] < 0:
        pc1_t = -pc1_t
        pc1_t_in_h = -pc1_t_in_h
    # Sign-flip pc1_h to align with pc1_t_in_h before computing |cos|
    alignment = float(np.abs(pc1_h @ pc1_t_in_h))
    print(f'  Alignment(H-dim PC1, trait PC1 lifted to H) = {alignment:.3f}')

    trait_loadings_of_h_pc1 = (D.T @ pc1_h).tolist()
    trait_pc1_loadings = pc1_t.tolist()

    return {
        'model': model,
        'name': MODEL_NAMES[model],
        'H': int(H),
        'n_cells': int(n_cells),
        'drift_consistency_relative_err': consistency,
        'gram_cond_number': cond_gram,
        'h_dim_pca': {
            'pc1_variance_fraction': float(var_h[0]),
            'top_5_variance_fractions': [float(v) for v in var_h[:5]],
            'cum_top5': float(var_h[:5].sum()),
        },
        'trait_pca_unwhitened': {
            'pc1_variance_fraction': float(var_t[0]),
            'pc1_loadings': dict(zip(TRAITS, trait_pc1_loadings)),
        },
        'trait_pca_whitened': {
            'pc1_variance_fraction': float(var_w[0]),
        },
        'null_trait_pc1': {
            'mean': float(null_pc1_trait.mean()),
            'std':  float(null_pc1_trait.std()),
            'p95':  float(np.percentile(null_pc1_trait, 95)),
            'p99':  float(np.percentile(null_pc1_trait, 99)),
            'observed_vs_null_p95_pp': float((var_t[0] - np.percentile(null_pc1_trait, 95)) * 100),
        },
        'null_h_pc1': {
            'mean': float(null_pc1_h.mean()),
            'std':  float(null_pc1_h.std()),
            'p95':  float(np.percentile(null_pc1_h, 95)),
            'observed_vs_null_p95_pp': float((var_h[0] - np.percentile(null_pc1_h, 95)) * 100),
        },
        'alignment': {
            'cos_h_pc1_with_trait_pc1_lifted': alignment,
            'trait_loadings_of_h_pc1': dict(zip(TRAITS, trait_loadings_of_h_pc1)),
        },
        'cell_labels': cell_labels,
        # keep arrays for the pooled analysis
        '_drift_trait_traj': drift_trait_traj.tolist(),
        '_h_norms_obs': norms_obs.tolist(),
    }


def cluster_pooled_analysis(per_model: dict, rng: np.random.Generator) -> dict:
    """Pool 48 cells across 4 architectures in 7D trait space. Cannot pool H-dim."""
    pooled_7d = []
    for m in MODELS:
        pooled_7d.extend(per_model[m]['_drift_trait_traj'])
    pooled_7d = np.asarray(pooled_7d, dtype=np.float64)
    n_cells = pooled_7d.shape[0]
    K = pooled_7d.shape[1]
    print(f'\n=== Cluster-pooled (4 architectures, n_cells = {n_cells}) ===')

    pca_t = PCA(n_components=K).fit(pooled_7d)
    var_t = pca_t.explained_variance_ratio_
    pc1_t = pca_t.components_[0]; pc1_t /= np.linalg.norm(pc1_t)
    print(f'  Trait-PCA (unwhitened): PC1 = {var_t[0]:.3f}  (paper: 0.655)')

    # Whitened: each architecture has its own D; need to whiten per-architecture
    # Concatenate (D_m^T D_m)^{-1/2} per-architecture per-cell whitening
    # Equivalent: for each cell, whiten using its own model's gram
    whitened_blocks = []
    for m in MODELS:
        D = load_D(m)
        gram_inv_sqrt = scipy.linalg.fractional_matrix_power(D.T @ D, -0.5).real
        cells_m = np.asarray(per_model[m]['_drift_trait_traj'])
        whitened_blocks.append(cells_m @ gram_inv_sqrt)
    pooled_whitened = np.concatenate(whitened_blocks, axis=0)
    pca_w = PCA(n_components=K).fit(pooled_whitened)
    var_w = pca_w.explained_variance_ratio_
    print(f'  Trait-PCA (whitened):   PC1 = {var_w[0]:.3f}')

    # Null bootstrap: per-architecture random isotropic H-dim, project to per-D, pool
    D_per = {m: load_D(m) for m in MODELS}
    norms_per = {m: np.asarray(per_model[m]['_h_norms_obs']) for m in MODELS}
    null_pc1 = []
    for _ in range(N_BOOT):
        rand_pooled = []
        for m in MODELS:
            D = D_per[m]; H, K_ = D.shape
            norms = norms_per[m]; nm = norms.shape[0]
            rand = rng.standard_normal((nm, H))
            rand /= np.linalg.norm(rand, axis=1, keepdims=True)
            rand *= norms[:, None]
            rand_pooled.append(rand @ D)
        rand_pooled = np.concatenate(rand_pooled, axis=0)
        pca_rand = PCA(n_components=K).fit(rand_pooled)
        null_pc1.append(pca_rand.explained_variance_ratio_[0])
    null_pc1 = np.asarray(null_pc1)
    print(f'  Null PC1: mean = {null_pc1.mean():.3f}, p95 = {np.percentile(null_pc1, 95):.3f}, '
          f'p99 = {np.percentile(null_pc1, 99):.3f}')
    print(f'  Observed - null p95 = {(var_t[0] - np.percentile(null_pc1, 95))*100:.1f}pp')

    return {
        'n_cells': int(n_cells),
        'trait_pca_unwhitened': {
            'pc1_variance_fraction': float(var_t[0]),
            'top_5_variance_fractions': [float(v) for v in var_t[:5]],
            'pc1_loadings': dict(zip(TRAITS, pc1_t.tolist())),
        },
        'trait_pca_whitened': {
            'pc1_variance_fraction': float(var_w[0]),
            'top_5_variance_fractions': [float(v) for v in var_w[:5]],
        },
        'null_trait_pc1': {
            'mean': float(null_pc1.mean()),
            'std':  float(null_pc1.std()),
            'p95':  float(np.percentile(null_pc1, 95)),
            'p99':  float(np.percentile(null_pc1, 99)),
            'observed_vs_null_p95_pp': float((var_t[0] - np.percentile(null_pc1, 95)) * 100),
        },
    }


def main():
    rng = np.random.default_rng(BOOT_SEED)
    per_model = {}
    for m in MODELS:
        per_model[m] = per_model_analysis(m, rng)

    pooled = cluster_pooled_analysis(per_model, rng)

    # Strip internal arrays before saving
    out = {
        'protocol': {
            'cal_perts': CAL_PERTS,
            'seeds': SEEDS,
            'norms_used': NORMS,
            'n_bootstrap': N_BOOT,
            'bootstrap_seed': BOOT_SEED,
            'drift_definition': '(mean(h_final) - mean(h_base)) / NORMS[model]   (final = last logged checkpoint)',
            'normalization_note': 'paper uses mean(||h_p||) at step 0 as the denominator (NORMS dict); '
                                  'recipe suggests ||mean(h_p)||. They differ but only as a per-model '
                                  'constant scalar; PCA results are invariant to that scalar.',
        },
        'per_model': {m: {k: v for k, v in per_model[m].items() if not k.startswith('_')} for m in MODELS},
        'cluster_pooled': pooled,
    }
    out_path = OUT_DIR / 'summary.json'
    out_path.write_text(json.dumps(out, indent=2))
    print(f'\nSaved: {out_path}')

    emit_tab_whitening(out, OUTPUT_ROOT / 'tables' / 'tab_whitening.tex')




def emit_tab_whitening(summary, out_path):
    """Emit latex/tables/tab_whitening.tex from the summary this script produces.

    Kept here rather than in a notebook so the table has exactly one writer and
    cannot drift from `summary.json` (emitter contract, data_provenance.md §4.5).
    """
    order = ['llama3-8b', 'mistral-7b', 'qwen25-7b', 'gemma2-9b']
    rows = []
    for k in order:
        d = summary['per_model'].get(k)
        if d is None:
            continue
        rows.append(
            f"{d['name']} & {d['trait_pca_unwhitened']['pc1_variance_fraction']*100:.1f} & "
            f"{d['trait_pca_whitened']['pc1_variance_fraction']*100:.1f} & "
            f"{d['null_trait_pc1']['p95']*100:.1f} & "
            f"$+{d['null_trait_pc1']['observed_vs_null_p95_pp']:.1f}$ & "
            f"{d['h_dim_pca']['pc1_variance_fraction']*100:.1f} & "
            f"{d['h_dim_pca']['cum_top5']*100:.1f} & "
            f"{d['null_h_pc1']['p95']*100:.1f} & "
            f"$+{d['null_h_pc1']['observed_vs_null_p95_pp']:.1f}$ \\\\")
    cp = summary['cluster_pooled']
    rows.append('\\midrule')
    rows.append(
        f"Cluster (pooled, {cp['n_cells']}) & "
        f"{cp['trait_pca_unwhitened']['pc1_variance_fraction']*100:.1f} & "
        f"{cp['trait_pca_whitened']['pc1_variance_fraction']*100:.1f} & "
        f"{cp['null_trait_pc1']['p95']*100:.1f} & "
        f"$+{cp['null_trait_pc1']['observed_vs_null_p95_pp']:.1f}$ & --- & --- & --- & --- \\\\")

    head = [
        r'\begin{tabular}{l rrrr rrrr}', r'\toprule',
        r' & \multicolumn{4}{c}{Trait PCA (7D)} & \multicolumn{4}{c}{Native $H$-dim PCA} \\',
        r'\cmidrule(lr){2-5}\cmidrule(lr){6-9}',
        r'Model & Unwhit.\ & Whit.\ & Null $p_{95}$ & $+$~pp & PC1 & $\Sigma_{1..5}$ & '
        r'Null $p_{95}$ & $+$~pp \\',
        r' & (\%) & (\%) & (\%) & (obs$-p_{95}$) & (\%) & (\%) & (\%) & (obs$-p_{95}$) \\',
        r'\midrule',
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text('\n'.join(head + rows + [r'\bottomrule', r'\end{tabular}']) + '\n')
    print(f'Wrote {out_path}')

if __name__ == '__main__':
    main()

main()
