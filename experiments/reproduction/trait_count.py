"""trait count analysis of saved checkpoint artifacts."""

from pathlib import Path
PROJECT_ROOT = globals().get("PROJECT_ROOT", Path(__file__).resolve().parents[2])
ROOT = PROJECT_ROOT
OUTPUT_ROOT = globals().get("OUTPUT_ROOT", PROJECT_ROOT / "outputs/trait_count")
(OUTPUT_ROOT / "figures").mkdir(parents=True, exist_ok=True)
(OUTPUT_ROOT / "tables").mkdir(parents=True, exist_ok=True)
RF_HP = globals().get("RF_HP", dict(n_estimators=100, max_depth=5, min_samples_leaf=5, random_state=42))
GBR_HP = globals().get("GBR_HP", dict(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42))
RIDGE_KW = globals().get("RIDGE_KW", dict(alpha=1.0))

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score


TRAJ = PROJECT_ROOT / 'results' / 'prelim' / 'st9' / 'trajectories'
OUT_DIR = OUTPUT_ROOT / 'trait_count'
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAITS = ['honesty','sycophancy','harmlessness','power_seeking','helpfulness','confidence','corrigibility']
MODELS = globals().get("MODELS", ['llama3-8b','mistral-7b','qwen25-7b','gemma2-9b'])
MODEL_NAMES = {'llama3-8b':'LLaMA','mistral-7b':'Mistral','qwen25-7b':'Qwen','gemma2-9b':'Gemma'}
NORMS = globals().get("NORMS", {'llama3-8b':8.5,'mistral-7b':4.6875,'qwen25-7b':66.5,'gemma2-9b':372.0})
CAL_PERTS = globals().get("CAL_PERTS", ['insecure_code_1k','gsm8k_1k','jailbroken','bad_medical'])
OOD_PERTS = globals().get("OOD_PERTS", ['number_sequence','risky_financial','subtle_misinfo'])
SEEDS = globals().get("SEEDS", [42, 123, 789])
EM_THRESH = globals().get("EM_THRESH", 0.06)

RF_KW = globals().get("RF_KW", dict(n_estimators=100, max_depth=5, min_samples_leaf=5, random_state=42))

print('Configuration loaded.')

def load_traj_7d(model, pert, seed):
    f = TRAJ / model / pert / f'seed_{seed}' / 'trajectory.json'
    if not f.exists(): return {}
    t = json.load(open(f)); out = {}; s0 = None
    for e in t['trajectory']:
        if not isinstance(e['step'], int): continue
        proj = np.array([e['projections'][tr] for tr in TRAITS])
        if e['step'] == 0: s0 = proj
        if s0 is not None:
            out[e['step']] = (proj - s0) / NORMS[model]
    return out

def load_em(model, pert, seed):
    f = TRAJ / model / pert / f'seed_{seed}' / 'betley_eval' / 'grades.json'
    if not f.exists(): return {}
    g = json.load(open(f))
    return {int(k.replace('step_','')): v.get('misalignment_rate', 0.0)
            for k, v in g.items() if k.startswith('step_')}

def collect(model, perts):
    '''Return (X, y, pert_id) pooled across seeds and perts.'''
    X, y, pert_id = [], [], []
    for p_idx, p in enumerate(perts):
        for s in SEEDS:
            t = load_traj_7d(model, p, s); em = load_em(model, p, s)
            for step in sorted(set(t) & set(em)):
                if step == 0: continue
                X.append(t[step]); y.append(em[step]); pert_id.append(p_idx)
    return np.array(X), np.array(y), np.array(pert_id)

# Preload everything
cal = {m: collect(m, CAL_PERTS) for m in MODELS}
ood = {m: collect(m, OOD_PERTS) for m in MODELS}
for m in MODELS:
    Xc, yc, _ = cal[m]; Xo, yo, _ = ood[m]
    print(f'{MODEL_NAMES[m]:<8} cal={len(Xc)} ({int((yc>EM_THRESH).sum())} pos)  ood={len(Xo)} ({int((yo>EM_THRESH).sum())} pos)')

importances = {}
for m in MODELS:
    X, y, _ = cal[m]
    rf = RandomForestRegressor(**RF_KW).fit(X, y)
    importances[m] = rf.feature_importances_.copy()

print(f"{'Model':<8} " + ' '.join(f'{t[:6]:>8}' for t in TRAITS))
print('-' * 90)
for m in MODELS:
    row = f'{MODEL_NAMES[m]:<8} ' + ' '.join(f'{v:>8.3f}' for v in importances[m])
    print(row)

mean_imp = np.mean(np.stack([importances[m] for m in MODELS]), axis=0)
print(f'\n{"mean":<8} ' + ' '.join(f'{v:>8.3f}' for v in mean_imp))
rank = np.argsort(-mean_imp)
print('\nTraits ranked by mean RF importance (high -> low):')
for i, t_idx in enumerate(rank):
    print(f'  {i+1}. {TRAITS[t_idx]:<14} mean importance = {mean_imp[t_idx]:.3f}')

json.dump({
    'per_model': {m: importances[m].tolist() for m in MODELS},
    'mean': mean_imp.tolist(),
    'mean_rank': [TRAITS[i] for i in rank],
    'traits': TRAITS,
}, open(OUT_DIR / 'rf_feature_importance.json', 'w'), indent=2)
print(f"\nWrote {OUT_DIR / 'rf_feature_importance.json'}")

def lopo_cv_metrics(trait_indices):
    '''Pooled LOPO-CV across all 4 models and 4 calibration perts. Returns (balanced_acc_pct, auroc).'''
    all_pred, all_true = [], []
    for m in MODELS:
        X, y, pid = cal[m]
        X_sub = X[:, trait_indices]
        for held_out in range(len(CAL_PERTS)):
            train_mask = pid != held_out; test_mask = pid == held_out
            if not test_mask.any() or not train_mask.any(): continue
            rf = RandomForestRegressor(**RF_KW).fit(X_sub[train_mask], y[train_mask])
            pred = np.clip(rf.predict(X_sub[test_mask]), 0, 1)
            all_pred.append(pred); all_true.append(y[test_mask])
    preds = np.concatenate(all_pred); trues = np.concatenate(all_true)
    actual = trues > EM_THRESH; alarm = preds > EM_THRESH
    n_p = int(actual.sum()); n_n = int((~actual).sum())
    tpr = (alarm & actual).sum() / n_p if n_p else 0.0
    tnr = (~alarm & ~actual).sum() / n_n if n_n else 0.0
    bacc = (tpr + tnr) / 2 * 100
    auroc = roc_auc_score(actual, preds) if (n_p and n_n) else float('nan')
    return bacc, auroc

# Start with all 7 traits; greedily remove one at a time until K=1
remaining = list(range(7)); removal_order = []; curve = []
while len(remaining) >= 1:
    K = len(remaining)
    bacc_full, auroc_full = lopo_cv_metrics(remaining)
    curve.append({'K': K, 'subset': [TRAITS[i] for i in remaining],
                  'bacc': bacc_full, 'auroc': auroc_full})
    if K == 1: break
    # Try removing each; pick the one whose removal retains the HIGHEST bacc (tie-break by AUROC)
    scores = {}
    for t in remaining:
        subset = [x for x in remaining if x != t]
        scores[t] = lopo_cv_metrics(subset)
    drop = max(scores.keys(), key=lambda t: (scores[t][0], scores[t][1]))
    removal_order.append(drop)
    remaining = [x for x in remaining if x != drop]

# Also the final K=1 entry is already in curve (last iteration)

print(f'{"K":>3}  {"BalAcc":>7}  {"AUROC":>7}  Subset')
print('-' * 80)
for c in curve:
    print(f'{c["K"]:>3}  {c["bacc"]:>6.2f}%  {c["auroc"]:>7.4f}  ' + ', '.join(c['subset']))

print(f'\nRemoval order (first -> last dropped): ' + ', '.join(TRAITS[i] for i in removal_order))

json.dump({
    'curve': curve,
    'removal_order': [TRAITS[i] for i in removal_order],
}, open(OUT_DIR / 'backward_elimination.json', 'w'), indent=2)
print(f"\nWrote {OUT_DIR / 'backward_elimination.json'}")

# Map K -> subset (from curve; curve was built in decreasing-K order)
subset_at_K = {c['K']: c['subset'] for c in curve}

def ood_eval(trait_names):
    trait_indices = [TRAITS.index(t) for t in trait_names]
    rfs = {}
    for m in MODELS:
        X, y, _ = cal[m]
        rfs[m] = RandomForestRegressor(**RF_KW).fit(X[:, trait_indices], y)
    preds, trues = [], []
    for m in MODELS:
        X, y, _ = ood[m]
        pred = np.clip(rfs[m].predict(X[:, trait_indices]), 0, 1)
        preds.append(pred); trues.append(y)
    preds = np.concatenate(preds); trues = np.concatenate(trues)
    actual = trues > EM_THRESH; alarm = preds > EM_THRESH
    n_p = int(actual.sum()); n_n = int((~actual).sum())
    fn = int((actual & ~alarm).sum()); fp = int((~actual & alarm).sum())
    return {
        'fnr': fn/n_p*100 if n_p else 0.0, 'fpr': fp/n_n*100 if n_n else 0.0,
        'acc': (n_p+n_n-fn-fp)/(n_p+n_n)*100, 'fn': fn, 'fp': fp,
        'auroc': float(roc_auc_score(actual, preds)) if n_p and n_n else float('nan'),
        'n_pos': n_p, 'n_neg': n_n,
    }

ood_rows = []
print(f'{"K":>3}  {"FNR":>7}  {"FPR":>7}  {"Acc":>7}  {"AUROC":>7}  Subset')
print('-' * 90)
for K in range(7, 0, -1):
    subset = subset_at_K[K]
    r = ood_eval(subset)
    ood_rows.append({'K': K, 'subset': subset, **r})
    print(f'{K:>3}  {r["fnr"]:>6.2f}%  {r["fpr"]:>6.2f}%  {r["acc"]:>6.2f}%  {r["auroc"]:>7.4f}  ' + ', '.join(subset))

json.dump({'per_K': ood_rows}, open(OUT_DIR / 'ood_at_K.json', 'w'), indent=2)
print(f"\nWrote {OUT_DIR / 'ood_at_K.json'}")

# Final-checkpoint drift matrix in full 7D, one row per (model, cal pert, seed)
rows_full = []
for m in MODELS:
    for p in CAL_PERTS:
        for s in SEEDS:
            t = load_traj_7d(m, p, s)
            steps = sorted(s2 for s2 in t if s2 != 0)
            if not steps: continue
            rows_full.append(t[steps[-1]])
X_full = np.array(rows_full)  # (48, 7)
pca_full = PCA(n_components=min(7, X_full.shape[0])).fit(X_full)
pc1_full = pca_full.components_[0] / np.linalg.norm(pca_full.components_[0])
ve_full = float(pca_full.explained_variance_ratio_[0])
print(f'Full 7D PC1: variance explained = {ve_full*100:.1f}%  (n = {X_full.shape[0]})')

pc1_rows = []
print(f'\n{"K":>3}  {"VE %":>6}  {"cos(PC1_K, PC1_7)":>20}   Subset')
print('-' * 95)
for K in range(7, 0, -1):
    subset = subset_at_K[K]; idx = [TRAITS.index(t) for t in subset]
    X_K = X_full[:, idx]
    pca_K = PCA(n_components=min(K, X_K.shape[0])).fit(X_K)
    pc1_K = pca_K.components_[0] / np.linalg.norm(pca_K.components_[0])
    # Embed pc1_K back into 7D zero-padded by non-retained traits for a direct cosine
    pc1_K_full = np.zeros(7)
    for j, t_idx in enumerate(idx): pc1_K_full[t_idx] = pc1_K[j]
    pc1_K_full /= np.linalg.norm(pc1_K_full)
    cos = abs(float(np.dot(pc1_full, pc1_K_full)))
    ve = float(pca_K.explained_variance_ratio_[0])
    pc1_rows.append({'K': K, 'subset': subset, 've': ve, 'cos_pc1': cos})
    print(f'{K:>3}  {ve*100:>5.1f}%  {cos:>19.4f}   ' + ', '.join(subset))

json.dump({'per_K': pc1_rows, 'full_ve': ve_full}, open(OUT_DIR / 'pc1_cosine_at_K.json', 'w'), indent=2)
print(f"\nWrote {OUT_DIR / 'pc1_cosine_at_K.json'}")

combined = []
for c, o, p in zip(curve, ood_rows, pc1_rows):
    combined.append({**c, 'ood_fnr': o['fnr'], 'ood_fpr': o['fpr'], 'ood_acc': o['acc'],
                     'ood_auroc': o['auroc'], 'pc1_cos_vs_7': p['cos_pc1']})

print(f'{"K":>3}  {"cal BalAcc":>10}  {"cal AUROC":>9}  {"OOD FNR":>8}  {"OOD FPR":>8}  {"OOD AUROC":>10}  {"PC1 cos":>8}')
print('-' * 75)
for r in combined:
    print(f'{r["K"]:>3}  {r["bacc"]:>9.2f}%  {r["auroc"]:>9.4f}  '
          f'{r["ood_fnr"]:>7.2f}%  {r["ood_fpr"]:>7.2f}%  {r["ood_auroc"]:>10.4f}  {r["pc1_cos_vs_7"]:>8.4f}')

json.dump(combined, open(OUT_DIR / 'summary.json', 'w'), indent=2)
print(f"\nWrote {OUT_DIR / 'summary.json'}")

# Backward-elimination trait-count ablation. Reads the summary this notebook just
# wrote, so the table can never drift from the analysis that produced it.
import json as _json
from pathlib import Path as _Path

_summ = _json.load(open(OUTPUT_ROOT / 'trait_count' / 'summary.json'))
_rows = sorted(_summ, key=lambda r: -r['K'])
_full = next(r for r in _rows if r['K'] == 7)


def _retained(prev, cur):
    """Describe row `cur` relative to the next-larger subset `prev`."""
    if cur['K'] == 7:
        return 'all 7'
    if prev is not None:
        dropped = set(prev['subset']) - set(cur['subset'])
        if len(dropped) == 1 and cur['K'] >= 5:
            return 'drops ' + next(iter(dropped)).replace('_', r'\_')
    return r'{\{}' + ', '.join(t.replace('_', r'\_') for t in cur['subset']) + r'{\}}'


L = [r'\begin{tabular}{r r r r r r l}', r'\toprule',
     r'$K$ & Cal BalAcc & Cal AUROC & OOD FNR & OOD AUROC & '
     r'$\cos(\mathrm{PC1}_K,\mathrm{PC1}_7)$ & Retained traits \\',
     r'\midrule']
_prev = None
for r in _rows:
    fnr = f"{r['ood_fnr']:.2f}\\%"
    # flag the cliff: any K whose OOD FNR exceeds the full-basis value by >5pp
    if r['ood_fnr'] - _full['ood_fnr'] > 5.0:
        fnr = r'$\mathbf{' + f"{r['ood_fnr']:.1f}" + r'\%}$'
    L.append(f"{r['K']} & ${r['bacc']:.1f}\\%$ & ${r['auroc']:.3f}$ & {fnr} & "
             f"${r['ood_auroc']:.3f}$ & ${r['pc1_cos_vs_7']:.3f}$ & {_retained(_prev, r)} \\\\")
    _prev = r
L += [r'\bottomrule', r'\end{tabular}']

_out = OUTPUT_ROOT / 'tables' / 'tab_trait_count.tex'
_out.write_text('\n'.join(L) + '\n')
print(f'Wrote {_out}')
print('\n'.join(L))
print(f"\nK=3 subset: {[r for r in _rows if r['K']==3][0]['subset']}")
