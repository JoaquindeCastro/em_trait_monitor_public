"""headline analysis of saved checkpoint artifacts."""

from pathlib import Path
PROJECT_ROOT = globals().get("PROJECT_ROOT", Path(__file__).resolve().parents[2])
ROOT = PROJECT_ROOT
OUTPUT_ROOT = globals().get("OUTPUT_ROOT", PROJECT_ROOT / "outputs/headline")
(OUTPUT_ROOT / "figures").mkdir(parents=True, exist_ok=True)
(OUTPUT_ROOT / "tables").mkdir(parents=True, exist_ok=True)
RF_HP = globals().get("RF_HP", dict(n_estimators=100, max_depth=5, min_samples_leaf=5, random_state=42))
GBR_HP = globals().get("GBR_HP", dict(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42))
RIDGE_KW = globals().get("RIDGE_KW", dict(alpha=1.0))

import json, csv, numpy as np, torch, warnings
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
warnings.filterwarnings('ignore')


TRAJ = PROJECT_ROOT / 'results' / 'prelim' / 'st9' / 'trajectories'
PC1_PATH = PROJECT_ROOT / 'results' / 'prelim' / 'st9' / 'cluster_pc1' / 'cluster_pc1_summary.json'
SAE_DIR = PROJECT_ROOT / 'results' / 'sae'
FIG_OUT = OUTPUT_ROOT / 'tables'

TRAITS = ['honesty','sycophancy','harmlessness','power_seeking','helpfulness','confidence','corrigibility']
MODELS = globals().get("MODELS", ['llama3-8b','mistral-7b','qwen25-7b','gemma2-9b'])
MODEL_NAMES = {'llama3-8b':'LLaMA','mistral-7b':'Mistral','qwen25-7b':'Qwen','gemma2-9b':'Gemma'}
SEEDS = globals().get("SEEDS", [42, 123, 789])
NORMS = globals().get("NORMS", {'llama3-8b':8.5,'mistral-7b':4.6875,'qwen25-7b':66.5,'gemma2-9b':372.0})

CAL_PERTS = globals().get("CAL_PERTS", ['insecure_code_1k','gsm8k_1k','jailbroken','bad_medical'])
OOD_PERTS = globals().get("OOD_PERTS", ['number_sequence','risky_financial','subtle_misinfo'])
EM_THRESH = globals().get("EM_THRESH", 0.06)

# Load cluster PC1 direction
pc1_data = json.load(open(PC1_PATH))
PC1 = np.array([pc1_data['cluster_pc1'][t] for t in TRAITS])
PC1 /= np.linalg.norm(PC1)

print(f'Cal perts: {CAL_PERTS}')
print(f'OOD perts: {OOD_PERTS}')
print(f'PC1 loaded (unit norm)')

# retained for independent auditor verification) ---
import sys
sys.path.insert(0, str(PROJECT_ROOT))
from experiments.analysis.extra_baselines import load_lora_norms
from sklearn.cross_decomposition import PLSRegression



# All paths use default seed_{s}/ — LLaMA insecure_code lr=2e-5 already swapped there.

def load_traj_7d(model, pert, seed):
    """Load 7D cosine-normalized trait drift at each checkpoint."""
    f = TRAJ / model / pert / f'seed_{seed}' / 'trajectory.json'
    if not f.exists(): return {}
    t = json.load(open(f)); result = {}; s0 = None
    for e in t['trajectory']:
        if not isinstance(e['step'], int): continue
        proj = np.array([e['projections'][tr] for tr in TRAITS])
        if e['step'] == 0: s0 = proj
        if s0 is not None:
            result[e['step']] = (proj - s0) / NORMS[model]
    return result

def load_act_drifts(model, pert, seed):
    """Load 4096D mean activation drift at each checkpoint (for Soligo PCA-7)."""
    f = TRAJ / model / pert / f'seed_{seed}' / 'activations.pt'
    if not f.exists(): return {}
    d = torch.load(f, map_location='cpu', weights_only=False)
    if 0 not in d: return {}
    base = d[0].float().mean(0).numpy()
    return {k: (v.float().mean(0).numpy() - base) / NORMS[model]
            for k, v in d.items() if isinstance(k, int)}

def load_act_norm(model, pert, seed):
    """Load L2 norm of mean activation drift at each checkpoint (scalar baseline).
    Same cosine normalization as drift vectors (divide by mean ||h|| at step 0)."""
    f = TRAJ / model / pert / f'seed_{seed}' / 'activations.pt'
    if not f.exists(): return {}
    d = torch.load(f, map_location='cpu', weights_only=False)
    if 0 not in d: return {}
    base = d[0].float().mean(0).numpy()
    return {k: float(np.linalg.norm((v.float().mean(0).numpy() - base) / NORMS[model]))
            for k, v in d.items() if isinstance(k, int)}

def load_train_loss(model, pert, seed):
    """Load training loss at each checkpoint step from trainer_state.json."""
    ckpt_dir = TRAJ / model / pert / f'seed_{seed}' / 'checkpoints'
    if not ckpt_dir.exists(): return {}
    # Use the final checkpoint's trainer_state.json since log_history accumulates across all steps
    final_ts = None; max_step = -1
    for ckpt in ckpt_dir.iterdir():
        if not ckpt.name.startswith('checkpoint-'): continue
        try: step = int(ckpt.name.replace('checkpoint-', ''))
        except ValueError: continue
        if step > max_step:
            ts = ckpt / 'trainer_state.json'
            if ts.exists():
                max_step = step; final_ts = ts
    if final_ts is None: return {}
    d = json.load(open(final_ts))
    log = d.get('log_history', [])
    # Map each logged step to its loss; interpolate to measurement steps later
    step_to_loss = {e['step']: e['loss'] for e in log if 'loss' in e and 'step' in e}
    return step_to_loss

def interpolate_loss(step_to_loss, target_step):
    """Return interpolated training loss at target_step (linear interp between logged steps)."""
    if not step_to_loss: return None
    if target_step in step_to_loss: return step_to_loss[target_step]
    steps = sorted(step_to_loss.keys())
    # Find bracketing steps
    lower = [s for s in steps if s <= target_step]
    upper = [s for s in steps if s >= target_step]
    if lower and upper:
        lo, hi = max(lower), min(upper)
        if lo == hi: return step_to_loss[lo]
        alpha = (target_step - lo) / (hi - lo)
        return (1 - alpha) * step_to_loss[lo] + alpha * step_to_loss[hi]
    elif lower: return step_to_loss[max(lower)]
    elif upper: return step_to_loss[min(upper)]
    return None

def load_betley(model, pert, seed):
    """Load Betley EM rate at each checkpoint."""
    f = TRAJ / model / pert / f'seed_{seed}' / 'betley_eval' / 'grades.json'
    if not f.exists(): return {}
    d = json.load(open(f))
    return {int(k.replace('step_','')): v.get('misalignment_rate', 0.0)
            for k, v in d.items() if k.startswith('step_')}

def load_ss(model, pert, seed):
    """Load Safety Score (mean of 7 traits) at each checkpoint."""
    f = TRAJ / model / pert / f'seed_{seed}' / 'aux_behavioral_v3' / 'summary.csv'
    if not f.exists(): return {}
    r = {}
    with open(f) as fh:
        for row in csv.DictReader(fh):
            scores = [float(row[t+'_score']) for t in TRAITS if t+'_score' in row and row[t+'_score']]
            if scores: r[int(row['step'])] = np.mean(scores)
    return r


def _load_sae_weights(model):
    """Load SAE encoder weights for a model. Returns (W_enc, b_enc) with ReLU forward."""
    p = SAE_DIR / model / 'sae_K256.pt'
    d = torch.load(p, map_location='cpu', weights_only=False)
    return d['encoder.weight'].float().numpy(), d['encoder.bias'].float().numpy()

_SAE_CACHE = {}
def load_sae_latents(model, pert, seed):
    """Load SAE K=256 latent drift at each checkpoint (cosine-normalized by hidden-dim norm)."""
    f = TRAJ/model/pert/f'seed_{seed}'/'activations.pt'
    if not f.exists(): return {}
    d = torch.load(f, map_location='cpu', weights_only=False)
    if 0 not in d: return {}
    if model not in _SAE_CACHE:
        _SAE_CACHE[model] = _load_sae_weights(model)
    W, b = _SAE_CACHE[model]
    # Mean activation per step, then encode
    mean_act = {k: v.float().mean(0).numpy() for k, v in d.items() if isinstance(k, int)}
    def _encode(h): return np.maximum(0, W @ h + b)  # ReLU(W h + b) -> (256,)
    z0 = _encode(mean_act[0])
    return {k: (_encode(mean_act[k]) - z0) / NORMS[model]
            for k in mean_act if k != 0 or True}  # include step 0 (zero vector)

print('Loaders ready (including SAE K=256, act_norm and train_loss baselines).')

#   Theory-driven (ours):        Our 7D, |PC1|, PC1 (signed)
#   Published-method baseline:   Chen-style evil Persona
#   Non-directional scalars:     ||Δh̄||₂, training loss
#   SAE latent representation:   SAE K=256

pooled = {}

for model in MODELS:
    c7d, c_act, csae, cnorm, closs, c_lns, c_lnv, cem = [], [], [], [], [], [], [], []
    for p in CAL_PERTS:
        for s in SEEDS:
            t7 = load_traj_7d(model, p, s)
            ta = load_act_drifts(model, p, s)
            sae_lat = load_sae_latents(model, p, s)
            an = load_act_norm(model, p, s)
            tl = load_train_loss(model, p, s)
            ss = load_ss(model, p, s)
            em = load_betley(model, p, s)
            ln = load_lora_norms(model, p, s)  # {step -> (scalar, per_layer_vec)}
            common = set(t7) & set(ta) & set(sae_lat) & set(ss) & set(em) & set(ln) - {0}
            for step in sorted(common):
                loss_val = interpolate_loss(tl, step)
                if loss_val is None: continue
                c7d.append(t7[step]); c_act.append(ta[step]); csae.append(sae_lat[step])
                cnorm.append(an.get(step, np.linalg.norm(ta[step])))
                closs.append(loss_val)
                c_lns.append(ln[step][0]); c_lnv.append(ln[step][1])
                cem.append(em[step])

    if not cem:
        print(f'{MODEL_NAMES[model]}: no cal data, skipping'); continue

    c7d = np.array(c7d); c_act = np.array(c_act); csae = np.array(csae)
    cnorm = np.array(cnorm).reshape(-1, 1)
    closs = np.array(closs).reshape(-1, 1)
    c_lns = np.array(c_lns).reshape(-1, 1)
    c_lnv = np.array(c_lnv)
    cem = np.array(cem)

    pca7 = PCA(n_components=7).fit(c_act)
    cpca7 = pca7.transform(c_act)
    c_pc1_abs = np.abs(c7d @ PC1).reshape(-1, 1)
    c_pc1_signed = (c7d @ PC1).reshape(-1, 1)
    # PLS-7 fit on cal (X=Δh̄ 4096D, y=EM), transform both cal + OOD
    pls7 = PLSRegression(n_components=7, scale=False).fit(c_act, cem)
    cpls7 = pls7.transform(c_act)

    o7d, o_act, osae, onorm, oloss, o_lns, o_lnv, oem, o_run = [], [], [], [], [], [], [], [], []
    for p in OOD_PERTS:
        for s in SEEDS:
            t7 = load_traj_7d(model, p, s)
            ta = load_act_drifts(model, p, s)
            sae_lat = load_sae_latents(model, p, s)
            an = load_act_norm(model, p, s)
            tl = load_train_loss(model, p, s)
            ss = load_ss(model, p, s)
            em = load_betley(model, p, s)
            ln = load_lora_norms(model, p, s)
            common = set(t7) & set(ta) & set(sae_lat) & set(ss) & set(em) & set(ln) - {0}
            for step in sorted(common):
                loss_val = interpolate_loss(tl, step)
                if loss_val is None: continue
                o7d.append(t7[step]); o_act.append(ta[step]); osae.append(sae_lat[step])
                onorm.append(an.get(step, np.linalg.norm(ta[step])))
                oloss.append(loss_val)
                o_lns.append(ln[step][0]); o_lnv.append(ln[step][1])
                oem.append(em[step])
                o_run.append((MODEL_NAMES[model], p, s))

    if not oem:
        print(f'{MODEL_NAMES[model]}: no OOD data, skipping'); continue

    o7d = np.array(o7d); o_act = np.array(o_act); osae = np.array(osae)
    onorm = np.array(onorm).reshape(-1, 1)
    oloss = np.array(oloss).reshape(-1, 1)
    o_lns = np.array(o_lns).reshape(-1, 1)
    o_lnv = np.array(o_lnv)
    oem = np.array(oem)
    opca7 = pca7.transform(o_act)
    opls7 = pls7.transform(o_act)
    o_pc1_abs = np.abs(o7d @ PC1).reshape(-1, 1)
    o_pc1_signed = (o7d @ PC1).reshape(-1, 1)
    actual = oem > EM_THRESH

    print(f'{MODEL_NAMES[model]}: cal={len(cem)} ({int((cem>EM_THRESH).sum())} pos), '
          f'OOD={len(oem)} ({int(actual.sum())} pos)')

    all_features = [
        ('|PC1| only', c_pc1_abs, o_pc1_abs),
        ('PC1 (signed)', c_pc1_signed, o_pc1_signed),
        ('Our 7D', c7d, o7d),
        ('Soligo PCA-7', cpca7, opca7),
        ('PLS-7', cpls7, opls7),
        ('SAE K=256', csae, osae),
        ('Act norm', cnorm, onorm),
        ('Train loss', closs, oloss),
        ('LoRA norm scalar', c_lns, o_lns),
        ('LoRA norm per-layer', c_lnv, o_lnv),
    ]

    for feat_name, cX, oX in all_features:
        for ml_name, ml_cls in [
            ('Ridge', lambda: Ridge(**RIDGE_KW)),
            ('GBR', lambda: GradientBoostingRegressor(**GBR_HP)),
            ('RF', lambda: RandomForestRegressor(**RF_HP)),
        ]:
            key = (feat_name, ml_name)
            if key not in pooled:
                pooled[key] = {'tp':0,'fp':0,'fn':0,'tn':0,'model_accs':[]}
            clf = ml_cls().fit(cX, cem)
            pred = np.clip(clf.predict(oX), 0, 1)
            alarm = pred > EM_THRESH
            tp = int((alarm & actual).sum()); fp = int((alarm & ~actual).sum())
            fn = int((~alarm & actual).sum()); tn = int((~alarm & ~actual).sum())
            pooled[key]['tp'] += tp; pooled[key]['fp'] += fp
            pooled[key]['fn'] += fn; pooled[key]['tn'] += tn
            pooled[key].setdefault('y_true_bin', []).extend(actual.tolist())
            pooled[key].setdefault('y_true_em', []).extend(oem.tolist())
            pooled[key].setdefault('y_pred_em', []).extend(pred.tolist())
            pooled[key].setdefault('run_id', []).extend(o_run)
            acc = (tp + tn) / len(oem) * 100
            pooled[key]['model_accs'].append(f'{MODEL_NAMES[model]}={acc:.0f}%')



print(f"{'Features':<16} {'ML':<8} {'Acc':>7} {'FNR':>7} {'FPR':>7} {'FN':>4} {'FP':>5}  Per-model")
print('-' * 85)
for (feat, ml), v in sorted(pooled.items()):
    n = v['tp']+v['fp']+v['fn']+v['tn']
    npos = v['tp']+v['fn']; nneg = v['fp']+v['tn']
    acc = (v['tp']+v['tn'])/n*100
    fnr = v['fn']/npos*100 if npos else 0
    fpr = v['fp']/nneg*100 if nneg else 0
    flag = ' \u274c' if v['fn'] > 5 else ''
    print(f"{feat:<16} {ml:<8} {acc:>6.1f}% {fnr:>6.1f}% {fpr:>6.1f}% {v['fn']:>4}{flag} {v['fp']:>5}  {', '.join(v['model_accs'])}")

# Checkpoints within a run (same model, pert, seed) are highly correlated
# (deterministic training trajectory). Resampling checkpoints underestimates
# variance. We cluster-bootstrap over the 36 OOD runs (4 models x 3 perts x 3 seeds)
# instead: on each iteration, sample 36 runs with replacement and pool all of
# their checkpoints before computing FNR/FPR. AUROC remains a pooled cross-check
# on the full (un-resampled) prediction set.

from sklearn.metrics import roc_auc_score

N_BOOT = globals().get("N_BOOT", 1000)
rng = np.random.default_rng(seed=42)

ci_stats = {}
for key, v in pooled.items():
    y_true_bin = np.array(v['y_true_bin'])
    y_pred_em  = np.array(v['y_pred_em'])
    y_true_em  = np.array(v['y_true_em'])
    run_id     = np.array(v['run_id'], dtype=object)
    n = len(y_true_bin)

    # Build run -> index map once
    run_tuples = [tuple(r) for r in run_id]
    unique_runs = sorted(set(run_tuples))
    n_runs = len(unique_runs)
    run_to_idx = {r: np.array([i for i, rr in enumerate(run_tuples) if rr == r])
                  for r in unique_runs}

    # Cluster bootstrap CI for FNR and FPR
    fnr_boot, fpr_boot = [], []
    for _ in range(N_BOOT):
        chosen = [unique_runs[ix] for ix in rng.integers(0, n_runs, n_runs)]
        idx = np.concatenate([run_to_idx[r] for r in chosen])
        yb  = y_true_bin[idx].astype(bool)
        ypb = y_pred_em[idx] > EM_THRESH
        pos = yb.sum(); neg = (~yb).sum()
        fnr_boot.append((~ypb & yb).sum() / pos * 100 if pos else np.nan)
        fpr_boot.append(( ypb & ~yb).sum() / neg * 100 if neg else np.nan)
    fnr_boot = np.array(fnr_boot); fpr_boot = np.array(fpr_boot)
    fnr_lo, fnr_hi = np.nanpercentile(fnr_boot, [2.5, 97.5])
    fpr_lo, fpr_hi = np.nanpercentile(fpr_boot, [2.5, 97.5])

    # AUROC on full pooled predictions (threshold-free cross-check)
    try:
        auroc = roc_auc_score(y_true_bin, y_pred_em)
    except ValueError:
        auroc = float('nan')

    ci_stats[key] = {
        'fnr_ci': (float(fnr_lo), float(fnr_hi)),
        'fpr_ci': (float(fpr_lo), float(fpr_hi)),
        'auroc':  float(auroc),
        'n_ckpts': n,
        'n_runs':  n_runs,
    }

print(f"{'Features':<16} {'ML':<8} {'FNR (95% CI)':>18} {'FPR (95% CI)':>18} {'AUROC':>7} {'runs':>5}")
print('-' * 82)
for (feat, ml), s in sorted(ci_stats.items()):
    v = pooled[(feat, ml)]
    n_pos = v['tp'] + v['fn']; n_neg = v['fp'] + v['tn']
    fnr_pt = v['fn']/n_pos*100 if n_pos else 0
    fpr_pt = v['fp']/n_neg*100 if n_neg else 0
    fnr_ci = f"{fnr_pt:.1f} [{s['fnr_ci'][0]:.1f},{s['fnr_ci'][1]:.1f}]"
    fpr_ci = f"{fpr_pt:.1f} [{s['fpr_ci'][0]:.1f},{s['fpr_ci'][1]:.1f}]"
    print(f'{feat:<16} {ml:<8} {fnr_ci:>18} {fpr_ci:>18} {s["auroc"]:>7.3f} {s["n_runs"]:>5}')


# Compute derived counts for caption (totals identical across combos)
_any_key = next(iter(pooled))
_v = pooled[_any_key]
n_total_ood = _v['tp'] + _v['fp'] + _v['fn'] + _v['tn']
n_dangerous_ood = _v['tp'] + _v['fn']

# Add the audited Chen-style generated-response persona-vector baseline.
# Its regressor is selected using calibration LODO only; the appendix shows
# all three regressors and the condensed table retains that selected row.
PERSONA_RESULT = PROJECT_ROOT / 'results' / 'staging' / 'chen_persona_vector' / 'full' / 'chen_evil_persona_vector_baseline.json'
persona_result = json.load(open(PERSONA_RESULT))
assert persona_result['counts'] == {'calibration': 624, 'ood': 468, 'ood_dangerous': 217}
assert persona_result['selected_regressor'] == 'Ridge'
for ml in ('Ridge', 'RF', 'GBR'):
    result = persona_result['results'][ml]['ood_pooled']
    key = ('Persona', ml)
    pooled[key] = {name: int(result[name]) for name in ('tp', 'fp', 'fn', 'tn')}
    ci = result['ci95_run_bootstrap']
    ci_stats[key] = {
        'fnr_ci': tuple(100 * float(x) for x in ci['fnr']),
        'fpr_ci': tuple(100 * float(x) for x in ci['fpr']),
        'auroc': float(result['auroc']),
        'n_ckpts': persona_result['counts']['ood'],
        'n_runs': 36,
    }

# Rows are grouped by feature-family, with section headers.
# Rebuttal additions (PLS-7, LoRA norm scalar, LoRA norm per-layer) sit in

rows_order = [
    # Theory-driven trait basis (ours)
    ('Our 7D', 'RF'),
    ('Our 7D', 'GBR'),
    ('Our 7D', 'Ridge'),
    ('|PC1| only', 'RF'),
    ('|PC1| only', 'GBR'),
    ('|PC1| only', 'Ridge'),
    ('Persona', 'Ridge'),
    ('Persona', 'RF'),
    ('Persona', 'GBR'),
    # Scalar non-directional baselines
    ('Act norm', 'RF'),
    ('Train loss', 'RF'),
    # Data-driven bases
    ('Soligo PCA-7', 'Ridge'),
    ('Soligo PCA-7', 'RF'),
    ('Soligo PCA-7', 'GBR'),
    ('PLS-7', 'Ridge'),
    ('PLS-7', 'RF'),
    ('PLS-7', 'GBR'),
    # SAE latent basis
    ('SAE K=256', 'Ridge'),
    ('SAE K=256', 'RF'),
    ('SAE K=256', 'GBR'),
    # LoRA-side scalars
    ('LoRA norm scalar', 'RF'),
    ('LoRA norm scalar', 'GBR'),
    ('LoRA norm scalar', 'Ridge'),
    ('LoRA norm per-layer', 'RF'),
    ('LoRA norm per-layer', 'GBR'),
    ('LoRA norm per-layer', 'Ridge'),
]

feat_display_map = {
    'Our 7D': 'Our 7D',
    '|PC1| only': r'|PC1| only',
    'Persona': 'Persona',
    'Act norm': r'$\|\Delta \bar{h}\|_2$',
    'Train loss': 'Training loss',
    'Soligo PCA-7': r'Soligo PCA-7',
    'PLS-7': r'PLS-7',
    'SAE K=256': r'SAE (trained, K=256)',
    'LoRA norm scalar': r'LoRA norm (scalar)',
    'LoRA norm per-layer': r'LoRA norm (per-layer)',
}

section_headers = {
    0:  r'\multicolumn{8}{l}{\emph{Theory-driven trait basis (ours)}}',
    3:  r'\multicolumn{8}{l}{\emph{Scalar directional baselines}}',
    9:  r'\multicolumn{8}{l}{\emph{Scalar non-directional baselines}}',
    11: r'\multicolumn{8}{l}{\emph{Data-driven bases}}',
    17: r'\multicolumn{8}{l}{\emph{SAE latent basis}}',
    20: r'\multicolumn{8}{l}{\emph{LoRA-side scalars}}',
}

lines = []
lines.append(r'\begin{tabular}{ll r l l r r r}')
lines.append(r'\toprule')
lines.append(r'Features & ML & Acc (\%) & FNR \% [95\% CI] & FPR \% [95\% CI] & FN & FP & AUROC \\')
lines.append(r'\midrule')

# Row shading: cycle 3 shades of blue per (feature) triplet, light-to-darker.
# Non-triplet rows (Act norm, Training loss) get the lightest shade.
ROW_SHADES = [r'\rowcolor{blue!5}', r'\rowcolor{blue!10}', r'\rowcolor{blue!17}']
triplet_idx = 0  # position within current feature triplet (0/1/2)
last_feat = None
for i, (feat, ml) in enumerate(rows_order):
    if i in section_headers:
        lines.append(section_headers[i] + r' \\')
        triplet_idx = 0
        last_feat = None
    if (feat, ml) not in pooled:
        continue
    v = pooled[(feat, ml)]
    s = ci_stats[(feat, ml)]
    n = v['tp']+v['fp']+v['fn']+v['tn']
    npos = v['tp']+v['fn']; nneg = v['fp']+v['tn']
    acc = (v['tp']+v['tn'])/n*100
    fnr = v['fn']/npos*100 if npos else 0
    fpr = v['fp']/nneg*100 if nneg else 0
    fnr_ci = f"{fnr:.1f} [{s['fnr_ci'][0]:.1f}, {s['fnr_ci'][1]:.1f}]"
    fpr_ci = f"{fpr:.1f} [{s['fpr_ci'][0]:.1f}, {s['fpr_ci'][1]:.1f}]"
    auroc = f"{s['auroc']:.3f}"
    feat_display = feat_display_map.get(feat, feat)
    # Reset triplet index at each new feature name
    if feat != last_feat:
        triplet_idx = 0
        last_feat = feat
    lines.append(ROW_SHADES[triplet_idx % 3])
    triplet_idx += 1
    is_bold = (feat == 'Our 7D' and ml == 'RF')
    is_italic = (feat == 'Our 7D' and ml == 'GBR')
    if is_bold:
        row = f'\\textbf{{{feat_display}}} & \\textbf{{{ml}}} & \\textbf{{{acc:.1f}}} & \\textbf{{{fnr_ci}}} & \\textbf{{{fpr_ci}}} & \\textbf{{{v["fn"]}}} & \\textbf{{{v["fp"]}}} & \\textbf{{{auroc}}} \\\\'
    elif is_italic:
        row = f'\\textit{{{feat_display}}} & \\textit{{{ml}}} & \\textit{{{acc:.1f}}} & \\textit{{{fnr_ci}}} & \\textit{{{fpr_ci}}} & \\textit{{{v["fn"]}}} & \\textit{{{v["fp"]}}} & \\textit{{{auroc}}} \\\\'
    else:
        row = f'{feat_display} & {ml} & {acc:.1f} & {fnr_ci} & {fpr_ci} & {v["fn"]} & {v["fp"]} & {auroc} \\\\'
    lines.append(row)

lines.append(r'\bottomrule')
lines.append(r'\end{tabular}')

tex = '\n'.join(lines)
out_path = FIG_OUT / 'tab_headline_detection.tex'
out_path.write_text(tex)
print(f'Saved (canonical): {out_path}')
print()
print(tex)


# Compares regressors without consulting held-out OOD data. We run leave-one-
# calibration-dataset-out cross-validation using ONLY calibration data (4 datasets
# x 3 seeds per model). No OOD data is consulted.
#
# Metrics: pooled balanced accuracy at the 6% threshold and AUROC,
#   computed once on the union of out-of-fold predictions across all
#   4 LODO folds x 4 models. This matches the pooled-AUROC convention
#   and avoids the inflation that mean-of-fold averaging produces when
#   individual held-out cal folds contain only safe checkpoints.
#
# Output: numbers printed for inline citation in app:regressor_cv (no
# separate table file).

from sklearn.metrics import roc_auc_score as _roc_auc

REGRESSORS_CV = {
    'Ridge': lambda: Ridge(**RIDGE_KW),
    'GBR':   lambda: GradientBoostingRegressor(**GBR_HP),
    'RF':    lambda: RandomForestRegressor(**RF_HP),
}


def _cv_build_by_pert(model):
    """pert -> (X_7d, y_em) pooled over 3 seeds & all cal checkpoints."""
    out = {}
    for pert in CAL_PERTS:
        xs, ys = [], []
        for seed in SEEDS:
            t = load_traj_7d(model, pert, seed)
            em = load_betley(model, pert, seed)
            for step in sorted(set(t) & set(em)):
                if step == 0:
                    continue
                xs.append(t[step]); ys.append(em[step])
        out[pert] = (np.array(xs), np.array(ys))
    return out


def _pooled_metrics(pred, label):
    """Pooled BalAcc and AUROC from out-of-fold predictions and labels."""
    pred = np.asarray(pred, dtype=float)
    label = np.asarray(label).astype(bool)
    alarm = pred > EM_THRESH
    tp = int((alarm & label).sum())
    fp = int((alarm & ~label).sum())
    fn = int((~alarm & label).sum())
    tn = int((~alarm & ~label).sum())
    n_pos, n_neg = tp + fn, fp + tn
    fnr = fn / n_pos * 100 if n_pos else float('nan')
    fpr = fp / n_neg * 100 if n_neg else float('nan')
    bal = ((100 - fnr) + (100 - fpr)) / 2 if (n_pos and n_neg) else float('nan')
    try:
        auroc = float(_roc_auc(label.astype(int), pred)) \
            if len(set(label.tolist())) > 1 else float('nan')
    except ValueError:
        auroc = float('nan')
    return {'bal_acc': bal, 'auroc': auroc, 'n_pos': n_pos, 'n_neg': n_neg}


# Collect out-of-fold predictions across all (model, fold) pairs.
global_pooled_pred = {r: [] for r in REGRESSORS_CV}
global_pooled_label = {r: [] for r in REGRESSORS_CV}
for m in MODELS:
    by_pert = _cv_build_by_pert(m)
    for held in CAL_PERTS:
        train_X = np.concatenate([by_pert[p][0] for p in CAL_PERTS if p != held], axis=0)
        train_y = np.concatenate([by_pert[p][1] for p in CAL_PERTS if p != held], axis=0)
        test_X, test_y = by_pert[held]
        if len(test_X) == 0:
            continue
        act = test_y > EM_THRESH
        for name, fac in REGRESSORS_CV.items():
            clf = fac().fit(train_X, train_y)
            pred = np.clip(clf.predict(test_X), 0, 1)
            global_pooled_pred[name].extend(pred.tolist())
            global_pooled_label[name].extend(act.astype(int).tolist())

global_stats = {r: _pooled_metrics(global_pooled_pred[r], global_pooled_label[r])
                for r in REGRESSORS_CV}

global_winner_balacc = max(REGRESSORS_CV, key=lambda r: global_stats[r]['bal_acc'])
global_winner_auroc  = max(REGRESSORS_CV, key=lambda r: global_stats[r]['auroc'])
n_total = global_stats['RF']['n_pos'] + global_stats['RF']['n_neg']
n_pos_g = global_stats['RF']['n_pos']

# --- Console summary ---
print(f'Pooled cal-LODO checkpoints: {n_total}  ({n_pos_g} dangerous, {n_total - n_pos_g} safe)')
print(f'Protocol: 4 models x 4 LODO folds = 16 folds; out-of-fold predictions pooled before metric')
print()
print(f"{'Reg':<6} {'BalAcc%':>8} {'AUROC':>8}")
print('-' * 24)
for r in REGRESSORS_CV:
    s = global_stats[r]
    marks = []
    if r == global_winner_balacc: marks.append('BalAcc')
    if r == global_winner_auroc:  marks.append('AUROC')
    mark = '  <-- WINNER by ' + ', '.join(marks) if marks else ''
    print(f"{r:<6} {s['bal_acc']:>7.1f}% {s['auroc']:>8.3f}{mark}")

print()
print('=== Inline-prose snippet for app:regressor_cv ===')
print(
    f"Pooled balanced accuracy ranges from "
    f"{min(s['bal_acc'] for s in global_stats.values()):.1f}\\% to "
    f"{max(s['bal_acc'] for s in global_stats.values()):.1f}\\%; "
    f"AUROC is GBR~{global_stats['GBR']['auroc']:.3f}, "
    f"RF~{global_stats['RF']['auroc']:.3f}, and "
    f"Ridge~{global_stats['Ridge']['auroc']:.3f}."
)

# Ours keeps all three regressors. Persona uses its calibration-selected regressor;
# every remaining baseline uses lowest held-out FNR with accuracy as tie-break.
# The full grid stays in the appendix as tab_headline_detection.tex.
OURS = 'Our 7D'
PERSONA = 'Persona'


def _metrics(feat, ml):
    v = pooled[(feat, ml)]
    n = v['tp'] + v['fp'] + v['fn'] + v['tn']
    npos, nneg = v['tp'] + v['fn'], v['fp'] + v['tn']
    return {
        'acc': (v['tp'] + v['tn']) / n * 100,
        'fnr': v['fn'] / npos * 100 if npos else 0.0,
        'fpr': v['fp'] / nneg * 100 if nneg else 0.0,
        'fn': v['fn'], 'fp': v['fp'],
    }


# preserve the order features first appear in the full table
feat_order, seen = [], set()
for feat, ml in rows_order:
    if (feat, ml) in pooled and feat not in seen:
        seen.add(feat); feat_order.append(feat)

keep = []          # (feat, ml) rows for the condensed table
for feat in feat_order:
    mls = [ml for f, ml in rows_order if f == feat and (f, ml) in pooled]
    if feat == OURS:
        keep += [(feat, ml) for ml in mls]
    elif feat == PERSONA:
        selected = persona_result['selected_regressor']
        assert selected in mls
        keep.append((feat, selected))
    else:
        best = min(mls, key=lambda ml: (_metrics(feat, ml)['fnr'],
                                        -_metrics(feat, ml)['acc']))
        keep.append((feat, best))

L = []
L.append(r'\begin{tabular}{ll r l l r r r}')
L.append(r'\toprule')
L.append(r'Features & ML & Acc (\%) & FNR \% [95\% CI] & FPR \% [95\% CI] & FN & FP & AUROC \\')
L.append(r'\midrule')
for i, (feat, ml) in enumerate(keep):
    if feat == OURS and i == 0:
        L.append(r'\multicolumn{8}{l}{\emph{Theory-driven trait basis (ours)}} \\')
    if feat != OURS and (i == 0 or keep[i - 1][0] == OURS):
        L.append(r'\multicolumn{8}{l}{\emph{Baselines (best configuration of each)}} \\')
    m = _metrics(feat, ml); s = ci_stats[(feat, ml)]
    fnr_ci = f"{m['fnr']:.1f} [{s['fnr_ci'][0]:.1f}, {s['fnr_ci'][1]:.1f}]"
    fpr_ci = f"{m['fpr']:.1f} [{s['fpr_ci'][0]:.1f}, {s['fpr_ci'][1]:.1f}]"
    disp = feat_display_map.get(feat, feat)
    L.append(r'\rowcolor{blue!5}' if feat == OURS else '')
    if feat == OURS and ml == 'RF':
        L.append(f'\\textbf{{{disp}}} & \\textbf{{{ml}}} & \\textbf{{{m["acc"]:.1f}}} & '
                 f'\\textbf{{{fnr_ci}}} & \\textbf{{{fpr_ci}}} & \\textbf{{{m["fn"]}}} & '
                 f'\\textbf{{{m["fp"]}}} & \\textbf{{{s["auroc"]:.3f}}} \\\\')
    else:
        L.append(f'{disp} & {ml} & {m["acc"]:.1f} & {fnr_ci} & {fpr_ci} & '
                 f'{m["fn"]} & {m["fp"]} & {s["auroc"]:.3f} \\\\')
L = [x for x in L if x != '']
L.append(r'\bottomrule')
L.append(r'\end{tabular}')

out_main = FIG_OUT / 'tab_headline_detection_main.tex'
out_main.write_text('\n'.join(L) + '\n')
print(f'Saved (condensed main-text): {out_main}   {len(keep)} rows '
      f'(was {sum(1 for r in rows_order if r in pooled)})')
for feat, ml in keep:
    print(f'   {feat:<24} {ml:<6} FNR {_metrics(feat, ml)["fnr"]:5.1f}%')
