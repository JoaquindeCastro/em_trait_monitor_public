"""basis ablation analysis of saved checkpoint artifacts."""

from pathlib import Path
PROJECT_ROOT = globals().get("PROJECT_ROOT", Path(__file__).resolve().parents[2])
ROOT = PROJECT_ROOT
OUTPUT_ROOT = globals().get("OUTPUT_ROOT", PROJECT_ROOT / "outputs/basis_ablation")
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
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import FancyBboxPatch
warnings.filterwarnings('ignore')


TRAJ_BASE = PROJECT_ROOT / 'results' / 'prelim' / 'st9' / 'trajectories'
ST1 = PROJECT_ROOT / 'results' / 'prelim' / 'st1'
PC1_PATH = PROJECT_ROOT / 'results' / 'prelim' / 'st9' / 'cluster_pc1' / 'cluster_pc1_summary.json'
FIG_OUT = OUTPUT_ROOT / 'figures'

TRAITS = ['honesty','sycophancy','harmlessness','power_seeking','helpfulness','confidence','corrigibility']
TRAIT_LABELS = ['Honesty','Sycophancy','Harmless.','Power-seek.','Helpful.','Confidence','Corrigib.']
MODELS = globals().get("MODELS", ['llama3-8b','mistral-7b','qwen25-7b','gemma2-9b'])
SEEDS = globals().get("SEEDS", [42, 123, 789])
CAL_PERTS = globals().get("CAL_PERTS", ['insecure_code_1k','gsm8k_1k','jailbroken','bad_medical'])
OOD_PERTS = globals().get("OOD_PERTS", ['number_sequence','risky_financial','subtle_misinfo'])
NORMS = globals().get("NORMS", {'llama3-8b':8.5,'mistral-7b':4.6875,'qwen25-7b':66.5,'gemma2-9b':372.0})

pc1_data = json.load(open(PC1_PATH))
PC1 = np.array([pc1_data['cluster_pc1'][t] for t in TRAITS])
PC1 /= np.linalg.norm(PC1)  # unit-normalize for consistency with §4
var_explained = [pc1_data['cluster_pc1_variance_explained'],
                 pc1_data['cluster_pc2_variance_explained'],
                 pc1_data['cluster_pc3_variance_explained']]

print(f'PC1 variance: {[f"{v*100:.1f}%" for v in var_explained]}')
print(f'PC1 loadings: {dict(zip(TRAITS, PC1.round(3)))}')
print(f'PC1 norm: {np.linalg.norm(PC1):.4f} (unit)')
print(f'Cal perts: {CAL_PERTS} (sycophancy excluded)')


def load_traj_7d(m, p, s):
    f = TRAJ_BASE/m/p/f'seed_{s}'/'trajectory.json'
    if not f.exists(): return {}
    t = json.load(open(f)); result = {}; s0 = None
    for e in t['trajectory']:
        step = e['step']
        if not isinstance(step, int): continue
        proj = np.array([e['projections'][tr] for tr in TRAITS])
        if step == 0: s0 = proj
        if s0 is not None: result[step] = (proj - s0) / NORMS[m]
    return result

def load_act_drifts(m, p, s):
    f = TRAJ_BASE/m/p/f'seed_{s}'/'activations.pt'
    if not f.exists(): return {}
    d = torch.load(f, map_location='cpu', weights_only=False)
    if 0 not in d: return {}
    base = d[0].float().mean(0).numpy()
    return {k: (v.float().mean(0).numpy() - base) / NORMS[m]
            for k, v in d.items() if isinstance(k, int)}

def load_betley(m, p, s):
    f = TRAJ_BASE/m/p/f'seed_{s}'/'betley_eval'/'grades.json'
    if not f.exists(): return {}
    d = json.load(open(f))
    return {int(k.replace('step_','')): v.get('misalignment_rate', 0.0)
            for k, v in d.items() if k.startswith('step_')}


# Table-1-parity loaders (same 5-way intersection: trajectory + activations + SAE + SafetyScore + Betley)
import csv as _csv
SAE_DIR = PROJECT_ROOT / 'results' / 'sae'
_SAE_CACHE = {}
def _load_sae_weights(m):
    d = torch.load(SAE_DIR/m/'sae_K256.pt', map_location='cpu', weights_only=False)
    return d['encoder.weight'].float().numpy(), d['encoder.bias'].float().numpy()
def load_sae_latents(m, p, s):
    f = TRAJ_BASE/m/p/f'seed_{s}'/'activations.pt'
    if not f.exists(): return {}
    d = torch.load(f, map_location='cpu', weights_only=False)
    if 0 not in d: return {}
    if m not in _SAE_CACHE: _SAE_CACHE[m] = _load_sae_weights(m)
    W, b = _SAE_CACHE[m]
    mean_act = {k: v.float().mean(0).numpy() for k, v in d.items() if isinstance(k, int)}
    def _enc(h): return np.maximum(0, W @ h + b)
    z0 = _enc(mean_act[0])
    return {k: (_enc(mean_act[k]) - z0) / NORMS[m] for k in mean_act}
def load_ss(m, p, s):
    f = TRAJ_BASE/m/p/f'seed_{s}'/'aux_behavioral_v3'/'summary.csv'
    if not f.exists(): return {}
    r = {}
    with open(f) as fh:
        for row in _csv.DictReader(fh):
            scores = [float(row[t+'_score']) for t in TRAITS if t+'_score' in row and row[t+'_score']]
            if scores: r[int(row['step'])] = np.mean(scores)
    return r

# Load basis vectors
bases_per_model = {}
for m in MODELS:
    align = torch.load(ST1/m/'persona_vectors.pt', map_location='cpu', weights_only=True)
    semantic = torch.load(ST1/m/'semantic_vectors.pt', map_location='cpu', weights_only=True)
    randoms = []
    for i in range(10):
        f = ST1/m/f'random_vectors_{i}.pt'
        if f.exists(): randoms.append(torch.load(f, map_location='cpu', weights_only=True))
    align_M = np.stack([align[t].float().numpy() for t in TRAITS])
    sem_M = np.stack([semantic[k].float().numpy() for k in semantic.keys()])
    rand_Ms = [np.stack([r[k].float().numpy() for k in r.keys()]) for r in randoms]
    bases_per_model[m] = (align_M, sem_M, rand_Ms)

# Project all cells onto all bases.
# Use the same 6-way filter as the headline (Table 1) so the alignment row
# of this ablation reproduces Table 1's headline numbers exactly. This restricts
# to checkpoints where all of (trajectory, activations, SAE, Safety Score,
# Betley, train loss) are available.
def load_train_loss(m, p, s):
    ckpt_dir = TRAJ_BASE/m/p/f'seed_{s}'/'checkpoints'
    if not ckpt_dir.exists(): return {}
    final_ts = None; max_step = -1
    for ckpt in ckpt_dir.iterdir():
        if not ckpt.name.startswith('checkpoint-'): continue
        try: step = int(ckpt.name.replace('checkpoint-',''))
        except ValueError: continue
        if step > max_step:
            ts = ckpt/'trainer_state.json'
            if ts.exists(): max_step = step; final_ts = ts
    if final_ts is None: return {}
    d = json.load(open(final_ts))
    return {e['step']: e['loss'] for e in d.get('log_history', []) if 'loss' in e and 'step' in e}

def interpolate_loss(step_to_loss, target_step):
    if not step_to_loss: return None
    if target_step in step_to_loss: return step_to_loss[target_step]
    steps = sorted(step_to_loss.keys())
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

print('Projecting all cells (Table 1 6-way intersection)...')
cell_cache = {}
for m in MODELS:
    align_M, sem_M, rand_Ms = bases_per_model[m]
    for p in CAL_PERTS + OOD_PERTS:
        for s in SEEDS:
            f = TRAJ_BASE/m/p/f'seed_{s}'/'activations.pt'
            if not f.exists(): continue
            A = torch.load(f, map_location='cpu', weights_only=False)
            if 0 not in A: continue
            _t7 = load_traj_7d(m, p, s)
            _ta = load_act_drifts(m, p, s)
            _sae = load_sae_latents(m, p, s)
            _ss = load_ss(m, p, s)
            _em = load_betley(m, p, s)
            _tl = load_train_loss(m, p, s)
            _allowed = (set(_t7) & set(_ta) & set(_sae) & set(_ss) & set(_em)) | {0}
            base_mean = A[0].float().mean(0).numpy()
            out = {}
            for step, tensor in A.items():
                if not isinstance(step, int): continue
                if step not in _allowed: continue
                if step != 0 and interpolate_loss(_tl, step) is None: continue
                drift = (tensor.float().mean(0).numpy() - base_mean) / NORMS[m]
                # (avoids tiny float-precision drift between post-hoc projection and the
                # trajectory.json values that Table 1 reads). Semantic/random are computed
                # post-hoc since trajectory.json doesn't store those projections.
                out[step] = {
                    'align': _t7[step] if step in _t7 else align_M @ drift,
                    'semantic': sem_M @ drift,
                    **{f'random_{i}': rand_Ms[i] @ drift for i in range(len(rand_Ms))},
                }
            cell_cache[(m, p, s)] = out
print(f'  {len(cell_cache)} cells projected')


EM_THRESH = globals().get("EM_THRESH", 0.06)

def build_dataset(basis_key, perts):
    rows = []
    for m in MODELS:
        for p in perts:
            for s in SEEDS:
                if (m, p, s) not in cell_cache: continue
                proj = cell_cache[(m, p, s)]
                em = load_betley(m, p, s)
                for step in sorted(set(proj) & set(em)):
                    if step == 0: continue
                    rows.append({'model': m, 'X': proj[step][basis_key], 'em': em[step]})
    return rows

def evaluate(basis_key):
    cal = build_dataset(basis_key, CAL_PERTS)
    ood = build_dataset(basis_key, OOD_PERTS)
    results = {}
    for ml_name, ml_cls in [
        ('Ridge', lambda: Ridge(**RIDGE_KW)),
        ('GBR', lambda: GradientBoostingRegressor(**GBR_HP)),
        ('RF',  lambda: RandomForestRegressor(**RF_HP)),
    ]:
        tp=fp=fn=tn = 0
        for m in MODELS:
            Xc = np.array([r['X'] for r in cal if r['model']==m])
            yc = np.array([r['em'] for r in cal if r['model']==m])
            Xo = np.array([r['X'] for r in ood if r['model']==m])
            yo = np.array([r['em'] for r in ood if r['model']==m])
            if len(Xc) < 5 or len(Xo) == 0: continue
            clf = ml_cls().fit(Xc, yc)
            yp = np.clip(clf.predict(Xo), 0, 1)
            alarm = yp > EM_THRESH; actual = yo > EM_THRESH
            tp += int((alarm & actual).sum()); fp += int((alarm & ~actual).sum())
            fn += int((~alarm & actual).sum()); tn += int((~alarm & ~actual).sum())
        n_pos = tp+fn; n_neg = fp+tn
        fnr = fn/n_pos*100 if n_pos else 0
        fpr = fp/n_neg*100 if n_neg else 0
        acc = (tp+tn)/(tp+fp+fn+tn)*100
        results[ml_name] = {'fnr': fnr, 'fpr': fpr, 'acc': acc, 'fn': fn, 'fp': fp, 'tp': tp, 'tn': tn}
    return results

print('Evaluating bases...')
align_res = evaluate('align')
sem_res = evaluate('semantic')
# Average over 10 random bases
rand_all = [evaluate(f'random_{i}') for i in range(10)]
rand_agg = {}
for ml in ['Ridge','GBR','RF']:
    rand_agg[ml] = {
        'fnr': np.mean([r[ml]['fnr'] for r in rand_all]),
        'fnr_std': np.std([r[ml]['fnr'] for r in rand_all]),
        'fpr': np.mean([r[ml]['fpr'] for r in rand_all]),
        'fpr_std': np.std([r[ml]['fpr'] for r in rand_all]),
        'acc': np.mean([r[ml]['acc'] for r in rand_all]),
    }

# Print summary
print(f"\n{'Basis':<12} {'ML':<8} {'FNR':>7} {'FPR':>7} {'Acc':>7} {'FNR ratio':>10}")
print('-' * 55)
for ml in ['Ridge','GBR','RF']:
    a = align_res[ml]; s = sem_res[ml]; r = rand_agg[ml]
    print(f"{'Alignment':<12} {ml:<8} {a['fnr']:>6.1f}% {a['fpr']:>6.1f}% {a['acc']:>6.1f}% {'1.0x':>10}")
    sem_ratio = s['fnr'] / a['fnr'] if a['fnr'] > 0 else float('inf')
    rand_ratio = r['fnr'] / a['fnr'] if a['fnr'] > 0 else float('inf')
    print(f"{'Semantic':<12} {ml:<8} {s['fnr']:>6.1f}% {s['fpr']:>6.1f}% {s['acc']:>6.1f}% {sem_ratio:>9.0f}x")
    print(f"{'Random':<12} {ml:<8} {r['fnr']:>6.1f}% {r['fpr']:>6.1f}% {r['acc']:>6.1f}% {rand_ratio:>9.0f}x")
    print()

# Feedback applied:
# 1. Shared y-axis scale (0-70%) for both panels — fair visual comparison
# 2. "0%" on Alignment+Ridge made prominent with green star marker
# 3. RF highlighted as recommended operating point via annotation
# 4. Legend moved to upper-right of FPR panel (right panel)

COLOR_ALIGN = '#2563eb'    # blue
COLOR_SEMANTIC = '#f59e0b' # amber
COLOR_RANDOM = '#9ca3af'   # gray
COLOR_ZERO = '#16a34a'     # green for zero-FNR highlight

ml_names = ['Ridge', 'GBR', 'RF']
x = np.arange(len(ml_names))
width = 0.25
Y_MAX = 75  # shared scale for both panels

fig, (ax_fnr, ax_fpr) = plt.subplots(1, 2, figsize=(7.0, 3.2), sharey=True)

# --- Left panel: FNR (safety-critical) ---
fnr_align = [align_res[ml]['fnr'] for ml in ml_names]
fnr_sem = [sem_res[ml]['fnr'] for ml in ml_names]
fnr_rand = [rand_agg[ml]['fnr'] for ml in ml_names]
fnr_rand_err = [rand_agg[ml]['fnr_std'] for ml in ml_names]

b1 = ax_fnr.bar(x - width, fnr_align, width, color=COLOR_ALIGN, edgecolor='white', zorder=3)
b2 = ax_fnr.bar(x, fnr_sem, width, color=COLOR_SEMANTIC, edgecolor='white', zorder=3)
b3 = ax_fnr.bar(x + width, fnr_rand, width, color=COLOR_RANDOM, edgecolor='white',
                yerr=fnr_rand_err, capsize=3, error_kw={'linewidth':1}, zorder=3)

for i, ml in enumerate(ml_names):
    a_fnr = align_res[ml]['fnr']
    s_fnr = sem_res[ml]['fnr']
    r_fnr = rand_agg[ml]['fnr']

    # Prominent "0% ★" for alignment zero-FNR (green star)
    if a_fnr == 0:
        ax_fnr.text(i - width, 2.0, '0%', ha='center', va='bottom',
                    fontsize=10, fontweight='bold', color=COLOR_ZERO)
        ax_fnr.plot(i - width, 0.5, marker='*', markersize=10, color=COLOR_ZERO, zorder=5)

    # FNR ratio annotations
    if a_fnr > 0 and s_fnr > 0:
        ratio = s_fnr / a_fnr
        ax_fnr.text(i, s_fnr + 1.5, f'{ratio:.0f}x', ha='center', va='bottom',
                    fontsize=9.5, fontweight='bold', color=COLOR_SEMANTIC)
    if a_fnr > 0 and r_fnr > 0:
        ratio = r_fnr / a_fnr
        ax_fnr.text(i + width, r_fnr + rand_agg[ml]['fnr_std'] + 1.5, f'{ratio:.0f}x',
                    ha='center', va='bottom', fontsize=9.5, fontweight='bold', color='#6b7280')

ax_fnr.set_xticks(x)
ax_fnr.set_xticklabels(ml_names, fontsize=12)
ax_fnr.set_ylabel('Rate (%)', fontsize=12)
ax_fnr.set_title('FNR (lower = safer)', fontsize=13, fontweight='bold')
ax_fnr.set_ylim(0, Y_MAX)
ax_fnr.tick_params(axis='both', labelsize=12)
ax_fnr.grid(axis='y', alpha=0.2, zorder=0)

# --- Right panel: FPR (same y-axis scale) ---
fpr_align = [align_res[ml]['fpr'] for ml in ml_names]
fpr_sem = [sem_res[ml]['fpr'] for ml in ml_names]
fpr_rand = [rand_agg[ml]['fpr'] for ml in ml_names]
fpr_rand_err = [rand_agg[ml]['fpr_std'] for ml in ml_names]

ax_fpr.bar(x - width, fpr_align, width, color=COLOR_ALIGN, edgecolor='white',
           label='Alignment', zorder=3)
ax_fpr.bar(x, fpr_sem, width, color=COLOR_SEMANTIC, edgecolor='white',
           label='Semantic', zorder=3)
ax_fpr.bar(x + width, fpr_rand, width, color=COLOR_RANDOM, edgecolor='white',
           label='Random', yerr=fpr_rand_err, capsize=3, error_kw={'linewidth':1}, zorder=3)

ax_fpr.set_xticks(x)
ax_fpr.set_xticklabels(ml_names, fontsize=12)
ax_fpr.set_title('FPR (lower = fewer false alarms)', fontsize=13, fontweight='bold')
ax_fpr.tick_params(axis='both', labelsize=12)
ax_fpr.grid(axis='y', alpha=0.2, zorder=0)
ax_fpr.legend(fontsize=10.5, loc='upper right')

plt.tight_layout()
out_path = FIG_OUT / 'fig_basis_ablation.pdf'
fig.savefig(out_path, bbox_inches='tight', dpi=600)
print(f'Saved: {out_path}')
plt.show()

print('=== Fact-check numbers ===')
print(f"PC1 loadings (4m5p): {dict(zip(TRAITS, PC1.round(4)))}")
print(f"Variance: PC1={var_explained[0]*100:.1f}%, PC2={var_explained[1]*100:.1f}%, PC3={var_explained[2]*100:.1f}%")
print()
for ml in ['Ridge','GBR','RF']:
    a = align_res[ml]; s = sem_res[ml]
    print(f"{ml}: Align FNR={a['fnr']:.1f}% FPR={a['fpr']:.1f}% | "
          f"Sem FNR={s['fnr']:.1f}% FPR={s['fpr']:.1f}% | "
          f"Rand FNR={rand_agg[ml]['fnr']:.1f}±{rand_agg[ml]['fnr_std']:.1f}%")
