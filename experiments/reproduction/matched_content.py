"""matched content analysis of saved checkpoint artifacts."""

from pathlib import Path
PROJECT_ROOT = globals().get("PROJECT_ROOT", Path(__file__).resolve().parents[2])
ROOT = PROJECT_ROOT
OUTPUT_ROOT = globals().get("OUTPUT_ROOT", PROJECT_ROOT / "outputs/matched_content")
(OUTPUT_ROOT / "figures").mkdir(parents=True, exist_ok=True)
(OUTPUT_ROOT / "tables").mkdir(parents=True, exist_ok=True)
RF_HP = globals().get("RF_HP", dict(n_estimators=100, max_depth=5, min_samples_leaf=5, random_state=42))
GBR_HP = globals().get("GBR_HP", dict(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42))
RIDGE_KW = globals().get("RIDGE_KW", dict(alpha=1.0))

import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import fisher_exact
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

warnings.filterwarnings("ignore")


TRAJ = PROJECT_ROOT / 'results' / 'prelim' / 'st9' / 'trajectories'
PC1_PATH = PROJECT_ROOT / 'results' / 'prelim' / 'st9' / 'cluster_pc1' / 'cluster_pc1_summary.json'
TAB_OUT = OUTPUT_ROOT / 'tables'
FIG_OUT = OUTPUT_ROOT / 'figures'
TAB_OUT.mkdir(parents=True, exist_ok=True); FIG_OUT.mkdir(parents=True, exist_ok=True)

TRAITS = ['honesty','sycophancy','harmlessness','power_seeking','helpfulness','confidence','corrigibility']
MODELS = globals().get("MODELS", ['mistral-7b','llama3-8b','qwen25-7b','gemma2-9b'])
MODEL_LABEL = {'mistral-7b':'Mistral 7B', 'llama3-8b':'LLaMA 3.1 8B',
               'qwen25-7b':'Qwen 2.5 7B', 'gemma2-9b':'Gemma 2 9B'}
NORMS = globals().get("NORMS", {'mistral-7b':4.6875, 'llama3-8b':8.5, 'qwen25-7b':66.5, 'gemma2-9b':372.0})
CAL_UNIF = ['insecure_code_1k','gsm8k_1k','jailbroken','bad_medical']
PERTS = {'extreme_sports_pool1500':'dangerous', 'safe_sports_pool1500':'safe'}
SEEDS = globals().get("SEEDS", [42, 123, 789])
EM_TAU = globals().get("EM_TAU", 0.06)
N_BOOT = globals().get("N_BOOT", 1000)
BOOT_SEED = globals().get("BOOT_SEED", 42)

RF_HP  = dict(n_estimators=100, max_depth=5, min_samples_leaf=5, random_state=42)
GBR_HP = globals().get("GBR_HP", dict(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42))

pc1 = json.load(open(PC1_PATH))
PC1_V = np.array([pc1['cluster_pc1'][t] for t in TRAITS])
PC1_V = PC1_V / np.linalg.norm(PC1_V)

def load_traj(model, pert, seed):
    f = TRAJ/model/pert/f'seed_{seed}'/'trajectory.json'
    if not f.exists(): return {}
    t = json.load(open(f))
    s0e = next((e for e in t['trajectory'] if e['step']==0), None)
    if s0e is None: return {}
    s0 = np.array([s0e['projections'][tr] for tr in TRAITS])
    return {e['step']: (np.array([e['projections'][tr] for tr in TRAITS]) - s0) / NORMS[model]
            for e in t['trajectory'] if isinstance(e['step'], int)}

def load_em(model, pert, seed):
    f = TRAJ/model/pert/f'seed_{seed}'/'betley_eval'/'grades.json'
    if not f.exists(): return {}
    d = json.load(open(f))
    return {int(k.replace('step_','')): (v.get('n_misaligned',0), v.get('n_scoreable',0),
                                          v.get('misalignment_rate',0.0))
            for k,v in d.items() if k.startswith('step_')}

def fit_cal_regressors(model):
    X, y = [], []
    for p in CAL_UNIF:
        for s in SEEDS:
            t = load_traj(model, p, s)
            em = load_em(model, p, s)
            common = sorted(set(t) & set(em) - {0})
            for st in common:
                X.append(t[st]); y.append(em[st][2])
    X, y = np.array(X), np.array(y)
    rf  = RandomForestRegressor(**RF_HP).fit(X, y)
    gbr = GradientBoostingRegressor(**GBR_HP).fit(X, y)
    return rf, gbr, len(y), int((y>EM_TAU).sum())

MODEL_REGS = {}
for m in MODELS:
    rf, gbr, n_cal, n_dang = fit_cal_regressors(m)
    MODEL_REGS[m] = (rf, gbr)
    print(f'{m}: cal={n_cal} rows, {n_dang} dangerous (>τ={EM_TAU*100:.0f}%)')


records = []  # one row per (model, pert, seed) at FINAL step
trajectories = {}  # (model, pert, seed) -> dict[step] -> {em_rate, pred_rf, pred_gbr, drift_mag, cos_pc1, n_mis, n_scor}

for m in MODELS:
    rf, gbr = MODEL_REGS[m]
    for pert, label in PERTS.items():
        for s in SEEDS:
            t = load_traj(m, pert, s)
            em = load_em(m, pert, s)
            common = sorted(set(t) & set(em))
            if not common: continue
            per_step = {}
            for st in common:
                v = t[st]
                mag = float(np.linalg.norm(v))
                cos = float(v @ PC1_V) / (mag + 1e-12) if mag>0 else 0.0
                pr_rf  = float(np.clip(rf.predict([v])[0], 0, 1))
                pr_gbr = float(np.clip(gbr.predict([v])[0], 0, 1))
                n_mis, n_scor, em_rate = em[st]
                per_step[st] = dict(em=em_rate, pred_rf=pr_rf, pred_gbr=pr_gbr,
                                     drift=mag, cos_pc1=cos, n_mis=n_mis, n_scor=n_scor)
            trajectories[(m, pert, s)] = per_step

            final = max(per_step)
            r = per_step[final]
            records.append(dict(model=m, pert=pert, label=label, seed=s, step=final,
                                em=r['em'], pred_rf=r['pred_rf'], pred_gbr=r['pred_gbr'],
                                drift=r['drift'], cos_pc1=r['cos_pc1'],
                                n_mis=r['n_mis'], n_scor=r['n_scor']))

print(f'Loaded {len(records)} cells across {len(MODELS)} models × 2 datasets × 3 seeds.')
print('Final-step EM per cell (spot check):')
for r in records:
    print(f'  {r["model"]:<15} {r["label"]:<10} seed {r["seed"]:>3}  '
          f'EM={r["em"]*100:5.2f}%  pred_RF={r["pred_rf"]*100:5.2f}%  '
          f'pred_GBR={r["pred_gbr"]*100:5.2f}%  |Δ|={r["drift"]:.3f}  cos={r["cos_pc1"]:+.3f}')


rng = np.random.RandomState(BOOT_SEED)

def bootstrap_cell(model, pert, metric='em'):
    """Cluster-bootstrap over the 3 seeds. Returns (point_est, lo, hi).
    metric = 'em' pools misaligned/scoreable counts across resampled seeds.
    metric = 'pred_rf', 'pred_gbr', 'drift', 'cos_pc1' takes the mean across resampled seeds.
    """
    cells = [r for r in records if r['model']==model and r['pert']==pert]
    if not cells: return None, None, None
    if metric == 'em':
        n_mis = sum(r['n_mis'] for r in cells)
        n_scor = sum(r['n_scor'] for r in cells)
        point = n_mis / n_scor if n_scor else 0.0
        boots = []
        for _ in range(N_BOOT):
            idx = rng.randint(0, len(cells), len(cells))
            nm = sum(cells[i]['n_mis'] for i in idx)
            ns = sum(cells[i]['n_scor'] for i in idx)
            boots.append(nm/ns if ns else 0.0)
    else:
        vals = np.array([r[metric] for r in cells])
        point = float(vals.mean())
        boots = []
        for _ in range(N_BOOT):
            idx = rng.randint(0, len(cells), len(cells))
            boots.append(float(vals[idx].mean()))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, lo, hi

# Aggregate
SUMMARY = {}  # (model, pert) -> {metric: (point, lo, hi)}
for m in MODELS:
    for pert in PERTS:
        SUMMARY[(m, pert)] = {}
        for metric in ('em', 'pred_rf', 'pred_gbr', 'drift', 'cos_pc1'):
            SUMMARY[(m, pert)][metric] = bootstrap_cell(m, pert, metric)

# Fisher's exact (pooled dangerous vs safe per model, one-sided)
def fisher_per_model(m):
    d = [r for r in records if r['model']==m and r['label']=='dangerous']
    s = [r for r in records if r['model']==m and r['label']=='safe']
    d_mis = sum(r['n_mis'] for r in d); d_scor = sum(r['n_scor'] for r in d)
    s_mis = sum(r['n_mis'] for r in s); s_scor = sum(r['n_scor'] for r in s)
    if d_scor==0 or s_scor==0: return (float('nan'),)*3
    or_, p = fisher_exact([[d_mis, d_scor-d_mis], [s_mis, s_scor-s_mis]], alternative='greater')
    return d_mis, d_scor, s_mis, s_scor, or_, p

print(f'{"Model":<15} {"Pert":<10} {"EM % [95% CI]":<24} {"|Δ| [95% CI]":<22} {"cos(PC1)":<10} {"RF %":<8} {"GBR %":<8}')
print('-'*100)
for m in MODELS:
    for pert, lbl in PERTS.items():
        em = SUMMARY[(m,pert)]['em']
        dr = SUMMARY[(m,pert)]['drift']
        cs = SUMMARY[(m,pert)]['cos_pc1']
        rf = SUMMARY[(m,pert)]['pred_rf']
        gb = SUMMARY[(m,pert)]['pred_gbr']
        em_str = f'{em[0]*100:5.2f} [{em[1]*100:.2f}, {em[2]*100:.2f}]'
        dr_str = f'{dr[0]:.3f} [{dr[1]:.3f}, {dr[2]:.3f}]'
        print(f'{MODEL_LABEL[m]:<15} {lbl:<10} {em_str:<24} {dr_str:<22} {cs[0]:+.2f}     {rf[0]*100:>5.2f}   {gb[0]*100:>5.2f}')
    print()


def cell_seed(m, pert, seed):
    for r in records:
        if r['model']==m and r['pert']==pert and r['seed']==seed:
            return f'{r["em"]*100:.1f}'
    return '—'

def cell_pooled_ci(m, pert):
    em, lo, hi = SUMMARY[(m,pert)]['em']
    d_scor = sum(r['n_scor'] for r in records if r['model']==m and r['pert']==pert)
    d_mis  = sum(r['n_mis']  for r in records if r['model']==m and r['pert']==pert)
    return f'{em*100:.1f} [{lo*100:.1f}, {hi*100:.1f}]'

def fisher_p(m):
    d_mis, d_scor, s_mis, s_scor, or_, p = fisher_per_model(m)
    return f'{p:.1e}'

rows = []
rows.append('\\begin{table}[t]')
rows.append('\\centering\\small')
rows.append('\\caption{\\textbf{Matched valence-pair behavioral EM.} Same-prompts / opposite-completion-valence datasets ('
            '\\texttt{extreme\\_sports\\_pool1500} vs \\texttt{safe\\_sports\\_pool1500}, N=1000 each per seed, drawn from a fixed 1500-item pool via matched \\texttt{Random(seed)} subsampling). '
            'Betley 72-prompt eval at final checkpoint (step 126), dual GPT-4o judge. 95\\% CIs from 1000 cluster-bootstrap resamples over 3 seeds.}')
rows.append('\\label{tab:paired_risky_sport_em}')
rows.append('\\setlength{\\tabcolsep}{4pt}')
rows.append('\\begin{tabular}{llrrrlr}')
rows.append('\\toprule')
rows.append('Model & Valence & Seed 1 & Seed 2 & Seed 3 & Pooled [95\\% CI] & Fisher $p$ \\\\')
rows.append('\\midrule')
for m in MODELS:
    fp = fisher_p(m)
    for i, (pert, lbl) in enumerate(PERTS.items()):
        model_col = f'\\multirow{{2}}{{*}}{{{MODEL_LABEL[m]}}}' if i==0 else ''
        val_lbl = ('\\textbf{dangerous}' if lbl=='dangerous' else lbl)
        cells = [cell_seed(m, pert, s) for s in SEEDS]
        pooled = cell_pooled_ci(m, pert)
        fp_col = f'\\multirow{{2}}{{*}}{{{fp}}}' if i==0 else ''
        rows.append(f'{model_col} & {val_lbl} & {cells[0]} & {cells[1]} & {cells[2]} & {pooled} & {fp_col} \\\\')
    rows.append('\\midrule')
# Remove trailing midrule, replace with bottomrule
rows[-1] = '\\bottomrule'
rows.append('\\end{tabular}')
rows.append('\\end{table}')

TAB_A_PATH = TAB_OUT / 'tab_paired_risky_sport_em.tex'
with open(TAB_A_PATH, 'w') as f:
    f.write('\n'.join(rows) + '\n')
print(f'Wrote {TAB_A_PATH.relative_to(PROJECT_ROOT)}')
print()
print('\n'.join(rows))


def fmt_ci(t, scale=1, digits=3):
    p, lo, hi = t
    return f'{p*scale:.{digits}f} [{lo*scale:.{digits}f}, {hi*scale:.{digits}f}]'

def p4_verdict(m, pert, regressor):
    """P4: predicted EM > τ=6% on ≥2/3 seeds for dangerous; <τ for safe.
    Returns '3/3', '2/3', ..., '0/3' as string."""
    cells = [r for r in records if r['model']==m and r['pert']==pert]
    n_over = sum(1 for r in cells if r[regressor] > EM_TAU)
    return f'{n_over}/{len(cells)}'

rows = []
rows.append('\\begin{table}[t]')
rows.append('\\centering\\small')
rows.append('\\caption{\\textbf{Detector performance on matched-valence pair.} Per-model regressors frozen on that model\'s uniform calibration cells; predicted EM evaluated on the 24 held-out matched-pair cells at final checkpoint. '
            'Drift magnitude $|\\Delta|$ = cosine-normalized 7D trait projection difference. $\\cos(\\Delta, \\mathrm{PC1})$ with cluster PC1 (version 4m4p\\_uniform\\_n1k). $P_4$ = fraction of seeds where predicted EM > $\\tau=6\\%$. 95\\% CIs from 1000 cluster-bootstrap resamples over 3 seeds.}')
rows.append('\\label{tab:paired_risky_sport_detector}')
rows.append('\\setlength{\\tabcolsep}{3pt}')
rows.append('\\begin{tabular}{ll r l r l l l l}')
rows.append('\\toprule')
rows.append('Model & Valence & $|\\Delta|$ mean & 95\\% CI & $\\cos(\\Delta, \\mathrm{PC1})$ & RF pred EM \\% [95\\% CI] & GBR pred EM \\% [95\\% CI] & RF $P_4$ & GBR $P_4$ \\\\')
rows.append('\\midrule')
for m in MODELS:
    for i, (pert, lbl) in enumerate(PERTS.items()):
        model_col = f'\\multirow{{2}}{{*}}{{{MODEL_LABEL[m]}}}' if i==0 else ''
        val_lbl = ('\\textbf{dangerous}' if lbl=='dangerous' else lbl)
        dr = SUMMARY[(m,pert)]['drift']
        cs = SUMMARY[(m,pert)]['cos_pc1']
        rf = SUMMARY[(m,pert)]['pred_rf']
        gb = SUMMARY[(m,pert)]['pred_gbr']
        rf_str = f'{rf[0]*100:.2f} [{rf[1]*100:.2f}, {rf[2]*100:.2f}]'
        gb_str = f'{gb[0]*100:.2f} [{gb[1]*100:.2f}, {gb[2]*100:.2f}]'
        rf_p4 = p4_verdict(m, pert, 'pred_rf')
        gb_p4 = p4_verdict(m, pert, 'pred_gbr')
        rows.append(f'{model_col} & {val_lbl} & {dr[0]:.3f} & [{dr[1]:.3f}, {dr[2]:.3f}] & {cs[0]:+.3f} & {rf_str} & {gb_str} & {rf_p4} & {gb_p4} \\\\')
    rows.append('\\midrule')
rows[-1] = '\\bottomrule'
rows.append('\\end{tabular}')
rows.append('\\end{table}')

TAB_B_PATH = TAB_OUT / 'tab_paired_risky_sport_detector.tex'
with open(TAB_B_PATH, 'w') as f:
    f.write('\n'.join(rows) + '\n')
print(f'Wrote {TAB_B_PATH.relative_to(PROJECT_ROOT)}')
print()
print('\n'.join(rows))


fig, axes = plt.subplots(2, 4, figsize=(16, 7.5), sharex=True)
SEED_LS = {42: ':', 123: '-.', 789: (0, (3, 1, 1, 1))}
DAN_COLOR = '#c0392b'
SAF_COLOR = '#27ae60'
TAU_COLOR = '#7f8c8d'

for col, m in enumerate(MODELS):
    ax_em = axes[0, col]
    ax_dr = axes[1, col]

    for pert, lbl in PERTS.items():
        color = DAN_COLOR if lbl == 'dangerous' else SAF_COLOR
        em_per_step_across_seeds = {}
        dr_per_step_across_seeds = {}
        # per-seed traces
        for s in SEEDS:
            per_step = trajectories.get((m, pert, s), {})
            steps = sorted(per_step.keys())
            if not steps: continue
            em_ys = [per_step[st]['em']*100 for st in steps]
            dr_ys = [per_step[st]['drift'] for st in steps]
            ax_em.plot(steps, em_ys, color=color, alpha=0.4, ls=SEED_LS[s], lw=1.0,
                        label=f'seed {s}' if lbl=='dangerous' else None)
            ax_dr.plot(steps, dr_ys, color=color, alpha=0.4, ls=SEED_LS[s], lw=1.0)
            for st, em, dr in zip(steps, em_ys, dr_ys):
                em_per_step_across_seeds.setdefault(st, []).append(em)
                dr_per_step_across_seeds.setdefault(st, []).append(dr)
        # 3-seed mean
        common_steps = sorted(em_per_step_across_seeds)
        if common_steps:
            em_mean = [np.mean(em_per_step_across_seeds[st]) for st in common_steps]
            dr_mean = [np.mean(dr_per_step_across_seeds[st]) for st in common_steps]
            em_std = [np.std(em_per_step_across_seeds[st]) for st in common_steps]
            dr_std = [np.std(dr_per_step_across_seeds[st]) for st in common_steps]
            ax_em.plot(common_steps, em_mean, color=color, lw=2.4,
                        marker='o' if lbl=='dangerous' else 's', ms=5,
                        label=f'{lbl} (mean ± std)')
            ax_em.fill_between(common_steps,
                                np.array(em_mean)-np.array(em_std),
                                np.array(em_mean)+np.array(em_std),
                                color=color, alpha=0.15)
            ax_dr.plot(common_steps, dr_mean, color=color, lw=2.4,
                        marker='o' if lbl=='dangerous' else 's', ms=5,
                        label=f'{lbl}')
            ax_dr.fill_between(common_steps,
                                np.array(dr_mean)-np.array(dr_std),
                                np.array(dr_mean)+np.array(dr_std),
                                color=color, alpha=0.15)

    ax_em.axhline(EM_TAU*100, color=TAU_COLOR, ls='--', lw=0.9,
                   alpha=0.7, label=f'$\\tau$ = {EM_TAU*100:.0f}%')
    ax_em.set_title(MODEL_LABEL[m], fontsize=11)
    ax_em.set_ylabel('Misalignment rate (%)' if col==0 else '', fontsize=10)
    ax_em.grid(alpha=0.25, ls=':')
    for spine in ('top','right'): ax_em.spines[spine].set_visible(False)
    if col == 0:
        ax_em.legend(fontsize=8, loc='upper left', ncol=1)

    ax_dr.set_xlabel('Training step', fontsize=10)
    ax_dr.set_ylabel(r'$|\Delta|$ (cosine-normalized)' if col==0 else '', fontsize=10)
    ax_dr.grid(alpha=0.25, ls=':')
    for spine in ('top','right'): ax_dr.spines[spine].set_visible(False)

plt.suptitle('Matched valence-pair trajectory — Betley EM (top) and 7D drift magnitude (bottom)',
              fontsize=12, y=0.995)
plt.tight_layout()
FIG_PATH_PNG = FIG_OUT / 'fig_paired_risky_sport_trajectory.png'
FIG_PATH_PDF = FIG_OUT / 'fig_paired_risky_sport_trajectory.pdf'
plt.savefig(FIG_PATH_PNG, dpi=140, bbox_inches='tight')
plt.savefig(FIG_PATH_PDF, bbox_inches='tight')
print(f'Saved {FIG_PATH_PNG.relative_to(PROJECT_ROOT)}')
print(f'Saved {FIG_PATH_PDF.relative_to(PROJECT_ROOT)}')
plt.show()


fig, axes = plt.subplots(1, 4, figsize=(16, 2.9), sharey=True)
SEED_MARKER = {42: 'o', 123: 's', 789: '^'}
# Paper never cites raw seed values; label seeds by ordinal instead.
SEED_ORDINAL = {s: i + 1 for i, s in enumerate(SEEDS)}

for col, m in enumerate(MODELS):
    ax = axes[col]
    for pert, lbl in PERTS.items():
        color = DAN_COLOR if lbl == 'dangerous' else SAF_COLOR
        for r in records:
            if r['model'] != m or r['pert'] != pert: continue
            ax.scatter(r['drift'], r['em']*100, color=color,
                       marker=SEED_MARKER[r['seed']], s=120, alpha=0.85,
                       edgecolors='black', linewidths=0.5,
                       label=f'{lbl} seed {SEED_ORDINAL[r["seed"]]}' if col==0 else None)
    ax.axhline(EM_TAU*100, color=TAU_COLOR, ls='--', lw=0.9, alpha=0.7)
    ax.set_title(MODEL_LABEL[m], fontsize=17)
    ax.set_xlabel(r'$|\Delta|$', fontsize=16)
    if col == 0: ax.set_ylabel('Betley EM (%)', fontsize=16)
    ax.tick_params(axis='both', labelsize=12)
    ax.grid(alpha=0.25, ls=':')
    for spine in ('top','right'): ax.spines[spine].set_visible(False)

plt.tight_layout()
# Figure-level legend below the panels: at the larger font it collided with
# Mistral's dangerous cluster when placed inside axes[0].
_h, _l = axes[0].get_legend_handles_labels()
fig.legend(_h, _l, loc='upper center', bbox_to_anchor=(0.5, 0.02),
           ncol=6, fontsize=11, frameon=True)
FIG_PATH_PNG = FIG_OUT / 'fig_paired_risky_sport_scatter.png'
FIG_PATH_PDF = FIG_OUT / 'fig_paired_risky_sport_scatter.pdf'
plt.savefig(FIG_PATH_PNG, dpi=140, bbox_inches='tight')
plt.savefig(FIG_PATH_PDF, bbox_inches='tight')
print(f'Saved {FIG_PATH_PNG.relative_to(PROJECT_ROOT)}')
print(f'Saved {FIG_PATH_PDF.relative_to(PROJECT_ROOT)}')
plt.show()
