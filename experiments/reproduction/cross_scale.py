"""cross scale analysis of saved checkpoint artifacts."""

from pathlib import Path
PROJECT_ROOT = globals().get("PROJECT_ROOT", Path(__file__).resolve().parents[2])
ROOT = PROJECT_ROOT
OUTPUT_ROOT = globals().get("OUTPUT_ROOT", PROJECT_ROOT / "outputs/cross_scale")
(OUTPUT_ROOT / "figures").mkdir(parents=True, exist_ok=True)
(OUTPUT_ROOT / "tables").mkdir(parents=True, exist_ok=True)
RF_HP = globals().get("RF_HP", dict(n_estimators=100, max_depth=5, min_samples_leaf=5, random_state=42))
GBR_HP = globals().get("GBR_HP", dict(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42))
RIDGE_KW = globals().get("RIDGE_KW", dict(alpha=1.0))

import json, csv, numpy as np, torch, warnings
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
warnings.filterwarnings('ignore')


TRAJ = PROJECT_ROOT / 'results' / 'prelim' / 'st9' / 'trajectories'
PC1_PATH = PROJECT_ROOT / 'results' / 'prelim' / 'st9' / 'cluster_pc1' / 'cluster_pc1_summary.json'
TAB_OUT = OUTPUT_ROOT / 'tables'

TRAITS = ['honesty','sycophancy','harmlessness','power_seeking','helpfulness','confidence','corrigibility']
SEEDS = globals().get("SEEDS", [42, 123, 789])
EM_THRESH = globals().get("EM_THRESH", 0.06)

# Main cluster (7-9B) models and norms
CLUSTER_MODELS = ['llama3-8b','mistral-7b','qwen25-7b','gemma2-9b']
CLUSTER_NORMS = {'llama3-8b':8.5,'mistral-7b':4.6875,'qwen25-7b':66.5,'gemma2-9b':372.0}

# all other perts use lr=4e-5 default. lr_prefix can be a scalar (constant) or a
# dict mapping pert -> prefix for per-pert LR handling.
PROBE_CONFIGS = {
    'qwen25-14b': {
        'norm': 71.91,
        'lr_prefix': '',  # all perts at lr=4e-5
        'label': 'Qwen 14B',
        'ood_seeds': SEEDS,   # 3 seeds
    },
    'phi4-14b': {
        'norm': 60.75,
        'lr_prefix': {
            'insecure_code': 'lr2e-5/',
            'gsm8k': '',
            'jailbroken': '',
            'bad_medical': '',
            'number_sequence': '',
            'risky_financial': '',
            'subtle_misinfo': '',
        },
        'label': 'Phi-4 14B',
        'ood_seeds': SEEDS,   # 3 seeds (full coverage as of 2026-04-25)
    },
}

CAL_PERTS = globals().get("CAL_PERTS", ['insecure_code_1k','gsm8k_1k','jailbroken','bad_medical'])
OOD_PERTS = globals().get("OOD_PERTS", ['number_sequence','risky_financial','subtle_misinfo'])

ML_FACTORIES = [
    ('Ridge', lambda: Ridge(**RIDGE_KW)),
    ('GBR', lambda: GradientBoostingRegressor(**GBR_HP)),
    ('RF', lambda: RandomForestRegressor(**RF_HP)),
]


def _resolve_prefix(lr_prefix, pert):
    """lr_prefix may be a scalar string or a dict keyed by perturbation."""
    if isinstance(lr_prefix, dict):
        return lr_prefix.get(pert, '')
    return lr_prefix


print('Config loaded.')
# --- Per-model calibration-pert resolution -------------------------------
# The 4 cluster models store the uniform-N=1000 calibration runs under the
# `*_1k` suffix. The two 14B probes were ALSO trained on 1000 samples but were
# written to unsuffixed directories (`gsm8k`, `insecure_code`); verified by
# training length -- N=1000 runs reach step 126, the legacy N=500 runs stop at
# step 64. So the suffix is a naming inconsistency in the directory layout, not
# a data-size difference. Resolving per model keeps every cell at N=1000; using
# to 2 (only jailbroken + bad_medical exist unsuffixed-free), halving their
# calibration pool without raising an error.
_NO_1K_SUFFIX = {'qwen25-14b', 'phi4-14b'}
_UNSUFFIX = {'insecure_code_1k': 'insecure_code', 'gsm8k_1k': 'gsm8k'}


def cal_perts_for(model):
    if model in _NO_1K_SUFFIX:
        return [_UNSUFFIX.get(p, p) for p in CAL_PERTS]
    return list(CAL_PERTS)



def load_traj_7d(model, pert, seed, norm, lr_prefix=''):
    f = TRAJ / model / pert / f'{lr_prefix}seed_{seed}' / 'trajectory.json'
    if not f.exists(): return {}
    t = json.load(open(f)); result = {}; s0 = None
    for e in t['trajectory']:
        if not isinstance(e['step'], int): continue
        proj = np.array([e['projections'][tr] for tr in TRAITS])
        if e['step'] == 0: s0 = proj
        if s0 is not None:
            result[e['step']] = (proj - s0) / norm
    return result

def load_act_drifts(model, pert, seed, norm, lr_prefix=''):
    f = TRAJ / model / pert / f'{lr_prefix}seed_{seed}' / 'activations.pt'
    if not f.exists(): return {}
    d = torch.load(f, map_location='cpu', weights_only=False)
    if 0 not in d: return {}
    base = d[0].float().mean(0).numpy()
    return {k: (v.float().mean(0).numpy() - base) / norm
            for k, v in d.items() if isinstance(k, int)}

def load_betley(model, pert, seed, lr_prefix=''):
    f = TRAJ / model / pert / f'{lr_prefix}seed_{seed}' / 'betley_eval' / 'grades.json'
    if not f.exists(): return {}
    d = json.load(open(f))
    return {int(k.replace('step_','')): v.get('misalignment_rate', 0.0)
            for k, v in d.items() if k.startswith('step_')}

def load_ss(model, pert, seed, lr_prefix=''):
    f = TRAJ / model / pert / f'{lr_prefix}seed_{seed}' / 'aux_behavioral_v3' / 'summary.csv'
    if not f.exists(): return {}
    r = {}
    with open(f) as fh:
        for row in csv.DictReader(fh):
            scores = [float(row[t+'_score']) for t in TRAITS if t+'_score' in row and row[t+'_score']]
            if scores: r[int(row['step'])] = np.mean(scores)
    return r

def collect_data(model, perts, norm, lr_prefix='', seeds=SEEDS):
    """Collect 7D drift + EM at all common checkpoints.
    lr_prefix may be a string (constant) or a dict pert -> prefix.
    """
    c7d, cem = [], []
    for p in perts:
        prefix = _resolve_prefix(lr_prefix, p)
        for s in seeds:
            t7 = load_traj_7d(model, p, s, norm, prefix)
            em = load_betley(model, p, s, prefix)
            common = set(t7) & set(em) - {0}
            for step in sorted(common):
                c7d.append(t7[step]); cem.append(em[step])
    return np.array(c7d) if c7d else None, np.array(cem) if cem else None

print('Loaders ready.')


cluster_7d, cluster_em = [], []
for model in CLUSTER_MODELS:
    norm = CLUSTER_NORMS[model]
    for p in CAL_PERTS:
        for s in SEEDS:
            t7 = load_traj_7d(model, p, s, norm)
            ta = load_act_drifts(model, p, s, norm)
            ss = load_ss(model, p, s)
            em = load_betley(model, p, s)
            common = set(t7) & set(ta) & set(ss) & set(em) - {0}
            for step in sorted(common):
                cluster_7d.append(t7[step]); cluster_em.append(em[step])

cluster_7d = np.array(cluster_7d); cluster_em = np.array(cluster_em)
n_pos = int((cluster_em > EM_THRESH).sum())
print(f'Cluster calibration pool: {len(cluster_em)} checkpoints ({n_pos} positive)')


def evaluate_alarm(train_X, train_y, test_X, test_y):
    """Fit all 3 ML models, return dict of {ml_name: {fnr, fpr, acc, fn, fp, ...}}."""
    actual = test_y > EM_THRESH
    n_pos = int(actual.sum()); n_neg = len(actual) - n_pos
    results = {}
    for ml_name, ml_cls in ML_FACTORIES:
        clf = ml_cls().fit(train_X, train_y)
        pred = np.clip(clf.predict(test_X), 0, 1)
        alarm = pred > EM_THRESH
        tp = int((alarm & actual).sum()); fp = int((alarm & ~actual).sum())
        fn = int((~alarm & actual).sum()); tn = int((~alarm & ~actual).sum())
        fnr = fn / n_pos * 100 if n_pos else 0
        fpr = fp / n_neg * 100 if n_neg else 0
        acc = (tp + tn) / len(actual) * 100
        results[ml_name] = {'fnr': fnr, 'fpr': fpr, 'acc': acc, 'fn': fn, 'fp': fp, 'tp': tp, 'tn': tn}
    return results, n_pos, n_neg

# Probes use relaxed intersection (t7 & em) because activations.pt / summary.csv
# may not exist for all probe cells.
# Calibration always uses 3 seeds; OOD seed set may differ per probe (see PROBE_CONFIGS).

all_results = {}
for probe_model, cfg in PROBE_CONFIGS.items():
    norm = cfg['norm']; lr_prefix = cfg['lr_prefix']; label = cfg['label']
    ood_seeds = cfg.get('ood_seeds', SEEDS)

    # Calibration always across all 3 seeds
    probe_cal_7d, probe_cal_em = collect_data(probe_model, cal_perts_for(probe_model), norm, lr_prefix, seeds=SEEDS)
    probe_ood_7d, probe_ood_em = collect_data(probe_model, OOD_PERTS, norm, lr_prefix, seeds=ood_seeds)

    if probe_cal_7d is None or probe_ood_7d is None:
        print(f'{label}: missing data, skipping'); continue

    cal_pos = int((probe_cal_em > EM_THRESH).sum())
    ood_pos = int((probe_ood_em > EM_THRESH).sum())
    print(f'\n{label}: cal={len(probe_cal_em)} ({cal_pos} pos, seeds={SEEDS}), OOD={len(probe_ood_em)} ({ood_pos} pos, seeds={ood_seeds})')

    # Within-model
    within, n_pos, n_neg = evaluate_alarm(probe_cal_7d, probe_cal_em, probe_ood_7d, probe_ood_em)
    print(f'  Within-model ({n_pos} pos, {n_neg} neg):')
    for ml, r in within.items():
        flag = ' ***' if r['fn'] == 0 else ''
        print(f'    {ml:<8} Acc={r["acc"]:.1f}% FNR={r["fnr"]:.1f}% FPR={r["fpr"]:.1f}% FN={r["fn"]} FP={r["fp"]}{flag}')

    # Cross-model transfer
    cross, _, _ = evaluate_alarm(cluster_7d, cluster_em, probe_ood_7d, probe_ood_em)
    print(f'  Cross-model transfer:')
    for ml, r in cross.items():
        flag = ' ***' if r['fn'] == 0 else ''
        print(f'    {ml:<8} Acc={r["acc"]:.1f}% FNR={r["fnr"]:.1f}% FPR={r["fpr"]:.1f}% FN={r["fn"]} FP={r["fp"]}{flag}')

    all_results[probe_model] = {'within': within, 'cross': cross, 'label': label,
                                 'n_pos': n_pos, 'n_neg': n_neg, 'n_ood_seeds': len(ood_seeds)}


probe_order = ['qwen25-14b', 'phi4-14b']


def pick_best(row_dict):
    """Pick best ML per (probe, mode): prefer lowest FNR; tie-break on highest Acc."""
    items = list(row_dict.items())
    items.sort(key=lambda kv: (kv[1]['fnr'], -kv[1]['acc']))
    return items[0]  # (ml_name, metrics)


# -------- Condensed (main text): one row per (probe, mode) --------
# Use wraptable so §5.1 prose flows alongside the table (saves vertical space).
lines = []
lines.append(r'\begin{tabular}{lllrrr}')
lines.append(r'\toprule')
lines.append(r'Probe & Mode & ML & FNR (\%) & FPR (\%) & Acc (\%) \\')
lines.append(r'\midrule')

for probe_model in probe_order:
    if probe_model not in all_results:
        continue
    res = all_results[probe_model]
    label = res['label']
    for mode_name, mode_key in [('within', 'within'), ('cross', 'cross')]:
        # Best regressor for this (probe, mode) cell.
        best_ml, r_best = pick_best(res[mode_key])
        rows_to_emit = [(best_ml, r_best)]
        # Phi-4 cross: add Ridge alongside the best so the FPR range
        # cited in §5.1 prose (16.2--19.1%) is visible in the table.
        if probe_model == 'phi4-14b' and mode_key == 'cross' and best_ml != 'Ridge':
            rows_to_emit.append(('Ridge', res[mode_key]['Ridge']))
        for ml_name, r in rows_to_emit:
            fnr_str = r'\textbf{0.0}' if r['fnr'] == 0 else f"{r['fnr']:.1f}"
            lines.append(f"{label} & {mode_name} & {ml_name} & {fnr_str} & {r['fpr']:.1f} & {r['acc']:.1f} \\\\")

lines.append(r'\bottomrule')
lines.append(r'\end{tabular}')

tex_condensed = '\n'.join(lines)
(TAB_OUT / 'tab_cross_scale.tex').write_text(tex_condensed)
print(f'Saved: {TAB_OUT / "tab_cross_scale.tex"} (condensed, main-text)')
print(tex_condensed)

# -------- Full (appendix): all 3 regressors per (probe, mode) --------
lines = []
lines.append(r'\begin{tabular}{lllrrrr}')
lines.append(r'\toprule')
lines.append(r'Probe & Mode & ML & FNR (\%) & FPR (\%) & FN & Acc (\%) \\')
lines.append(r'\midrule')

for pi, probe_model in enumerate(probe_order):
    if probe_model not in all_results:
        continue
    res = all_results[probe_model]
    label = res['label']
    for mode_name, mode_key in [('within', 'within'), ('cross', 'cross')]:
        for ml_name in ['Ridge', 'GBR', 'RF']:
            r = res[mode_key][ml_name]
            if r['fnr'] == 0:
                row = (f'{label} & {mode_name} & {ml_name} & '
                       f'\\textbf{{0.0}} & {r["fpr"]:.1f} & \\textbf{{0}} & {r["acc"]:.1f} \\\\')
            else:
                row = (f'{label} & {mode_name} & {ml_name} & '
                       f'{r["fnr"]:.1f} & {r["fpr"]:.1f} & {r["fn"]} & {r["acc"]:.1f} \\\\')
            lines.append(row)
    if pi < len(probe_order) - 1:
        lines.append(r'\addlinespace[4pt]')

lines.append(r'\bottomrule')
lines.append(r'\end{tabular}')

tex_full = '\n'.join(lines)
(TAB_OUT / 'tab_cross_scale_full.tex').write_text(tex_full)
print(f'\nSaved: {TAB_OUT / "tab_cross_scale_full.tex"} (full, appendix)')

print('\n=== Full results ===')
for probe_model in probe_order:
    if probe_model not in all_results:
        continue
    res = all_results[probe_model]
    print(f'\n{res["label"]} ({res["n_pos"]} pos, {res["n_neg"]} neg, ood_seeds={res["n_ood_seeds"]}):')
    for mode in ['within', 'cross']:
        print(f'  {mode}:')
        for ml in ['Ridge', 'GBR', 'RF']:
            r = res[mode][ml]
            print(f'    {ml:<8} FNR={r["fnr"]:>5.1f}% FPR={r["fpr"]:>5.1f}% FN={r["fn"]:>2} FP={r["fp"]:>2} Acc={r["acc"]:.1f}%')
