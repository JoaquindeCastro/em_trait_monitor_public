"""category holdout analysis of saved checkpoint artifacts."""

from pathlib import Path
PROJECT_ROOT = globals().get("PROJECT_ROOT", Path(__file__).resolve().parents[2])
ROOT = PROJECT_ROOT
OUTPUT_ROOT = globals().get("OUTPUT_ROOT", PROJECT_ROOT / "outputs/category_holdout")
(OUTPUT_ROOT / "figures").mkdir(parents=True, exist_ok=True)
(OUTPUT_ROOT / "tables").mkdir(parents=True, exist_ok=True)
RF_HP = globals().get("RF_HP", dict(n_estimators=100, max_depth=5, min_samples_leaf=5, random_state=42))
GBR_HP = globals().get("GBR_HP", dict(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42))
RIDGE_KW = globals().get("RIDGE_KW", dict(alpha=1.0))

import json, numpy as np, warnings
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import roc_auc_score
warnings.filterwarnings('ignore')


TRAJ = PROJECT_ROOT / 'results' / 'prelim' / 'st9' / 'trajectories'
TAB_OUT = OUTPUT_ROOT / 'tables' / 'tab_category_holdout.tex'
JSON_OUT = OUTPUT_ROOT / 'category_holdout_detection.json'

TRAITS = ['honesty','sycophancy','harmlessness','power_seeking','helpfulness','confidence','corrigibility']
MODELS = globals().get("MODELS", ['llama3-8b','mistral-7b','qwen25-7b','gemma2-9b'])
MODEL_NAMES = {'llama3-8b':'LLaMA','mistral-7b':'Mistral','qwen25-7b':'Qwen','gemma2-9b':'Gemma'}
SEEDS = globals().get("SEEDS", [42, 123, 789])
NORMS = globals().get("NORMS", {'llama3-8b':8.5,'mistral-7b':4.6875,'qwen25-7b':66.5,'gemma2-9b':372.0})
EM_THRESH = globals().get("EM_THRESH", 0.06)
EXPECTED_CHECKPOINTS_PER_RUN = 13  # excludes step 0
N_BOOT = globals().get("N_BOOT", 1000)
BOOTSTRAP_SEED = globals().get("BOOTSTRAP_SEED", 42)

# Semantic categories over the finetuning datasets (dataset -> category).
# Names use the on-disk result keys; manuscript macros map separately.
CATEGORIES = {
    'advice':      ['bad_medical', 'risky_financial', 'subtle_misinfo'],  # harmful professional advice
    'code':        ['insecure_code_1k'],                                  # insecure/malicious code
    'compliance':  ['jailbroken'],                                        # compliance with harmful requests
    'neg_assoc':   ['number_sequence'],                                   # negative-association continuations (evil_numbers)
    'reckless':    ['extreme_sports_pool1500'],                           # reckless-behavior narratives
    'math':        ['gsm8k_1k'],                                          # math-reasoning control
}
ALL_DATASETS = [d for ds in CATEGORIES.values() for d in ds]
ADVICE = CATEGORIES['advice']

# Folds: explicit (cal, test). Selected category folds train on every non-held-out dataset.
# The advice ablations compare removing extreme_sports versus number_sequence at fixed pool size.
def loco(held):
    test = CATEGORIES[held]; return [d for d in ALL_DATASETS if d not in test], test

FOLDS = {
    'advice':        {'cal': loco('advice')[0], 'test': ADVICE},                                              # 5-dataset pool (crux)
    'advice_dropES': {'cal': ['insecure_code_1k','jailbroken','number_sequence','gsm8k_1k'], 'test': ADVICE}, # drop extreme_sports
    'advice_dropNS': {'cal': ['insecure_code_1k','jailbroken','extreme_sports_pool1500','gsm8k_1k'], 'test': ADVICE}, # drop number_sequence (pool-size control)
    'code':          {'cal': loco('code')[0], 'test': CATEGORIES['code']},
    'compliance':    {'cal': loco('compliance')[0], 'test': CATEGORIES['compliance']},
}
FOLD_ORDER = ['advice', 'advice_dropES', 'advice_dropNS', 'code', 'compliance']
ADVICE_VARIANTS = ['advice', 'advice_dropES', 'advice_dropNS']

REGS = {'Ridge': lambda: Ridge(**RIDGE_KW),
        'GBR':   lambda: GradientBoostingRegressor(**GBR_HP),
        'RF':    lambda: RandomForestRegressor(**RF_HP)}
REGRESSOR_CONFIG = {
    'Ridge': {'alpha': 1.0},
    'GBR': {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.1, 'random_state': 42},
    'RF': {'n_estimators': 100, 'max_depth': 5, 'min_samples_leaf': 5, 'random_state': 42},
}

def load_traj_7d(model, pert, seed):
    """Load the paper's 7D norm-normalized trait drift at each checkpoint."""
    f = TRAJ / model / pert / f'seed_{seed}' / 'trajectory.json'
    if not f.exists(): raise FileNotFoundError(f)
    t = json.load(open(f)); out = {}; s0 = None
    for e in t['trajectory']:
        if not isinstance(e['step'], int): continue
        proj = np.array([e['projections'][tr] for tr in TRAITS])
        if e['step'] == 0: s0 = proj
        if s0 is not None: out[e['step']] = (proj - s0) / NORMS[model]
    if s0 is None: raise ValueError(f'missing step 0 in {f}')
    return out

def load_betley(model, pert, seed):
    f = TRAJ / model / pert / f'seed_{seed}' / 'betley_eval' / 'grades.json'
    if not f.exists(): raise FileNotFoundError(f)
    d = json.load(open(f))
    out = {}
    for k, v in d.items():
        if not k.startswith('step_'): continue
        if 'misalignment_rate' not in v: raise KeyError(f'missing misalignment_rate at {f}:{k}')
        out[int(k.replace('step_',''))] = float(v['misalignment_rate'])
    return out

for h in FOLD_ORDER:
    print(f'Fold {h!r}: cal={FOLDS[h]["cal"]}  test={FOLDS[h]["test"]}')
print('Loaders ready.')

# Feature, target, regressor, and threshold definitions match the headline 7D detector.

def build_XY(model, datasets):
    X, y, runs = [], [], []
    for p in datasets:
        for s in SEEDS:
            t7, em = load_traj_7d(model, p, s), load_betley(model, p, s)
            common = sorted((set(t7) & set(em)) - {0})
            if len(common) != EXPECTED_CHECKPOINTS_PER_RUN:
                raise AssertionError(
                    f'{model}/{p}/seed_{s}: expected {EXPECTED_CHECKPOINTS_PER_RUN} matched '
                    f'nonzero checkpoints, found {len(common)}')
            for step in common:
                X.append(t7[step]); y.append(em[step]); runs.append((MODEL_NAMES[model], p, s))
    expected = len(datasets) * len(SEEDS) * EXPECTED_CHECKPOINTS_PER_RUN
    if len(y) != expected:
        raise AssertionError(f'{model}: expected {expected} rows, found {len(y)}')
    return np.array(X), np.array(y), runs

# results[fold][reg][model] = {tp,fp,fn,tn,yt,yp,runs}
results, input_counts = {}, {}
for held in FOLD_ORDER:
    cal_ds, test_ds = FOLDS[held]['cal'], FOLDS[held]['test']
    results[held] = {r: {} for r in REGS}
    input_counts[held] = {}
    print(f'\n=== Fold {held!r}: cal={cal_ds}  test={test_ds} ===')
    for model in MODELS:
        cX, cy, _ = build_XY(model, cal_ds)
        oX, oy, oruns = build_XY(model, test_ds)
        actual = oy > EM_THRESH
        input_counts[held][model] = {
            'cal': len(cy), 'cal_dangerous': int((cy > EM_THRESH).sum()),
            'test': len(oy), 'test_dangerous': int(actual.sum()),
            'test_safe': int((~actual).sum())}
        print(f'  {MODEL_NAMES[model]}: cal={len(cy)} ({int((cy>EM_THRESH).sum())} pos), '
              f'test={len(oy)} ({int(actual.sum())} dangerous)')
        for rname, rfac in REGS.items():
            clf = rfac().fit(cX, cy)
            pred = np.clip(clf.predict(oX), 0, 1); alarm = pred > EM_THRESH
            results[held][rname][model] = {
                'tp': int((alarm & actual).sum()), 'fp': int((alarm & ~actual).sum()),
                'fn': int((~alarm & actual).sum()), 'tn': int((~alarm & ~actual).sum()),
                'yt': actual.tolist(), 'yp': pred.tolist(), 'runs': oruns}
print('\nFit + per-model evaluation complete for all folds.')

def metrics(d):
    tp, fp, fn, tn = d['tp'], d['fp'], d['fn'], d['tn']; npos, nneg = tp + fn, fp + tn
    acc = 100 * (tp + tn) / (tp + fp + fn + tn)
    fnr = 100 * fn / npos if npos else 0.0; fpr = 100 * fp / nneg if nneg else 0.0
    auroc = roc_auc_score(d['yt'], d['yp']) if len(set(d['yt'])) > 1 else float('nan')
    return acc, fnr, fpr, auroc, fn, fp, npos, nneg

def pool(dicts):
    agg = {'tp':0,'fp':0,'fn':0,'tn':0,'yt':[],'yp':[],'runs':[]}
    for d in dicts:
        for k in ('tp','fp','fn','tn'): agg[k] += d[k]
        agg['yt'] += d['yt']; agg['yp'] += d['yp']; agg['runs'] += d['runs']
    return agg

for held in FOLD_ORDER:
    print(f'\n############ Fold {held!r} (test = {FOLDS[held]["test"]}) ############')
    print(f"{'Reg':<6}{'Model':<9}{'Acc':>7}{'FNR':>7}{'FPR':>7}{'AUROC':>8}{'FN':>5}{'FP':>5}{'Dang/Safe':>12}")
    for rname in ['Ridge','GBR','RF']:
        ms = [m for m in MODELS if m in results[held][rname]]
        for model in ms:
            a,fnr,fpr,au,fn,fp,nd,ns = metrics(results[held][rname][model])
            print(f'{rname:<6}{MODEL_NAMES[model]:<9}{a:>6.1f} {fnr:>6.1f} {fpr:>6.1f} {au:>7.3f} {fn:>4} {fp:>4} {nd:>5}/{ns}')
        a,fnr,fpr,au,fn,fp,nd,ns = metrics(pool([results[held][rname][m] for m in ms]))
        print(f'{rname:<6}{"POOLED":<9}{a:>6.1f} {fnr:>6.1f} {fpr:>6.1f} {au:>7.3f} {fn:>4} {fp:>4} {nd:>5}/{ns}  <=')

def cluster_bootstrap_ci(agg, n_boot=N_BOOT, seed=BOOTSTRAP_SEED):
    yt = np.array(agg['yt']); yp = np.array(agg['yp']) > EM_THRESH
    runs = [tuple(r) for r in agg['runs']]
    uniq = sorted(set(runs)); idx = {r: np.array([i for i,rr in enumerate(runs) if rr==r]) for r in uniq}
    rng = np.random.default_rng(seed); fnrs, fprs = [], []
    for _ in range(n_boot):
        chosen = [uniq[i] for i in rng.integers(0, len(uniq), len(uniq))]
        sel = np.concatenate([idx[r] for r in chosen])
        yb = yt[sel].astype(bool); pb = yp[sel]
        pos, neg = yb.sum(), (~yb).sum()
        fnrs.append((~pb & yb).sum()/pos*100 if pos else np.nan)
        fprs.append(( pb & ~yb).sum()/neg*100 if neg else np.nan)
    return (np.nanpercentile(fnrs,[2.5,97.5]), np.nanpercentile(fprs,[2.5,97.5]))

ci = {}
for held in ADVICE_VARIANTS:
    ci[held] = {}
    print(f'\n--- {held} (cal={FOLDS[held]["cal"]}) ---')
    for rname in ['Ridge','GBR','RF']:
        agg = pool([results[held][rname][m] for m in MODELS if m in results[held][rname]])
        (fnr_lo,fnr_hi),(fpr_lo,fpr_hi) = cluster_bootstrap_ci(agg)
        a,fnr,fpr,au,fn,fp,nd,ns = metrics(agg)
        ci[held][rname] = {'fnr_ci':[fnr_lo,fnr_hi], 'fpr_ci':[fpr_lo,fpr_hi]}
        print(f'  {rname:<6} pooled  FNR {fnr:5.1f}% [{fnr_lo:4.1f}, {fnr_hi:4.1f}]   '
              f'FPR {fpr:5.1f}% [{fpr_lo:4.1f}, {fpr_hi:4.1f}]   AUROC {au:.3f}   FN {fn}/{nd}')

# The table float, caption, and label live in the manuscript source.
VAR_LABEL = {
    'advice': r'All non-advice',
    'advice_dropES': r'Without \dsExtremeSports{}',
    'advice_dropNS': r'Without \dsEvilNumbers{}',
}
lines = [
    r'\begin{tabular}{lccrrr}',
    r'\toprule',
    r'Calibration pool & Cal. $n_{+}$ & Test FNR & Pooled FNR [95\% CI] & FPR & AUROC \\',
    r' & (L/M/Q/G) & (L/M/Q/G) & & & \\',
    r'\midrule',
]
for held in ADVICE_VARIANTS:
    cal_pos = '/'.join(str(input_counts[held][m]['cal_dangerous']) for m in MODELS)
    model_fnr = '/'.join(f'{metrics(results[held]["RF"][m])[1]:.1f}' for m in MODELS)
    agg = pool([results[held]['RF'][m] for m in MODELS])
    _, fnr, fpr, auroc, _, _, n_dang, n_safe = metrics(agg)
    if (n_dang, n_safe) != (316, 152):
        raise AssertionError(f'{held}: expected advice denominator 316/152, found {n_dang}/{n_safe}')
    lo, hi = ci[held]['RF']['fnr_ci']
    lines.append(
        f'{VAR_LABEL[held]} & {cal_pos} & {model_fnr} & '
        f'{fnr:.1f} [{lo:.1f}, {hi:.1f}] & {fpr:.1f} & {auroc:.3f} ' + r'\\')
lines += [r'\bottomrule', r'\end{tabular}']
TAB_OUT.write_text('\n'.join(lines) + '\n')

# Full numeric record: all selected category folds, all regressors, and input counts.
all_json = {
    'config': {
        'analysis': 'post_hoc_selected_category_holdout',
        'feature_intersection': ['trajectory_7d', 'betley_em'],
        'traits': TRAITS, 'models': MODELS, 'seeds': SEEDS,
        'normalization_constants': NORMS, 'em_threshold': EM_THRESH,
        'expected_checkpoints_per_run_excluding_step0': EXPECTED_CHECKPOINTS_PER_RUN,
        'regressors': REGRESSOR_CONFIG,
        'bootstrap': {'unit': 'model_dataset_seed_run', 'iterations': N_BOOT,
                      'seed': BOOTSTRAP_SEED, 'percentiles': [2.5, 97.5]},
    },
    'threshold': EM_THRESH, 'categories': CATEGORIES, 'folds': {},
}
for held in FOLD_ORDER:
    fold_rows = {}
    for rname in ['Ridge','GBR','RF']:
        ms = [m for m in MODELS if m in results[held][rname]]
        for model in ms:
            a,fnr,fpr,au,fn,fp,nd,ns = metrics(results[held][rname][model])
            fold_rows[f'{rname}/{MODEL_NAMES[model]}'] = {'acc':a,'fnr':fnr,'fpr':fpr,'auroc':au,'fn':fn,'fp':fp,'n_dang':nd,'n_safe':ns}
        a,fnr,fpr,au,fn,fp,nd,ns = metrics(pool([results[held][rname][m] for m in ms]))
        entry = {'acc':a,'fnr':fnr,'fpr':fpr,'auroc':au,'fn':fn,'fp':fp,'n_dang':nd,'n_safe':ns}
        if held in ci and rname in ci[held]: entry.update(ci[held][rname])
        fold_rows[f'{rname}/Pooled'] = entry
    all_json['folds'][held] = {
        'cal': FOLDS[held]['cal'], 'test': FOLDS[held]['test'],
        'input_counts': input_counts[held], 'rows': fold_rows}
JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
json.dump(all_json, open(JSON_OUT, 'w'), indent=2)
print(f'wrote {TAB_OUT}\nwrote {JSON_OUT}\n')
print('--- category-holdout table ---'); print('\n'.join(lines))
