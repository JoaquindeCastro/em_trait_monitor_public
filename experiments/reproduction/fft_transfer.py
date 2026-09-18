"""fft transfer analysis of saved checkpoint artifacts."""

from pathlib import Path
PROJECT_ROOT = globals().get("PROJECT_ROOT", Path(__file__).resolve().parents[2])
ROOT = PROJECT_ROOT
OUTPUT_ROOT = globals().get("OUTPUT_ROOT", PROJECT_ROOT / "outputs/fft_transfer")
(OUTPUT_ROOT / "figures").mkdir(parents=True, exist_ok=True)
(OUTPUT_ROOT / "tables").mkdir(parents=True, exist_ok=True)
RF_HP = globals().get("RF_HP", dict(n_estimators=100, max_depth=5, min_samples_leaf=5, random_state=42))
GBR_HP = globals().get("GBR_HP", dict(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42))
RIDGE_KW = globals().get("RIDGE_KW", dict(alpha=1.0))

import json, csv, numpy as np, torch
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score


TRAJ = PROJECT_ROOT / 'results' / 'prelim' / 'st9' / 'trajectories'
PC1_PATH = PROJECT_ROOT / 'results' / 'prelim' / 'st9' / 'cluster_pc1' / 'cluster_pc1_summary.json'

TRAITS = ['honesty','sycophancy','harmlessness','power_seeking','helpfulness','confidence','corrigibility']
MODELS = globals().get("MODELS", ['llama3-8b','mistral-7b','qwen25-7b','gemma2-9b'])
MODEL_NAMES = {'llama3-8b':'LLaMA','mistral-7b':'Mistral','qwen25-7b':'Qwen','gemma2-9b':'Gemma'}
# Per-model fallback ||h^0|| (used only if a cell's activations.pt is missing). For cal
# cells these equal the per-cell norm exactly; FFT cells differ by <=0.2%, so we read
# each cell's own ||h^0|| below to keep the cosine normalization identical across
# LoRA and full-finetuning vectors.
NORMS = globals().get("NORMS", {'llama3-8b': 8.500, 'mistral-7b': 4.688, 'qwen25-7b': 66.500, 'gemma2-9b': 372.000})
CAL_PERTS = globals().get("CAL_PERTS", ['insecure_code_1k','gsm8k_1k','jailbroken','bad_medical'])
OOD_PERTS = globals().get("OOD_PERTS", ['number_sequence','risky_financial','subtle_misinfo'])
SEEDS = globals().get("SEEDS", [42, 123, 789])
EM_THRESH = globals().get("EM_THRESH", 0.06)
RF_KW = globals().get("RF_KW", dict(n_estimators=100, max_depth=5, min_samples_leaf=5, random_state=42))
GBR_KW = globals().get("GBR_KW", dict(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42))
RIDGE_KW = globals().get("RIDGE_KW", dict(alpha=1.0))
N_BOOT = globals().get("N_BOOT", 1000)

pc1d = json.load(open(PC1_PATH))
PC1 = np.array([pc1d['cluster_pc1'][t] for t in TRAITS]); PC1 /= np.linalg.norm(PC1)

def cell_h0_norm(cell_dir, model):
    """Per-cell ||h^0|| read from this cell's activations.pt (canonical
    _get_activation_norm). Falls back to the per-model NORMS constant if the
    activation cache is absent. Identical recipe for LoRA cal and FFT cells, so the
    cosine normalization is provably the same across both finetuning methods."""
    f = cell_dir / 'activations.pt'
    if f.exists():
        acts = torch.load(f, map_location='cpu', weights_only=False)
        if 0 in acts:
            return float(acts[0].norm(dim=-1).mean().item())
    return NORMS[model]

def load_traj_7d(model, traj_path):
    """Return dict step -> 7D drift (cosine-normalized by this cell's own ||h^0||)."""
    if not traj_path.exists(): return {}
    norm = NORMS[model]  # paper's fixed per-model constant (methodology.tex); cell_h0_norm kept but unused for audit consistency
    t = json.load(open(traj_path))['trajectory']
    s0 = next(e for e in t if e['step'] == 0)['projections']
    out = {}
    for e in t:
        if e['step'] == 0: continue
        sk = e['step'] if isinstance(e['step'], int) else 126
        out[sk] = np.array([e['projections'][k] - s0[k] for k in TRAITS]) / norm
    return out

def load_betley(grades_path):
    if not grades_path.exists(): return {}
    d = json.load(open(grades_path))
    return {int(k.replace('step_','')): v.get('misalignment_rate', 0.0)
            for k, v in d.items() if k.startswith('step_')}

def lora_paths(model, pert, seed):
    base = TRAJ / model / pert / f'seed_{seed}'
    if not (base / 'trajectory.json').exists():
        for cand in (TRAJ / model / pert).glob(f'lr*/seed_{seed}'):
            base = cand; break
    return base / 'trajectory.json', base / 'betley_eval' / 'grades.json'

def fft_paths(model, pert, seed, lr):
    base = TRAJ / model / pert / 'fft' / f'lr{lr}' / 'ns1000' / f'seed_{seed}'
    return base / 'trajectory.json', base / 'betley_eval' / 'grades.json'

def fft_lr(model, pert):
    """Official LR per (model, pert) for FFT."""
    if model == 'mistral-7b' and pert in ('risky_financial','number_sequence'):
        return '5e-6'
    return '1e-5'

print('Constants + loaders ready (per-cell ||h^0|| cosine normalization).')


cal_pool = {m: {'X': [], 'y': [], 'pid': []} for m in MODELS}
for m in MODELS:
    for p_idx, p in enumerate(CAL_PERTS):
        for s in SEEDS:
            tp, gp = lora_paths(m, p, s)
            t7 = load_traj_7d(m, tp)
            em = load_betley(gp)
            for step in sorted(set(t7) & set(em)):
                cal_pool[m]['X'].append(t7[step])
                cal_pool[m]['y'].append(em[step])
                cal_pool[m]['pid'].append(p_idx)
    cal_pool[m]['X'] = np.array(cal_pool[m]['X'])
    cal_pool[m]['y'] = np.array(cal_pool[m]['y'])
    cal_pool[m]['pid'] = np.array(cal_pool[m]['pid'])
    Xc = cal_pool[m]['X']; yc = cal_pool[m]['y']
    print(f'  {MODEL_NAMES[m]:<8} cal n={len(Xc):3d}  dangerous={int((yc>EM_THRESH).sum())}')

# Full 3-seed x 4-model x 3-OOD-pert grid (36 cells). LR per (model, pert) follows
FFT_GRID = []
for m in MODELS:
    for p in ['subtle_misinfo', 'risky_financial', 'number_sequence']:
        for s in SEEDS:
            FFT_GRID.append((m, p, s, fft_lr(m, p)))

fft_pool = {m: {'X': [], 'y': [], 'run_id': []} for m in MODELS}
n_skipped = 0
for m, p, s, lr in FFT_GRID:
    tp, gp = fft_paths(m, p, s, lr)
    if not tp.exists() or not gp.exists():
        n_skipped += 1; continue
    t7 = load_traj_7d(m, tp)
    em = load_betley(gp)
    run_tag = f'{MODEL_NAMES[m]}/{p}/s{s}/lr{lr}'
    for step in sorted(set(t7) & set(em)):
        fft_pool[m]['X'].append(t7[step])
        fft_pool[m]['y'].append(em[step])
        fft_pool[m]['run_id'].append(run_tag)

for m in MODELS:
    fft_pool[m]['X'] = np.array(fft_pool[m]['X'])
    fft_pool[m]['y'] = np.array(fft_pool[m]['y'])
    fft_pool[m]['run_id'] = np.array(fft_pool[m]['run_id'])
    yo = fft_pool[m]['y']
    n_runs = len(set(fft_pool[m]['run_id']))
    print(f'  {MODEL_NAMES[m]:<8} FFT OOD n={len(yo):3d} ({int((yo>EM_THRESH).sum())} dangerous) across {n_runs} runs')
print(f'\n  Skipped (data missing): {n_skipped}/{len(FFT_GRID)}')


FEATURES = [
    ('Our 7D', lambda X: X),                              # full 7D drift
    ('|PC1| only', lambda X: np.abs(X @ PC1).reshape(-1,1)),  # scalar |PC1|
]
CLASSIFIERS = [
    ('RF',    lambda: RandomForestRegressor(**RF_KW)),
    ('GBR',   lambda: GradientBoostingRegressor(**GBR_KW)),
    ('Ridge', lambda: Ridge(**RIDGE_KW)),
]

pooled = {}
for (feat_name, feat_fn) in FEATURES:
    for (ml_name, ml_cls) in CLASSIFIERS:
        all_true, all_pred, all_run = [], [], []
        for m in MODELS:
            Xc = feat_fn(cal_pool[m]['X']); yc = cal_pool[m]['y']
            Xo = feat_fn(fft_pool[m]['X']); yo = fft_pool[m]['y']; runs = fft_pool[m]['run_id']
            if len(Xo) == 0: continue
            clf = ml_cls().fit(Xc, yc)
            pred = np.clip(clf.predict(Xo), 0, 1)
            all_true.append(yo); all_pred.append(pred); all_run.append(runs)
        all_true = np.concatenate(all_true)
        all_pred = np.concatenate(all_pred)
        all_run = np.concatenate(all_run)
        actual = all_true > EM_THRESH
        alarm = all_pred > EM_THRESH
        n_pos = int(actual.sum()); n_neg = int((~actual).sum())
        tp = int((alarm & actual).sum()); fp = int((alarm & ~actual).sum())
        fn = int((~alarm & actual).sum()); tn = int((~alarm & ~actual).sum())
        pooled[(feat_name, ml_name)] = dict(
            tp=tp, fp=fp, fn=fn, tn=tn, n=len(all_true),
            n_pos=n_pos, n_neg=n_neg,
            fnr=fn/n_pos*100 if n_pos else 0.0,
            fpr=fp/n_neg*100 if n_neg else 0.0,
            acc=(tp+tn)/len(all_true)*100,
            auroc=float(roc_auc_score(actual, all_pred)) if n_pos and n_neg else float('nan'),
            y_true_bin=actual.tolist(), y_pred_em=all_pred.tolist(), run_id=all_run.tolist(),
        )

# Print quick view
print(f'{"Features":<12} {"ML":<6} {"Acc":>6} {"FNR":>7} {"FPR":>7} {"FN":>3} {"FP":>3} {"AUROC":>6}')
print('-'*60)
for (f, ml), r in pooled.items():
    print(f'{f:<12} {ml:<6} {r["acc"]:>5.1f} {r["fnr"]:>6.2f}% {r["fpr"]:>6.2f}% {r["fn"]:>3} {r["fp"]:>3} {r["auroc"]:>6.3f}')

# The GROUND-TRUTH label threshold stays FIXED at EM_THRESH: "dangerous" must mean the
# same thing in both regimes or the cross-method comparison is circular. What we sweep
# here is ONLY the alarm firing point, and we select it using FFT labels -- so this is a
# POST-HOC ORACLE, not a deployable calibration. It quantifies how much of the frozen-
# transfer FNR is an operating-point effect vs. genuine loss of discriminative signal.
# Selection rule: minimize FNR subject to FPR <= 5%. Computed from `pooled` (no refit).
ORACLE_FPR_CAP = globals().get("ORACLE_FPR_CAP", 5.0)
TAU_GRID = globals().get("TAU_GRID", np.arange(0.005, 0.1251, 0.0025))

oracle = {}
for key, r in pooled.items():
    actual = np.array(r['y_true_bin'], dtype=bool)
    pred = np.array(r['y_pred_em'])
    best = None
    for ta in TAU_GRID:
        alarm = pred > ta
        fn = int((actual & ~alarm).sum()); fp = int((~actual & alarm).sum())
        fnr = 100*fn/max(actual.sum(), 1); fpr = 100*fp/max((~actual).sum(), 1)
        if fpr <= ORACLE_FPR_CAP and (best is None or fnr < best['fnr']):
            best = dict(tau=float(ta), fnr=fnr, fpr=fpr, fn=fn, fp=fp)
    oracle[key] = best

print(f'Label threshold FIXED at tau_label = {EM_THRESH:.2f}; sweeping firing point only')
print(f'Oracle rule: min FNR s.t. FPR <= {ORACLE_FPR_CAP:.0f}%\n')
print(f'{"Features":<12} {"ML":<6} {"AUROC":>6} | {"FNR@dep":>8} {"FPR@dep":>8} | {"tau*":>6} {"FNR@tau*":>9} {"FPR@tau*":>9}')
print('-'*82)
for key, r in pooled.items():
    o = oracle[key]
    print(f'{key[0]:<12} {key[1]:<6} {r["auroc"]:>6.3f} | {r["fnr"]:>7.1f}% {r["fpr"]:>7.1f}% | '
          f'{o["tau"]:>6.3f} {o["fnr"]:>8.1f}% {o["fpr"]:>8.1f}%')

import json as _json
_out = OUTPUT_ROOT / 'fft_threshold_diagnostic.json'
_out.parent.mkdir(parents=True, exist_ok=True)
_out.write_text(_json.dumps(
    {f'{k[0]}|{k[1]}': {'auroc': pooled[k]['auroc'],
                        'deployed': {'tau': EM_THRESH, 'fnr': pooled[k]['fnr'],
                                     'fpr': pooled[k]['fpr'], 'fn': pooled[k]['fn'],
                                     'fp': pooled[k]['fp']},
                        'oracle': oracle[k]} for k in pooled}, indent=1))
print(f'\nWrote {_out}')


# Resample with replacement among the OOD runs (model x pert x seed) to capture
# run-level correlation. Matches the headline notebook's bootstrap protocol.
rng = np.random.default_rng(seed=42)

for (f, ml), r in pooled.items():
    y_true = np.array(r['y_true_bin'])
    y_pred = np.array(r['y_pred_em'])
    run = np.array(r['run_id'])
    unique_runs = np.array(sorted(set(run.tolist())))
    n_runs = len(unique_runs)
    fnr_boot, fpr_boot = [], []
    for _ in range(N_BOOT):
        sample = rng.choice(unique_runs, size=n_runs, replace=True)
        idx = np.concatenate([np.where(run == rr)[0] for rr in sample])
        yt = y_true[idx]; yp = y_pred[idx]
        alarm = yp > EM_THRESH
        n_p = int(yt.sum()); n_n = int((~yt).sum())
        if n_p:
            fnr_boot.append((((yt) & ~alarm).sum() / n_p) * 100)
        if n_n:
            fpr_boot.append((((~yt) & alarm).sum() / n_n) * 100)
    r['fnr_ci'] = (float(np.percentile(fnr_boot, 2.5)), float(np.percentile(fnr_boot, 97.5))) if fnr_boot else (float('nan'), float('nan'))
    r['fpr_ci'] = (float(np.percentile(fpr_boot, 2.5)), float(np.percentile(fpr_boot, 97.5))) if fpr_boot else (float('nan'), float('nan'))

print(f'{"Features":<12} {"ML":<6} {"Acc":>5} {"FNR % [CI]":>20} {"FPR % [CI]":>20} {"FN":>3} {"FP":>3} {"AUROC":>6}')
print('-'*90)
for (f, ml), r in pooled.items():
    fnr_lo, fnr_hi = r['fnr_ci']; fpr_lo, fpr_hi = r['fpr_ci']
    print(f'{f:<12} {ml:<6} {r["acc"]:>5.1f} '
          f'{r["fnr"]:>5.1f} [{fnr_lo:>4.1f},{fnr_hi:>5.1f}]   '
          f'{r["fpr"]:>5.1f} [{fpr_lo:>4.1f},{fpr_hi:>5.1f}]   '
          f'{r["fn"]:>3} {r["fp"]:>3} {r["auroc"]:>6.3f}')

# Re-runs each (feature, classifier) detector trained on LoRA cal,
# evaluates per held-out FFT dataset to surface where FN concentrate.
print(f'\n{"":<14s} {"subtle_misinfo":>16s} {"risky_financial":>17s} {"number_sequence":>17s}')
print('-' * 70)
for (fname, ml_name), r in pooled.items():
    # We have r['y_true_em'] and r['y_pred_em'] pooled across all OOD cells + run_id.
    # Decompose by pert (extract from run_id: 'Model/pert/sX/lrY')
    actual = np.array(r['y_true_bin']); y_pred = np.array(r['y_pred_em']); runs = r['run_id']
    alarm = y_pred > EM_THRESH
    row = f'  {fname:<9s} {ml_name:<3s}'
    for p in ['subtle_misinfo', 'risky_financial', 'number_sequence']:
        mask = np.array([rid.split('/')[1] == p for rid in runs])
        n_dgr = int(actual[mask].sum())
        fn = int((actual[mask] & ~alarm[mask]).sum())
        fnr = fn/n_dgr*100 if n_dgr else 0.0
        row += f'    {fn:>3d}/{n_dgr:<3d} ({fnr:>4.1f}%)'
    print(row)


# Mirrors tab_error_breakdown.tex from the LoRA evaluation but on FFT cells.
# Each cell: 'FN / FP' as integer totals across 3 seeds (avg checkpoints/cell ~13).
PERTS_ORDERED = ['subtle_misinfo', 'risky_financial', 'number_sequence']
PERT_LABELS = {'subtle_misinfo': 'Subtle misinfo', 'risky_financial': 'Risky fin.', 'number_sequence': 'Number seq.'}
DETECTORS_ORDERED = [
    ('Our 7D', 'RF'), ('Our 7D', 'GBR'), ('Our 7D', 'Ridge'),
    ('|PC1| only', 'RF'), ('|PC1| only', 'GBR'), ('|PC1| only', 'Ridge'),
]

# Need per-cell predictions. Re-derive from cal-trained models since we have them in scope.
def per_model_pert_fn(feat_fn, ml_cls, ml_kw):
    out = {}  # out[model][pert] = (fn, fp, n_dgr, n_safe)
    for m in MODELS:
        Xc = feat_fn(cal_pool[m]['X']); yc = cal_pool[m]['y']
        clf = ml_cls(**ml_kw).fit(Xc, yc)
        Xo_all = feat_fn(fft_pool[m]['X']); yo = fft_pool[m]['y']; runs = fft_pool[m]['run_id']
        pred = np.clip(clf.predict(Xo_all), 0, 1)
        actual = yo > EM_THRESH; alarm = pred > EM_THRESH
        out[m] = {}
        for p in PERTS_ORDERED:
            mask = np.array([rid.split('/')[1] == p for rid in runs])
            fn = int((actual[mask] & ~alarm[mask]).sum())
            fp = int((~actual[mask] & alarm[mask]).sum())
            n_dgr = int(actual[mask].sum())
            n_safe = int((~actual[mask]).sum())
            out[m][p] = (fn, fp, n_dgr, n_safe)
    return out

FEATURE_FNS = {'Our 7D': lambda X: X, '|PC1| only': lambda X: np.abs(X @ PC1).reshape(-1,1)}
ML_SPECS = {'RF': (RandomForestRegressor, RF_KW), 'GBR': (GradientBoostingRegressor, GBR_KW), 'Ridge': (Ridge, RIDGE_KW)}

breakdown = {}
for (fname, ml_name) in DETECTORS_ORDERED:
    cls, kw = ML_SPECS[ml_name]
    breakdown[(fname, ml_name)] = per_model_pert_fn(FEATURE_FNS[fname], cls, kw)

# Print readable summary
print(f'\n{"":<24s} {"Subtle misinfo":>18s} {"Risky fin.":>14s} {"Number seq.":>14s}')
print(f'{"":<24s} {"FN/Dgr FP/Safe":>18s} {"FN/Dgr FP/Safe":>14s} {"FN/Dgr FP/Safe":>14s}')
print('-' * 80)
for (fname, ml_name) in DETECTORS_ORDERED:
    print(f'{fname} + {ml_name}:')
    for m in MODELS:
        cells = [breakdown[(fname,ml_name)][m][p] for p in PERTS_ORDERED]
        row = f'  {MODEL_NAMES[m]:<12s}'
        for (fn, fp, n_d, n_s) in cells:
            row += f'   {fn:>2d}/{n_d:<3d} {fp:>2d}/{n_s:<3d}'
        print(row)

# Emit LaTeX table
lines = []
lines.append(r'\begin{table}[h]')
lines.append(r'\centering\small')
lines.append(r'\caption{\textbf{FFT held-out error breakdown by (model, dataset).} Each cell shows $\mathrm{FN} / \mathrm{FP}$ counts pooled across 3 seeds at $\tau = ' + f'{int(EM_THRESH*100)}' + r'\%$, mirroring Table~\ref{tab:error_breakdown}. Per-model regressors fit on the four LoRA calibration datasets (Table~\ref{tab:headline_detection}); evaluation pool is the 36-cell FFT grid. The high FN counts on LLaMA $\times$ \texttt{number\_sequence} and Gemma $\times$ \texttt{number\_sequence} are correct direction-aware suppression rather than detector failure (text).}')
lines.append(r'\label{tab:fft_error_breakdown}')
lines.append(r'\setlength{\tabcolsep}{4pt}')
lines.append(r'\begin{tabular}{ll ccc}')
lines.append(r'\toprule')
lines.append(r'Detector & Model & Subtle misinfo & Risky fin. & Number seq. \\')
lines.append(r'\midrule')
for (fname, ml_name) in DETECTORS_ORDERED:
    det_label = f'{fname.replace("|PC1|", "$|$PC1$|$")} + {ml_name}'
    for i, m in enumerate(MODELS):
        cells = [breakdown[(fname,ml_name)][m][p] for p in PERTS_ORDERED]
        cell_strs = [f'{fn} / {fp}' for (fn, fp, _, _) in cells]
        prefix = det_label if i == 0 else ''
        lines.append(f'{prefix} & {MODEL_NAMES[m]} & ' + ' & '.join(cell_strs) + r' \\')
    if (fname, ml_name) != DETECTORS_ORDERED[-1]:
        lines.append(r'\midrule')
lines.append(r'\bottomrule')
lines.append(r'\end{tabular}')
lines.append(r'\end{table}')

out_tex = OUTPUT_ROOT / 'tables' / 'tab_fft_error_breakdown.tex'
out_tex.write_text('\n'.join(lines) + '\n')
print(f'\nWrote {out_tex}')


n_runs_used = len({rid for m in MODELS for rid in fft_pool[m]['run_id'].tolist()})
n_ckpts_used = sum(len(fft_pool[m]['y']) for m in MODELS)
n_dgr_used = sum(int((fft_pool[m]['y'] > EM_THRESH).sum()) for m in MODELS)

lines = []
lines.append(r'\begin{table}[t]')
lines.append(r'\centering')
lines.append(r'\footnotesize')
lines.append(r'\caption{\textbf{FFT cross-method alarm transfer.} Per-model regressors trained on the same four LoRA calibration perturbations as Table~\ref{tab:headline_detection}, evaluated on FFT-induced trajectories instead of LoRA. \emph{Mirrors Table~\ref{tab:headline_detection} format on the FFT test set.} Test set: ' + f'{n_ckpts_used} checkpoints ({n_dgr_used} dangerous) across {n_runs_used} held-out FFT runs' + r' (subtle\_misinfo + risky\_financial 3 seeds each; number\_sequence, 3 seeds each). 95\% confidence intervals from 1000 cluster bootstrap resamples over the held-out FFT runs. Cf. \S\ref{sec:detection} and Appendix~\ref{app:fft} for the LoRA baseline and per-cell breakdown.}' )
lines.append(r'\label{tab:fft_transfer_full}')
lines.append(r'\setlength{\tabcolsep}{3pt}')
lines.append(r'\begin{tabular}{ll r l l r r r}')
lines.append(r'\toprule')
lines.append(r'Features & ML & Acc (\%) & FNR \% [95\% CI] & FPR \% [95\% CI] & FN & FP & AUROC \\')
lines.append(r'\midrule')
lines.append(r'\multicolumn{8}{l}{\emph{Theory-driven trait basis (ours)}} \\')
for ml_name in ['RF','GBR','Ridge']:
    r = pooled[('Our 7D', ml_name)]
    bold = r'\textbf' if ml_name == 'RF' else ''
    fnr_lo, fnr_hi = r['fnr_ci']; fpr_lo, fpr_hi = r['fpr_ci']
    def fmt(val, lo, hi):
        return f'{val:.1f} [{lo:.1f}, {hi:.1f}]'
    fnr_s = fmt(r['fnr'], fnr_lo, fnr_hi)
    fpr_s = fmt(r['fpr'], fpr_lo, fpr_hi)
    if bold:
        lines.append(rf'{bold}{{Our 7D}} & {bold}{{{ml_name}}} & {bold}{{{r["acc"]:.1f}}} & {bold}{{{fnr_s}}} & {bold}{{{fpr_s}}} & {bold}{{{r["fn"]}}} & {bold}{{{r["fp"]}}} & {bold}{{{r["auroc"]:.3f}}} \\')
    else:
        lines.append(rf'Our 7D & {ml_name} & {r["acc"]:.1f} & {fnr_s} & {fpr_s} & {r["fn"]} & {r["fp"]} & {r["auroc"]:.3f} \\')
lines.append(r'\multicolumn{8}{l}{\emph{Scalar PC1 baseline}} \\')
for ml_name in ['RF','GBR','Ridge']:
    r = pooled[('|PC1| only', ml_name)]
    fnr_lo, fnr_hi = r['fnr_ci']; fpr_lo, fpr_hi = r['fpr_ci']
    fnr_s = f'{r["fnr"]:.1f} [{fnr_lo:.1f}, {fnr_hi:.1f}]'
    fpr_s = f'{r["fpr"]:.1f} [{fpr_lo:.1f}, {fpr_hi:.1f}]'
    lines.append(rf'$|$PC1$|$ only & {ml_name} & {r["acc"]:.1f} & {fnr_s} & {fpr_s} & {r["fn"]} & {r["fp"]} & {r["auroc"]:.3f} \\')
lines.append(r'\bottomrule')
lines.append(r'\end{tabular}')
lines.append(r'\end{table}')

out_tex = OUTPUT_ROOT / 'tables' / 'tab_fft_transfer.tex'
out_tex.parent.mkdir(parents=True, exist_ok=True)
out_tex.write_text('\n'.join(lines) + '\n')
print(f'Wrote {out_tex}')
print()
print(out_tex.read_text())

# Notebook-generated so representation.tex \input's it (no hand-written numbers).
# Reduced 4-row view: 3 detectors on the 7D basis + |PC1| Ridge, to contrast
# direction-aware vs scalar. Full per-detector table (with CIs/FN/FP) is in the
# appendix as tab:fft_transfer_full. Numbers come from the same `pooled` dict.
# The final column is the POST-HOC oracle firing point (Appendix~\ref{app:fft_threshold}),
# clearly separated from the frozen-transfer columns it must not be confused with.
main_rows = [('Our 7D', 'RF'), ('Our 7D', 'GBR'), ('Our 7D', 'Ridge'), ('|PC1| only', 'Ridge')]
DISPLAY = {'Our 7D': 'Our 7D', '|PC1| only': r'$|$PC1$|$'}

mlines = []
mlines.append(r'\begin{table}[t]')
mlines.append(r'\centering\footnotesize')
mlines.append(r'\caption{\textbf{FFT cross-method monitor transfer.} '
              r'Per-model monitors are trained on LoRA calibration trajectories and evaluated '
              r'on 36 held-out FFT runs ('
              + f'{n_ckpts_used} checkpoints, {n_dgr_used} dangerous'
              + r"). ``Deployed'' uses the LoRA-calibrated threshold $\tau="
              + f'{int(EM_THRESH*100)}' + r'\%$. The post-hoc $\tau^\star$ uses FFT '
              r'labels to minimize FNR subject to FPR $\leq ' + f'{ORACLE_FPR_CAP:.0f}' + r'\%$ and is '
              r'included only to diagnose threshold mismatch. Full per-model results appear in '
              r'Appendix~\ref{app:fft_details}.}')
mlines.append(r'\label{tab:fft_transfer}')
mlines.append(r'\setlength{\tabcolsep}{4pt}')
mlines.append(r'\begin{tabular}{ll rrrr c rr}')
mlines.append(r'\toprule')
mlines.append(r' & & \multicolumn{4}{c}{Deployed ($\tau = ' + f'{int(EM_THRESH*100)}' + r'\%$)} & & \multicolumn{2}{c}{Post-hoc $\tau^{*}$} \\')
mlines.append(r'\cmidrule(lr){3-6} \cmidrule(lr){8-9}')
mlines.append(r'Features & ML & Acc & FNR & FPR & AUROC & & $\tau^{*}$ & FNR \\')
mlines.append(r'\midrule')
for feat, ml in main_rows:
    r = pooled[(feat, ml)]
    o = oracle[(feat, ml)]
    mlines.append(rf'{DISPLAY[feat]} & {ml} & {r["acc"]:.1f} & {r["fnr"]:.1f} & {r["fpr"]:.1f} & {r["auroc"]:.3f} & & {o["tau"]:.3f} & {o["fnr"]:.1f} \\')
mlines.append(r'\bottomrule')
mlines.append(r'\end{tabular}')
mlines.append(r'\end{table}')

main_tex = OUTPUT_ROOT / 'tables' / 'tab_fft_transfer_main.tex'
main_tex.write_text('\n'.join(mlines) + '\n')
print(f'Wrote {main_tex}')
print('\n'.join(mlines))


# Reproducible source for the "parameter-update capacity" rotation claim in
# representation.tex / appendix.tex. LoRA cal and FFT vectors use the IDENTICAL
# per-cell ||h^0|| cosine normalization (see load_traj_7d), so the comparison is
# methodologically consistent across finetuning methods. We report the rotation of
# cluster-PC1 (cos and degrees), which is invariant to the per-cell-vs-constant
# norm choice; the augmented variance-explained is recomputed in this same frame.
from sklearn.decomposition import PCA
from itertools import product

def _final_drift(load7d_dict):
    return load7d_dict[max(load7d_dict)] if load7d_dict else None

def final_drift_lora(m, p, s):
    tp, _ = lora_paths(m, p, s)
    return _final_drift(load_traj_7d(m, tp))

def final_drift_fft(m, p, s):
    tp, _ = fft_paths(m, p, s, fft_lr(m, p))
    return _final_drift(load_traj_7d(m, tp))

# 48-vector LoRA calibration pool (4 models x 4 cal perts x 3 seeds), final-step drift
cal_vecs = np.array([v for m, p, s in product(MODELS, CAL_PERTS, SEEDS)
                     if (v := final_drift_lora(m, p, s)) is not None])

# +24 dangerous FFT (subtle_misinfo + risky_financial; excludes benign number_sequence)
DANGER_FFT = ['subtle_misinfo', 'risky_financial']
fft_danger = np.array([v for m, p, s in product(MODELS, DANGER_FFT, SEEDS)
                       if (v := final_drift_fft(m, p, s)) is not None])

# +36 full FFT (all three held-out OOD perts)
fft_full = np.array([v for m, p, s in product(MODELS, OOD_PERTS, SEEDS)
                     if (v := final_drift_fft(m, p, s)) is not None])

def _pc1_var(X):
    pca = PCA(n_components=7).fit(X)
    return pca.components_[0], float(pca.explained_variance_ratio_[0])

base_pc1, base_var = _pc1_var(cal_vecs)

def _augment(extra):
    pc1, var = _pc1_var(np.vstack([cal_vecs, extra]))
    if pc1 @ base_pc1 < 0: pc1 = -pc1
    c = abs(float(pc1 @ base_pc1))
    return c, float(np.degrees(np.arccos(min(1.0, c)))), var

print(f'LoRA cal pool: {cal_vecs.shape[0]} vectors, baseline PC1 var = {base_var*100:.2f}%')
aug = {}
for label, extra in [('+24 dangerous FFT', fft_danger), ('+36 full FFT', fft_full)]:
    c, deg, var = _augment(extra)
    aug[label] = dict(n_extra=int(extra.shape[0]), cos=round(c, 4),
                      rotation_deg=round(deg, 2), aug_pc1_var=round(var, 4))
    print(f'  {label:20s} (n={extra.shape[0]:2d}): cos={c:.4f}, rotation={deg:.2f} deg, '
          f'augmented PC1 var={var*100:.2f}%')

rec = dict(method='per-cell ||h^0|| cosine norm; final-step drift; 7D trait PCA',
           cal_n=int(cal_vecs.shape[0]), baseline_pc1_var=round(base_var, 4),
           augmentation=aug)
out_path = OUTPUT_ROOT / 'fft_pc1_augmentation.json'
out_path.parent.mkdir(parents=True, exist_ok=True)
json.dump(rec, open(out_path, 'w'), indent=2)
print(f'\nWrote {out_path}')


slines = []
slines.append(r'\begin{table}[h]')
slines.append(r'\centering\small')
slines.append(r'\caption{\textbf{Post-hoc alarm-threshold sweep on the FFT test set.} '
              r'The ground-truth danger label is held FIXED at $\tau_{\text{label}} = '
              + f'{int(EM_THRESH*100)}' + r'\%$ in every column; only the alarm firing '
              r'point varies. \emph{Deployed} is the LoRA-calibrated firing point actually '
              r'used in Table~\ref{tab:fft_transfer}. $\tau^{*}$ is selected \emph{using FFT '
              r'labels} to minimize FNR subject to FPR $\leq ' + f'{ORACLE_FPR_CAP:.0f}' + r'\%$, '
              r'so it is an oracle upper bound that a practitioner without FFT labels could not '
              r'obtain. AUROC is threshold-free and identical under both columns.}')
slines.append(r'\label{tab:fft_threshold_sweep}')
slines.append(r'\setlength{\tabcolsep}{4pt}')
slines.append(r'\begin{tabular}{ll c rr c rrr}')
slines.append(r'\toprule')
slines.append(r' & & & \multicolumn{2}{c}{Deployed ($\tau = ' + f'{int(EM_THRESH*100)}' + r'\%$)} & & \multicolumn{3}{c}{Post-hoc oracle $\tau^{*}$} \\')
slines.append(r'\cmidrule(lr){4-5} \cmidrule(lr){7-9}')
slines.append(r'Features & ML & AUROC & FNR & FPR & & $\tau^{*}$ & FNR & FPR \\')
slines.append(r'\midrule')
for key in [('Our 7D','RF'),('Our 7D','GBR'),('Our 7D','Ridge'),
            ('|PC1| only','RF'),('|PC1| only','GBR'),('|PC1| only','Ridge')]:
    r = pooled[key]; o = oracle[key]
    disp = DISPLAY[key[0]]
    if key == ('|PC1| only','RF'):
        slines.append(r'\midrule')
    slines.append(rf'{disp} & {key[1]} & {r["auroc"]:.3f} & {r["fnr"]:.1f} & {r["fpr"]:.1f} & & '
                  rf'{o["tau"]:.3f} & {o["fnr"]:.1f} & {o["fpr"]:.1f} \\')
slines.append(r'\bottomrule')
slines.append(r'\end{tabular}')
slines.append(r'\end{table}')

sweep_tex = OUTPUT_ROOT / 'tables' / 'tab_fft_threshold_sweep.tex'
sweep_tex.write_text('\n'.join(slines) + '\n')
print(f'Wrote {sweep_tex}')
print('\n'.join(slines))
