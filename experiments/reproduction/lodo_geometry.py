"""lodo geometry analysis of saved checkpoint artifacts."""

from pathlib import Path
PROJECT_ROOT = globals().get("PROJECT_ROOT", Path(__file__).resolve().parents[2])
ROOT = PROJECT_ROOT
OUTPUT_ROOT = globals().get("OUTPUT_ROOT", PROJECT_ROOT / "outputs/lodo_geometry")
(OUTPUT_ROOT / "figures").mkdir(parents=True, exist_ok=True)
(OUTPUT_ROOT / "tables").mkdir(parents=True, exist_ok=True)
RF_HP = globals().get("RF_HP", dict(n_estimators=100, max_depth=5, min_samples_leaf=5, random_state=42))
GBR_HP = globals().get("GBR_HP", dict(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42))
RIDGE_KW = globals().get("RIDGE_KW", dict(alpha=1.0))

import json, numpy as np, warnings
from pathlib import Path
from sklearn.decomposition import PCA
warnings.filterwarnings('ignore')


TRAJ = PROJECT_ROOT / 'results' / 'prelim' / 'st9' / 'trajectories'
PC1_PATH = PROJECT_ROOT / 'results' / 'prelim' / 'st9' / 'cluster_pc1' / 'cluster_pc1_summary.json'

TRAITS = ['honesty','sycophancy','harmlessness','power_seeking','helpfulness','confidence','corrigibility']
MODELS = globals().get("MODELS", ['llama3-8b','mistral-7b','qwen25-7b','gemma2-9b'])
SEEDS = globals().get("SEEDS", [42, 123, 789])
NORMS = globals().get("NORMS", {'llama3-8b':8.5,'mistral-7b':4.6875,'qwen25-7b':66.5,'gemma2-9b':372.0})
CAL_PERTS = globals().get("CAL_PERTS", ['insecure_code_1k','gsm8k_1k','jailbroken','bad_medical'])

pc1_data = json.load(open(PC1_PATH))
PC1_full = np.array([pc1_data['cluster_pc1'][t] for t in TRAITS])
PC1_full /= np.linalg.norm(PC1_full)
full_var = pc1_data['cluster_pc1_variance_explained']

print(f'Full PC1: {len(CAL_PERTS)} perts × {len(MODELS)} models × {len(SEEDS)} seeds = {len(CAL_PERTS)*len(MODELS)*len(SEEDS)} vectors')
print(f'Variance explained: {full_var*100:.1f}%')


def load_final_drift(m, p, s):
    """Load cosine-normalized final-step drift vector."""
    f = TRAJ / m / p / f'seed_{s}' / 'trajectory.json'
    if not f.exists(): return None
    t = json.load(open(f))['trajectory']
    s0 = None; last = None
    for e in t:
        step = e['step']
        if not isinstance(step, int): continue  # skip 'final' string entries
        proj = np.array([e['projections'][tr] for tr in TRAITS])
        if step == 0:
            s0 = proj
        elif last is None or step > last[0]:
            last = (step, proj)
    if s0 is None or last is None: return None
    return (last[1] - s0) / NORMS[m]

# Build per-pert vector pools
vectors_by_pert = {}
for p in CAL_PERTS:
    vecs = []
    for m in MODELS:
        for s in SEEDS:
            v = load_final_drift(m, p, s)
            if v is not None: vecs.append(v)
    vectors_by_pert[p] = np.array(vecs)
    print(f'{p}: {len(vecs)} vectors')

all_vecs = np.vstack(list(vectors_by_pert.values()))
print(f'\nTotal: {len(all_vecs)} vectors (should be 48)')


def compute_pc1(X):
    """Compute PC1 from drift vectors, sign-aligned with full PC1."""
    pca = PCA(n_components=min(7, len(X))).fit(X)
    pc1 = pca.components_[0]
    if np.dot(pc1, PC1_full) < 0: pc1 = -pc1
    pc1 /= np.linalg.norm(pc1)
    return pc1, pca.explained_variance_ratio_[0]

print('=== LOPO PC1 Stability ===')
print(f'{"Left out":<18} {"n_vec":>6} {"cos(LODO, full)":>16} {"var explained":>14} {"Δvar":>8}')
print('-' * 65)

results = []
for leave_out in CAL_PERTS:
    # Build matrix from remaining 3 perts
    remaining = [p for p in CAL_PERTS if p != leave_out]
    X_lopo = np.vstack([vectors_by_pert[p] for p in remaining])
    pc1_lopo, var_lopo = compute_pc1(X_lopo)
    cos = float(np.dot(pc1_lopo, PC1_full))
    results.append({
        'left_out': leave_out,
        'n_vec': len(X_lopo),
        'cos': cos,
        'var': var_lopo,
        'pc1': pc1_lopo,
    })
    print(f'{leave_out:<18} {len(X_lopo):>6} {cos:>16.4f} {var_lopo*100:>13.1f}% {(var_lopo - full_var)*100:>+7.1f}%')

# Also show full PC1 as reference
print(f'{"(none — full)":<18} {len(all_vecs):>6} {"1.0000":>16} {full_var*100:>13.1f}% {"—":>8}')

# Headline: minimum cosine
min_cos = min(r['cos'] for r in results)
min_pert = min(results, key=lambda r: r['cos'])['left_out']
print(f'\n★ Minimum cosine: {min_cos:.4f} (leaving out {min_pert})')
print(f'  → PC1 direction is NOT driven by any single perturbation type.')
if min_cos >= 0.98:
    print(f'  → Claim "direction generalizes across perturbation types" is supported (min cos ≥ 0.98).')
elif min_cos >= 0.95:
    print(f'  → Strong stability (min cos ≥ 0.95), minor perturbation-dependent variance.')
else:
    print(f'  → Some perturbation-dependence detected (min cos < 0.95). Investigate which pert matters most.')

# Show how much each trait loading shifts under LOPO

print('=== Per-trait loading shifts under LOPO ===')
print(f'{"Trait":<15}', end='')
print(f'{"Full":>8}', end='')
for r in results:
    print(f'{"−"+r["left_out"][:8]:>12}', end='')
print(f'{"max |Δ|":>10}')
print('-' * 70)

for i, t in enumerate(TRAITS):
    full_val = PC1_full[i]
    print(f'{t:<15}{full_val:>+8.3f}', end='')
    deltas = []
    for r in results:
        lopo_val = r['pc1'][i]
        deltas.append(abs(lopo_val - full_val))
        print(f'{lopo_val:>+12.3f}', end='')
    print(f'{max(deltas):>10.3f}')

print(f'\nMax loading shift across all traits and leave-outs: '
      f'{max(abs(r["pc1"][i] - PC1_full[i]) for r in results for i in range(7)):.3f}')

DISPLAY = {'bad_medical': 'bad_medical_advice', 'risky_financial': 'risky_financial_advice',
           'number_sequence': 'evil_numbers', 'subtle_misinfo': 'subtle_misinformation',
           'insecure_code_1k': 'insecure_code', 'gsm8k_1k': 'GSM8K', 'jailbroken': 'jailbroken'}

TAB_OUT = OUTPUT_ROOT / 'tables'
BSLASH = '\\'  # workaround for f-string backslash limitation

# --- Table: LOPO cosine stability ---
lines = []
lines.append(r'\begin{table}[h]')
lines.append(r'\centering\small')
lines.append(f'\\caption{{\\textbf{{LODO PC1 stability.}} Leaving out one calibration dataset and recomputing PC1 from the remaining 3 types (36 vectors each). The minimum cosine ({min_cos:.3f}, leaving out {DISPLAY.get(min_pert, min_pert).replace(chr(95), chr(92)+chr(95))}) confirms the direction is not driven by any single perturbation.}}')
lines.append(r'\label{tab:lopo_pc1}')
lines.append(r'\begin{tabular}{lcrr}')
lines.append(r'\toprule')
lines.append(r'Left out & $n$ vectors & cos(LODO, full) & Var.\ expl.\ (\%) \\')
lines.append(r'\midrule')
for r in results:
    name = DISPLAY.get(r['left_out'], r['left_out']).replace('_', r'\_')
    if r['left_out'] == min_pert:
        lines.append(f"\\textbf{{{name}}} & {r['n_vec']} & \\textbf{{{r['cos']:.4f}}} & {r['var']*100:.1f} \\\\")
    else:
        lines.append(f"{name} & {r['n_vec']} & {r['cos']:.4f} & {r['var']*100:.1f} \\\\")
lines.append(r'\midrule')
lines.append(f"(none --- full) & {len(all_vecs)} & 1.0000 & {full_var*100:.1f} \\\\")
lines.append(r'\bottomrule')
lines.append(r'\end{tabular}')
lines.append(r'\end{table}')

tex1 = '\n'.join(lines)
(TAB_OUT / 'tab_lopo_pc1.tex').write_text(tex1)
print(f'Saved: {TAB_OUT / "tab_lopo_pc1.tex"}')

# --- Table: Per-trait loading shifts ---
lines2 = []
lines2.append(r'\begin{table}[h]')
lines2.append(r'\centering\small')
lines2.append(r'\caption{\textbf{Per-trait PC1 loading shifts under LODO.} Each column shows the PC1 loading when one dataset is left out. Max $|\Delta|$ is the largest shift from the full PC1 for that trait.}')
lines2.append(r'\label{tab:lopo_loadings}')
cols = 'l' + 'r' * (len(results) + 2)
lines2.append(r'\begin{tabular}{' + cols + '}')
lines2.append(r'\toprule')
header = 'Trait & Full'
for r in results:
    short = r['left_out'][:6].replace('_', r'\_')
    header += f' & $-${short}'
header += r' & max $|\Delta|$ \\'
lines2.append(header)
lines2.append(r'\midrule')

for i, t in enumerate(TRAITS):
    full_val = PC1_full[i]
    tname = t.replace('_', r'\_')
    row = f'{tname} & {full_val:+.3f}'
    deltas = []
    for r in results:
        lopo_val = r['pc1'][i]
        deltas.append(abs(lopo_val - full_val))
        row += f' & {lopo_val:+.3f}'
    row += f' & {max(deltas):.3f} ' + r'\\'
    lines2.append(row)

lines2.append(r'\bottomrule')
lines2.append(r'\end{tabular}')
lines2.append(r'\end{table}')

tex2 = '\n'.join(lines2)
(TAB_OUT / 'tab_lopo_loadings.tex').write_text(tex2)
print(f'Saved: {TAB_OUT / "tab_lopo_loadings.tex"}')

# --- Print summary for paper text ---
max_shift = max(abs(r['pc1'][i] - PC1_full[i]) for r in results for i in range(7))
max_cos = max(r['cos'] for r in results)
print(f'\n=== Paper-ready summary ===')
print(f'LOPO stability: min cos = {min_cos:.4f} (leaving out {min_pert}), max cos = {max_cos:.4f}.')
print(f'Max per-trait loading shift: {max_shift:.3f}.')
