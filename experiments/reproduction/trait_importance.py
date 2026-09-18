"""trait importance analysis of saved checkpoint artifacts."""

from pathlib import Path
PROJECT_ROOT = globals().get("PROJECT_ROOT", Path(__file__).resolve().parents[2])
ROOT = PROJECT_ROOT
OUTPUT_ROOT = globals().get("OUTPUT_ROOT", PROJECT_ROOT / "outputs/trait_importance")
(OUTPUT_ROOT / "figures").mkdir(parents=True, exist_ok=True)
(OUTPUT_ROOT / "tables").mkdir(parents=True, exist_ok=True)
RF_HP = globals().get("RF_HP", dict(n_estimators=100, max_depth=5, min_samples_leaf=5, random_state=42))
GBR_HP = globals().get("GBR_HP", dict(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42))
RIDGE_KW = globals().get("RIDGE_KW", dict(alpha=1.0))

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


IMP = json.load(open(globals().get("IMPORTANCE_PATH", ROOT / 'results/analysis/per_model_trait_importance.json')))
ENUM = json.load(open(globals().get("ENUMERATION_PATH", ROOT / 'results/analysis/full_trait_subset_enumeration.json')))

TRAITS = IMP['traits']
MODELS = globals().get("MODELS", IMP['models'])
MODEL_LABELS = {'llama3-8b': 'LLaMA-3 8B', 'mistral-7b': 'Mistral 7B',
                'qwen25-7b': 'Qwen 2.5 7B', 'gemma2-9b': 'Gemma 2 9B'}
print(f'traits: {TRAITS}')
print(f'models: {MODELS}')
print(f'ceiling (excluded from correlations): {IMP["rank_correlation"].get("ceiling_excluded", [])}')
print(f'protocol: {IMP["config"]["protocol"]}')

d_auroc = np.array([[IMP['importance'][t][m]['d_auroc'] for m in MODELS] for t in TRAITS])

fig, ax = plt.subplots(figsize=(6.5, 4.5))
vmax = float(np.max(np.abs(d_auroc)))
im = ax.imshow(d_auroc, cmap='RdBu_r', vmin=-vmax, vmax=+vmax, aspect='auto')
ax.set_xticks(range(len(MODELS)))
ax.set_xticklabels([MODEL_LABELS[m] for m in MODELS], rotation=15, ha='right')
ax.set_yticks(range(len(TRAITS)))
ax.set_yticklabels(TRAITS)
for i in range(len(TRAITS)):
    for j in range(len(MODELS)):
        ax.text(j, i, f'{d_auroc[i,j]:+.3f}', ha='center', va='center',
                fontsize=9, color='white' if abs(d_auroc[i,j]) > vmax*0.55 else 'black')
cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
cbar.set_label(r'$\Delta$AUROC (full 7D $-$ LOO 6D)')
ax.set_title('Per-model LOO-trait importance ($\Delta$AUROC)')
plt.tight_layout()
out = OUTPUT_ROOT / 'figures/fig_per_model_trait_importance.pdf'
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, bbox_inches='tight')
print(f'saved: {out}')
plt.show()

ranks = np.array(IMP['rank_matrix'])  # (models, traits)
df_rank = pd.DataFrame(ranks.T, index=TRAITS, columns=[MODEL_LABELS[m] for m in MODELS])
df_rank

corr = IMP['rank_correlation']
pairs = pd.DataFrame([{
    'pair': f"{MODEL_LABELS[p['models'][0]]} vs {MODEL_LABELS[p['models'][1]]}",
    'Spearman rho': p['spearman_rho'], 'Spearman p': p['spearman_p'],
    'Kendall tau': p['kendall_tau'],
} for p in corr['pairs']])
print(f"median Spearman rho: {corr.get('spearman_median', float('nan')):+.3f}   "
      f"range [{corr.get('spearman_min', float('nan')):+.3f}, {corr.get('spearman_max', float('nan')):+.3f}]")
print(f"median Kendall tau:  {corr.get('kendall_median', float('nan')):+.3f}")
print(f"models used: {corr.get('models_used', 'all')}    ceiling excluded: {corr.get('ceiling_excluded', [])}")
pairs

conc = IMP['concordance']
rows = []
for t in TRAITS:
    c = conc[t]
    rows.append({'trait': t, 'top': c['top_count'], 'middle': c['middle_count'],
                 'bottom': c['bottom_count'], 'label': c['label']})
df_conc = pd.DataFrame(rows)
df_conc

# Emit appendix tex table: trait, per-model rank, concordance summary label.
LABEL_HUMAN = {
    'consistently_critical':   'consistently critical',
    'consistently_disposable': 'consistently disposable',
    'mostly_critical':         'mostly critical (3/4)',
    'mostly_disposable':       'mostly disposable (3/4)',
    'model_dependent':         'model-dependent',
}
def rank_cell(r):
    return f'{r:.1f}' if r != int(r) else f'{int(r)}'

lines = [
    r'\begin{table}[h]',
    r'\centering\small',
    r'\caption{\textbf{Per-model trait importance and cross-model consistency.} '
    r'For each (model, trait) we report the model-specific rank derived from LOO-trait '
    r'$\Delta$AUROC on cal-LOPO-CV (1 = most important, 7 = least). Cross-model '
    r'concordance bins each rank as \emph{top-3} / \emph{mid-1} / \emph{bot-3} and '
    r'summarises agreement across the 4 models. Gemma is at ceiling '
    r'(AUROC $=0.988$, ties on every LOO); its rank is reported for completeness but '
    r'contributes only to the concordance labels, not to the median Spearman '
    r'$\rho = ' + f"{corr.get('spearman_median', float('nan')):+.2f}" + r'$ (informative-model pairs only). '
    r'Only \texttt{sycophancy} lands bottom-3 for every model; \texttt{power\_seeking} '
    r'for 3 of 4. All other traits are model-dependent.}',
    r'\label{tab:per_model_trait_importance}',
    r'\begin{tabular}{lcccc l}',
    r'\toprule',
    r'Trait & ' + ' & '.join(MODEL_LABELS[m] for m in MODELS) + r' & Concordance \\',
    r'\midrule',
]
for t_idx, t in enumerate(TRAITS):
    row = [t.replace('_', r'\_')]
    for m_idx in range(len(MODELS)):
        r = ranks[m_idx, t_idx]
        row.append(rank_cell(r))
    row.append(LABEL_HUMAN.get(conc[t]['label'], conc[t]['label']))
    lines.append(' & '.join(row) + r' \\')
lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']

tex_out = OUTPUT_ROOT / 'tables/tab_per_model_trait_importance.tex'
tex_out.write_text('\n'.join(lines) + '\n')
print(f'wrote {tex_out}')
print('\n'.join(lines))

OOD = json.load(open(globals().get("SUBSETS_PATH", ROOT / 'results/analysis/universal_ksubset_ood.json')))
rows = []
for name, r in OOD['results'].items():
    row = {'subset': name, 'K': r['K']}
    row['pooled BalAcc'] = r['pooled']['balacc']
    row['pooled AUROC'] = r['pooled']['auroc']
    row['FN'] = r['pooled']['fn']
    row['FP'] = r['pooled']['fp']
    row['cos(subset PC1, 7D PC1)'] = r['pc1_geometry']['cos_full_pc1']
    row['angle (°)'] = r['pc1_geometry']['angle_deg']
    rows.append(row)
df_ood = pd.DataFrame(rows)
df_ood

# Per-model FN table (dangerous missed) for these 4 candidate subsets.
fn_rows = []
for name, r in OOD['results'].items():
    fn_rows.append({'subset': name,
                    **{MODEL_LABELS[m]: r['per_model'][m]['fn'] for m in MODELS}})
pd.DataFrame(fn_rows).set_index('subset')

# Emit appendix companion tex table.
# Emits ONLY the tabular, per the emitter contract (data_provenance.md 4.5).
# The float and caption live in latex/sections/appendix.tex beside
# \label{tab:universal_ksubset_ood} -- edit the caption THERE, not here.
# A caption written here would be stripped by normalize_tables.py and would
# silently never reach the paper (failure mode 10).
def _fmt_int(v): return str(int(v))
def _fmt3(v):    return f'{v:.3f}'
def _fmt1(v):    return f'{v:.1f}'

lines = [
    r'\begin{tabular}{lrrrr}',
    r'\toprule',
    r'Subset & AUROC & FN & FP & PC1 cos.\ (angle) \\',
    r'\midrule',
]
for name, r in OOD['results'].items():
    display_name = {'HHH': 'HHH',
                    'MinMax_K3': 'MinMax K$=$3',
                    'HHC': 'HHC',
                    'HHH_conf': 'HHH $+$ confidence',
                    'HHH_syco': 'HHH $+$ sycophancy',
                    'HHH_ps': 'HHH $+$ power-seeking',
                    'HHHC': 'HHHC',
                    'All7': 'All 7'}[name]
    p = r['pooled']; g = r['pc1_geometry']
    row = [display_name, _fmt3(p['auroc']),
           _fmt_int(p['fn']), _fmt_int(p['fp']),
           f"{_fmt3(g['cos_full_pc1'])} (${_fmt1(g['angle_deg'])}^\\circ$)"]
    lines.append(' & '.join(row) + r' \\')
lines += [r'\bottomrule', r'\end{tabular}']

tex_out = OUTPUT_ROOT / 'tables/tab_universal_ksubset_ood.tex'
tex_out.write_text('\n'.join(lines) + '\n')
print(f'wrote {tex_out}')
print('\n'.join(lines))
