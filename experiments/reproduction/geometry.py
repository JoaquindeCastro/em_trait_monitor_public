"""geometry analysis of saved checkpoint artifacts."""

from pathlib import Path
PROJECT_ROOT = globals().get("PROJECT_ROOT", Path(__file__).resolve().parents[2])
ROOT = PROJECT_ROOT
OUTPUT_ROOT = globals().get("OUTPUT_ROOT", PROJECT_ROOT / "outputs/geometry")
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

# Sign convention: PC1 aligned so harmlessness loading is positive.
# Positive (blue) = trait is high in the aligned (safe) state, decreases under drift.
# Negative (red) = trait increases under misalignment drift.
# Alarm uses |drift · PC1|, so the sign is purely for labeling.

plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
})

fig, (ax_load, ax_scree) = plt.subplots(1, 2, figsize=(5.5, 3.0),
                                         gridspec_kw={'width_ratios': [3, 1], 'wspace': 0.35})

# --- Left: PC1 loadings ---
colors = ['#2563eb' if v > 0 else '#dc2626' for v in PC1]
bars = ax_load.bar(range(len(TRAITS)), PC1, color=colors, edgecolor='white', linewidth=0.5, width=0.7)

for i, (bar, val) in enumerate(zip(bars, PC1)):
    offset = 0.02 if val > 0 else -0.02
    va = 'bottom' if val > 0 else 'top'
    ax_load.text(i, val + offset, f'{val:+.3f}', ha='center', va=va, fontsize=7.5, fontweight='bold')

ax_load.set_xticks(range(len(TRAITS)))
ax_load.set_xticklabels(TRAIT_LABELS, rotation=35, ha='right', fontsize=9)
ax_load.set_ylabel('PC1 loading')
ax_load.axhline(0, color='black', linewidth=0.5)
ax_load.set_ylim(-0.58, 0.78)
ax_load.grid(axis='y', alpha=0.2)

# Sign convention subtitle
ax_load.text(0.5, 1.0,
             r'$\mathbf{+}$ aligned-state trait (decreases under drift)    '
             r'$\mathbf{-}$ misaligned-state trait (increases under drift)',
             transform=ax_load.transAxes, fontsize=6.5, ha='center', va='bottom',
             color='#555555', fontstyle='italic')

# --- Right: Scree plot ---
pcs = [1, 2, 3]
var_pct = [v * 100 for v in var_explained]
ax_scree.bar(pcs, var_pct, color='#6366f1', edgecolor='white', width=0.6)
for i, v in zip(pcs, var_pct):
    ax_scree.text(i, v + 1.5, f'{v:.1f}%', ha='center', fontsize=8,
                  fontweight='bold' if i == 1 else 'normal')
ax_scree.set_xticks(pcs)
ax_scree.set_xticklabels(['PC1', 'PC2', 'PC3'], fontsize=9)
ax_scree.set_ylabel('Variance explained (%)')
ax_scree.set_ylim(0, 85)
ax_scree.grid(axis='y', alpha=0.2)

plt.tight_layout()
out_path = FIG_OUT / 'fig_pc1_loadings.pdf'
fig.savefig(out_path, bbox_inches='tight', dpi=600)
print(f'Saved: {out_path}')
plt.show()
