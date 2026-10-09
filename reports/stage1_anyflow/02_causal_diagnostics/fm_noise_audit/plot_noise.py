"""Standalone plot of the completed audit; no model or GPU use."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
data = json.loads((ROOT / 'analysis.json').read_text())
groups = data['groups']
sigmas = sorted(float(k.removeprefix('sigma_')) for k in groups if k.startswith('sigma_'))
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
colors = {'original': '#333333', 'causal0': '#d18116', 'causal48': '#247bb5'}
for role, label in [('original', 'Original'), ('causal0', 'Causal FM0'), ('causal48', 'Causal FM48')]:
    ys = [groups[f'sigma_{s:.6f}'][f'{role}_raw_velocity_mse'] for s in sigmas]
    axes[0].plot(sigmas, ys, 'o-', label=label, color=colors[role])
for role, label in [('causal0', 'FM0 minus Original'), ('causal48', 'FM48 minus Original')]:
    ys = [groups[f'sigma_{s:.6f}'][f'{role}_raw_velocity_mse'] -
          groups[f'sigma_{s:.6f}']['original_raw_velocity_mse'] for s in sigmas]
    axes[1].plot(sigmas, ys, 'o-', label=label, color=colors[role])
for ax in axes[:2]:
    ax.set_xlabel('Sigma (0 = clean, 1 = noise)')
    ax.set_ylabel('Velocity MSE against noise - GT')
    ax.grid(alpha=.25)
    ax.legend(fontsize=9)
axes[0].set_title('Matched GT states: 6 states per sigma')
axes[1].set_title('Difference of MSEs, not MSE of predictions')
coverage = data['training_coverage']
names = ['low', 'mid', 'high']
for offset, key, label in [(-.24, 'sample_share', 'Sample share'), (0, 'weight_mass_share', 'Weight mass'),
                           (.24, 'weighted_loss_share', 'Weighted loss mass')]:
    axes[2].bar([i + offset for i in range(3)], [100 * coverage[b][key] for b in names],
                width=.24, label=label)
axes[2].set_xticks(range(3), ['Low\nN=4', 'Mid\nN=35', 'High\nN=153'])
axes[2].set_ylim(0, 100)
axes[2].set_ylabel('Share (%)')
axes[2].set_title('Training coverage: not gradient mass')
axes[2].legend(fontsize=9)
fig.suptitle('Real ABot FM48 audit: local MSE improvement does not establish action or video quality', fontsize=12)
fig.savefig(ROOT / 'noise_audit.png', dpi=170)
plt.close(fig)
