"""Export a local-metric tradeoff figure; run only after all probes complete."""
from pathlib import Path
import json
import statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parent
data=json.loads((OUT/'analysis.json').read_text())
assert data['complete'], 'Do not plot a partial checkpoint comparison as final'
variants=('initial128','control136','auxiliary136')
labels=('AnyFlow128','136 control','136 + consistency')
fig,axes=plt.subplots(2,3,figsize=(13.5,7),layout='constrained')
for col,region in enumerate(('high','middle','low')):
    groups=[[r for r in data['rows'] if r['variant']==v and r['noise_region']==region] for v in variants]
    assert all(len(group)==6 for group in groups)
    for row,metrics in enumerate((
        [('self_consistency_endpoint_rmse','Finite vs current diagonal','#2759b0'),
         ('reference_refinement_rmse','Diagonal refinement change','#777777')],
        [('finite_rmse_to_teacher_at_r','Finite to teacher at r','#bd4b2d'),
         ('diagonal_rmse_to_teacher_at_r','Diagonal to teacher at r','#369873')],
    )):
        ax=axes[row,col]
        for key,label,color in metrics:
            means=[statistics.mean(r[key] for r in group) for group in groups]
            low=[min(r[key] for r in group) for group in groups]
            high=[max(r[key] for r in group) for group in groups]
            ax.plot(range(3),means,'o-',label=label,color=color)
            ax.fill_between(range(3),low,high,color=color,alpha=.12)
        ax.set_xticks(range(3),labels,rotation=15,ha='right')
        ax.set_ylabel('Latent endpoint RMSE')
        ax.set_title(f"{region.capitalize()}: {groups[0][0]['sigma']:.3f} -> {groups[0][0]['target_sigma']:.3f}")
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
fig.suptitle('Stage1 local diagnostics: self-consistency and pseudo-target error are separate\n'
    'Means across A/D x 3 chunks; shading is case range, not a confidence interval',fontsize=13)
fig.supxlabel('At r>0, pseudo-target=(1-r)*teacher_clean+r*same_noise. Diagonal is not GT.\n'
    'Do not compare absolute errors across interval widths or infer video quality from these curves.',fontsize=10)
fig.savefig(OUT/'local_metric_tradeoff.png',dpi=180)
print(OUT/'local_metric_tradeoff.png')
