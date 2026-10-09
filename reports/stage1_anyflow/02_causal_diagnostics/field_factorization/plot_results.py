"""Display all measured cases; absolute velocity and action delta are distinct."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

out=Path(__file__).resolve().parent
d=json.loads((out/'analysis.json').read_text())
roles=['original','causal_original','causal_initializer','causal_fm32','causal_anyflow32','causal_anyflow128']
labels=['Original','Causal\nno adaptation','Prior init','FM32','AnyFlow32\ndiagonal','AnyFlow128\ndiagonal']
fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
for axis,metric,title,ylim in [(axes[0],'mean_velocity_cosine','Whole velocity vs Original',(0.983,1.001)),
                               (axes[1],'mean_delta_cosine','Current A/D velocity difference vs Original',(-.13,1.04))]:
    for i,r in enumerate(roles):
        rows=[x for x in d['records'] if x['role']==r]
        values=[(x['velocity_cosine_A']+x['velocity_cosine_D'])/2 if metric=='mean_velocity_cosine' else x['delta_cosine'] for x in rows]
        offsets=[(j-5.5)*.032 for j in range(len(values))]
        axis.scatter([i+x for x in offsets],values,s=18,c='#2c6aa0',alpha=.65,zorder=3)
        mean=d['summary'][r][metric]
        axis.hlines(mean,i-.25,i+.25,color='#d86c20',lw=3,zorder=4)
        axis.annotate(f'{mean:.3f}',(i,mean),xytext=(0,10),textcoords='offset points',ha='center',fontsize=9)
    axis.set_xticks(range(len(roles)),labels,fontsize=9)
    axis.set(title=title,ylabel='Cosine similarity',ylim=ylim)
    axis.grid(axis='y',alpha=.2)
fig.suptitle('Same-state causalization diagnostic: similar total prediction, misaligned action difference',fontsize=12)
fig.supxlabel('12 fixed generated-endpoint interpolants; full history recomputation; unified SDPA backend.\nDots: cases. Orange: mean. No video-quality or persistent-KV equivalence claim.',fontsize=9)
fig.savefig(out/'field_geometry.png',dpi=180)
fig.savefig(out/'field_geometry.svg')
