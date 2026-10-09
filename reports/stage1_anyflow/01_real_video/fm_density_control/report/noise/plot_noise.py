"""Plot completed fixed-state density comparisons; no model or GPU."""
import json
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE=Path(__file__).resolve().parent
data=json.loads((BASE/'metrics.json').read_text())
assert data['status']=='complete_matched_statistics' and data['teacher_full_outputs_identical']
rows=data['rows'];assert len(rows)==54
sigmas=sorted({r['sigma'] for r in rows})
values={name:[mean(r[name+'_raw_velocity_mse'] for r in rows if r['sigma']==s) for s in sigmas]
        for name in ('original','fm0','shift12','shift2_22')}
fig,axes=plt.subplots(1,2,figsize=(12.5,4.6))
for name,label,color,style in (
    ('original','Original H3','#252525','-'),('fm0','Causal FM0','#969696','--'),
    ('shift12','FM48: density shift12','#3269ad','-'),
    ('shift2_22','FM48: density shift2.22','#cc601d','-')):
    axes[0].plot(sigmas,values[name],style,marker='o',markersize=3,label=label,color=color)
axes[0].set(xlabel='Sigma (noise level)',ylabel='Raw velocity MSE',title='Fixed GT states: all 3 chunks')
axes[0].legend(fontsize=8);axes[0].grid(alpha=.2)
change=[100*(a/b-1) for a,b in zip(values['shift2_22'],values['shift12'])]
axes[1].axhline(0,color='#555555',lw=1)
axes[1].plot(sigmas,change,'o-',color='#7a4095',markersize=4)
axes[1].set(xlabel='Sigma (noise level)',ylabel='Candidate vs control MSE change (%)',
    title='Lower is better; sigma=1 worsens in 6/6 states')
axes[1].grid(alpha=.2)
axes[1].annotate(f'{change[-1]:+.2f}%',(sigmas[-1],change[-1]),xytext=(-40,-25),textcoords='offset points')
fig.suptitle('FM48 density control: weight function fixed, sampling distribution changed',fontsize=12)
fig.text(.5,.01,'Each point: 2 held-out clips x 3 chunks. Correlated fixed states; not action or rollout quality.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.04,1,.94))
fig.savefig(BASE/'noise_density.png',dpi=160)
fig.savefig(BASE/'noise_density.svg')
