"""Plot measured instantaneous action geometry, keeping the cosine scale [-1,1]."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parent
data=json.loads((OUT/'analysis.json').read_text())['rows']
labels=[f"{r['history']} history / chunk {r['chunk']} / sigma {r['sigma']:.3f}" for r in data]
y=np.arange(len(data))
fig,(ax,bx)=plt.subplots(1,2,figsize=(13,8),gridspec_kw={'width_ratios':[1.35,1]},sharey=True)
ax.barh(y-.17,[r['matched_cosine'] for r in data],height=.32,label='Matched teacher conditions',color='#2964ad')
ax.barh(y+.17,[r['native_cosine'] for r in data],height=.32,label='Native teacher prefix policy',color='#d67c26')
ax.set_xlim(-1,1);ax.axvline(0,color='#555',lw=.8)
ax.set_yticks(y,labels);ax.invert_yaxis();ax.set_xlabel('cos(delta student, delta teacher)')
ax.set_title('Direction agreement: instantaneous r = t');ax.legend(loc='lower left',fontsize=9)
bx.barh(y,[r['matched_norm_ratio'] for r in data],height=.6,color='#2964ad')
bx.axvline(1,color='#555',ls='--',lw=1,label='Equal response norm')
bx.set_xlabel('Student / matched-teacher action-delta norm');bx.set_title('Response magnitude')
bx.legend(loc='lower right',fontsize=9)
for i,r in enumerate(data):
    bx.text(r['matched_norm_ratio']+.025,i,f"{r['matched_norm_ratio']:.2f}",va='center',fontsize=9)
bx.set_xlim(0,max(r['matched_norm_ratio'] for r in data)*1.18)
for a in (ax,bx):
    a.grid(axis='x',alpha=.2);a.set_axisbelow(True);a.spines[['top','right']].set_visible(False)
fig.suptitle('AnyFlow128: only the CURRENT chunk action changes A to D',fontsize=15,y=.98)
fig.text(.5,.02,'Same generated history and explicitly noised endpoint; not captured solver states. '
    'Local geometry is not a video-quality or action-direction acceptance test.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.05,1,.95))
fig.savefig(OUT/'action_geometry.png',dpi=170)
fig.savefig(OUT/'action_geometry.svg')
