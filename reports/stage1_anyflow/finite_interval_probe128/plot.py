"""Standalone diagnostic figure; uses base Python's existing Matplotlib."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT=Path(__file__).resolve().parent
r=json.loads((OUT/'analysis.json').read_text())
labels=[];matrix=[];gap=[];refine=[]
for action in ('A','D'):
    records=r['actions'][action]['coarse']
    for chunk in range(3):
        labels.append(f'{action}, chunk {chunk}')
        matrix.append([100*next(x for x in records if x['chunk']==chunk and x['native_step']==step)['velocity_relative_rmse'] for step in (0,4,7)])
        last=r['actions'][action]['low_interval'][chunk]
        gap.append(last['finite_vs_reference_rmse']['16'])
        refine.append(last['refinement_8_to16'])
fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
im=axes[0].imshow(matrix,cmap='YlOrRd',vmin=0,vmax=22,aspect='auto')
axes[0].set_xticks(range(3),['High noise\n1.000 → 0.940','Middle\n0.689 → 0.571','Last interval\n0.241 → 0'])
axes[0].set_yticks(range(6),labels)
axes[0].set_title('Velocity difference vs 8-substep reference')
for i,row in enumerate(matrix):
    for j,value in enumerate(row):axes[0].text(j,i,f'{value:.1f}%',ha='center',va='center',color='black' if value<15 else 'white')
fig.colorbar(im,ax=axes[0],label='Relative velocity RMSE (%)',shrink=.8)
y=np.arange(6)
axes[1].barh(y-.17,gap,height=.32,label='Finite map vs 16-substep endpoint')
axes[1].barh(y+.17,refine,height=.32,label='8 vs 16-substep endpoint')
axes[1].set_yticks(y,labels);axes[1].invert_yaxis()
axes[1].set_xlabel('Latent endpoint RMSE');axes[1].set_title('Last interval: gap persists under refinement')
axes[1].legend(fontsize=8,loc='upper center',bbox_to_anchor=(.5,-.13))
fig.suptitle('Frozen AnyFlow128, clean teacher history — numerical references are not ground truth',fontsize=12)
fig.savefig(OUT/'interval_errors.png',dpi=160)
