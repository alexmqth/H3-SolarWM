"""Independent short/long future-span interventions, canonical and index checks."""
from pathlib import Path
import sys,json,os,hashlib
import torch
E=Path(__file__).resolve().parents[1];sys.path.insert(0,str(E))
import fixed_layout as M
os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:4]);torch.set_num_threads(4)
h=M.sha(E/'fixed_layout.py');builder,visible=M.frozen_cpu_functions();rows=[]
for scene,name in zip(M.SCENES,M.FIXTURES):
 f=torch.load(M.EXP014/name,map_location='cpu',weights_only=False);cp=f['packed'];head=cp['action_text_spans_local'][0][0]
 widths=[hi-lo for lo,hi in cp['action_text_spans_local']]
 canonical_ids=[[31]*w for w in widths]
 p,_=M.candidate(builder,cp,canonical_ids);assert M.same_layout(p,cp)
 for tag,prompt in f['prompts'].items():
  for stop in [12,17]:
   a,ap=visible(cp,prompt,stop,390);b,bp=visible(p,prompt,stop,390)
   assert M.same_layout(a,b) and torch.equal(ap,bp)
 # Exercise actual visible variable widths without neural embeddings.
 ids=[list(range(50,50+(8+(i%7)))) for i in range(37)]
 original,oid=M.candidate(builder,cp,ids)
 allrows=torch.cat([original[x] for x in ['text_pos','img_pos','audio_pos']]);assert allrows.unique().numel()==allrows.numel()
 assert int(allrows.min())>=0 and int(allrows.max())<original['seq_len']
 trials=[]
 for stop in [12,17]:
  vp,vt=visible(original,oid[:,None],stop,390)
  for mode in ['all_one','alternate_short','long']:
   changed=[x[:] for x in ids]
   for i in range(stop,37):
    n=1 if mode=='all_one' else (1+i%2 if mode=='alternate_short' else 40+i%3)
    changed[i]=[999-i]*n
   q,qid=M.candidate(builder,cp,changed);vq,vqt=visible(q,qid[:,None],stop,390)
   assert M.same_layout(vp,vq) and torch.equal(vt,vqt),(scene,stop,mode)
   trials.append({'stop':stop,'future_mode':mode,'visible_all_fields_and_token_ids_exact':True})
 p12,_=visible(original,oid[:,None],12,390);p17,_=visible(original,oid[:,None],17,390)
 s12=M.semantic_parts(p12,12);s17=M.semantic_parts(p17,12);assert all(torch.equal(s12[k],s17[k]) for k in s12)
 rows.append({'scene':scene,'canonical_A_D_visible_exact':True,'all_physical_indices_disjoint':True,'history_semantics_exact':True,'future_trials':trials})
assert M.sha(E/'fixed_layout.py')==h,'candidate changed during audit'
r={'status':'PASS_CPU_LAYOUT_ONLY','candidate_sha256':h,'GPU_calls':0,'model_forward':0,'scenes':rows,'limitations':['Token IDs/synthetic lengths only for mutation; no new mixed text embeddings','Actual DiT/refiner/backend masks and KV values not tested','No generated video or training claim']}
(E/'judge/INDEPENDENT_CPU_AUDIT.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({'status':r['status'],'scenes':len(rows),'future_trials':sum(len(x['future_trials']) for x in rows),'candidate_sha256':h},indent=2))
