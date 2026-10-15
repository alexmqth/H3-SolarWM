from pathlib import Path
import json,hashlib,av
R=Path('/home/qma/work/GWM'); S=R/'submission/experiments'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
for name in ['EXP-019_v3_mixed_action_validation','EXP-020_original_mixed_action_reference']:
 d=S/name; m=json.loads((d/'SOURCE_MANIFEST.json').read_text())
 items=m['files'].values() if 'files' in m else ({'path':p,'sha256':v} for p,v in m.items())
 sources=[]
 for x in items:
  p=R/x['path']; assert sha(p)==x['sha256'],p
  sources.append(x['path'])
 manifests=sorted(d.glob('G2_*_ARTIFACT_MANIFEST.json')) if '019' in name else [d/'ARTIFACT_MANIFEST.json']
 checked=[]; videos=[]
 for mp in manifests:
  for x in json.loads(mp.read_text())['files']:
   for p in (R/x['source'],d/x['artifact']):
    assert p.stat().st_size==x['bytes'] and sha(p)==x['sha256'],p
   checked.append(x['artifact'])
   p=d/x['artifact']
   if p.suffix=='.mp4':
    with av.open(str(p)) as c:
     st=c.streams.video[0]; fs=list(c.decode(st)); pts=[f.pts for f in fs]
     assert len(fs)==56 and st.average_rate==24 and all(a<b for a,b in zip(pts,pts[1:]))
     videos.append({'path':x['artifact'],'frames':len(fs),'fps':24,'resolution':[st.width,st.height]})
 if '019' in name:
  b=json.loads((d/'CUMULATIVE_BUDGET.json').read_text()); rows=b['rows']
  assert sum(x['forward'] for x in rows)==166
  assert abs(sum(x['wall_seconds'] for x in rows)-2052.979802771006)<1e-6
  for scene in ['7199292c','9dc2e588','b784d995']:
   a=d/'artifacts'/('G2_'+scene); z=json.loads((a/'budget.json').read_text()); y=json.loads((a/'result.json').read_text())
   assert z['counts']==dict(forward=39,commit=1,decode=2,text=0)
   assert z['elapsed']<900 and z['peak_allocated_gib']<44
   assert sum(e['kind']=='forward' for e in z['events'])==39
   assert [x['nfe_per_chunk'] for x in y['runs']]==[30,8]
   assert y['runs'][0]['initial_noise_sha256']==y['runs'][1]['initial_noise_sha256']
   assert all(x['history_tensor_unchanged'] and x['history_kv_unchanged'] for x in y['runs'])
  counts={'total_dit':166,'generation_sampling':152,'regression_probe':8,'clean_commit':6,'text':14,'video_decode':8,'training_updates':0}
 else:
  z=json.loads((d/'artifacts/b784d995/result.json').read_text()); assert z['counts']==dict(dit=30,text=4,image_encode=1,video_decode=1,audio_decode=1)
  assert z['status']=='COMPLETE_PENDING_JUDGE' and z['wall_seconds']<1200
  counts=z['counts']|{'training_updates':0}
 out={'reviewer':'Judge','date':'2026-10-15','independent_cpu_integrity':'PASS','source_sha_files_checked':sources,'artifact_source_and_copy_sha_checked':checked,'videos_fully_decoded':videos,'verified_counts':counts,'scope':'CPU-only independent verification; visual assessment in FINAL_REVIEW.md; no new GPU calls'}
 (d/'judge/FINAL_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n')
 print(name,len(sources),'sources;',len(checked),'artifact pairs;',len(videos),'videos PASS')
