from pathlib import Path
import json,hashlib,math
import av
from PIL import Image,ImageDraw
r=Path('/home/qma/work/GWM');e=r/'submission/experiments/EXP-025_v3_fm_full_coverage';j=e/'judge'
def load(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
l=load(e/'GPU_LEDGER.json');assert l['status']=='COMPLETE_PENDING_JUDGE' and l['error'] is None
assert l['counts']==dict(dit=570,backward=256,update=256,vae_decode=12,vae_encode=0,text=0)
assert abs(sum(l['phase_seconds'].values())-l['elapsed_seconds'])<.1
assert l['elapsed_seconds']<13800 and all(v<l['phase_caps'][k] for k,v in l['phase_seconds'].items())
pre=load(e/'CPU_PREFLIGHT.json');assert pre['implementation_sha256']==sha(e/'run_exp025.py')
for p,h in pre['source_sha256'].items():assert sha(r/p)==h
before=load(e/'eval_before/probes.json');after=load(e/'eval_after/probes.json')
rows=[]
for role,x in before.items():
 for block in ['C1','C2']:
  for a,b in zip(x[block],after[role][block]):
   assert a['sigma']==b['sigma'];rows.append(dict(role=role,block=block,sigma=a['sigma'],before=a['fm_mse'],after=b['fm_mse'],relative_change_pct=100*(b['fm_mse']/a['fm_mse']-1)))
aggregates={}
for key,subset in [('all',rows)]+[(block,[x for x in rows if x['block']==block]) for block in ['C1','C2']]+[(split+'_'+block,[x for x in rows if x['block']==block and x['role'].startswith(split)]) for split in ['train','dev'] for block in ['C1','C2']]:
 a=sum(x['before'] for x in subset)/len(subset);b=sum(x['after'] for x in subset)/len(subset)
 aggregates[key]=dict(mean_before=a,mean_after=b,ratio_of_means_change_pct=100*(b/a-1),mean_individual_relative_change_pct=sum(x['relative_change_pct'] for x in subset)/len(subset),improved=sum(x['after']<x['before'] for x in subset),count=len(subset))
records=[];sheets=[]
raw_rows=load(e/'eval_before/results.json')+load(e/'eval_after/results.json');assert len(raw_rows)==12
for row in raw_rows:
 p=r/row['video'];assert sha(p)==row['sha256']
 with av.open(str(p)) as c:
  stream=c.streams.video[0];stream.thread_count=2;assert stream.width==832 and stream.height==480 and float(stream.average_rate)==24
  pts=[float(f.pts*f.time_base) for f in c.decode(video=0)]
 assert len(pts)==56 and all(abs((b-a)-1/24)<1e-5 for a,b in zip(pts,pts[1:]))
 records.append(dict(path=row['video'],sha256=sha(p),frames=56,fps=24,pts_uniform=True))
# Comparisons may still be rendering: refuse a partial final audit.
gallery=load(e/'comparisons/MEDIA_AUDIT.json');assert len(gallery)==6,'Worker still producing gallery'
for row in gallery:
 p=r/row['video'];assert sha(p)==row['video_sha256']
 with av.open(str(p)) as c:
  stream=c.streams.video[0];stream.thread_count=2;assert (stream.width,stream.height)==(2496,528) and float(stream.average_rate)==24
  frames=[];pts=[]
  for f in c.decode(video=0):frames.append(f.to_image());pts.append(float(f.pts*f.time_base))
 assert len(frames)==56 and all(abs((b-a)-1/24)<1e-5 for a,b in zip(pts,pts[1:]))
 records.append(dict(path=row['video'],sha256=sha(p),frames=56,fps=24,pts_uniform=True))
 ids=list(range(35,56)) if row['kind']=='GT39_generated17' else list(range(56))
 for page,start in enumerate(range(0,len(ids),8)):
  sub=ids[start:start+8];canvas=Image.new('RGB',(1248,len(sub)*284),(20,20,20));draw=ImageDraw.Draw(canvas)
  for n,idx in enumerate(sub):
   canvas.paste(frames[idx].resize((1248,264)),(0,n*284));draw.text((8,n*284+266),f'{row["role"]} {row["kind"]} RGB {idx}',fill='white')
  path=j/f'{row["role"]}_{row["kind"]}_page{page}.jpg';canvas.save(path,quality=88);sheets.append(str(path.relative_to(r)))
out=dict(status='PASS_FINAL_CPU_EVIDENCE_CHECK',gpu_seconds=l['elapsed_seconds'],phase_seconds=l['phase_seconds'],counts=l['counts'],source_hashes_match=True,training_checkpoint_review='MILESTONE_256.json',probe_aggregates=aggregates,probes=rows,videos=records,visual_review_sheets=sheets,quality='VISUAL_REVIEW_PENDING',gpu_calls_by_judge=0)
(j/'FINAL_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k not in ['probes','videos','visual_review_sheets']},indent=2))
