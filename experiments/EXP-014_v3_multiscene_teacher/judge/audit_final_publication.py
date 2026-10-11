from pathlib import Path
import json,hashlib,datetime
import av,numpy as np
root=Path('/home/qma/work/GWM'); e=root/'submission/experiments/EXP-014_v3_multiscene_teacher'; raw=root/'H3-World/outputs/EXP-014_v3_multiscene_teacher'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def decode(p):
 with av.open(str(p)) as c:
  s=c.streams.video[0]; assert float(s.average_rate)==24
  fs=list(c.decode(video=0)); pts=[f.pts for f in fs]
  assert len(fs)==56 and all(a<b for a,b in zip(pts,pts[1:]))
  return [f.to_ndarray(format='rgb24') for f in fs]
count=0
for scene,rows in read(e/'artifacts/recovery_v2/copied_artifact_manifest.json').items():
 for rel,v in rows.items():
  src=root/v['source']; dst=e/'artifacts/recovery_v2'/scene/rel
  assert src.stat().st_size==dst.stat().st_size==v['bytes'] and sha(src)==sha(dst)==v['sha256'];count+=1
comparisons={}
for scene in ['s1_7199292c','s3_b784d995']:
 p=e/f'artifacts/recovery_v2_comparisons/{scene}_AA_vs_recovered_AD_56.mp4'; frames=decode(p)
 assert frames[0].shape==(552,1664,3)
 rows={}
 for branch,x,source in [('AA',0,raw/f'G2/{scene}/FM30/AA/rollout_56.mp4'),('AD',832,raw/f'recovery_v2/{scene}/AD/rollout_56.mp4')]:
  src=decode(source)
  vals=[float(np.abs(f[44:524,x+4:x+828].astype(np.float32)-s[:,4:828].astype(np.float32)).mean()) for f,s in zip(frames,src)]
  assert max(vals)<8,(scene,branch,max(vals))
  rows[branch]={'mean':float(np.mean(vals)),'max':max(vals)}
 comparisons[scene]={'sha256':sha(p),'frames':56,'source_alignment_MAD':rows}
out={'status':'PASS_PUBLICATION_INTEGRITY_ONLY','raw_copies_checked':count,'comparisons':comparisons}
(e/'judge/RECOVERY_PUBLICATION_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n')
ledgers=list((raw/'teacher').glob('*/budget.json'))+list((raw/'recovery_v2').glob('*/budget.json')); assert len(ledgers)==6
lo=hi=0;sampling=commit=dec=0;details=[]
for p in ledgers:
 d=read(p); sampling+=d['sampling_forwards']; commit+=d['commit_forwards'];dec+=d['vae_decodes']
 if d['gpu_seconds']==0:
  times=[datetime.datetime.fromisoformat(v['at']) for v in d['events']]
  lower=(times[-1]-times[0]).total_seconds()
  upper=(datetime.datetime.fromisoformat('2026-10-10T23:55:17+00:00')-times[0]).total_seconds()
 else:lower=upper=d['gpu_seconds']
 lo+=lower;hi+=upper;details.append({'path':str(p.relative_to(root)),'sha256':sha(p),'seconds_bounds':[lower,upper]})
w=read(e/'WORKER_CUMULATIVE_BUDGET.json')['teacher_generation']
assert (sampling,commit,dec)==(398,4,12)
assert abs(lo-w['gpu_seconds_lower_bound'])<1e-5 and abs(hi-w['gpu_seconds_upper_bound'])<1e-5
enc=read(raw/'P1/budget.json'); assert (enc['text_encoder_calls'],enc['image_vae_encodes'])==(12,4)
result={'status':'PASS','reserved_denoiser_forwards':sampling+commit,'confirmed_completed_minimum':sampling+commit-2,'uncertain_interrupted_calls':2,'decodes_completed':dec,'teacher_seconds_bounds':[lo,hi],'encoding_seconds':enc['gpu_seconds'],'total_gpu_hours_bounds':[(lo+enc['gpu_seconds'])/3600,(hi+enc['gpu_seconds'])/3600],'training_updates':0,'source_ledgers':details}
(e/'judge/FINAL_BUDGET_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'publication':out,'budget':result},indent=2))
