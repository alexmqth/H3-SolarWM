import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
from pathlib import Path
import json, hashlib, math
from collections import Counter
import torch, av
from PIL import Image,ImageDraw
root=Path('/home/qma/work/GWM'); e=root/'submission/experiments/EXP-023_v3_fm_small_data'; out=e/'judge'; out.mkdir(exist_ok=True)
def load(n):return json.loads((e/n).read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
m=load('checkpoint_128.json'); assert sha(root/m['path'])==m['sha256']
c=torch.load(root/m['path'],map_location='cpu',weights_only=False)
u=load('TRAIN_UPDATES.json'); assert len(u)==128 and c['step']==128 and c['order_pos']==128
assert [x['step'] for x in u]==list(range(1,129))
assert [x['clip_id'] for x in u]==c['order'][:128]
counts=Counter(x['clip_id'] for x in u);assert len(counts)==32 and set(counts.values())=={4}
assert all(math.isfinite(x['loss']) and math.isfinite(x['grad_l2']) and x['grad_l2']>0 for x in u)
assert all(torch.isfinite(v).all() for x in c['adapter'] for v in x.values())
assert all(k in c for k in ['optimizer','cpu_rng','cuda_rng','order_rng','noise_rng'])
manifest=load('TRAIN_RECOVERY_SOURCE_MANIFEST.json'); mismatches=[p for p,s in manifest.items() if sha(root/p)!=s];assert not mismatches,mismatches
names=['ENCODE_PILOT','ENCODE_REST','EVAL_PRE','EVAL_PRE_RETRY','EVAL_PRE_RETRY2','TRAIN','TRAIN_RETRY','EVAL_POST']
budgets=[load(n+'_BUDGET.json') for n in names]; total=load('CUMULATIVE_BUDGET.json')
count_sum={k:sum(b['counts'].get(k,0) for b in budgets) for k in total['counts']};assert count_sum==total['counts']
seconds=sum(b['wall_seconds'] for b in budgets); assert abs(seconds-total['gpu_process_seconds_sum'])<1e-5
video_rows=[]; sheets=[]
for g in load('FINAL_VIDEO_AUDIT.json')['gallery']:
 for row in [g,g['before'],g['after']]:
  p=root/row['video'];assert sha(p)==row['sha256']; pts=[]; frames=[]
  with av.open(str(p)) as container:
   stream=container.streams.video[0]; stream.thread_count=2; assert float(stream.average_rate)==24
   for f in container.decode(video=0):
    pts.append(float(f.pts*f.time_base))
    if row is g:frames.append(f.to_image())
   dims=(stream.width,stream.height)
  assert len(pts)==56 and all(abs((b-a)-1/24)<1e-5 for a,b in zip(pts,pts[1:])),p
  video_rows.append({'path':row['video'],'sha256':sha(p),'frames':len(pts),'fps':24,'dimensions':dims,'pts_uniform':True})
  if row is g:
   ids=list(range(35,56)) if g['history'].startswith('GT') else list(range(56))
   # Every displayed comparison frame, 8 rows per sheet, legible 416px-wide columns.
   for page,start in enumerate(range(0,len(ids),8)):
    subset=ids[start:start+8]; canvas=Image.new('RGB',(1248,len(subset)*290),(20,20,20)); d=ImageDraw.Draw(canvas)
    for j,i in enumerate(subset):
     canvas.paste(frames[i].resize((1248,270)),(0,j*290));d.text((5,j*290+272),f'{g["role"]} {g["history"]} RGB {i}',fill='white')
    p2=out/f'{p.stem}_page{page}.jpg';canvas.save(p2,quality=88);sheets.append(str(p2.relative_to(root)))
probes=load('POST_COMPARISON.json'); before=sum(x['fm_mse_0'] for x in probes['probe'])/8;after=sum(x['fm_mse_128'] for x in probes['probe'])/8
result={'status':'PASS_CPU_EVIDENCE_RECONCILIATION','checkpoint128_sha256':m['sha256'],'updates':128,'training_clips':32,'passes_per_clip':4,'checkpoint_order_matches':True,'finite_parameters_and_positive_recorded_gradients':True,'optimizer_state_items':len(c['optimizer']['state']),'recovery_manifest_files_verified':len(manifest),'budget_counts':count_sum,'gpu_process_seconds_sum':seconds,'dit_note':'444 budget reservations includes one failed C2 entry before model execution; not 444 successful forwards','videos':video_rows,'visual_review_sheets':sheets,'probe_mse_before':before,'probe_mse_after':after,'ratio_of_means_change_pct':100*(after/before-1),'mean_of_individual_relative_changes_pct':sum(x['relative_change_pct'] for x in probes['probe'])/8,'limitations':['No new GPU calls','Cache first-backward invariant checked by recorded signature, not fresh all-byte tensor comparison','Decoded all video frames; visual verdict is separate']}
(out/'FINAL_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['videos','visual_review_sheets']},indent=2));print('SHEETS',len(sheets))
