from pathlib import Path
import json,hashlib,subprocess,math
import av,numpy as np
from PIL import Image,ImageDraw
root=Path('/home/qma/work/GWM');e=root/'submission/experiments/EXP-027_abot_baseline_diagnosis';p=e/'recovery_01';out=e/'judge';sheets=out/'frame_review';sheets.mkdir(exist_ok=True)
rows=json.loads((p/'VIDEO_RESULTS.json').read_text());cpu=json.loads((p/'CPU_REVIEW.json').read_text());files=[(root/x['video'],x['sha256']) for x in rows]
for x in cpu['records']:
 files.extend((root/x[k]['path'],x[k]['sha256']) for k in ('reference','paired'))
media=[];refs={}
for f,h in files:
 assert hashlib.sha256(f.read_bytes()).hexdigest()==h
 with av.open(str(f)) as c:
  stream=c.streams.video[0];fps=float(stream.average_rate);decoded=list(c.decode(video=0))
  assert len(decoded)==56 and fps==24 and all(decoded[i].pts<decoded[i+1].pts for i in range(55))
  if 'reference' in f.parts:refs[f.name.split('_source')[0]]=np.stack([x.to_ndarray(format='rgb24') for x in decoded])
  media.append({'path':str(f.relative_to(root)),'sha256':h,'frames':len(decoded),'fps':fps,'width':stream.width,'height':stream.height,'pts_increasing':True})
images=[];metrics=[]
for clip in sorted({x['clip_id'] for x in rows}):
 pair={x['checkpoint_updates']:x for x in rows if x['clip_id']==clip};raw={k:np.load((root/v['video']).with_suffix('.rgb.npy'),mmap_mode='r') for k,v in pair.items()};assert all(v.shape==(56,480,832,3) for v in raw.values());assert np.array_equal(raw[0][:39],raw[64][:39])
 metrics.append({'clip':clip,'prefix39_exactly_equal_pre_post':True,'generated17_mean_abs_pre_post':float(np.abs(raw[0][39:].astype(np.float32)-raw[64][39:].astype(np.float32)).mean())})
 for page,start in enumerate((38,44,50)):
  frame_ids=list(range(start,start+6));w,h=312,180;bar=24;im=Image.new('RGB',(w*6,(h+bar)*3),'#222222');d=ImageDraw.Draw(im)
  for row,(label,frames) in enumerate((('Source',refs[clip]),('0updates',raw[0]),('64updates',raw[64]))):
   for col,i in enumerate(frame_ids):
    x,y=col*w,row*(h+bar);im.paste(Image.fromarray(frames[i]).resize((w,h)),(x,y+bar));d.text((x+5,y+4),f'{clip[:8]} {label} f{i}',fill='white')
  file=sheets/f'{clip[:8]}_page{page+1}.jpg';im.save(file,quality=93);images.append(str(file))
a=json.loads((e/'GPU_LEDGER.json').read_text());b=json.loads((p/'GPU_LEDGER.json').read_text());combined=json.loads((p/'COMBINED_LEDGER.json').read_text());assert a['status']=='FAILED_STOPPED' and b['status']=='COMPLETE_PENDING_JUDGE';assert a['counts']['dit']+b['counts']['dit']==303 and b['counts']['update']==64 and b['counts']['vae_decode']==4;assert abs(a['elapsed_seconds']+b['elapsed_seconds']-combined['cumulative_gpu_process_seconds'])<1e-6;assert combined['cumulative_gpu_process_seconds']<6000
assert len(rows)==4 and len(media)==8
status=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True);assert any(line.strip()=='0, 0' for line in status.splitlines())
result={'status':'LEDGER_AND_ALL_MEDIA_VERIFIED_VISUAL_REVIEW_PENDING','media':media,'raw_frame_checks':metrics,'review_sheets':images,'all_generated_frames_covered':'38 boundary plus every generated frame 39..55, source/0/64, two clips','gpu_seconds':combined['cumulative_gpu_process_seconds'],'counts':combined['cumulative_counts'],'successful_dit_calls':302,'failed_partial_dit_attempts':1,'gpu0_release_snapshot':status,'rss_limit':'final snapshot only, no verified peak','sampling_seconds':[{'clip':x['clip_id'],'updates':x['checkpoint_updates'],'seconds':x['sampling_c2_seconds']} for x in rows]}
(out/'FINAL_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'videos':len(media),'sheets':images,'gpu_seconds':result['gpu_seconds']},indent=2))
