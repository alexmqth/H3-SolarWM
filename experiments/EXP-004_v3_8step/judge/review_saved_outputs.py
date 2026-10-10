from pathlib import Path
from PIL import Image,ImageDraw
import numpy as np,json,hashlib,av,sys
branch=sys.argv[1];start=int(sys.argv[2]);stop=start+5;n0=39 if start==12 else 56;n=56 if start==12 else 73
D=Path('H3-World/outputs/EXP-004_v3_8step')/branch;O=Path('submission/experiments/EXP-004_v3_8step/judge');B=Path('H3-World/outputs/EXP-002_native_cached')/branch
j=json.loads((D/f'chunk_{start}_{stop}.json').read_text());a=np.load(D/f'published_{n}.npy',mmap_mode='r');assert a.shape==(n,480,832,3)
old=np.load(B/'published_56.npy',mmap_mode='r')[:39] if start==12 else np.load(D/'published_56.npy',mmap_mode='r')
baseline=json.loads((B/f'chunk_{start}_{stop}.json').read_text())
checks={'prefix_frozen':np.array_equal(a[:n0],old),'RGB_sha':hashlib.sha256(a.tobytes()).hexdigest()==j['published_RGB_sha256'],'endpoint_sha':hashlib.sha256((D/f'chunk_{start}_{stop}.pt').read_bytes()).hexdigest()==j['endpoint_sha256'],'memory_cache_unchanged':j['memory_cache_read_unchanged'],'cache_file_unchanged':j['cache_file_unchanged_during_sampling'],'frozen_parameters':j['frozen_parameter_versions_unchanged'],'eight_sampling_forwards':j['sampling_forwards']==8,'native8_sigma_list':len(j['sigmas'])==9 and j['sigmas'][0]==1 and j['sigmas'][-1]==0}
keys=['initial_noise_sha256','anchor_sha256','audio_sha256','prompt_sha256','position_sha256','first12_cache_sha256']
if start==12:keys+=['history_tensor_sha256','prior_RGB_sha256']
checks['matched_baseline_conditions']={k:j[k]==baseline[k] for k in keys}
assert all(v for k,v in checks.items() if k!='matched_baseline_conditions') and all(checks['matched_baseline_conditions'].values()),checks
if start==17:checks['uses_own8step_second']=j['second_endpoint_sha256']==json.loads((D/'chunk_12_17.json').read_text())['endpoint_sha256'];assert checks['uses_own8step_second']
with av.open(str(D/f'rollout_{n}.mp4')) as c:
 s=c.streams.video[0];pts=[f.pts for f in c.decode(video=0)];assert len(pts)==n and str(s.average_rate)=='24' and all(y>x for x,y in zip(pts,pts[1:]))
checks.update(frames=n,fps=24,flow=j['flow']['horizontal_flow_px']['mean'],baseline30_flow=baseline['flow']['horizontal_flow_px']['mean'],sampling_seconds=j['sampling_seconds'],baseline30_sampling_seconds=baseline['sampling_seconds'],boundary_gray_MAD=j['boundary_gray_MAD'],GPU_peak_allocated_MiB=j['GPU_peak_allocated_MiB'])
(O/f'{branch}_{n}_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
sheet=Image.new('RGB',(1664,1300),'white');draw=ImageDraw.Draw(sheet)
for k,i in enumerate(range(n0-1,n)):
 x=k%4*416;y=k//4*260;sheet.paste(Image.fromarray(a[i]).resize((416,240)),(x,y));draw.text((x,y+240),f'8step {branch} RGB{i}',fill='black')
sheet.save(O/f'{branch}_{n}.jpg');Image.fromarray(a[-1]).save(f'/tmp/exp004_{branch}_{n}_last.png');print(json.dumps(checks))
