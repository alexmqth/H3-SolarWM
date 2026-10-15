from pathlib import Path
import sys,json,hashlib
from unittest.mock import patch
import numpy as np,torch
R=Path('/home/qma/work/GWM');P=R/'submission/experiments/EXP-023_v3_fm_small_data';sys.path.insert(0,str(P))
import fm_run as fm
import finalize_cpu as gallery
from common import ENCODED,inputs,write
out=P/'judge/media_recovery_cpu';out.mkdir(parents=True,exist_ok=False)
_,_,_,train,dev,roles=inputs();clip=roles['train_constant']['clip_id']
f=torch.load(ENCODED/clip/'condition.pt',map_location='cpu',weights_only=True);record=json.loads((ENCODED/clip/'record.json').read_text());data={'f':f,'record':record}
assert f['initial_noise'].dtype==torch.bfloat16
frames=np.load(P/'media_smoke_identity/source_identity_path_test.rgb.npy')
assert frames.shape==(56,480,832,3)
checks=[]
for role,phase,history in [('train_constant','eval_pre','GT first39 displayed and committed'),('dev_switch','eval_post','self-generated C1 and C2')]:
 path=out/(role+'.mp4');raw=fm.persist_rgb(frames,path)
 identity=fm.video_identity(data,role,phase,history);write(path.with_suffix('.identity.json'),identity)
 fm.save_video(frames,path);arr,fps=gallery.read_video(path)
 assert arr.shape==frames.shape and fps==24 and np.array_equal(np.load(raw),frames)
 checks.append({'case':role,'BF16_identity':True,'raw_recoverable':True,'decoded_frames':56})
for failure in ['identity','codec']:
 path=out/(failure+'_failure.mp4');raw=fm.persist_rgb(frames,path)
 try:
  if failure=='identity':
   with patch.object(fm,'video_identity',side_effect=TypeError('injected identity failure')):fm.video_identity(data,'train_constant','eval_pre','GT')
  else:
   with patch.object(fm.subprocess,'Popen',side_effect=FileNotFoundError('injected codec failure')):fm.save_video(frames,path)
 except (TypeError,FileNotFoundError):pass
 else:raise AssertionError('fault injection did not execute')
 assert raw.is_file() and np.array_equal(np.load(raw),frames) and not path.exists()
 # CPU-only recovery from already persisted frames, no model execution.
 fm.save_video(np.load(raw),path);assert gallery.read_video(path)[0].shape==frames.shape
 checks.append({'case':failure+'_injected','raw_survives':True,'cpu_recovery':True})
canvas=np.zeros((56,540,2496,3),dtype=np.uint8)
for c in range(3):canvas[:,30:510,c*832:(c+1)*832]=frames
gallery.write_video(canvas,out/'comparison.mp4');assert gallery.read_gallery(out/'comparison.mp4')==(56,24.0)
checks.append({'case':'actual_comparison_writer','decoded_frames':56,'size':[2496,540]})
result={'status':'PASS_JUDGE_CPU_MEDIA_RECOVERY','gpu_calls':0,'checks':checks,'source_sha256':{n:hashlib.sha256((P/n).read_bytes()).hexdigest() for n in ['fm_run.py','finalize_cpu.py']},'scope':'actual BF16 condition and source RGB; labels exercise metadata branches only, no generated video or model quality evidence'}
(P/'judge/MEDIA_RECOVERY_CPU_REVIEW.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
