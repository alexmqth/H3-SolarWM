"""Independent saved-tensor/ledger/video integrity audit. No GPU calls."""
from pathlib import Path
import hashlib,json
import torch,av,numpy as np
from PIL import Image,ImageDraw
E=Path(__file__).resolve().parents[1];R=E.parents[2];O=R/'H3-World/outputs/EXP-016_v3_real56_vae'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
b=read(O/'budget.json');res=read(O/'result.json');assert b['status']=='complete' and res['status']=='complete_pending_judge'
assert (b['image_encode'],b['video_encode'],b['video_decode'],b['model_loads'])==(4,8,4,1)
assert b['gpu_wall_seconds']<=600 and res['peak_allocated_gib']<=44
parent=read(E.parent/'EXP-015_v3_real56_data/OUTPUT_MANIFEST.json');checks=[]
for row in parent['scenes']:
 sid=row['scene'];m=read(O/sid/'metrics.json');assert m['status']=='complete_pending_visual_review'
 assert m['source_video_sha256']==row['output_files']['real56.mp4']['sha256'] and m['source_I0_sha256']==row['output_files']['I0.png']['sha256']
 for f in m['files'].values():
  p=Path(f['path']);assert p.stat().st_size==f['bytes'] and sha(p)==f['sha256']
 tensors={k:torch.load(m['files'][k]['path'],map_location='cpu',weights_only=True) for k in ['I0_latent','full17','prefix12']}
 for k,t in tensors.items():assert torch.isfinite(t).all() and list(t.shape)==m['files'][k]['shape'] and str(t.dtype)==m['files'][k]['dtype']
 assert tuple(tensors['full17'].shape)==(1,24,17,30,52) and tuple(tensors['prefix12'].shape)==(1,24,12,30,52)
 assert tuple(tensors['I0_latent'].shape)==(1,24,1,30,52)
 assert torch.equal(tensors['full17'][:,:,:12],tensors['prefix12']);assert m['prefix_comparison']['max_abs']==0
 clips={}
 for key in ['reconstruction','source_vs_reconstruction']:
  with av.open(m['files'][key]['path']) as c:
   s=c.streams.video[0];s.thread_count=2;fs=list(c.decode(video=0));assert len(fs)==56 and s.average_rate==24
   assert all(a.pts<b.pts for a,b in zip(fs,fs[1:]));assert all((f.width,f.height)==tuple(m['files'][key]['resolution']) for f in fs)
   clips[key]=[f.to_ndarray(format='rgb24') for f in fs]
 out=Image.new('RGB',(1664,1008),'white');d=ImageDraw.Draw(out)
 for i,arr in enumerate(clips['reconstruction']):
  x=i%8*208;y=i//8*144;out.paste(Image.fromarray(arr).resize((208,120)),(x,y+24));d.text((x+2,y+4),str(i),fill='black')
 out.save(E/f'judge/{sid}_all56.jpg')
 Image.fromarray(clips['source_vs_reconstruction'][55]).save(E/f'judge/{sid}_source_recon_f55.jpg')
 checks.append({'scene':sid,'prefix_saved_tensors_bit_exact':True,'latent_dtype':str(tensors['full17'].dtype),'saved_files_sha_pass':True,'reconstruction_full_decode_frames':56,'comparison_full_decode_frames':56,'MAD_mean_before_mp4':m['roundtrip']['MAD_mean'],'PSNR_mean_before_mp4':m['roundtrip']['PSNR_mean']})
o={'status':'PASS_ENCODING_PROTOCOL_AND_FILES','task':'EXP-016/v1','GPU_calls_by_this_audit':0,'counts':{k:b[k] for k in ['image_encode','video_encode','video_decode','model_loads']},'gpu_wall_seconds':b['gpu_wall_seconds'],'gpu_hours':b['gpu_wall_seconds']/3600,'peak_allocated_gib':res['peak_allocated_gib'],'budget_sha256':sha(O/'budget.json'),'checks':checks,'visual_acceptance':'separate Judge review; no generation/training claim'}
(E/'judge/INDEPENDENT_RESULT_AUDIT.json').write_text(json.dumps(o,indent=2)+'\n');print(json.dumps(o,indent=2))
