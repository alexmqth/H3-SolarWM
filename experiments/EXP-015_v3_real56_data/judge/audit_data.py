"""Judge audit: raw binary annotations, native spans, future isolation and video integrity."""
from pathlib import Path
import json,hashlib,tarfile,sys
import numpy as np
import av
from PIL import Image
E=Path(__file__).resolve().parents[1];R=E.parents[2]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
sys.path.insert(0,str(R/'H3-World/code/abot'))
import abot_action as A
import action_script as S
m=read(E/'OUTPUT_MANIFEST.json');expected=read(E.parent/'EXP-013_v3_multiscene_data_plan/candidate_manifest.json')['selected']
assert len(m['scenes'])==4
checks=[]
keys=['W','A','S','D','Q','E','I','J','K','L','Space']
for row,sel in zip(m['scenes'],expected):
 sid=sel['episode_id'];assert row['episode_id']==sid and row['split']=='train'
 idx=[sel['src_start']+5*(j//4)+j%4 for j in range(56)];assert idx==row['source_frame_indices']
 base=R/f'H3-World/data/abot_bridge/raw/data/{sid[:2]}/{sid}'
 assert sha(base/'video.mp4')==row['source_video_sha256'] and sha(base/'annotations.tar')==row['annotations_sha256']
 for name,item in row['output_files'].items():
  p=Path(item['path']);assert p.stat().st_size==item['bytes'] and sha(p)==item['sha256']
 out={k:Path(v['path']) for k,v in row['output_files'].items()}
 with tarfile.open(base/'annotations.tar') as t:ann=json.load(t.extractfile('action.json'))
 raw=np.load(out['raw56.npy']);assert raw.shape==(56,17) and np.isfinite(raw).all()
 target=np.array([[int(ann['frames'][j]['keys'].get(k,False)) for k in keys] for j in idx])
 assert np.array_equal(raw[:,:11],target)
 ep=A.read_episode(str(base/'annotations.tar'));recon=A.window_action_matrix(ep,sel['src_start'],56,A.episode_translation_scale(ep))
 assert np.array_equal(raw,recon)
 widths=([1,4,4,4,4]*4)[:17];ends=np.cumsum(widths);starts=np.r_[0,ends[:-1]]
 assert ends[11]==39 and ends[-1]==56
 pooled=np.load(out['pooled17.npy']);p=np.zeros((17,17),np.float32)
 for j,(a,b) in enumerate(zip(starts,ends)):
  p[j,:11]=raw[a:b,:11].max(axis=0);p[j,11:14]=np.clip(raw[a:b,11:14].sum(axis=0)/4,-3,3);p[j,14:]=np.clip(raw[a:b,14:].sum(axis=0)/4,-4,4)
 assert np.array_equal(pooled,p)
 k9=np.load(out['keys9_17.npy']);assert np.array_equal(k9,S.keys9(pooled))
 script=read(out['action_script17.json']);assert script==S.annotate_from_keys9(k9)
 altered=raw.copy();altered[39:,:11]=1-altered[39:,:11];altered[39:,11:]=10000
 alt=S.keys9(A.bin_to_latent(altered,17));assert np.array_equal(alt[:12],k9[:12]);assert S.annotate_from_keys9(alt)[:12]==script[:12]
 with av.open(str(out['real56.mp4'])) as c:
  stream=c.streams.video[0];stream.thread_count=2;assert stream.average_rate==24
  fs=list(c.decode(video=0));assert len(fs)==56 and all((f.width,f.height)==(832,480) for f in fs)
  assert all(a.pts<b.pts for a,b in zip(fs,fs[1:]))
  assert np.array_equal(fs[0].to_ndarray(format='rgb24'),np.asarray(Image.open(out['I0.png']).convert('RGB')))
 checks.append({'episode_id':sid,'raw_key_rows':56,'key_columns':11,'native_spans':17,'full_decode_frames':56,'C1_future_mutation_invariant':True,'script_equal_native':True,'continuous_source_helper_max_abs_error':float(np.max(np.abs(raw-recon)))})
result={'status':'PASS_CPU_DATA_AND_ACTION_INTEGRITY','task':'EXP-015/v1','complete_decoded_frames':224,'manifest_sha256':sha(E/'OUTPUT_MANIFEST.json'),'checks':checks,'scope':'Binary labels independently read raw JSON; continuous labels recomputed with existing source helper; pixel/source alignment and visual quality reviewed separately','GPU_calls':0,'training_updates':0}
(E/'judge/INDEPENDENT_DATA_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
