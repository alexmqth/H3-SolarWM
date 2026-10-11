"""CPU synthetic-span reproducer using actual frozen builder and visible_inputs source."""
from pathlib import Path
import ast,hashlib,json
import numpy as np
import torch
E=Path(__file__).resolve().parents[1];R=E.parents[2]
P=R/'H3-World/outputs/2026-10-09-04/stage1_local_topology/runtime/DiffSynth-Studio-h3-v2/diffsynth/pipelines/minimax_h3_audio_video.py'
L=R/'H3-World/outputs/2026-10-09-22/chunk_partition_cb/runtime/code/causal/local_topology.py'
class DummyUnit:
 def __init__(self,*args,**kwargs):pass
ns={'np':np,'torch':torch,'PipelineUnit':DummyUnit,'MiniMaxH3Pipeline':object}
for path,name in [(P,'MiniMaxH3Unit_PackedSequenceBuilder'),(L,'visible_inputs')]:
 node=next(x for x in ast.parse(path.read_text()).body if getattr(x,'name',None)==name)
 exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),ns)
b=ns['MiniMaxH3Unit_PackedSequenceBuilder']()
def pack(extra):
 spans=[];pos=100
 for k in range(37):
  end=pos+20+(extra if k>=12 else 0);spans.append((pos,end));pos=end
 full=b._build_packed_fl2va(pos,37,30,52,250,[0],action_text_spans=spans)
 prompt=torch.zeros((pos,2))
 return ns['visible_inputs'](full,prompt,12,390)
a,ta=pack(0);c,tc=pack(1)
assert torch.equal(ta,tc) and a['seq_len']==c['seq_len']
vid_a=a['img_position_ids'][0,a['img_pos'][390:],0];vid_c=c['img_position_ids'][0,c['img_pos'][390:],0]
assert torch.equal(vid_c-vid_a,torch.full_like(vid_a,25))
assert torch.equal(a['img_position_ids'][0,:100],c['img_position_ids'][0,:100])
out={'status':'CONFIRMED_LAYOUT_RISK','scope':'Synthetic action lengths, actual frozen CPU builder and visible_inputs; no tokenizer/model generation claim','future_change':'one extra token in each action span12..36; all visible content identical','visible_prompt_equal':True,'visible_sequence_length_equal':True,'visible_video_temporal_coordinate_shift':25,'head_coordinates_unchanged':True,'conclusion':'Removing future rows does not remove dependence on total future text length. Existing equal-length A/D frozen evidence is not invalidated; mixed-action training requires an explicit content-independent layout gate. No protocol edits performed.','source_sha256':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [P,L]},'gpu_calls':0}
(E/'judge/FUTURE_LAYOUT_RISK.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
