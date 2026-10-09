"""Validate real paired-context artifacts and a tiny mixed-precision KV probe."""
import ast
import hashlib
import json
from pathlib import Path
import sys
import torch

BASE=Path(__file__).resolve().parent;RT=BASE/'runtime'
sys.path[:0]=[str(RT/'code'),str(RT/'DiffSynth-Studio-h3-v2')]
import geometry_primitives as g
from causal.h3_training import make_small_h3,synthetic_h3_batch
from causal.h3_precision import configure_precision
from causal.h3_cached import H3ChunkCache,chunk_forward,expand_packed_two_anchors

torch.set_num_threads(2);torch.manual_seed(13)
provenance=json.loads((BASE/'geometry_primitive_provenance.json').read_text())
code=(BASE/'geometry_primitives.py').read_text()
for node in ast.parse(code).body:
    if isinstance(node,(ast.FunctionDef,ast.ClassDef)):
        assert hashlib.sha256(ast.get_source_segment(code,node).encode()).hexdigest()==provenance[node.name]['definition_sha256']
prep=json.loads((BASE/'counterfactual/preparation.json').read_text());assert prep['status']=='complete' and len(prep['records'])==6
for r in prep['records']:
    p=Path(r['file']);assert hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256']
    pair=torch.load(p,map_location='cpu',weights_only=True)
    a,d=pair['prompts']['A'],pair['prompts']['D'];assert a.shape==d.shape
    spans=pair['packed']['action_text_spans_local'];allowed=torch.zeros(len(a),dtype=torch.bool)
    for lo,hi in spans[pair['start']:pair['stop']]:allowed[int(lo):int(hi)]=True
    assert torch.equal(a[~allowed],d[~allowed]) and not torch.equal(a[allowed],d[allowed])
    assert all(torch.isfinite(x).all() for x in [a,d])

m=make_small_h3().eval()
for block in m.blocks:block.attn.qkv_proj=block.attn.qkv_proj.base
m=m.bfloat16().requires_grad_(False);configure_precision(m,'h3_fp32')
b=synthetic_h3_batch(frames=12);packed=expand_packed_two_anchors(b['packed'],frame_rows=4)
pa=b['prompt_embeds'].bfloat16();pd=pa.clone()
for lo,hi in b['packed']['action_text_spans_local'][5:10]:pd[int(lo):int(hi)]+=1
anchor=torch.cat([b['anchor_rows']]*2).bfloat16();history=b['clean_video'][:,:,:5]
cond={'audio_noise':b['audio_latents'].bfloat16()}
def common(prompt):return dict(full_packed=packed,prompt=prompt,anchor=anchor,audio=cond['audio_noise'],chunk_frames=5,anchor_slot=1,action_prefix_mode='causal',action_feedback=True)
with torch.no_grad():
    caches=[]
    for prompt in [pa,pd]:
        c=H3ChunkCache(5,'cpu');chunk_forward(m,history,index=0,cache=c,sigma=0.,commit=True,**common(prompt));caches.append(c)
    assert g.cache_digest(caches[0])==g.cache_digest(caches[1]);before=g.cache_digest(caches[0]);commits=caches[0].commits
    z=.4*b['clean_video'][:,:,5:10]+.6*b['noise'][:,:,5:10]
    student=[chunk_forward(m,z,index=1,cache=caches[0],sigma=.6,anchor_frame_index=4,**common(prompt)) for prompt in [pa,pd]]
    teacher=[g.forward(m,z,history,cond,prompt,anchor,packed,.6,mode='original') for prompt in [pa,pd]]
    assert all(torch.isfinite(x).all() for x in student+teacher)
    assert g.rms(student[0]-student[1])>0 and g.rms(teacher[0]-teacher[1])>0
    assert g.cache_digest(caches[0])==before and caches[0].commits==commits
    repeated=chunk_forward(m,z,index=1,cache=caches[0],sigma=.6,anchor_frame_index=4,**common(pa))
    assert torch.equal(repeated,student[0])
result=dict(status='passed',real_paired_artifacts=6,exact_same_A_D_layout=True,outside_current_equal=True,
    audited_primitives_source_exact=True,tiny_history_KV_independent_of_future_A_D=True,
    tiny_both_action_responses_nonzero=True,tiny_repeat_error=0.,tiny_readonly_cache=True,
    scope='real condition tensors plus tiny H3 BF16-stack/FP32-boundary mechanics, not33B quality')
(BASE/'counterfactual_cpu.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
