"""CPU real-layout check; does not load a pretrained model or run a DiT."""
from pathlib import Path
import sys,json
BASE=Path(__file__).resolve().parent;H3=BASE.parents[2]
sys.path[:0]=[str(H3/'code'),str(H3/'DiffSynth-Studio-h3-v2')]
import torch
from audit_core import mask,tensor_hash,ScopedActionAttention
from causal.local_topology import visible_inputs,OriginalWindowAttention
from causal.h3_cached import H3ChunkCache,ChunkAttention

torch.set_num_threads(4)
src=H3/'outputs/2026-10-09-22/chunk_partition_cb/source_coarse/inputs'
data={a:torch.load(src/f'parking_{a}.pt',map_location='cpu',weights_only=True) for a in 'AD'}
assert torch.equal(data['A']['packed']['img_position_ids'],data['D']['packed']['img_position_ids'])
p,t=visible_inputs(data['A']['packed'],data['A']['pairs'][0]['prompts']['A'],17,390)
v0=p['action_video_start'];boundary=v0+12*390
rp=mask(p,17,390,12,causal=False,public_feedback=False)
r2=mask(p,17,390,12,causal=True,public_feedback=False)
change=rp & ~r2;expected=torch.zeros_like(change);expected[v0:boundary,boundary:]=True
assert torch.equal(change,expected) and not bool((r2 & ~rp).any())
assert torch.equal(rp[:v0],r2[:v0]) # includes every historical/current action feedback row
for i,(lo,hi) in enumerate(p['action_text_rows'].tolist()):
    assert bool(r2[lo:hi,v0+i*390:v0+(i+1)*390].all())
    assert bool(r2[v0+i*390:v0+(i+1)*390,lo:hi].all())
common=dict(cache=H3ChunkCache(),index=1,prefix=v0,frame_rows=390,frame_start=12,
            action_rows=p['action_text_rows'],action_feedback=True)
routes={}
for route in ('own','within','cross','full'):
    c=ScopedActionAttention(**common);c.route=route;c.chunk_start=12
    routes[route]=c.masks(5*390,12*390,'cpu')[1]
native=ChunkAttention(**common,action_prefix_mode='causal').masks(5*390,12*390,'cpu')[1]
assert torch.equal(routes['full'],native)
assert torch.equal(routes['within']|routes['cross'],routes['full'])
assert torch.equal(routes['within']&routes['cross'],routes['own'])
result=dict(passed=True,device='cpu',neural_forwards=0,latent_intervals=[[0,12],[12,17]],
    spatial_tokens_per_latent=390,prefix_rows=v0,R1P_R2_removed_edges=int(change.sum()),
    only_difference='history-video queries reading current-video keys; action feedback and prefix mask identical',
    global_positions_AD_equal=True,visible_positions_sha256=tensor_hash(p['img_position_ids']),
    full_route_matches_production_causal=True,within_cross_union_full=True,within_cross_intersection_own=True,
    additional_video_action_edges={k:int((m & ~routes['own']).sum()) for k,m in routes.items()})
(BASE/'real_layout_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
