"""Check zero-weight equivalence and generate tiny-H3 auxiliary integration data."""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys
from types import SimpleNamespace

OUT = Path(__file__).resolve().parent
RUNTIME = OUT/'runtime'
sys.path[:0] = [str(RUNTIME/'code'), str(RUNTIME/'code/abot'), str(RUNTIME/'DiffSynth-Studio-h3-v2')]
import torch
from causal.h3_training import make_small_h3, synthetic_h3_batch
from causal.h3_precision import configure_precision
from causal.pretrained_lora import load_adapter
from causal.stage1_lora import load_stage1_lora
from causal.anyflow import load_anyflow
from causal.h3_cached import chunk_forward
from causal.train_stage1_anyflow import clean_cache, condition


def equal(a,b):
    if torch.is_tensor(a):
        return torch.equal(a,b)
    if isinstance(a,dict):
        return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):
        return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


torch.set_num_threads(2)
checks={}
for name in ('causal_adapter.pt','anyflow_adapter.pt','stage1_lora.pt'):
    a=torch.load(OUT/'cpu_original'/name,map_location='cpu',weights_only=True)
    b=torch.load(OUT/'cpu_control'/name,map_location='cpu',weights_only=True)
    a.pop('metadata');b.pop('metadata');checks[name]=equal(a,b)
a=torch.load(OUT/'cpu_original/trainer_state.pt',map_location='cpu',weights_only=True)
b=torch.load(OUT/'cpu_control/trainer_state.pt',map_location='cpu',weights_only=True)
for key in ('optimizer','logical_rng_state','cpu_rng_state','cuda_rng_state'):
    checks[key]=equal(a[key],b[key])
assert all(checks.values()),checks

spec=importlib.util.spec_from_file_location('refinement',OUT.parent/'stage1_finite_interval_refinement128/probe.py')
probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe);probe.torch=torch
torch.manual_seed(13)
model=make_small_h3().eval()
for block in model.blocks:
    block.attn.qkv_proj=block.attn.qkv_proj.base
init=OUT/'cpu_control/step_00'
load_adapter(model,init/'causal_adapter.pt','cpu')
configure_precision(model,'h3_fp32')
load_anyflow(model,init/'anyflow_adapter.pt','cpu')
load_stage1_lora(model,init/'stage1_lora.pt')
model.requires_grad_(False)
batch=synthetic_h3_batch(frames=12,seed=31)
case=dict(label='synthetic',clean=batch['clean_video'],packed=batch['packed'],
    prompt=batch['prompt_embeds'],audio=batch['audio_latents'],
    anchors=[batch['anchor_rows']]*3,actions=None,action_adapter=None)
args=SimpleNamespace(chunk_frames=5,history_chunks=5,anchor_mode='fixed',
                     action_prefix_mode='causal',action_feedback=True)
refdir=OUT/'cpu_reference';refdir.mkdir(exist_ok=False)
report=dict(status='running',checkpoint=str(init),records=[],cache_checks=[],
    clean_sha256=probe.tensor_sha(case['clean']),prompt_sha256=probe.tensor_sha(case['prompt']),
    audio_sha256=probe.tensor_sha(case['audio']),action_sha256=None,
    anchor_sha256=[probe.tensor_sha(x) for x in case['anchors']])
versions=[(p,p._version) for p in model.parameters()]
with torch.no_grad():
    for chunk in range(3):
        cache=clean_cache(model,case,chunk,args)
        before=probe.cache_digest(cache);commits=cache.commits
        common=condition(case,chunk,args)
        clean=case['clean'][:,:,chunk*5:(chunk+1)*5]
        noise=batch['noise'][:,:,chunk*5:(chunk+1)*5]
        t=probe.SIGMAS[7]
        def velocity(z,sigma,target):
            return chunk_forward(model,z,sigma=sigma,target_sigma=target,
                                 index=chunk,cache=cache,**common)
        record=probe.interval(velocity,(1-t)*clean+t*noise,t,0.)
        record.update(chunk=chunk,native_step=7)
        path=refdir/f'chunk_{chunk}.pt'
        torch.save(dict(tensors=probe.INTERVAL_TENSORS,record=record,
                        action='synthetic',checkpoint=str(init)),path)
        record.update(target_file=str(path),target_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        report['records'].append(record)
        assert cache.commits==commits and probe.cache_digest(cache)==before
        report['cache_checks'].append(dict(chunk=chunk,unchanged=True))
        cache.clear()
assert all(p._version==v for p,v in versions)
report.update(status='complete',parameter_versions_unchanged=True)
(refdir/'probe_synthetic.json').write_text(json.dumps(report,indent=2)+'\n')
# A changed action must be rejected even when all tensor shapes still match.
from causal.finite_interval_consistency import load_reference
load_reference(case,0,refdir)
bad=dict(case,actions=torch.zeros(12,1))
try:
    load_reference(bad,0,refdir)
    raise AssertionError('Changed action was accepted')
except ValueError as exc:
    assert 'actions changed' in str(exc)
checks['changed_action_rejected']=True
(OUT/'cpu_zero_equivalence.json').write_text(json.dumps(checks,indent=2)+'\n')
print(json.dumps(checks))
