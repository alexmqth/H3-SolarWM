"""Audit actual saved pre-update fork weights, Adam and RNG against source128."""
import argparse
import json
from pathlib import Path
import torch


def equal(a,b):
    if torch.is_tensor(a):return torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


def audit(source,destination):
    checks={}
    for name in ('causal_adapter.pt','action_adapter.pt','stage1_lora.pt','anyflow_adapter.pt'):
        a=torch.load(source/name,map_location='cpu',weights_only=True)
        b=torch.load(destination/name,map_location='cpu',weights_only=True)
        a.pop('metadata');b.pop('metadata');checks[name]=equal(a,b)
    a=torch.load(source/'trainer_state.pt',map_location='cpu',weights_only=True)
    b=torch.load(destination/'trainer_state.pt',map_location='cpu',weights_only=True)
    for key in ('optimizer','optimizer_step','updates','teacher_artifacts',
                'logical_rng_state','cpu_rng_state','cuda_rng_state'):
        checks[key]=equal(a[key],b[key])
    mutable={'out_dir','resume_from','steps','checkpoint_every','device',
             'interval_consistency_weight','interval_reference_dir'}
    old={k:v for k,v in a['config'].items() if k not in mutable}
    new={k:v for k,v in b['config'].items() if k not in mutable}
    checks['all_other_config_equal']=old==new
    assert all(checks.values()),checks
    return dict(source=str(source),destination=str(destination),checks=checks,
                objective_extension={k:b['config'][k] for k in ('interval_consistency_weight','interval_reference_dir')},
                note='Explicit objective fork, not an unchanged-objective continuation')


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('source',type=Path);ap.add_argument('destination',type=Path);ap.add_argument('output',type=Path)
    args=ap.parse_args();r=audit(args.source,args.destination)
    args.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
