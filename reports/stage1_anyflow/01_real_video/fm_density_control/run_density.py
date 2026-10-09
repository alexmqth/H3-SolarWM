"""Run the frozen trainer with a mandatory step00 equality guard.

This wrapper does not alter model, loss, optimizer, tensor values or RNG.
It checks initialization before the first model evaluation and audits the
expected sigma/weight/RNG stream after each logical batch.
"""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

BASE=Path(__file__).resolve().parent
CONTROL=BASE.parents[1]/'2026-10-08-18/stage1_real_abot_fm'
sys.path[:0]=[str(BASE/'runtime/code'),str(BASE/'runtime/DiffSynth-Studio-h3-v2')]
import torch
from causal import train_stage1_anyflow as trainer
from causal.anyflow import logical_time_pairs
from causal.anyflow_reference import gaussian_timestep_weights


def equal_tree(a,b):
    if torch.is_tensor(a):return torch.is_tensor(b) and torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal_tree(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal_tree(x,y) for x,y in zip(a,b))
    return a==b


def main():
    protocol=json.loads((BASE/'protocol.json').read_text())
    assert protocol['maximum_updates']==48
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    assert sha(CONTROL/'train_48/training.json')==protocol['control_training_sha256']
    for name,h in json.loads((BASE/'runtime_manifest.json').read_text()).items():
        assert sha(BASE/'runtime'/name)==h,name
    assert json.loads((BASE/'preflight.json').read_text())['status']=='passed'
    expected=json.loads((BASE/'preflight.json').read_text())['samples']
    audit=dict(status='pending',at=datetime.now().astimezone().isoformat(),batches=[])
    def save():
        temp=BASE/'stream_audit.tmp.json';temp.write_text(json.dumps(audit,indent=2)+'\n');temp.replace(BASE/'stream_audit.json')
    raw=trainer.logical_batch
    initialized=False;step=0
    def checked(model,case,chunk,args,*,generator,backward):
        nonlocal initialized,step
        if not initialized:
            load=lambda p:torch.load(p,map_location='cpu',weights_only=True)
            for name in ('causal_adapter.pt','stage1_lora.pt'):
                old=load(CONTROL/'train_48/step_00'/name);new=load(args.out_dir/'step_00'/name)
                old.pop('metadata');new.pop('metadata');assert equal_tree(old,new),name
            old=load(CONTROL/'train_48/step_00/trainer_state.pt')
            new=load(args.out_dir/'step_00/trainer_state.pt')
            for key in ('optimizer','logical_rng_state','teacher_artifacts'):
                assert equal_tree(old[key],new[key]),key
            assert args.training_timestep_shift==2.22 and args.training_weight_shift==12.
            assert trainer.timestep_shift(args,training=False)==trainer.timestep_weight_shift(args,training=False)==2.22
            initialized=True;audit.update(status='running',step00_tensors_equal=True,step00_optimizer_rng_data_equal=True);save()
        clone=torch.Generator();clone.set_state(generator.get_state())
        pairs=logical_time_pairs(clone,shift=trainer.timestep_shift(args,training=backward),batch_size=4)
        shape=case['clean'][:,:,chunk*5:chunk*5+5].shape;noise_hashes=[]
        for j in range(4):
            noise=torch.randn(shape,generator=clone,dtype=torch.float32)
            noise_hashes.append(hashlib.sha256(noise.numpy().tobytes()).hexdigest())
        record=raw(model,case,chunk,args,generator=generator,backward=backward)
        assert torch.equal(generator.get_state(),clone.get_state())
        for j,item in enumerate(record['samples']):
            assert item['sigma']==float(pairs.t[j])
            value=gaussian_timestep_weights(torch.tensor([item['sigma']*1000],device=case['clean'].device),
                shift=trainer.timestep_weight_shift(args,training=backward))[0]
            assert item['weight']==float(value)
            if backward:
                e=expected[step*4+j]
                assert (e['clip'],e['chunk'],e['candidate_sigma'],e['reconstructed_noise_sha256'])==(
                    case['label'],chunk,item['sigma'],noise_hashes[j])
        audit['batches'].append(dict(phase='train' if backward else 'validation',step=step+1 if backward else None,
            clip=case['label'],chunk=chunk,sigmas=[s['sigma'] for s in record['samples']],
            weight_shift=trainer.timestep_weight_shift(args,training=backward),rng_stream_verified=True))
        if backward:step+=1
        save();return record
    trainer.logical_batch=checked
    try:
        trainer.main()
        assert step==48 and initialized
        audit.update(status='complete',updates=step)
    except BaseException as e:
        audit.update(status='failed',error=repr(e),updates=step);raise
    finally:save()


if __name__=='__main__':main()
