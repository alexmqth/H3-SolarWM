"""CPU audit before the single-factor density experiment; no model download."""
from pathlib import Path
from types import SimpleNamespace
import copy
from datetime import datetime
import hashlib
import importlib.util
import json
import sys

import torch

BASE=Path(__file__).resolve().parent
CONTROL=BASE.parents[1]/'2026-10-08-18/stage1_real_abot_fm'
sys.path[:0]=[str(BASE/'runtime/code'),str(BASE/'runtime/DiffSynth-Studio-h3-v2')]
from causal import train_stage1_anyflow as current
from causal.anyflow import logical_time_pairs,install_anyflow
from causal.h3_training import make_small_h3,synthetic_h3_batch
from causal.stage1_lora import install_stage1_lora
from causal.training_state import load_training_state


def main():
    torch.set_num_threads(2)
    spec=importlib.util.spec_from_file_location('frozen_bridge_trainer',CONTROL/'runtime/code/causal/train_stage1_anyflow.py')
    legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
    results=[]
    for objective in ('fm','anyflow'):
        torch.manual_seed(981)
        model=make_small_h3().eval().requires_grad_(False)
        if objective=='anyflow':install_anyflow(model).requires_grad_(False)
        bank=install_stage1_lora(model,rank=2,alpha=2.)
        data=synthetic_h3_batch(frames=12,seed=918)
        case=dict(label='cpu_control',clean=data['clean_video'],packed=data['packed'],
            prompt=data['prompt_embeds'],audio=data['audio_latents'],anchors=[data['anchor_rows']]*3,
            actions=None,action_adapter=None)
        args=SimpleNamespace(flow_shift=2.22,training_timestep_shift=12.,validation_timestep_shift=2.22,
            chunk_frames=5,history_chunks=5,logical_batch=4,history_gradient_mode='full',
            anchor_mode='fixed',action_prefix_mode='causal',action_feedback=True,
            objective=objective,finite_difference_epsilon=5.,precision_profile='h3_fp32',smoke=True)
        g=torch.Generator().manual_seed(945)
        before=legacy.logical_batch(model,case,1,args,generator=g,backward=True)
        grads=[p.grad.clone() for p in bank.parameters()];rng=g.get_state()
        model.zero_grad(set_to_none=True);g=torch.Generator().manual_seed(945)
        after=current.logical_batch(model,case,1,args,generator=g,backward=True)
        assert before==after and torch.equal(rng,g.get_state())
        assert all(torch.equal(x,p.grad) for x,p in zip(grads,bank.parameters()))
        results.append(dict(objective=objective,full_history=True,loss_gradient_rng_identical_to_frozen_trainer=True))

    old=json.loads((CONTROL/'train_48/training.json').read_text())
    cfg=dict(old['config'],steps=49,training_weight_shift=None,validation_weight_shift=None)
    state=load_training_state(CONTROL/'train_48/step_48',cfg)
    assert state['optimizer_step']==48
    try:load_training_state(CONTROL/'train_48/step_48',dict(cfg,training_weight_shift=2.22))
    except ValueError as e:assert 'training_weight_shift' in str(e)
    else:raise AssertionError('Loss policy change silently accepted as exact resume')

    generator=torch.Generator().manual_seed(10013)
    samples=[]
    for i,u in enumerate(old['updates']):
        start=generator.get_state()
        pairs0=logical_time_pairs(generator,shift=12.)
        alternative=torch.Generator();alternative.set_state(start)
        pairs1=logical_time_pairs(alternative,shift=2.22)
        assert torch.equal(generator.get_state(),alternative.get_state())
        shape=(1,24,5 if u['chunk']<2 else 2,30,52)
        for j,s in enumerate(u['samples']):
            assert float(pairs0.t[j])==s['sigma']
            noise=torch.randn(shape,generator=generator,dtype=torch.float32)
            paired=torch.randn(shape,generator=alternative,dtype=torch.float32)
            assert torch.equal(noise,paired)
            samples.append(dict(step=i+1,clip=u['action'],chunk=u['chunk'],sample=j,
                control_sigma=float(pairs0.t[j]),candidate_sigma=float(pairs1.t[j]),
                reconstructed_noise_sha256=hashlib.sha256(noise.numpy().tobytes()).hexdigest()))
        assert torch.equal(generator.get_state(),alternative.get_state())
    assert torch.equal(generator.get_state(),state['logical_rng_state'])
    def bins(key):
        out=dict(low=0,mid=0,high=0)
        for row in samples:
            x=row[key];out['low' if x<=.2407808991 else 'mid' if x<=.6894410401 else 'high']+=1
        return out
    coverage=dict(control=bins('control_sigma'),candidate=bins('candidate_sigma'))
    result=dict(status='passed',at=datetime.now().astimezone().isoformat(),legacy_default=results,
        legacy_resume_validated=True,changed_loss_exact_resume_rejected=True,
        all192_control_sigmas_reconstructed=True,final_control_logical_rng_matches_saved_state=True,
        candidate_uses_identical_noise_stream=True,coverage=coverage,samples=samples,
        limitation='Noise hashes reconstructed from exact saved RNG/code; historical run did not save actual noise tensor hashes. CPU equivalence is not quality evidence.')
    (BASE/'preflight.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='samples'},indent=2))


if __name__=='__main__':main()
