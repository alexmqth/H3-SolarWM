from pathlib import Path
from types import SimpleNamespace
import copy
import sys

import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'code'),str(ROOT/'DiffSynth-Studio-h3-v2')]
from causal.train_stage1_anyflow import timestep_shift, timestep_weight_shift, logical_batch
from causal.anyflow import logical_time_pairs, anyflow_sample_loss, adaptive_scale
from causal.anyflow_reference import gaussian_timestep_weights
from causal.h3_training import make_small_h3, synthetic_h3_batch
from causal.anyflow import install_anyflow, validate_checkpoint_protocol
from causal.stage1_lora import install_stage1_lora


def test_training_validation_and_inference_shift_are_independent():
    args=SimpleNamespace(flow_shift=2.22,training_timestep_shift=12.,validation_timestep_shift=None)
    assert timestep_shift(args,training=True)==12.
    assert timestep_shift(args,training=False)==2.22
    assert timestep_shift(SimpleNamespace(flow_shift=2.22),training=True)==2.22
    cfg=dict(anchor_mode='rgb',chunk_frames=5,history_chunks=5,action_prefix_mode='causal',
             action_feedback=True,flow_shift=2.22,training_timestep_shift=12.,validation_timestep_shift=2.22)
    meta=dict(objective='TF-AnyFlow v1.5',config=cfg,optimizer_step=16)
    inference=dict(cfg,anchor_mode='dynamic_last_frame_rgb_dual')
    validate_checkpoint_protocol(meta,copy.deepcopy(meta),inference)
    with pytest.raises(ValueError,match='flow_shift mismatch'):
        validate_checkpoint_protocol(meta,copy.deepcopy(meta),dict(inference,flow_shift=12.))


@pytest.mark.parametrize('invalid',[0.,-1.,float('nan'),float('inf')])
def test_nonpositive_or_nonfinite_shift_rejected(invalid):
    with pytest.raises(ValueError):
        timestep_shift(SimpleNamespace(flow_shift=2.22,training_timestep_shift=invalid),training=True)
    with pytest.raises(ValueError):
        timestep_weight_shift(SimpleNamespace(flow_shift=2.22,training_weight_shift=invalid),training=True)


def test_weight_override_does_not_change_density_or_validation_policy():
    args=SimpleNamespace(flow_shift=2.22,training_timestep_shift=2.22,training_weight_shift=12.)
    assert timestep_shift(args,training=True)==2.22
    assert timestep_weight_shift(args,training=True)==12.
    assert timestep_weight_shift(args,training=False)==2.22
    del args.training_weight_shift
    assert timestep_weight_shift(args,training=True)==2.22


@pytest.mark.parametrize('objective',['anyflow','fm'])
@pytest.mark.parametrize('weight_shift',[None,2.22])
def test_real_small_h3_uses_training_density_and_matching_gaussian_weights(objective,weight_shift):
    torch.set_num_threads(2);torch.manual_seed(113)
    model=make_small_h3().eval().requires_grad_(False)
    if objective=='anyflow':install_anyflow(model).requires_grad_(False)
    bank=install_stage1_lora(model,rank=2,alpha=2.)
    batch=synthetic_h3_batch(frames=5,seed=51)
    case=dict(label='A',clean=batch['clean_video'],packed=batch['packed'],
              prompt=batch['prompt_embeds'],audio=batch['audio_latents'],anchors=[batch['anchor_rows']],
              actions=None,action_adapter=None)
    args=SimpleNamespace(flow_shift=2.22,training_timestep_shift=12.,validation_timestep_shift=2.22,
        training_weight_shift=weight_shift,
        chunk_frames=5,history_chunks=5,logical_batch=4,history_gradient_mode='full',
        anchor_mode='fixed',action_prefix_mode='causal',action_feedback=True,
        objective=objective,finite_difference_epsilon=5.,precision_profile='h3_fp32',smoke=True)
    old=copy.copy(args);old.training_timestep_shift=None
    before=logical_batch(model,case,0,args,generator=torch.Generator().manual_seed(101),backward=False)
    reference=logical_batch(model,case,0,old,generator=torch.Generator().manual_seed(101),backward=False)
    assert before==reference
    record=logical_batch(model,case,0,args,generator=torch.Generator().manual_seed(101),backward=True)
    pairs=logical_time_pairs(torch.Generator().manual_seed(101),shift=12.)
    for j,s in enumerate(record['samples']):
        assert s['sigma']==float(pairs.t[j])
        expected_r=float(pairs.r[j]) if objective=='anyflow' else float(pairs.t[j])
        assert s['target_sigma']==expected_r
        expected=float(gaussian_timestep_weights(torch.tensor([s['sigma']*1000]),
            shift=12. if weight_shift is None else weight_shift)[0])
        assert s['weight']==expected
    assert any(p.grad is not None and p.grad.norm()>0 for p in bank.parameters())


@pytest.mark.parametrize('objective',['fm','anyflow'])
def test_explicit_legacy_weight_preserves_full_history_gradient_and_rng(objective):
    torch.set_num_threads(2);torch.manual_seed(817)
    model=make_small_h3().eval().requires_grad_(False)
    if objective=='anyflow':install_anyflow(model).requires_grad_(False)
    bank=install_stage1_lora(model,rank=2,alpha=2.)
    batch=synthetic_h3_batch(frames=12,seed=53)
    case=dict(label='A',clean=batch['clean_video'],packed=batch['packed'],
        prompt=batch['prompt_embeds'],audio=batch['audio_latents'],anchors=[batch['anchor_rows']]*3,
        actions=None,action_adapter=None)
    args=SimpleNamespace(flow_shift=2.22,training_timestep_shift=12.,validation_timestep_shift=2.22,
        chunk_frames=5,history_chunks=5,logical_batch=4,history_gradient_mode='full',
        anchor_mode='fixed',action_prefix_mode='causal',action_feedback=True,
        objective=objective,finite_difference_epsilon=5.,precision_profile='h3_fp32',smoke=True)
    generator=torch.Generator().manual_seed(105)
    a=logical_batch(model,case,1,args,generator=generator,backward=True)
    grads=[p.grad.clone() for p in bank.parameters()];rng=generator.get_state()
    model.zero_grad(set_to_none=True);args.training_weight_shift=12.
    generator=torch.Generator().manual_seed(105)
    b=logical_batch(model,case,1,args,generator=generator,backward=True)
    assert a==b and torch.equal(rng,generator.get_state())
    assert all(torch.equal(x,p.grad) for x,p in zip(grads,bank.parameters()))
    assert a['differentiable_history_forwards']==4
