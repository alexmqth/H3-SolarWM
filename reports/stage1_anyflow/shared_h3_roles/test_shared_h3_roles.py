"""Actual tiny H3: role isolation while a checkpointed FMBS graph is live."""
from pathlib import Path
import sys

import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'code'),str(ROOT/'DiffSynth-Studio-h3-v2')]
from causal.anyflow import install_anyflow,finite_map_step
from causal.fmbs import simulate_h3_chunk
from causal.h3_cached import H3ChunkCache,chunk_forward
from causal.h3_precision import configure_precision
from causal.h3_training import make_small_h3,synthetic_h3_batch
from causal.shared_h3_roles import SharedH3Roles
from causal.stage1_lora import install_stage1_lora
from causal.pretrained_lora import CausalQKVLoRA
from diffsynth.pipelines.minimax_h3_audio_video import model_fn_minimax_h3


def original_forward(model,state,history,batch,sigma,*,checkpoint=False):
    whole=torch.cat((history,state),dim=2).float()
    update=torch.ones_like(whole);update[:,:,:history.shape[2]]=0
    return model_fn_minimax_h3(model,whole,batch['audio_latents'].float(),batch['packed'],
        batch['prompt_embeds'],timestep_video=torch.tensor(sigma*1000),timestep_audio=torch.tensor(1000.),
        keyframe_cond_anchor=batch['anchor_rows'].float(),input_latents_video=whole,
        denoise_mask_video=update,fixed_prefix_timesteps=True,
        use_gradient_checkpointing=checkpoint,use_gradient_checkpointing_offload=checkpoint)[0][:,:,history.shape[2]:]


def setup(dtype=torch.float32):
    torch.set_num_threads(2);torch.manual_seed(281)
    model=make_small_h3().eval().to(dtype).requires_grad_(False)
    configure_precision(model,'h3_fp32')
    batch=synthetic_h3_batch(frames=12,seed=81)
    batch['prompt_embeds']=batch['prompt_embeds'].to(dtype)
    original_params=list(model.parameters())
    with torch.no_grad():
        reference=original_forward(model,batch['noise'][:,:,10:],batch['clean_video'][:,:,:10],batch,.65)
    old=model.blocks[-1].attn.qkv_proj
    # The tiny test's older QKV wrapper lacks nn.Linear shape attributes.
    # Production AutoTorchLinear exposes them; preserve the tiny base intact.
    old.in_features=old.base.in_features;old.out_features=old.base.out_features
    visual=CausalQKVLoRA(old,rank=2,device='cpu')
    model.blocks[-1].attn.qkv_proj=visual
    with torch.no_grad():
        visual.lora_B.normal_(std=.04)
    bank=install_stage1_lora(model,rank=2,alpha=2)
    time=install_anyflow(model)
    with torch.no_grad():
        for m in bank.modules:
            for p in m.lora_B:p.normal_(std=.025)
        # Force target conditioning away from its original identity so the
        # frozen-teacher check cannot pass through a zero-initialization trick.
        time.delta.proj_out.bias.add_(.5)
    rng=torch.get_rng_state().clone()
    roles=SharedH3Roles(model,bank,critic_rank=2,critic_alpha=2,student_only_modules=(visual,))
    assert torch.equal(torch.get_rng_state(),rng)
    assert all(any(p is q for q in model.parameters()) for p in original_params)
    assert all(any(p is q for q in roles._shared) for p in original_params)
    with torch.no_grad():
        for m in roles.critic_bank.modules:
            for p in m.lora_B:p.normal_(std=.015)
    common=dict(full_packed=batch['packed'],prompt=batch['prompt_embeds'],anchor=batch['anchor_rows'],
        audio=batch['audio_latents'],chunk_frames=5,action_prefix_mode='causal',action_feedback=True)
    return model,bank,time,roles,batch,common,reference


@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
def test_original_teacher_is_identical_after_nonzero_student_critic_and_time(dtype):
    model,bank,time,roles,b,common,ref=setup(dtype)
    student_ids={id(p) for p in bank.parameters()};critic_ids={id(p) for p in roles.critic_bank.parameters()}
    assert not student_ids&critic_ids
    assert all(p.requires_grad for p in bank.parameters()+roles.critic_bank.parameters())
    with roles.context('teacher'):
        assert not hasattr(model,'anyflow_conditioner') and not torch.is_grad_enabled()
        actual=original_forward(model,b['noise'][:,:,10:],b['clean_video'][:,:,:10],b,.65)
    torch.testing.assert_close(actual,ref,atol=0,rtol=0)
    assert model.anyflow_conditioner is time and roles.active is None
    assert all(m.enabled for m in bank.modules) and not any(m.enabled for m in roles.critic_bank.modules)


def generated_history(model,roles,b,common):
    cache=H3ChunkCache(2,'cpu');grid=(1.,.9,.65,.3,0.);history=[]
    with roles.context('student'),torch.no_grad():
        for index in (0,1):
            current=b['noise'][:,:,index*5:index*5+5].clone()
            for t,r in zip(grid,grid[1:]):
                velocity=chunk_forward(model,current,sigma=t,target_sigma=r,index=index,cache=cache,**common)
                current=finite_map_step(current,velocity,t,r)
            history.append(current)
            chunk_forward(model,current,sigma=0,index=index,cache=cache,commit=True,**common)
    return cache,torch.cat(history,dim=2),grid


@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
def test_live_fmbs_gradient_survives_teacher_pass_and_critic_optimizer(dtype):
    model,bank,time,roles,b,common,_=setup(dtype)
    cache,history,grid=generated_history(model,roles,b,common)
    params=bank.parameters()+list(time.parameters());critic=roles.critic_bank.parameters()
    snapshots=[(x,x.clone(),x._version) for entries in cache.layers.values() for e in entries for x in (e.key,e.value,e.rope)]
    def endpoint():
        return roles.guard_backward(simulate_h3_chunk(model,b['noise'][:,:,10:],sigmas=grid,
            interval_index=2,chunk_index=2,cache=cache,conditions=common), 'student')
    with roles.context('student'):
        pure=endpoint();pure.square().mean().backward()
    expected=[torch.zeros_like(p) if p.grad is None else p.grad.clone() for p in params]
    for p in params:p.grad=None
    start_student=[p.detach().clone() for p in params]
    start_critic=[p.detach().clone() for p in critic]
    optimizer=torch.optim.AdamW(critic,lr=.001)
    with roles.context('student'):
        live=endpoint()
        with roles.context('teacher'):
            teacher=original_forward(model,live.detach(),history,b,.65)
            assert not teacher.requires_grad
        with roles.context('critic'):
            # Fake distribution uses detached actual student output. Update
            # critic while the student FMBS graph and checkpoint closures live.
            noise=b['noise'][:,:,10:].float();sigma=.65
            noisy=(1-sigma)*live.detach()+sigma*noise
            fake=original_forward(model,noisy,history,b,sigma,checkpoint=True)
            loss=roles.guard_backward((fake-(noise-live.detach())).square().mean(),'critic')
            loss.backward();optimizer.step();optimizer.zero_grad(set_to_none=True)
        assert roles.active=='student' and model.anyflow_conditioner is time
        assert all(p.grad is None for p in params)
        assert all(torch.equal(p,old) for p,old in zip(params,start_student))
        assert any(not torch.equal(p,old) for p,old in zip(critic,start_critic))
        live.square().mean().backward()
    torch.testing.assert_close(pure,live,atol=0,rtol=0)
    for p,grad in zip(params,expected):
        actual=torch.zeros_like(p) if p.grad is None else p.grad
        torch.testing.assert_close(actual,grad,atol=1e-7,rtol=1e-5)
    assert all(torch.equal(x,old) and x._version==v for x,old,v in snapshots)
    assert all(p.grad is None for p in roles._shared)
    print('shared_roles_gradient_norm',float(torch.sqrt(sum(g.square().sum() for g in expected))))


def test_wrong_backward_role_is_rejected_and_exception_restores_modules():
    model,bank,time,roles,b,common,_=setup()
    with roles.context('student'):
        result=roles.guard_backward(chunk_forward(model,b['noise'][:,:,:5],sigma=.8,target_sigma=.3,
            index=0,cache=H3ChunkCache(2,'cpu'),allow_grad_read=True,**common),'student')
    with pytest.raises(RuntimeError,match='student backward requires'):
        result.square().mean().backward()
    with pytest.raises(RuntimeError,match='sentinel'):
        with roles.context('student'):
            with roles.context('critic'):
                assert not hasattr(model,'anyflow_conditioner')
                raise RuntimeError('sentinel')
    assert model.anyflow_conditioner is time and roles.active is None
    assert all(m.enabled for m in bank.modules) and not any(m.enabled for m in roles.critic_bank.modules)


def test_subset_critic_and_invalid_installation_leave_student_intact():
    torch.set_num_threads(2);torch.manual_seed(12)
    model=make_small_h3().requires_grad_(False);bank=install_stage1_lora(model,rank=2,alpha=2)
    originals=[model.get_submodule(n) for n in bank.names]
    with pytest.raises(ValueError,match='subset'):
        SharedH3Roles(model,bank,critic_names=['unknown'])
    assert all(model.get_submodule(n) is m for n,m in zip(bank.names,originals))
    names=tuple(n for n in bank.names if n.startswith('blocks.1.') and '.attn.' in n)
    roles=SharedH3Roles(model,bank,critic_names=names,critic_rank=2,critic_alpha=2)
    assert roles.critic_bank.names==names
    for name,m in zip(bank.names,originals):
        current=model.get_submodule(name)
        assert (current.base is m) if name in names else current is m
    with pytest.raises(ValueError,match='Unknown'):
        with roles.context('fake_unknown'):pass
