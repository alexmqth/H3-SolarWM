import copy
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'code'), str(ROOT/'DiffSynth-Studio-h3-v2')]
from causal.h3_training import make_small_h3, synthetic_h3_batch
from causal.h3_cached import H3ChunkCache, chunk_forward
from causal.h3_precision import configure_precision
from causal.anyflow import install_anyflow
from causal.stage1_lora import install_stage1_lora
from causal.clean_history_graph import build_clean_history_graph


def setup(dtype=torch.float32,anyflow=True):
    torch.set_num_threads(2)
    torch.manual_seed(131)
    model=make_small_h3().eval().to(dtype).requires_grad_(False)
    configure_precision(model,'h3_fp32')
    if anyflow:
        install_anyflow(model).requires_grad_(False)
    bank=install_stage1_lora(model,rank=2,alpha=2.)
    with torch.no_grad():
        for m in bank.modules:
            for b in m.lora_B: b.normal_(std=.03)
    data=synthetic_h3_batch(frames=12,seed=19)
    common=dict(full_packed=data['packed'],prompt=data['prompt_embeds'].to(dtype),
        anchor=data['anchor_rows'].to(dtype),audio=data['audio_latents'].to(dtype),chunk_frames=5)
    return model,bank,data,common


def detached(model,clean,common):
    cache=H3ChunkCache(5,'cpu')
    with torch.no_grad():
        for i in range(2):
            chunk_forward(model,clean[:,:,5*i:5*i+5],index=i,cache=cache,
                          sigma=0.,commit=True,**common)
    return cache


def predict(model,cache,noise,common,checkpoint=False,offload=False):
    return chunk_forward(model,noise[:,:,10:],index=2,cache=cache,
        sigma=.6,target_sigma=.1 if hasattr(model,'anyflow_conditioner') else None,
        allow_grad_read=torch.is_grad_enabled(),
        use_gradient_checkpointing=checkpoint,use_gradient_checkpointing_offload=offload,**common)


@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
@pytest.mark.parametrize('checkpoint',[False,True])
@pytest.mark.parametrize('anyflow',[False,True])
def test_identical_values_immutable_history_and_real_backward(dtype,checkpoint,anyflow):
    model,bank,data,common=setup(dtype,anyflow)
    clean=data['clean_video'].float().requires_grad_()
    old=detached(model,clean,common)
    with torch.no_grad(): expected=predict(model,old,data['noise'],common)
    graph=build_clean_history_graph(model,clean,2,lambda i:common,checkpoint=checkpoint,
                                   offload=checkpoint)
    for layer in old.layers:
        for a,b in zip(old.layers[layer],graph.layers[layer]):
            assert a.index==b.index
            for key in ('key','value','rope'): assert torch.equal(getattr(a,key),getattr(b,key))
            assert b.key.requires_grad and b.value.requires_grad
    snapshots={layer:tuple(id(e) for e in entries) for layer,entries in graph.layers.items()}
    y=predict(model,graph,data['noise'],common,checkpoint,checkpoint)
    assert torch.equal(y,expected)
    y.square().mean().backward()
    assert clean.grad[:,:,:10].norm()>0
    assert clean.grad[:,:,10:].count_nonzero()==0
    assert snapshots=={layer:tuple(id(e) for e in entries) for layer,entries in graph.layers.items()}
    assert graph.commits==old.commits
    for group in ('qkv','out','ffn','refiner'):
        assert any(p.grad is not None and p.grad.norm()>0 for p in bank.parameters(group))
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
    # An independent loss must rebuild its own graph, with no retain_graph.
    for p in bank.parameters(): p.grad=None
    again=build_clean_history_graph(model,clean,2,lambda i:common,checkpoint=checkpoint)
    predict(model,again,data['noise'],common,checkpoint).square().mean().backward()
    assert any(p.grad is not None for p in bank.parameters())
    with pytest.raises(RuntimeError,match='requires no_grad'):
        e=graph.layers[0][-1];graph.commit(0,2,e.key,e.value,e.rope)


def test_full_parameter_gradient_matches_finite_difference_of_original_forward():
    model,bank,data,common=setup()
    params=bank.parameters()
    def gradient(full):
        for p in params:p.grad=None
        cache=(build_clean_history_graph(model,data['clean_video'],2,lambda i:common,checkpoint=True)
               if full else detached(model,data['clean_video'],common))
        value=predict(model,cache,data['noise'],common,True).square().mean()
        value.backward()
        return [torch.zeros_like(p) if p.grad is None else p.grad.clone() for p in params]
    cut=gradient(False);full=gradient(True)
    differences=[a-b for a,b in zip(full,cut)]
    norm=torch.sqrt(sum(x.square().sum() for x in differences))
    assert norm>1e-5
    directions=[x/norm for x in differences]
    expected=sum((g*d).sum() for g,d in zip(full,directions))
    cut_expected=sum((g*d).sum() for g,d in zip(cut,directions))
    originals=[p.detach().clone() for p in params]
    values=[];eps=2e-3
    with torch.no_grad():
        for sign in (1,-1):
            for p,initial,direction in zip(params,originals,directions):p.copy_(initial+sign*eps*direction)
            # The oracle is the original inference/commit path, rebuilt after
            # perturbing weights, not the new differentiable cache builder.
            values.append(predict(model,detached(model,data['clean_video'],common),data['noise'],common).square().mean())
        for p,initial in zip(params,originals):p.copy_(initial)
    fd=(values[0]-values[1])/(2*eps)
    torch.testing.assert_close(expected,fd,atol=2e-4,rtol=.02)
    assert abs(fd-expected)<abs(fd-cut_expected)/5
    print(f'finite_difference={fd.item():.8f} full_grad={expected.item():.8f} detached_grad={cut_expected.item():.8f} missing_norm={norm.item():.8f}')


def test_temporary_collectors_are_empty_and_sealed_during_backward(monkeypatch):
    import causal.clean_history_graph as history
    original=history.CleanGraphCapture
    collectors=[]
    def create():
        value=original();collectors.append(value);return value
    monkeypatch.setattr(history,'CleanGraphCapture',create)
    model,bank,data,common=setup()
    cache=history.build_clean_history_graph(model,data['clean_video'],2,lambda i:common,checkpoint=True)
    assert len(collectors)==2
    assert all(c.sealed and not c.entries for c in collectors)
    predict(model,cache,data['noise'],common,True).square().mean().backward()
    # Checkpoint closures still exist here. They must not retain newly copied
    # K/V entries or graph tensors through the transient capture container.
    assert all(c.sealed and not c.entries for c in collectors)
