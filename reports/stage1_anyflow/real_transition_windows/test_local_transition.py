"""Actual tiny-H3: inference equivalence, gradients, checkpoint and leakage."""
from pathlib import Path
import sys
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
B=ROOT/'outputs/2026-10-09-06/stage1_history_conditioning'
sys.path[:0]=[str(ROOT/'code'),str(ROOT/'DiffSynth-Studio-h3-v2'),str(B),
              str(ROOT/'outputs/2026-10-09-04/stage1_local_topology')]
from causal.h3_training import make_small_h3,synthetic_h3_batch
from causal.h3_precision import configure_precision
from causal.local_topology import visible_inputs
from causal.local_transition import transition_forward,observed_transition_losses
from causal.position_contract import initial_action_text_reference,fixed_action_position_origin
from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Unit_PackedSequenceBuilder
from history_conditioning import history_forward


def setup(dtype):
    torch.set_num_threads(2);torch.manual_seed(911)
    model=make_small_h3().to(dtype).eval();configure_precision(model,'h3_fp32')
    b=synthetic_h3_batch(frames=37,seed=321)
    return model,b


def inputs(b,index,dtype):
    start=index*12;stop=start+12
    p,t=visible_inputs(b['packed'],b['prompt_embeds'].to(dtype),stop,4)
    return dict(history=b['clean_video'][:,:,:start].float(),history_noise=b['noise'][:,:,:start].float(),
                full_packed=p,prompt=t,anchor=b['anchor_rows'].to(dtype),audio=b['audio_latents'],
                sigma=.5,index=index,chunk_frames=12,history_chunks=5)


@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
def test_inference_equivalent_gradients_checkpoint_and_detached_history(dtype):
    model,b=setup(dtype)
    params=[p for p in model.parameters() if p.requires_grad]
    for index in (0,1,2):
        kw=inputs(b,index,dtype);z=b['noise'][:,:,index*12:index*12+12].clone().float()
        kw['history'].requires_grad_(True);kw['history_noise'].requires_grad_(True)
        history_before=kw['history'].detach().clone()
        with torch.no_grad():ref=history_forward(model,z,mode='N',**kw)
        grads=[];outputs=[]
        for checkpoint in (False,True):
            model.zero_grad(set_to_none=True)
            out=transition_forward(model,z,checkpoint=checkpoint,**kw);outputs.append(out.detach())
            torch.testing.assert_close(out,ref,atol=0,rtol=0)
            loss=observed_transition_losses(out,None,torch.zeros_like(out),margin=0,action_weight=0)['total']
            loss.backward()
            grads.append([p.grad.clone() for p in params])
            assert all(torch.isfinite(g).all() for g in grads[-1])
            assert sum(float(g.abs().sum()) for g in grads[-1])>0
            assert kw['history'].grad is None and kw['history_noise'].grad is None
        for a,c in zip(*grads):torch.testing.assert_close(a,c,atol=0,rtol=0)
        torch.testing.assert_close(history_before,kw['history'],atol=0,rtol=0)


@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
def test_future_text_lengths_and_content_do_not_change_window_or_gradients(dtype):
    model,b=setup(dtype);text=b['prompt_embeds'];reference=initial_action_text_reference(b['packed'])
    for index in (0,1,2):
        stop=index*12+12;parts=[text[:16]];spans=[];cursor=16
        for i in range(37):
            rows=text[16+i*8:16+(i+1)*8] if i<stop else torch.randn(13,32)*10
            parts.append(rows);spans.append((cursor,cursor+len(rows)));cursor+=len(rows)
        alt=MiniMaxH3Unit_PackedSequenceBuilder()._build_packed_fl2va(cursor,37,4,4,2,[0],action_text_spans=spans)
        alt=fixed_action_position_origin(alt,reference_text_length=reference)
        lp,pt=visible_inputs(alt,torch.cat(parts).to(dtype),stop,4)
        kw=inputs(b,index,dtype);z=b['noise'][:,:,index*12:stop].float()
        params=[p for p in model.parameters() if p.requires_grad];grads=[];outs=[]
        for packed,prompt in ((kw['full_packed'],kw['prompt']),(lp,pt)):
            model.zero_grad(set_to_none=True)
            out=transition_forward(model,z,**dict(kw,full_packed=packed,prompt=prompt))
            out.float().square().mean().backward();outs.append(out.detach())
            grads.append([p.grad.clone() for p in params])
        torch.testing.assert_close(*outs,atol=0,rtol=0)
        for a,c in zip(*grads):torch.testing.assert_close(a,c,atol=0,rtol=0)


def test_ranking_does_not_backpropagate_into_observed_target():
    p=torch.tensor([2.],requires_grad=True);n=torch.tensor([1.],requires_grad=True)
    target=torch.tensor([0.],requires_grad=True)
    losses=observed_transition_losses(p,n,target,margin=.5,action_weight=.1)
    losses['total'].backward()
    assert p.grad>0 and n.grad<0 and target.grad is None
    assert float(losses['positive_FM'].detach())==4 and float(losses['negative_FM'].detach())==1
    # Correct prediction still has an absolute optimum; ranking is auxiliary.
    assert float(losses['action_hinge'].detach())==3.5
