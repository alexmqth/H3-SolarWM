"""Tiny real-H3 VJP comparison with differentiable GT history; CPU only."""
import json
from pathlib import Path
import sys

import torch

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[2]
sys.path[:0] = [str(ROOT/'code'), str(ROOT/'DiffSynth-Studio-h3-v2')]
from causal.h3_training import make_small_h3, synthetic_h3_batch
from causal.h3_precision import configure_precision
from causal.stage1_lora import install_stage1_lora
from causal.h3_cached import H3ChunkCache, chunk_forward
from causal.clean_history_graph import build_clean_history_graph
from paired_replay import paired_objective, paired_cotangents, backward_replay

torch.set_num_threads(2)
records = []
for dtype in (torch.float32, torch.bfloat16):
    for chunk in (1, 2):
        torch.manual_seed(933)
        model = make_small_h3().eval().to(dtype).requires_grad_(False)
        configure_precision(model, 'h3_fp32')
        bank = install_stage1_lora(model, rank=2, alpha=2)
        with torch.no_grad():
            for m in bank.modules:
                for p in m.lora_B: p.normal_(std=.02)
        params = bank.parameters()
        batch = synthetic_h3_batch(frames=12, seed=127)
        prompt = batch['prompt_embeds'].to(dtype)
        alternate = prompt.clone()
        spans = batch['packed']['action_text_spans_local']
        for lo, hi in spans[chunk*5:min(chunk*5+5,12)]:
            alternate[int(lo):int(hi)] += .3
        def conditions(index, which):
            return dict(full_packed=batch['packed'], prompt=prompt if which == 0 else alternate,
                audio=batch['audio_latents'].to(dtype), anchor=batch['anchor_rows'].to(dtype),
                chunk_frames=5, action_prefix_mode='causal', action_feedback=True)
        clean = batch['clean_video'].to(dtype)
        start = chunk*5; noisy = .4*clean[:,:,start:start+5].float() + .6*batch['noise'][:,:,start:start+5].float()
        def predict(which, grad):
            if grad:
                cache = build_clean_history_graph(model, clean, chunk, lambda i:conditions(i,which),
                    chunk_frames=5, history_chunks=5, checkpoint=True, offload=False)
            else:
                cache=H3ChunkCache(5,'cpu')
                for index in range(chunk):
                    chunk_forward(model,clean[:,:,index*5:index*5+5],sigma=0.,index=index,
                        cache=cache,commit=True,**conditions(index,which))
            return chunk_forward(model,noisy,sigma=.6,index=chunk,cache=cache,
                allow_grad_read=grad,use_gradient_checkpointing=grad,**conditions(chunk,which))
        # Fixed nondegenerate targets isolate VJP implementation, not H3 quality.
        ta = batch['noise'][:,:,start:start+5].float()
        td = .85*ta + .15
        with torch.no_grad():
            references=[predict(i,False).float() for i in (0,1)]
        values=[predict(i,True) for i in (0,1)]
        loss,_=paired_objective(*values,ta,td)
        loss.backward()
        direct=[p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in params]
        for p in params:p.grad=None
        cotangents,_=paired_cotangents(*references,ta,td)
        errors=[]
        for i in (0,1):errors.append(backward_replay(predict(i,True),references[i],cotangents[i],tolerance=0.))
        replay=[p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in params]
        for x,y in zip(direct,replay):torch.testing.assert_close(x,y,rtol=2e-5,atol=1e-7)
        assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
        assert sum(float(g.square().sum()) for g in replay)>0
        # A stale reference must stop before its backward mutates gradients.
        refused=False
        try:backward_replay(predict(0,True),references[0]+.1,cotangents[0])
        except RuntimeError:refused=True
        assert refused
        records.append(dict(dtype=str(dtype),chunk=chunk,reference_max_abs=errors,
            full_parameter_grad_max_abs=max(float((x-y).abs().max()) for x,y in zip(direct,replay)),
            direct_norm=sum(float(g.square().sum()) for g in direct)**.5,
            stale_reference_rejected=refused))
result=dict(status='passed',scope='tiny-H3 CPU sequential paired VJP versus simultaneous full-history gradient',
            cases=records,pretrained_H3_quality_verified=False)
(BASE/'cpu_replay.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
