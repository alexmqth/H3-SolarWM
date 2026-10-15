"""CPU regression that invokes fm_run.train_one's actual attention-scope helper."""
from __future__ import annotations

import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
FROZEN=ROOT/"H3-World/outputs/2026-10-09-22/chunk_partition_cb/runtime"
sys.path[:0]=[str(FROZEN/"code"),str(FROZEN/"code/causal"),
             str(FROZEN/"DiffSynth-Studio-h3-v2"),
             str(ROOT/"submission/reports/stage1_anyflow/02_causal_diagnostics/current_prefix_candidate")]

import torch
from torch.utils.checkpoint import checkpoint
import current_prefix as route
from causal.h3_cached import ChunkAttention,H3ChunkCache
import fm_run as fm


def trial(*, audit, checkpointed=True, raise_backward=False):
    torch.manual_seed(13)
    param=torch.nn.Parameter(torch.tensor(.71,dtype=torch.float64))
    q=torch.randn((4,1,4),dtype=torch.float64)
    k=torch.randn((4,1,4),dtype=torch.float64)
    base=torch.randn((4,1,4),dtype=torch.float64)
    video=torch.zeros_like(base)
    video[2:]=torch.tensor([[[.4,-.2,.8,.1]],[[-.3,.9,.2,-.1]]])
    rope=torch.zeros((4,1,1),dtype=torch.float64)
    control=ChunkAttention(H3ChunkCache(5,"cpu"),0,2,2,0,torch.tensor([[1,2]]),
                           "own",True,allow_grad_read=True)
    trace=[];phase="forward"
    original_route=route.current_prefix_attend
    original_class=ChunkAttention.attend

    def audited(self,*args,**kwargs):
        if raise_backward and phase=="backward":raise RuntimeError("INJECTED_BACKWARD_ERROR")
        trace.append(dict(phase=phase,route=ChunkAttention.attend.__module__))
        assert ChunkAttention.attend.__module__=="current_prefix"
        return original_route(self,*args,**kwargs)

    def layer(value):
        return control.attend(q,k,base+value*video,rope_freqs=rope,layer=0,
                              apply_rope=lambda x,_:x,scale=.5)[0].sum()

    error=None;output=None
    try:
        with fm.verified_attention_scope(audited if audit else None):
            if checkpointed:
                with torch.autograd.graph.save_on_cpu():
                    loss=checkpoint(layer,param,use_reentrant=False)
            else:loss=layer(param)
            output=float(loss.detach())
            phase="backward"
            loss.backward()
    except RuntimeError as exc:
        error=str(exc)
    assert ChunkAttention.attend is original_class
    assert route.current_prefix_attend is original_route
    return dict(output=output,grad=(float(param.grad) if param.grad is not None else None),
                trace=trace,error=error,route_restored=True)


def main():
    original_class=ChunkAttention.attend
    original_route=route.current_prefix_attend
    no_hook_direct=trial(audit=False,checkpointed=False)
    no_hook_checkpoint=trial(audit=False,checkpointed=True)
    hooked_first=trial(audit=True)
    hooked_second=trial(audit=True)
    for item in (no_hook_direct,no_hook_checkpoint,hooked_first,hooked_second):
        assert item["error"] is None and abs(item["grad"]-.9220111905797003)<1e-10
        assert abs(item["output"]-no_hook_direct["output"])<1e-10
    assert [x["phase"] for x in hooked_first["trace"]]==["forward","backward"]
    assert all(x["route"]=="current_prefix" for x in hooked_first["trace"])
    assert hooked_first==hooked_second
    with_fwd_error=False
    try:
        with fm.verified_attention_scope(lambda *args,**kwargs:None):
            raise RuntimeError("INJECTED_FORWARD_ERROR")
    except RuntimeError as exc:
        with_fwd_error=str(exc)=="INJECTED_FORWARD_ERROR"
    assert with_fwd_error and ChunkAttention.attend is original_class
    assert route.current_prefix_attend is original_route
    backward_error=trial(audit=True,raise_backward=True)
    assert backward_error["error"]=="INJECTED_BACKWARD_ERROR"
    assert backward_error["route_restored"]
    result=dict(status="PASS_CPU_PRODUCTION_SCOPE_HOOK",gpu_calls=0,
                actual_helper="fm_run.verified_attention_scope",
                no_hook_direct=no_hook_direct,no_hook_checkpoint=no_hook_checkpoint,
                hooked_first=hooked_first,hooked_second=hooked_second,
                forward_exception_restored=with_fwd_error,
                backward_exception=backward_error)
    path=HERE/"HOOK_LIFECYCLE_CPU_V2.json"
    if path.exists():raise FileExistsError(path)
    path.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()
