"""Fixed-row GEMM tiles for an implementation-equivalence audit only.

Padding zero rows and discarding their outputs does not alter linear/LoRA
semantics. It prevents sequence-length-dependent GEMM algorithm choices.
No merging, rounding of parameters, training or production-default change.
"""
from contextlib import contextmanager
import torch
from torch.nn import functional as F
from diffsynth.core.vram.layers import LoRAHotLoadMixin


def rows(operation,x):
    shape=x.shape
    flat=x.reshape(-1,shape[-1]);n=len(flat)
    if n==0:return operation(x)
    tile=8 if n<=8 else 256
    values=[]
    for start in range(0,n,tile):
        part=flat[start:start+tile];used=len(part)
        if used<tile:part=torch.cat([part,part.new_zeros(tile-used,part.shape[-1])],0)
        values.append(operation(part)[:used])
    y=torch.cat(values,0)
    return y.reshape(*shape[:-1],y.shape[-1])


@contextmanager
def canonical_linears():
    linear=F.linear;lora=LoRAHotLoadMixin.lora_forward
    def stable(x,weight,bias=None):
        if torch.is_grad_enabled():raise RuntimeError('Audit GEMM is inference-only')
        return rows(lambda t:linear(t,weight,bias),x)
    def stable_lora(self,x,out):
        if self.lora_merger is not None:raise RuntimeError('No merged LoRA in this audit')
        for a,b in zip(self.lora_A_weights,self.lora_B_weights):
            a=a.T.to(device=x.device,dtype=x.dtype);b=b.T.to(device=x.device,dtype=x.dtype)
            low=rows(lambda t:t@a,x)
            out=out+rows(lambda t:t@b,low)
        return out
    F.linear=stable;LoRAHotLoadMixin.lora_forward=stable_lora
    try:yield
    finally:F.linear=linear;LoRAHotLoadMixin.lora_forward=lora
