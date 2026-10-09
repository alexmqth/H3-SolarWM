"""Numerical audit backend: group queries with identical bool visibility.

Gather only permitted keys in their original order. Mathematically identical
masked attention, but equivalent causal nodes have the same kernel shapes in
full recompute and cropped cached calls. Inference-only, scoped per forward.
No weight, conditioning or visibility changes; not a deployment default.
"""
from contextlib import contextmanager
import numpy as np
import torch
from torch.nn import functional as F


@contextmanager
def canonical_attention():
    original=F.scaled_dot_product_attention
    groups={}
    def grouped(q,k,v,attn_mask=None,dropout_p=0.0,is_causal=False,scale=None,**kwargs):
        if attn_mask is None:
            return original(q,k,v,attn_mask=attn_mask,dropout_p=dropout_p,
                            is_causal=is_causal,scale=scale,**kwargs)
        if torch.is_grad_enabled() or attn_mask.dtype!=torch.bool or dropout_p or is_causal:
            raise ValueError('Canonical audit expects no_grad, bool explicit mask, no dropout')
        if q.shape[0]!=1 or attn_mask.shape[:2]!=(1,1):
            raise ValueError('Canonical audit supports batch1/head-shared visibility')
        key=(q.shape[-2],k.shape[-2])
        if key not in groups:
            a=attn_mask[0,0].detach().cpu().numpy()
            _,idx,inverse=np.unique(np.packbits(a,axis=1),axis=0,return_index=True,return_inverse=True)
            plan=[]
            for group,row in enumerate(idx):
                queries=np.flatnonzero(inverse==group);keys=np.flatnonzero(a[row])
                if not len(keys):raise ValueError('Fully masked query')
                plan.append((torch.tensor(queries,device=q.device),torch.tensor(keys,device=q.device)))
            groups[key]=(attn_mask.clone(),plan)
        reference,plan=groups[key]
        if not torch.equal(reference,attn_mask):
            raise ValueError('Mask changed within forward; require a fresh canonical context')
        output=torch.empty_like(q)
        for queries,keys in plan:
            partial=original(q.index_select(-2,queries),k.index_select(-2,keys),
                             v.index_select(-2,keys),scale=scale,**kwargs)
            output.index_copy_(-2,queries,partial)
        return output
    F.scaled_dot_product_attention=grouped
    try:yield
    finally:F.scaled_dot_product_attention=original
