import torch
from torch.nn import functional as F
from canonical_sdpa import canonical_attention


def test_grouping_equals_explicit_mask_and_restores_backend():
    torch.manual_seed(331);torch.set_num_threads(2)
    q=torch.randn(1,4,29,16);k=torch.randn(1,4,37,16);v=torch.randn_like(k)
    a=torch.zeros(1,1,29,37,dtype=torch.bool)
    a[:,:,:10,:12]=True;a[:,:,10:17,8:29]=True;a[:,:,17:,::2]=True
    original=F.scaled_dot_product_attention
    with torch.no_grad():
        expected=original(q,k,v,attn_mask=a)
        with canonical_attention():
            actual=F.scaled_dot_product_attention(q,k,v,attn_mask=a)
            torch.testing.assert_close(actual,expected,atol=3e-7,rtol=2e-6)
            repeat=F.scaled_dot_product_attention(q,k,v,attn_mask=a)
            torch.testing.assert_close(actual,repeat,atol=0,rtol=0)
    assert F.scaled_dot_product_attention is original
