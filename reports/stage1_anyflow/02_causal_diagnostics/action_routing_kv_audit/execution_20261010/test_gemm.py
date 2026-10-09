from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'DiffSynth-Studio-h3-v2'))
import torch
from torch.nn import functional as F
from canonical_gemm import canonical_linears


def test_fixed_rows_preserve_linear_values_and_restore():
    torch.manual_seed(771);torch.set_num_threads(2)
    x=torch.randn(270,17);weight=torch.randn(31,17);bias=torch.randn(31)
    original=F.linear
    with torch.no_grad():
        expected=original(x,weight,bias)
        with canonical_linears():
            actual=F.linear(x,weight,bias)
            torch.testing.assert_close(actual,expected,atol=3e-6,rtol=2e-5)
            cropped=F.linear(x[12:57],weight,bias)
            torch.testing.assert_close(cropped,actual[12:57],atol=0,rtol=0)
            small=F.linear(x[:3],weight,bias)
            torch.testing.assert_close(small,expected[:3],atol=3e-6,rtol=2e-5)
    assert F.linear is original
