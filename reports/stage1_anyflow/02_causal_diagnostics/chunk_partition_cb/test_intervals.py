from pathlib import Path
import sys
import pytest
import torch

BASE = Path(__file__).resolve().parent
sys.path[:0] = [str(BASE/'runtime/code'), str(BASE/'runtime/DiffSynth-Studio-h3-v2'),
                str(BASE/'source_history')]
from causal.h3_training import make_small_h3, synthetic_h3_batch
from causal.h3_precision import configure_precision
from causal.local_topology import visible_inputs
from history_conditioning import history_forward
from interval_forward import interval_forward


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
@pytest.mark.parametrize('mode,old_mode', [('clean','C'), ('sigma_noised','N')])
def test_explicit_intervals(dtype,mode,old_mode):
    torch.set_num_threads(2);torch.manual_seed(911)
    model=make_small_h3().eval().to(dtype).requires_grad_(False)
    configure_precision(model,'h3_fp32')
    b=synthetic_h3_batch(frames=37,seed=321)
    text=b['prompt_embeds'].to(dtype)
    with torch.no_grad():
        for start,stop in ((0,7),(0,12),(7,12),(12,17),(12,24)):
            layout,prompt=visible_inputs(b['packed'],text,stop,4)
            history=b['clean_video'][:,:,:start].float().clone()
            noise=b['noise'][:,:,:start].float().clone()
            state=b['noise'][:,:,start:stop].float().clone()
            snapshots=[x.clone() for x in (history,noise,state)]
            kw=dict(history=history,history_noise=noise,full_packed=layout,prompt=prompt,
                anchor=b['anchor_rows'].to(dtype),audio=b['audio_latents'],sigma=.6)
            out=interval_forward(model,state,start=start,mode=mode,**kw)
            # Old API is emulated with one initial chunk of length start;
            # current tensor may be shorter. This verifies identical semantics
            # without adopting index*width as the new interval contract.
            old=history_forward(model,state,index=int(start>0),chunk_frames=start or stop,
                history_chunks=5,mode=old_mode,**kw)
            torch.testing.assert_close(out,old,atol=0,rtol=0)
            future=text.clone()
            for lo,hi in b['packed']['action_text_spans_local'][stop:]:future[lo:hi]+=10
            lp,pt=visible_inputs(b['packed'],future,stop,4)
            alt=interval_forward(model,state,start=start,mode=mode,**dict(kw,full_packed=lp,prompt=pt))
            torch.testing.assert_close(out,alt,atol=0,rtol=0)
            action=text.clone()
            for lo,hi in b['packed']['action_text_spans_local'][start:stop]:action[lo:hi]+=3
            lp,pt=visible_inputs(b['packed'],action,stop,4)
            alt=interval_forward(model,state,start=start,mode=mode,**dict(kw,full_packed=lp,prompt=pt))
            assert float((out-alt).abs().max())>1e-6
            if start:
                alt=interval_forward(model,state,start=start,mode=mode,**dict(kw,history=history+2))
                assert float((out-alt).abs().max())>1e-6
            for x,y in zip((history,noise,state),snapshots):
                torch.testing.assert_close(x,y,atol=0,rtol=0)
