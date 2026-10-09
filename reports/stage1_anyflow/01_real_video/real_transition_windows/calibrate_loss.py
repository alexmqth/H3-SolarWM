"""Train-only fixed8-state action-loss scale calibration; zero optimizer."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
BASE=Path(__file__).resolve().parent;RT=BASE/'runtime'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(args):
    import torch
    import numpy as np
    import infer as abot
    from causal.h3_precision import configure_precision
    from causal.local_topology import visible_inputs
    from causal.local_transition import transition_forward
    torch.set_num_threads(4);torch.manual_seed(13)
    spec=json.loads((BASE/'calibration_protocol.json').read_text())
    assert sha(__file__)==spec['source_sha256']
    for rel,expected in json.loads((BASE/'runtime_manifest.json').read_text()).items():assert sha(RT/rel)==expected
    pre=json.loads((BASE/'gradient_preflight/preflight.json').read_text());assert pre['status']=='complete'
    out=BASE/'loss_calibration';out.mkdir(exist_ok=False);begin=time.perf_counter()
    result=dict(status='loading',at=datetime.now().astimezone().isoformat(),pid=os.getpid(),
        start_ticks=Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')',1)[1].split()[19],gpu=args.gpu,
        protocol_sha256=sha(BASE/'calibration_protocol.json'),records=[],optimizer_updates=0,forwards=0)
    def save():
        result['wall_seconds']=time.perf_counter()-begin
        p=out/'calibration.tmp.json';p.write_text(json.dumps(result,indent=2)+'\n');p.replace(out/'calibration.json')
    def move(x):
        if torch.is_tensor(x):return x.to('cuda:0')
        if isinstance(x,dict):return {k:move(v) for k,v in x.items()}
        return x
    def th(x):return hashlib.sha256(x.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
    save()
    try:
        pipe=abot.load_pipeline('cuda:0');released=RT/'checkpoints/H3-World/step-10000.safetensors'
        pipe.load_lora(pipe.dit,state_dict=abot.load_checkpoint_lora(released),hotload=True)
        model=pipe.dit.requires_grad_(False).eval()
        configure_precision(model,'h3_fp32',native_transformer_dir=RT/'DiffSynth-Studio-h3-v2/models/MiniMax/MiniMax-H3/FL2VA/transformer')
        pipe.load_models_to_device(['dit']);params=[]
        for i in range(len(model.blocks)-8,len(model.blocks)):
            for name in ('qkv_proj','out_proj'):
                m=getattr(model.blocks[i].attn,name)
                assert len(m.lora_A_weights)==len(m.lora_B_weights)==1
                for weights in (m.lora_A_weights,m.lora_B_weights):
                    p=torch.nn.Parameter(weights[0].detach().to('cuda:0',torch.float32).clone())
                    weights[0]=p;params.append(p)
        hashes=[th(p) for p in params];torch.cuda.reset_peak_memory_stats();result['status']='calibrating';save()
        for row in spec['train_examples']:
            assert sha(row['encoded_path'])==row['sha256']
            d=torch.load(row['encoded_path'],map_location='cpu',weights_only=True)
            assert d['source']['split']=='train' and '12' in d['negatives']
            clean=d['clean_latents'].to('cuda:0',torch.float32);noise=d['initial_noise'].to('cuda:0',torch.float32)
            anchor=d['anchor'].to('cuda:0');audio=d['audio_noise'].to('cuda:0')
            cond={}
            for branch,c in [('positive',d['positive']),('negative',d['negatives']['12'])]:
                p,t=visible_inputs(move(c['packed']),c['prompt_embeds'].to('cuda:0'),24,390)
                cond[branch]=dict(full_packed=p,prompt=t)
            for sigma in spec['sigmas']:
                current=(1-sigma)*clean[:,:,12:24]+sigma*noise[:,:,12:24]
                target=noise[:,:,12:24]-clean[:,:,12:24];losses={};positive_grads=None
                torch.cuda.synchronize();tick=time.perf_counter()
                for branch in ('positive','negative'):
                    for p in params:p.grad=None
                    v=transition_forward(model,current,history=clean[:,:,:12],history_noise=noise[:,:,:12],
                        anchor=anchor,audio=audio,sigma=sigma,index=1,checkpoint=True,checkpoint_offload=True,**cond[branch])
                    loss=(v.float()-target).square().mean();loss.backward();result['forwards']+=1
                    assert torch.isfinite(loss) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
                    losses[branch]=float(loss.detach())
                    if branch=='positive':positive_grads=[p.grad.detach().cpu().clone() for p in params]
                    else:
                        fm2=pair2=dot=0.
                        for p,gp in zip(params,positive_grads):
                            pair=gp-p.grad.detach().cpu();fm2+=float(gp.double().square().sum());pair2+=float(pair.double().square().sum());dot+=float((gp.double()*pair.double()).sum())
                    del v,loss
                torch.cuda.synchronize()
                rec=dict(clip_id=d['source']['clip_id'],sigma=sigma,positive_FM=losses['positive'],negative_FM=losses['negative'],
                    gap_negative_minus_positive=losses['negative']-losses['positive'],FM_gradient_norm=fm2**.5,
                    pair_gradient_norm=pair2**.5,FM_pair_gradient_cosine=dot/(max(fm2*pair2,1e-30)**.5),seconds=time.perf_counter()-tick)
                result['records'].append(rec);save();print(json.dumps(rec),flush=True)
            del clean,noise,d,cond
        assert len(result['records'])==8 and result['forwards']==16 and hashes==[th(p) for p in params]
        margin=max(1e-5,float(np.median([abs(r['gap_negative_minus_positive']) for r in result['records']])))
        active=[r for r in result['records'] if margin>r['gap_negative_minus_positive'] and r['pair_gradient_norm']>1e-12]
        if not active:raise ValueError('No active finite pair gradients; do not launch training')
        raw=.25*float(np.median([r['FM_gradient_norm']/r['pair_gradient_norm'] for r in active]))
        weight=min(10.,raw)
        result.update(status='complete',margin=margin,action_weight=weight,unclamped_action_weight=raw,
                      active_calibration_states=len(active),parameters_unchanged=True,GPU_peak_MiB=torch.cuda.max_memory_allocated()/2**20,
                      interpretation='Scales fixed from8 training states. Never a video-quality result. Action term targets25% initial FM gradient magnitude, lambda capped10.')
    except BaseException as exc:
        result.update(status='failed',error=repr(exc));raise
    finally:save()


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--gpu',type=int,required=True);args=ap.parse_args()
    free=int(subprocess.check_output(['nvidia-smi',f'--id={args.gpu}','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))
    if free<40000:raise RuntimeError('Need idle GPU with >=40000MiB free')
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu),ABOT_VRAM_RESERVE_GIB='20',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    sys.path[:0]=[str(RT/'code'),str(RT/'code/abot'),str(RT/'DiffSynth-Studio-h3-v2')]
    main(args)
