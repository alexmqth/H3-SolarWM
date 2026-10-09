"""Actual33B N-window backward with selected RELEASED LoRA; zero updates."""
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
    import infer as abot
    from causal.h3_precision import configure_precision
    from causal.local_topology import visible_inputs
    from causal.local_transition import transition_forward
    torch.set_num_threads(4);torch.manual_seed(13)
    spec=json.loads((BASE/'gradient_protocol.json').read_text())
    assert sha(__file__)==spec['source_sha256']
    for rel,expected in json.loads((BASE/'runtime_manifest.json').read_text()).items():assert sha(RT/rel)==expected,rel
    assert sha(spec['encoded_path'])==spec['encoded_sha256']
    d=torch.load(spec['encoded_path'],map_location='cpu',weights_only=True)
    assert d['source']['split']=='train' and d['format']=='h3world_native_single_I0_real_windows_v2'
    out=BASE/'gradient_preflight';out.mkdir(exist_ok=False);begin=time.perf_counter()
    result=dict(status='loading',at=datetime.now().astimezone().isoformat(),pid=os.getpid(),
                start_ticks=Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')',1)[1].split()[19],
                gpu=args.gpu,protocol_sha256=sha(BASE/'gradient_protocol.json'),records=[],optimizer_updates=0,
                source_sha256=sha(__file__),inference_identity_checked=False)
    def save():
        result['wall_seconds']=time.perf_counter()-begin
        p=out/'preflight.tmp.json';p.write_text(json.dumps(result,indent=2)+'\n');p.replace(out/'preflight.json')
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
        result['precision']=configure_precision(model,'h3_fp32',native_transformer_dir=RT/'DiffSynth-Studio-h3-v2/models/MiniMax/MiniMax-H3/FL2VA/transformer')
        pipe.load_models_to_device(['dit']);params=[];bank=[]
        clean=d['clean_latents'].to('cuda:0',torch.float32);noise=d['initial_noise'].to('cuda:0',torch.float32)
        audio=d['audio_noise'].to('cuda:0');anchor=d['anchor'].to('cuda:0')
        def conditions(index,branch):
            stop=index*12+12
            c=d['positive'] if branch=='positive' else d['negatives'][str(index*12)]
            p,t=visible_inputs(move(c['packed']),c['prompt_embeds'].to('cuda:0'),stop,390)
            return dict(history=clean[:,:,:index*12],history_noise=noise[:,:,:index*12],full_packed=p,prompt=t,
                        anchor=anchor,audio=audio,sigma=.5,index=index)
        z=.5*clean[:,:,12:24]+.5*noise[:,:,12:24]
        with torch.no_grad():before=transition_forward(model,z,**conditions(1,'positive'))
        for i in range(len(model.blocks)-8,len(model.blocks)):
            for name in ('qkv_proj','out_proj'):
                module=getattr(model.blocks[i].attn,name)
                assert len(module.lora_A_weights)==len(module.lora_B_weights)==1
                # FP32 master values equal released BF16 values exactly.
                pa=torch.nn.Parameter(module.lora_A_weights[0].detach().to('cuda:0',torch.float32).clone())
                pb=torch.nn.Parameter(module.lora_B_weights[0].detach().to('cuda:0',torch.float32).clone())
                module.lora_A_weights[0]=pa;module.lora_B_weights[0]=pb
                params.extend((pa,pb));bank.append(dict(name=module.name,A=pa,B=pb))
        assert not any(p.requires_grad for p in model.parameters()),'Base must stay frozen; hotloaded lists are explicit optimizer bank'
        with torch.no_grad():after=transition_forward(model,z,**conditions(1,'positive'))
        result['parameter_installation_max_abs']=float((after-before).abs().max())
        assert torch.equal(before,after)
        result['inference_identity_checked']=True;del before,after
        result['trainable_parameters']=sum(p.numel() for p in params)
        result['parameter_names']=[e['name'] for e in bank]
        hashes=[th(p) for p in params];torch.cuda.reset_peak_memory_stats();result['status']='backward';save()
        for index,branch in ((1,'positive'),(1,'negative'),(2,'positive')):
            for p in params:p.grad=None
            start=index*12;stop=start+12
            current=.5*clean[:,:,start:stop]+.5*noise[:,:,start:stop]
            target=noise[:,:,start:stop]-clean[:,:,start:stop]
            torch.cuda.synchronize();tick=time.perf_counter()
            v=transition_forward(model,current,checkpoint=True,checkpoint_offload=True,**conditions(index,branch))
            loss=(v.float()-target).square().mean();loss.backward();torch.cuda.synchronize()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
            gradnorm=float(torch.stack([p.grad.float().norm().square() for p in params]).sum().sqrt())
            assert gradnorm>0 and torch.isfinite(loss)
            result['records'].append(dict(history_latents=start,current_action=branch,sigma=.5,FM=float(loss.detach()),
                gradient_norm=gradnorm,wall_seconds=time.perf_counter()-tick,
                GPU_peak_MiB=torch.cuda.max_memory_allocated()/2**20))
            save();print(json.dumps(result['records'][-1]),flush=True);del v,loss
        assert [th(p) for p in params]==hashes
        result.update(status='complete',parameters_unchanged=True,GPU_peak_MiB=torch.cuda.max_memory_allocated()/2**20,
                      scope='Memory and gradient feasibility only. No measured trained quality, no calibrated margin, no optimizer instantiated.')
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
