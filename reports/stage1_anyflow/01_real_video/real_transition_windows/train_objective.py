"""Fixed4-update E2 arm. Same Original initialization/data/bank; loss differs."""
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
    from causal.pretrained_lora import save_h3_lora_adapter
    torch.set_num_threads(4);torch.manual_seed(13)
    spec=json.loads((BASE/'training_protocol.json').read_text())
    assert sha(__file__)==spec['source_sha256']
    for rel,expected in json.loads((BASE/'runtime_manifest.json').read_text()).items():assert sha(RT/rel)==expected
    cal=json.loads((BASE/'loss_calibration/calibration.json').read_text())
    assert cal['status']=='complete' and sha(BASE/'loss_calibration/calibration.json')==spec['calibration_sha256']
    assert sha(BASE/'baseline_review.json')==spec['baseline_review_sha256']
    assert json.loads((BASE/'baseline_review.json').read_text())['complete']
    assert spec['optimizer_steps']==4 and spec['logical_batch']==2
    out=BASE/f'train_{args.arm}';out.mkdir(exist_ok=False);begin=time.perf_counter()
    result=dict(status='loading',at=datetime.now().astimezone().isoformat(),pid=os.getpid(),
                start_ticks=Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')',1)[1].split()[19],gpu=args.gpu,
                arm=args.arm,protocol_sha256=sha(BASE/'training_protocol.json'),updates=[],checkpoints=[],optimizer_updates=0,
                margin=cal['margin'],action_weight=cal['action_weight'] if args.arm=='fm_action' else 0.,
                noisy_forwards=0,backward_calls=0)
    def save():
        result['wall_seconds']=time.perf_counter()-begin
        p=out/'training.tmp.json';p.write_text(json.dumps(result,indent=2)+'\n');p.replace(out/'training.json')
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
        pipe.load_models_to_device(['dit']);params=[];entries=[]
        for i in range(len(model.blocks)-8,len(model.blocks)):
            for name in ('qkv_proj','out_proj'):
                m=getattr(model.blocks[i].attn,name)
                assert len(m.lora_A_weights)==len(m.lora_B_weights)==1
                pa=torch.nn.Parameter(m.lora_A_weights[0].detach().to('cuda:0',torch.float32).clone())
                pb=torch.nn.Parameter(m.lora_B_weights[0].detach().to('cuda:0',torch.float32).clone())
                m.lora_A_weights[0]=pa;m.lora_B_weights[0]=pb;params.extend((pa,pb));entries.append((m.name,pa,pb))
        assert not any(p.requires_grad for p in model.parameters())
        result['trainable_parameters']=sum(p.numel() for p in params)
        result['initial_parameter_hashes']=[th(p) for p in params]
        base_versions=[(p,p._version) for p in model.parameters()]
        opt=torch.optim.AdamW(params,lr=spec['lr'],betas=(.9,.95),weight_decay=0.)
        def checkpoint(step):
            directory=out/f'step_{step:02d}';directory.mkdir()
            file=directory/'action_lora.pt'
            metadata=dict(config=spec,arm=args.arm,optimizer_step=step,precision=result['precision'],
                          architecture='native T2/N, single I0, window12',history='observed_GT_noised_at_current_sigma',
                          action_position_contract='fixed_origin_native_time_grid',generated_history_trained=False)
            save_h3_lora_adapter(file,entries,metadata)
            torch.save(dict(optimizer=opt.state_dict(),step=step,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state()),directory/'optimizer.pt')
            result['checkpoints'].append(dict(step=step,path=str(file),sha256=sha(file)));save()
        checkpoint(0);torch.cuda.reset_peak_memory_stats();result['status']='training';save()
        cached={}
        for step,batch in enumerate(spec['samples']):
            assert len(batch)==2
            opt.zero_grad(set_to_none=True);micros=[];tick=time.perf_counter()
            for sample in batch:
                path=sample['encoded_path'];assert sha(path)==sample['sha256']
                if path not in cached:cached[path]=torch.load(path,map_location='cpu',weights_only=True)
                d=cached[path];assert d['source']['split']=='train'
                index=sample['window'];start=index*12;stop=start+12;sigma=sample['sigma']
                assert str(start) in d['negatives']
                clean=d['clean_latents'].to('cuda:0',torch.float32);noise=d['initial_noise'].to('cuda:0',torch.float32)
                current=(1-sigma)*clean[:,:,start:stop]+sigma*noise[:,:,start:stop]
                target=noise[:,:,start:stop]-clean[:,:,start:stop]
                shared=dict(history=clean[:,:,:start],history_noise=noise[:,:,:start],anchor=d['anchor'].to('cuda:0'),
                            audio=d['audio_noise'].to('cuda:0'),sigma=sigma,index=index)
                conditions={}
                for branch,c in [('positive',d['positive']),('negative',d['negatives'][str(start)])]:
                    p,t=visible_inputs(move(c['packed']),c['prompt_embeds'].to('cuda:0'),stop,390)
                    conditions[branch]=dict(full_packed=p,prompt=t)
                values={}
                with torch.no_grad():
                    for branch in ('positive','negative'):
                        v=transition_forward(model,current,**shared,**conditions[branch]);result['noisy_forwards']+=1
                        values[branch]=float((v.float()-target).square().mean());del v
                raw_hinge=cal['margin']+values['positive']-values['negative']
                alpha=result['action_weight'] if raw_hinge>0 else 0.
                # Sequential replay is the exact hinge gradient with immutable
                # state/parameters. It avoids retaining two33B activation graphs.
                replay_errors={}
                for branch,coefficient in [('positive',1+alpha),('negative',-alpha)]:
                    if coefficient==0:continue
                    v=transition_forward(model,current,checkpoint=True,checkpoint_offload=True,**shared,**conditions[branch])
                    loss=(v.float()-target).square().mean();result['noisy_forwards']+=1
                    error=abs(float(loss.detach())-values[branch]);replay_errors[branch]=error
                    if error>1e-6:raise ValueError('Gradient replay differs from hinge selection pass')
                    (loss*(coefficient/2)).backward();result['backward_calls']+=1;del v,loss
                micros.append(dict(clip_id=d['source']['clip_id'],window=index,sigma=sigma,
                    positive_FM=values['positive'],negative_FM=values['negative'],action_hinge=max(0.,raw_hinge),
                    total=values['positive']+result['action_weight']*max(0.,raw_hinge),hinge_active=raw_hinge>0,replay_errors=replay_errors))
                del clean,noise,current,target,shared,conditions
            if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in params):raise FloatingPointError('Invalid bank gradient')
            norm=float(torch.nn.utils.clip_grad_norm_(params,1.))
            opt.step();result['optimizer_updates']+=1;torch.cuda.synchronize()
            result['updates'].append(dict(step=step+1,microbatches=micros,gradient_norm_before_clip=norm,seconds=time.perf_counter()-tick))
            assert all(p._version==v for p,v in base_versions)
            if step+1==4:checkpoint(4)
            save();print(json.dumps(result['updates'][-1]),flush=True)
        assert result['optimizer_updates']==4
        result.update(status='complete_pending_evaluation',base_parameters_unchanged=True,
                      GPU_peak_MiB=torch.cuda.max_memory_allocated()/2**20,
                      interpretation='Four-update objective smoke only; no efficacy claim until same-state and local visual evaluation. No automatic extension.')
    except BaseException as exc:
        result.update(status='failed',error=repr(exc));raise
    finally:save()


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--gpu',type=int,required=True);ap.add_argument('--arm',choices=['fm_only','fm_action'],required=True);args=ap.parse_args()
    free=int(subprocess.check_output(['nvidia-smi',f'--id={args.gpu}','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))
    if free<40000:raise RuntimeError('Need idle GPU with >=40000MiB free')
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu),ABOT_VRAM_RESERVE_GIB='20',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    sys.path[:0]=[str(RT/'code'),str(RT/'code/abot'),str(RT/'DiffSynth-Studio-h3-v2')]
    main(args)
