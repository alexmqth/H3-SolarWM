"""Read-only clean-history velocity/endpoint errors over the actual solver grid.

The fixed target is real ABot noise-minus-clean, with original natural combined
actions. This is not a new A/D intervention or a video-quality metric. Two
independent checkpoint processes share exact CPU noise/source hashes; each
builds its own KV and includes an Original-weight bidirectional teacher.
"""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

BASE=Path(__file__).resolve().parent
BRIDGE=BASE.parents[1]/'2026-10-08-18/stage1_real_abot_fm'
RT=BRIDGE/'runtime'
MANIFEST=BRIDGE.parents[2]/'data/abot_bridge/encoded_manifest.json'
CLIPS=('118eb5d8b75e1b8ac23a4e9ae77af9a9_A_1140','dfec8ed3237860eba14d67c089ecd041_D_1750')


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def state_errors(prediction,clean,noise,sigma):
    import torch
    p,c,e=prediction.double(),clean.double(),noise.double()
    target=e-c; residual=p-target
    mse=float(residual.square().mean())
    endpoint=((1-sigma)*c+sigma*e)-sigma*p
    endpoint_mse=float((endpoint-c).square().mean())
    # Independent endpoint reconstruction checks the noise/clean velocity sign.
    assert abs(endpoint_mse-sigma*sigma*mse)<1e-10*max(1,mse)
    return dict(raw_velocity_mse=mse,target_velocity_rms=float(target.square().mean().sqrt()),
        normalized_velocity_mse=mse/max(float(target.square().mean()),1e-30),
        clean_endpoint_mse=endpoint_mse,prediction_rms=float(p.square().mean().sqrt()),
        velocity_target_cosine=float(torch.nn.functional.cosine_similarity(p.flatten(),target.flatten(),dim=0)))


def main(args):
    import torch
    import infer as abot
    import geometry_primitives as g
    from causal.h3_cached import H3ChunkCache,chunk_forward
    from causal.h3_precision import configure_precision,validate_precision_checkpoint
    from causal.pretrained_lora import load_adapter
    from causal.stage1_lora import load_stage1_lora
    from causal.anyflow_sampling import configure_video_schedule
    from causal.anyflow_reference import gaussian_timestep_weights
    torch.set_num_threads(4);torch.manual_seed(13)
    run=BASE/f'step_{args.step:02d}';run.mkdir(exist_ok=False)
    checkpoint=BRIDGE/f'train_48/step_{args.step:02d}'
    out=run/'probe.json';started=time.perf_counter();cache=None
    result=dict(status='loading',pid=os.getpid(),gpu=args.gpu,step=args.step,optimizer_updates=0,
        started_at=datetime.now().astimezone().isoformat(),records=[],cache_audits=[],
        scope=__doc__,model_forwards=0,clean_forwards=0,
        source_sha256={str(p):sha(p) for p in (Path(__file__),BRIDGE/'geometry_primitives.py',MANIFEST,RT/'code/causal/h3_cached.py')})
    def save():
        result['wall_seconds']=time.perf_counter()-started
        temp=out.with_suffix('.tmp.json');temp.write_text(json.dumps(result,indent=2)+'\n');temp.replace(out)
    save()
    try:
        training=torch.load(checkpoint/'trainer_state.pt',map_location='cpu',weights_only=True)
        assert training['optimizer_step']==args.step and training['config']['objective']=='fm'
        assert training['config']['real_manifest_sha256']==sha(MANIFEST)
        assert all(sha(checkpoint/n)==h for n,h in training['weight_sha256'].items())
        result['checkpoint_sha256']=training['weight_sha256'];del training
        pipe=abot.load_pipeline('cuda:0')
        pipe.load_lora(pipe.dit,state_dict=abot.load_checkpoint_lora(RT/'checkpoints/H3-World/step-10000.safetensors'),hotload=True)
        model=pipe.dit.requires_grad_(False).eval()
        vmeta=load_adapter(model,checkpoint/'causal_adapter.pt','cuda:0')
        visual=[model.blocks[i].attn.qkv_proj for i in vmeta['block_indices']]
        precision=configure_precision(model,'h3_fp32',native_transformer_dir=RT/'DiffSynth-Studio-h3-v2/models/MiniMax/MiniMax-H3/FL2VA/transformer')
        bank,bmeta=load_stage1_lora(model,checkpoint/'stage1_lora.pt',device='cuda:0')
        validate_precision_checkpoint(vmeta['metadata'],precision);validate_precision_checkpoint(bmeta,precision)
        pipe.load_models_to_device(['dit']);model.requires_grad_(False)
        versions=[(p,p._version) for p in model.parameters()]
        grid8=configure_video_schedule(pipe.scheduler,steps=8,grid='native',flow_shift=2.22)
        grid30=configure_video_schedule(pipe.scheduler,steps=30,grid='native',flow_shift=2.22)
        sigmas=grid8[:-1]+[grid30[-2]]
        assert len(sigmas)==9 and len(set(sigmas))==9 and min(sigmas)>0
        result.update(sigmas=sigmas,grid8=grid8,grid30=grid30,status='running');save()
        def role(teacher):
            for module in [*visual,*bank.modules]:module.enabled=not teacher
        def move(x):
            if torch.is_tensor(x):return x.to('cuda:0')
            if isinstance(x,dict):return {k:move(v) for k,v in x.items()}
            return x
        manifest=json.loads(MANIFEST.read_text())
        with torch.no_grad():
            for ci,clip in enumerate(CLIPS):
                row=next(x for x in manifest['clips'] if x['clip_id']==clip)
                assert row['split']=='validation' and sha(row['encoded_file'])==row['sha256']
                raw=torch.load(row['encoded_file'],map_location='cpu',weights_only=True)
                clean=raw['clean_latents'].float().to('cuda:0')
                cond=move(raw['conditioning']);packed=move(raw['causal_packed'])
                anchors=[move(x) for x in raw['anchors']]
                noise=torch.randn(clean.shape,generator=torch.Generator().manual_seed(914001+ci),dtype=torch.float32).to('cuda:0')
                all_noise_hash=g.tensor_sha(noise)
                def common(k):
                    return dict(full_packed=packed,prompt=cond['prompt_embeds'],anchor=anchors[k],audio=cond['audio_noise'],
                        chunk_frames=5,anchor_frame_index=k*5-1 if k else None,anchor_slot=1,
                        action_prefix_mode='causal',action_feedback=True,action_adapter=None)
                role(False);cache=H3ChunkCache(5,'cpu')
                for chunk in (0,1,2):
                    start=chunk*5;stop=min(start+5,12);c=clean[:,:,start:stop];n=noise[:,:,start:stop]
                    before=g.cache_digest(cache);cv=g.cache_versions(cache);commits=cache.commits
                    for sigma in sigmas:
                        z=((1-sigma)*c+sigma*n).float();preds={}
                        role(False)
                        preds['causal']=chunk_forward(model,z,index=chunk,cache=cache,sigma=sigma,**common(chunk)).float()
                        role(True)
                        preds['original']=g.forward(model,z,clean[:,:,:start],cond,cond['prompt_embeds'],anchors[chunk],packed,sigma,mode='original')
                        result['model_forwards']+=2
                        item=dict(clip=clip,chunk=chunk,sigma=sigma,native8=sigma in grid8[:-1],
                            source_sha256=row['sha256'],noise_sha256=all_noise_hash,state_sha256=g.tensor_sha(z),
                            history_sha256=g.tensor_sha(clean[:,:,:start]),clean_target_sha256=g.tensor_sha(c),
                            prompt_sha256=g.tensor_sha(cond['prompt_embeds']),anchor_sha256=g.tensor_sha(anchors[chunk]),
                            audio_sha256=g.tensor_sha(cond['audio_noise']),
                            roles={k:dict(**state_errors(p,c,n,sigma),prediction_sha256=g.tensor_sha(p)) for k,p in preds.items()},
                            gaussian_weight={str(s):float(gaussian_timestep_weights(torch.tensor([sigma*1000.]),shift=s)[0]) for s in (12.,2.22)},
                            student_teacher=g.compare(preds['causal'],preds['original']))
                        assert all(torch.isfinite(x).all() for x in preds.values())
                        g.assert_versions(cv);assert cache.commits==commits
                        assert all(p._version==v for p,v in versions)
                        result['records'].append(item);save()
                        print(json.dumps(dict(clip=clip,chunk=chunk,sigma=sigma,mse={k:v['raw_velocity_mse'] for k,v in item['roles'].items()})),flush=True)
                    assert g.cache_digest(cache)==before
                    result['cache_audits'].append(dict(clip=clip,chunk=chunk,before=before,after=before,read_only=True,
                        cpu_kv_MiB=cache.nbytes/2**20,commits=commits))
                    if chunk<2:
                        role(False);chunk_forward(model,c,index=chunk,cache=cache,sigma=0.,commit=True,**common(chunk))
                        result['clean_forwards']+=1
                    save()
                cache.clear();cache=None
        result.update(status='complete',peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20)
    except BaseException as exc:result.update(status='failed',error=repr(exc));raise
    finally:
        if cache is not None:cache.clear()
        save()


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--gpu',type=int,required=True)
    ap.add_argument('--step',type=int,choices=[0,48],required=True);args=ap.parse_args()
    free=int(subprocess.check_output(['nvidia-smi',f'--id={args.gpu}','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))
    if free<34000:raise RuntimeError(f'Insufficient GPU{args.gpu} free VRAM: {free}')
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu),ABOT_VRAM_RESERVE_GIB='18',HF_HUB_OFFLINE='1',
        TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    sys.path[:0]=[str(RT/'code'),str(RT/'code/abot'),str(RT/'DiffSynth-Studio-h3-v2'),str(BRIDGE)]
    main(args)
