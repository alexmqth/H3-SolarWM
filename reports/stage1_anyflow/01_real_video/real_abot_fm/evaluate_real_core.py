"""Real held-out RGB conditions: Original or deployed causal FM inference.

Uses the exact saved real-clip text, first RGB anchor and video/audio noise.
GT-history and generated-history are separate treatments. No AnyFlow or
postprocessing; natural joint actions are NOT pure A/D interventions.
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
RUNTIME=BASE/'runtime'
MANIFEST=BASE.parents[2]/'data/abot_bridge/encoded_manifest.json'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main(args, state_modifier=None):
    import torch
    import infer as abot
    from causal.h3_precision import configure_precision,validate_precision_checkpoint
    from causal.pretrained_lora import load_adapter
    from causal.stage1_lora import load_stage1_lora
    from causal.h3_cached import H3ChunkCache,chunk_forward,last_frame_image_anchor
    from causal.anyflow_sampling import configure_video_schedule
    from benchmark import write_video
    from evaluate_videos import evaluate as video_metrics
    from evaluate_action_control import evaluate as flow_metrics
    from summarize_action_experiment import rgb_boundary
    torch.set_num_threads(4);torch.manual_seed(13)
    manifest=json.loads(MANIFEST.read_text())
    row=next(r for r in manifest['clips'] if r['clip_id']==args.clip)
    if row['split']!='validation':raise ValueError('Only held-out episodes allowed')
    if sha(row['encoded_file'])!=row['sha256']:raise ValueError('Encoded source changed')
    state=torch.load(row['encoded_file'],map_location='cpu',weights_only=True)
    if state['source_kind']!='real_ABot_episode':raise ValueError('Not real RGB GT')
    for k,h in [('video','video_sha256'),('action','action_sha256'),('first_frame','first_frame_sha256')]:
        if sha(state['source'][k])!=state['source'][h]:raise ValueError('Source file changed')
    if state_modifier is not None:
        state=state_modifier(state)
    if args.mode=='original' and (args.checkpoint or args.history!='generated' or args.steps!=30):
        raise ValueError('Original is native 30-step, without causal adapters/GT history')
    if args.mode=='causal' and not args.checkpoint:raise ValueError('Causal requires checkpoint incl step00')
    args.out.mkdir(parents=True,exist_ok=False)
    result=dict(status='loading',pid=os.getpid(),gpu=args.gpu,
        started_at=datetime.now().astimezone().isoformat(),clip_id=args.clip,
        encoded_sha256=row['sha256'],source=state['source'],seed=13,
        initial_conditioning=('saved scene head, first-frame image, video/audio noise; action rows explicitly replaced'
            if state.get('intervention') else 'exact saved text, first-frame image, video/audio noise'),
        action_script=state['action_script'],action_interpretation=state.get('intervention', 'real combined keys/camera; not controlled A/D'),
        mode=args.mode,history_source=args.history,steps=args.steps,
        checkpoint=str(args.checkpoint) if args.checkpoint else None,
        objective='ordinary flow matching, not AnyFlow',chunk_frames=5,history_chunks=5,
        flow_shift=2.22,action_prefix_mode='causal',action_feedback=True,
        cache_device='cpu',anchor_protocol='original single RGB' if args.mode=='original' else state['anchor_protocol'],
        timing_note='one shared-host run, cached conditioning; no warmup mean or speedup claim',
        denoiser_forwards=0,commit_forwards=0,chunk_seconds=[],
        intervention_provenance=state.get('intervention_provenance'))
    def save():
        tmp=args.out/'evaluation.tmp.json';tmp.write_text(json.dumps(result,indent=2)+'\n')
        tmp.replace(args.out/'evaluation.json')
    def sync():torch.cuda.synchronize()
    def tensor_hash(x):return hashlib.sha256(x.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
    save();cache=None;began=time.perf_counter()
    try:
        pipe=abot.load_pipeline('cuda:0')
        pipe.load_lora(pipe.dit,state_dict=abot.load_checkpoint_lora(RUNTIME/'checkpoints/H3-World/step-10000.safetensors'),hotload=True)
        model=pipe.dit.requires_grad_(False).eval()
        precision=configure_precision(model,'h3_fp32',native_transformer_dir=RUNTIME/'DiffSynth-Studio-h3-v2/models/MiniMax/MiniMax-H3/FL2VA/transformer')
        if args.mode=='causal':
            receipt=torch.load(args.checkpoint/'trainer_state.pt',map_location='cpu',weights_only=True)
            if receipt['config']['objective']!='fm' or not receipt['config']['real_data_manifest']:
                raise ValueError('Requires real-data FM checkpoint')
            for name,expected in receipt['weight_sha256'].items():
                if sha(args.checkpoint/name)!=expected:raise ValueError('Checkpoint changed')
            if receipt['config']['real_manifest_sha256']!=sha(MANIFEST):raise ValueError('Training data changed')
            result['optimizer_step']=receipt['optimizer_step']
            result['checkpoint_sha256']=receipt['weight_sha256']
            loaded=load_adapter(model,args.checkpoint/'causal_adapter.pt','cuda:0')
            validate_precision_checkpoint(loaded['metadata'],precision)
            bank,metadata=load_stage1_lora(model,args.checkpoint/'stage1_lora.pt',device='cuda:0')
            validate_precision_checkpoint(metadata,precision)
            del receipt
        pipe.load_models_to_device(['dit']);sync()
        result['load_seconds']=time.perf_counter()-began
        cond=state['conditioning']
        def move(x):return x.to('cuda:0') if torch.is_tensor(x) else x
        cond={k:({j:move(v) for j,v in x.items()} if isinstance(x,dict) else move(x)) for k,x in cond.items()}
        initial=cond['initial_noise'].float();audio=cond['audio_noise'].clone()
        prompt=cond['prompt_embeds'];clean=state['clean_latents'].to('cuda:0',torch.float32)
        anchors=[x.to('cuda:0') for x in state['anchors']]
        packed={k:move(v) for k,v in state['causal_packed'].items()}
        actions=state['action_cond'].to('cuda:0')
        result['input_fingerprints']={k:tensor_hash(v) for k,v in dict(video_noise=initial,audio_noise=audio,
            prompt=prompt,initial_anchor=cond['anchor'],actions=actions).items()}
        torch.save(state['conditioning'],args.out/'conditioning.pt')
        result['video_sigmas']=configure_video_schedule(pipe.scheduler,steps=args.steps,grid='native',flow_shift=2.22)
        pipe.scheduler_audio.set_timesteps(args.steps,shift=3.)
        model.requires_grad_(False)
        versions=[(p,p._version) for p in model.parameters()]
        torch.cuda.reset_peak_memory_stats();sync();tick=time.perf_counter()
        result['status']='sampling';save()
        with torch.no_grad():
            if args.mode=='original':
                video=initial.clone()
                # Native packed assembly uses index_copy_ into the video
                # dtype. Match chunk_forward's FP32 video/audio/anchor
                # policy; text stays in the condition projection's BF16.
                audio=audio.to(video.dtype)
                for j,t in enumerate(pipe.scheduler.timesteps):
                    vp,ap=pipe.model_fn(dit=model,video_latents=video,audio_latents=audio,
                        prompt_embeds=prompt,packed=cond['packed'],keyframe_cond_anchor=cond['anchor'].to(video.dtype),
                        timestep_video=t.to('cuda:0'),timestep_audio=pipe.scheduler_audio.timesteps[j].to('cuda:0'))
                    video=pipe.scheduler.step(vp,t,video)
                    audio=pipe.scheduler_audio.step(ap,pipe.scheduler_audio.timesteps[j],audio)
                    if not torch.isfinite(video).all():raise FloatingPointError('Original nonfinite')
                    result['denoiser_forwards']+=1
                    if j%5==0:print(f'[original] {args.clip} {j+1}/{args.steps}',flush=True)
            else:
                cache=H3ChunkCache(5,'cpu');done=[]
                for index,start in enumerate((0,5,10)):
                    sync();ctick=time.perf_counter()
                    current=initial[:,:,start:start+5].clone()
                    anchor=anchors[index] if args.history=='gt' else anchors[0]
                    if index and args.history=='generated':
                        pipe.load_models_to_device(['video_vae'])
                        tail=last_frame_image_anchor(pipe.video_vae,torch.cat(done,dim=2),dtype=pipe.torch_dtype)
                        anchor=torch.cat((cond['anchor'],tail))
                        pipe.load_models_to_device(['dit'])
                    common=dict(full_packed=packed,prompt=prompt,anchor=anchor,audio=audio,
                        chunk_frames=5,anchor_frame_index=start-1 if index else None,
                        anchor_slot=1,action_prefix_mode='causal',action_feedback=True,
                        action_cond=actions[start:start+current.shape[2]],action_adapter=None)
                    before=cache.commits
                    for j,t in enumerate(pipe.scheduler.timesteps):
                        vp=chunk_forward(model,current,index=index,cache=cache,sigma=float(t)/1000,**common)
                        current=pipe.scheduler.step(vp,t,current)
                        if not torch.isfinite(current).all():raise FloatingPointError('Causal nonfinite')
                        result['denoiser_forwards']+=1
                    if cache.commits!=before:raise RuntimeError('Noisy read mutated cache commits')
                    done.append(current)
                    commit=clean[:,:,start:start+current.shape[2]] if args.history=='gt' else current
                    chunk_forward(model,commit,index=index,cache=cache,sigma=0.,commit=True,**common)
                    result['commit_forwards']+=1
                    sync();result['chunk_seconds'].append(time.perf_counter()-ctick)
                    result['cpu_kv_peak_MiB']=cache.peak_bytes/2**20
                    save();print(f'[causal] {args.clip} {args.history} chunk{index} complete',flush=True)
                video=torch.cat(done,dim=2)
                result['per_layer_commits']=cache.commits;cache.clear()
            sync();result['sampling_seconds']=time.perf_counter()-tick
            result['sampling_peak_GPU_MiB']=torch.cuda.max_memory_allocated()/2**20
            if not all(p._version==v for p,v in versions):raise RuntimeError('Inference changed parameters')
            torch.save(video.cpu(),args.out/'latents.pt')
            # GT-history concatenates separately predicted chunks. It is an
            # oracle-history diagnostic, not an autonomous generated trajectory.
            pipe.load_models_to_device(['video_vae']);dtick=time.perf_counter()
            decoded=pipe.video_vae.decode_video(video,dtype=pipe.torch_dtype,tiled=True,tile_size=256,tile_overlap=64)
            frames=pipe.vae_output_to_video(decoded,min_value=0,max_value=1)
            if len(frames)!=39:raise ValueError('Incomplete decoded video')
            write_video(frames,args.out/'video.mp4');sync()
            result['decode_and_encode_seconds']=time.perf_counter()-dtick
        result['end_to_end_seconds']=time.perf_counter()-began
        result['inference_after_load_seconds']=result['end_to_end_seconds']-result['load_seconds']
        result['GPU_peak_MiB']=torch.cuda.max_memory_allocated()/2**20
        result.setdefault('cpu_kv_peak_MiB',0.)
        result['video_metrics']=video_metrics(args.out/'video.mp4')
        result['flow_metrics']=flow_metrics(args.out/'video.mp4')
        result['boundary']=rgb_boundary(args.out/'video.mp4',[17,34])
        assert (result['video_metrics']['frames'],result['video_metrics']['fps'])==(39,24.)
        result.update(status='complete',completed_at=datetime.now().astimezone().isoformat())
    except BaseException as exc:
        result.update(status='failed',error=repr(exc));raise
    finally:
        if cache is not None:cache.clear()
        save()
