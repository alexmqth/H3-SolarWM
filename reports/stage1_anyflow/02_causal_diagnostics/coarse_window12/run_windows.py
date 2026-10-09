"""E1 Original-weight visible-window A/D forks with native time and single I0.

Default: three 12-latent current windows from fixed full37 reference histories.
--first5: one matched first-window control with identical full37 inputs.
Both are local oracle tests, not autonomous free rollout or Stage1 acceptance.
"""
import argparse
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

BASE=Path(__file__).resolve().parent;RT=BASE/'runtime'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(args):
    import numpy as np
    from PIL import Image,ImageDraw,ImageFont
    import torch
    import infer as abot
    from causal.h3_precision import configure_precision
    from causal.local_topology import visible_inputs,window_forward,grounded_prefix
    from causal.h3_cached import H3ChunkCache,chunk_forward
    from causal.anyflow_sampling import configure_video_schedule
    from native_prefix import native_prefix_variant
    from single_anchor import original_image_window_variant
    from benchmark import write_video
    from evaluate_action_control import evaluate as flow_metrics
    window_forward=original_image_window_variant(window_forward)
    chunk_forward=native_prefix_variant(chunk_forward)
    torch.set_num_threads(4);torch.manual_seed(13)
    prep=json.loads((BASE/'inputs/preparation.json').read_text());assert prep['status']=='complete'
    for rel,h in json.loads((BASE/'runtime_manifest.json').read_text()).items():assert sha(RT/rel)==h
    inp=BASE/f'inputs/parking_{args.history}.pt';assert sha(inp)==prep['files'][inp.name]
    data=torch.load(inp,map_location='cpu',weights_only=True)
    width=5 if args.first5 else 12
    ranges=[(0,5)] if args.first5 else [(0,12),(12,24),(24,36)]
    rgb_bounds=[0,17] if args.first5 else data['rgb_bounds']
    out=BASE/('matched_first5' if args.first5 else f'coarse_{args.history}');out.mkdir(exist_ok=False)
    ticks=Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')',1)[1].split()[19]
    result=dict(status='loading',scope=__doc__,variant='matched_first5' if args.first5 else 'coarse12',
        pid=os.getpid(),start_ticks=ticks,gpu=args.gpu,started_at=datetime.now().astimezone().isoformat(),
        history_reference=args.history,history_kind=data['history_kind'],optimizer_updates=0,
        weights='Original H3 + released action LoRA only',source_sha256=sha(__file__),
        runtime_manifest_sha256=sha(BASE/'runtime_manifest.json'),input_sha256=sha(inp),
        preparation_sha256=sha(BASE/'inputs/preparation.json'),
        protocol_sha256=sha(BASE/'protocol.json'),launch_protocol_sha256=sha(BASE/'launch_protocol.json'),
        single_anchor_variant_sha256=sha(BASE/'single_anchor.py'),prefix_variant_sha256=sha(BASE/'native_prefix.py'),
        steps_per_window=30,current_window_latents=width,history_cap_windows=5,precision='h3_fp32',
        anchor='original I0 at original coordinates',prefix_time='native1-sigma',audio='fixed noise at native0',
        position_contract='frozen action-content-independent full37 coordinates; future rows physically deleted',
        denoiser_forwards=0,diagnostic_forwards=0,clean_commits=0,cpu_KV_MiB=0,
        window_policy='recompute all visible history/prefix each sigma; raw history immutable',
        timing_scope='single shared-host run with cached conditioning; not latency/speedup benchmark',
        rgb_bounds=rgb_bounds,preflight={},records=[],stage1_accepted=False)
    began=time.perf_counter()
    def save():
        result['wall_seconds']=time.perf_counter()-began
        tmp=out/'evaluation.tmp.json';tmp.write_text(json.dumps(result,indent=2)+'\n');tmp.replace(out/'evaluation.json')
    def th(x):
        x=x.detach().cpu().contiguous()
        return hashlib.sha256(str((tuple(x.shape),str(x.dtype))).encode()+x.view(torch.uint8).numpy().tobytes()).hexdigest()
    def sync():torch.cuda.synchronize()
    def move(x):
        if torch.is_tensor(x):return x.to('cuda:0')
        if isinstance(x,dict):return {k:move(v) for k,v in x.items()}
        return x
    def sheet(frames,path,first=0):
        im=Image.new('RGB',(208*7,138*math.ceil(len(frames)/7)),'white');draw=ImageDraw.Draw(im)
        for j,frame in enumerate(frames):
            x=j%7*208;y=j//7*138;draw.text((x+2,y+2),f'frame{first+j}',fill='black')
            im.paste(frame.resize((208,120)),(x,y+18))
        im.save(path)
    save()
    try:
        pipe=abot.load_pipeline('cuda:0');released=RT/'checkpoints/H3-World/step-10000.safetensors'
        pipe.load_lora(pipe.dit,state_dict=abot.load_checkpoint_lora(released),hotload=True)
        model=pipe.dit.requires_grad_(False).eval();result['released_action_LoRA_sha256']=sha(released)
        result['precision_receipt']=configure_precision(model,'h3_fp32',
            native_transformer_dir=RT/'DiffSynth-Studio-h3-v2/models/MiniMax/MiniMax-H3/FL2VA/transformer')
        pipe.load_models_to_device(['dit']);sync();result['load_seconds']=time.perf_counter()-began
        versions=[(p,p._version) for p in model.parameters()]
        history=data['reference_latents'].to('cuda:0',torch.float32)
        initial=data['initial_noise'].to('cuda:0',torch.float32)
        audio=data['audio_noise'].to('cuda:0');anchor=data['anchor'].to('cuda:0')
        packed=move(data['packed'])
        result['history_latent_sha256']=th(history);result['full_initial_noise_sha256']=th(initial)
        result['audio_sha256']=th(audio);result['anchor_sha256']=th(anchor)
        sigmas=configure_video_schedule(pipe.scheduler,steps=30,grid='native',flow_shift=2.22)
        result['sigmas']=sigmas;result['probe_sigma_indices']=[0,15,27]
        def conditions(index,action):
            # first5 reuses pair0 prompt, then physically drops rows5+.
            text=data['pairs'][index]['prompts'][action].to('cuda:0')
            layout,text=visible_inputs(packed,text,ranges[index][1],390)
            return dict(full_packed=layout,prompt=text,anchor=anchor,audio=audio,
                        chunk_frames=width,anchor_slot=0)
        def predict(state,index,action,sigma):
            return window_forward(model,state,history=history[:,:,:ranges[index][0]],
                sigma=sigma,index=index,history_chunks=5,**conditions(index,action))
        torch.cuda.reset_peak_memory_stats();all_frames={'A':[],'D':[]}
        with torch.no_grad():
            first=initial[:,:,:width];sigma=sigmas[15]
            direct=predict(first,0,'A',sigma);repeat=predict(first,0,'A',sigma)
            assert torch.equal(direct,repeat),'Nonzero repeat floor'
            with grounded_prefix(chunk_frames=width):
                cached=chunk_forward(model,first,index=0,sigma=sigma,cache=H3ChunkCache(5,'cpu'),
                    action_prefix_mode='own',action_feedback=True,**conditions(0,'A'))
            error=(cached.float()-direct.float());relative=float(error.square().mean().sqrt()/direct.float().square().mean().sqrt().clamp_min(1e-12))
            result['diagnostic_forwards']=3
            result['preflight']=dict(repeat_max_abs=0.,first_window_T1_T2_max_abs=float(error.abs().max()),
                first_window_T1_T2_relative_rms=relative,note='Split versus whole SDPA can round differently; predeclared2%RMS guard')
            if relative>.02:raise RuntimeError('First-window T1/T2 identity exceeds predeclared guard')
            del cached,direct,repeat,error,first
            result['status']='sampling';save()
            for index,(start,stop) in enumerate(ranges):
                predictions={};probes=[];r0,r1=rgb_bounds[index:index+2]
                for action in 'AD':
                    pipe.load_models_to_device(['dit']);sync();tick=time.perf_counter()
                    current=initial[:,:,start:stop].clone();noise_hash=th(current);history_hash=th(history[:,:,:start])
                    cond=conditions(index,action)
                    for j,t in enumerate(pipe.scheduler.timesteps):
                        if action=='A' and j in (0,15,27):state=current.detach().cpu().clone()
                        vp=predict(current,index,action,float(t)/1000);result['denoiser_forwards']+=1
                        if action=='A' and j in (0,15,27):
                            probes.append(dict(index=index,step=j,sigma=float(t)/1000,state=state,
                                velocity_A=vp.detach().cpu(),history_sha256=history_hash,state_sha256=th(state),
                                condition_action='A',state_kind='actual A solver state from fixed reference history'))
                        current=pipe.scheduler.step(vp,t,current)
                        if not torch.isfinite(current).all():raise FloatingPointError('Nonfinite endpoint')
                        if j%5==0:save();print(f'[{result["variant"]} ref{args.history}] window{index} {action} {j+1}/30',flush=True)
                    sync();sample_seconds=time.perf_counter()-tick
                    assert th(history[:,:,:start])==history_hash
                    torch.save(current.cpu(),out/f'window{index}_{action}.pt');predictions[action]=current
                    result['records'].append(dict(window=index,action=action,latent_range=[start,stop],rgb_range=[r0,r1],
                        initial_noise_sha256=noise_hash,history_sha256=history_hash,endpoint_sha256=th(current),
                        prompt_sha256=th(cond['prompt']),anchor_sha256=th(anchor),
                        visible_action_frames=len(cond['full_packed']['action_text_rows']),
                        layout_positions_sha256=th(cond['full_packed']['img_position_ids']),
                        sampling_seconds=sample_seconds,GPU_peak_MiB=torch.cuda.max_memory_allocated()/2**20))
                    save()
                torch.save(probes,out/f'window{index}_solver_states.pt')
                # Both actions must be evaluated on each captured A solver
                # state for later geometry; D's own path is never subtracted.
                pipe.load_models_to_device(['video_vae']);sync();tick=time.perf_counter()
                for action in 'AD':
                    known=torch.cat([history[:,:,:start],predictions[action]],dim=2)
                    decoded=pipe.video_vae.decode_video(known,dtype=pipe.torch_dtype,
                        process_image=False,tiled=True,tile_size=256,tile_overlap=64)
                    frames=pipe.vae_output_to_video(decoded,min_value=0,max_value=1)
                    assert len(frames)==r1,(index,len(frames),r1)
                    current_frames=frames[r0:r1];all_frames[action].extend(current_frames)
                    path=out/f'window{index}_{action}.mp4';write_video(current_frames,path)
                    row=next(r for r in result['records'] if r['window']==index and r['action']==action)
                    row['flow']=flow_metrics(path);row['video_sha256']=sha(path)
                    gray=np.stack([np.asarray(f.convert('L'),dtype=np.float32) for f in current_frames])
                    row['frame_gray_MAD']=float(np.abs(np.diff(gray,axis=0)).mean())
                    sheet(current_frames,out/f'window{index}_{action}_all_frames.jpg',r0)
                    del decoded,frames,known
                sync();result.setdefault('decode_pair_seconds',[]).append(time.perf_counter()-tick)
                save();print(f'[decoded] ref{args.history} window{index} RGB[{r0},{r1})',flush=True)
                del predictions
            for action in 'AD':
                assert len(all_frames[action])==rgb_bounds[-1]
                write_video(all_frames[action],out/f'{action}_local.mp4')
                sheet(all_frames[action],out/f'{action}_all_frames.jpg')
            font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16);grid=[]
            for j,(a,d) in enumerate(zip(all_frames['A'],all_frames['D'])):
                i=next(i for i in range(len(ranges)) if rgb_bounds[i]<=j<rgb_bounds[i+1])
                im=Image.new('RGB',(1664,520),'black');im.paste(a,(0,40));im.paste(d,(832,40));draw=ImageDraw.Draw(im)
                draw.text((8,10),f'{width}-latent T2 | reference {args.history} | current A | window{i} | 30 steps',font=font,fill='white')
                draw.text((840,10),'current D | same history/noise | ORACLE LOCAL FORKS',font=font,fill='white');grid.append(im)
            write_video(grid,out/'AD_local_forks.mp4')
            assert all(p._version==v for p,v in versions),'Frozen weights changed'
        result.update(status='complete_pending_visual_review',parameter_versions_unchanged=True,
            GPU_peak_MiB=torch.cuda.max_memory_allocated()/2**20,
            history_latents_unchanged=th(history)==result['history_latent_sha256'],
            latent_history_MiB=history.numel()*history.element_size()/2**20,
            local_action_pairs=[dict(window=i,A=next(r['flow']['horizontal_flow_px']['mean'] for r in result['records'] if r['window']==i and r['action']=='A'),
                D=next(r['flow']['horizontal_flow_px']['mean'] for r in result['records'] if r['window']==i and r['action']=='D')) for i in range(len(ranges))],
            future_RGB_revision_policy='Append only new prefix-decoded interval; boundaries are oracle history resets')
    except BaseException as exc:
        result.update(status='failed',error=repr(exc));raise
    finally:save()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--gpu',type=int,required=True)
    p.add_argument('--history',choices=['A','D'],required=True);p.add_argument('--first5',action='store_true');a=p.parse_args()
    if a.first5 and a.history!='A':p.error('Matched first5 has no history; run once with reference A')
    free=int(subprocess.check_output(['nvidia-smi',f'--id={a.gpu}','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))
    if free<40000:raise RuntimeError('Need idle GPU with>=40000MiB free')
    os.environ.update(CUDA_VISIBLE_DEVICES=str(a.gpu),ABOT_VRAM_RESERVE_GIB='18',HF_HUB_OFFLINE='1',
        TRANSFORMERS_OFFLINE='1',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',TOKENIZERS_PARALLELISM='false')
    sys.path[:0]=[str(RT/'code'),str(RT/'code/abot'),str(RT/'code/causal'),str(RT/'DiffSynth-Studio-h3-v2')]
    main(a)
