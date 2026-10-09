"""E1 R-prefix/T2 local parking A/D positive control, no future chunks.

Each current chunk forks from the SAME frozen Original-generated history.
This is an oracle-history local experiment, never an autonomous trajectory.
All video output uses prefix-only decode and never revises emitted RGB.
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
RT=BASE/'runtime'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(args):
    import av
    import numpy as np
    from PIL import Image,ImageDraw
    import torch
    import infer as abot
    from causal.h3_precision import configure_precision
    from causal.local_topology import visible_inputs,window_forward,grounded_prefix
    from causal.h3_cached import H3ChunkCache,chunk_forward
    from causal.anyflow_sampling import configure_video_schedule
    from benchmark import write_video
    from evaluate_action_control import evaluate as flow_metrics
    torch.set_num_threads(4);torch.manual_seed(13)
    prep=json.loads((BASE/'inputs/preparation.json').read_text())
    if prep['status']!='complete':raise RuntimeError('VAE/input preparation is not complete')
    manifest=json.loads((BASE/'runtime_manifest.json').read_text())
    for rel,h in manifest.items():
        if sha(RT/rel)!=h:raise RuntimeError(f'Frozen source changed: {rel}')
    inp=BASE/f'inputs/parking_{args.history}.pt'
    assert sha(inp)==prep['files'][inp.name]
    saved=torch.load(inp,map_location='cpu',weights_only=True)
    out=BASE/f'positive_{args.history}';out.mkdir(exist_ok=False)
    stat=Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')',1)[1].split()
    result=dict(status='loading',scope=__doc__,pid=os.getpid(),start_ticks=stat[19],gpu=args.gpu,
        started_at=datetime.now().astimezone().isoformat(),history_reference=args.history,
        history_kind=saved['history_kind'],optimizer_updates=0,weights='Original H3 + released action LoRA only',
        source_sha256=sha(__file__),runtime_manifest_sha256=sha(BASE/'runtime_manifest.json'),
        input_sha256=sha(inp),preparation_sha256=sha(BASE/'inputs/preparation.json'),
        steps_per_chunk=30,chunk_frames=5,history_cap=5,precision='h3_fp32',
        anchor='RGB dual from initial RGB and available history prefix decode',
        timing_scope='single shared-host feasibility run, cached text/anchors; not a speedup benchmark',
        denoiser_forwards=0,diagnostic_forwards=0,clean_commits=0,cpu_KV_MiB=0,
        window_policy='recompute all known prefix every sigma; history latents read-only',
        preflight={},records=[])
    began=time.perf_counter()
    def save():
        result['wall_seconds']=time.perf_counter()-began
        tmp=out/'evaluation.tmp.json';tmp.write_text(json.dumps(result,indent=2)+'\n')
        tmp.replace(out/'evaluation.json')
    def sync():torch.cuda.synchronize()
    def th(x):
        x=x.detach().cpu().contiguous()
        return hashlib.sha256(str((tuple(x.shape),str(x.dtype))).encode()+x.view(torch.uint8).numpy().tobytes()).hexdigest()
    def move(x):
        if torch.is_tensor(x):return x.to('cuda:0')
        if isinstance(x,dict):return {k:move(v) for k,v in x.items()}
        return x
    save()
    try:
        pipe=abot.load_pipeline('cuda:0')
        released=RT/'checkpoints/H3-World/step-10000.safetensors'
        pipe.load_lora(pipe.dit,state_dict=abot.load_checkpoint_lora(released),hotload=True)
        result['released_action_LoRA_sha256']=sha(released)
        model=pipe.dit.requires_grad_(False).eval()
        result['precision_receipt']=configure_precision(model,'h3_fp32',
            native_transformer_dir=RT/'DiffSynth-Studio-h3-v2/models/MiniMax/MiniMax-H3/FL2VA/transformer')
        pipe.load_models_to_device(['dit']);sync()
        result['load_seconds']=time.perf_counter()-began
        versions=[(p,p._version) for p in model.parameters()]
        reference=saved['reference_latents'].to('cuda:0',torch.float32)
        initial=saved['initial_noise'].to('cuda:0',torch.float32)
        audio=saved['audio_noise'].to('cuda:0')
        packed=move(saved['packed'])
        anchors=[x.to('cuda:0') for x in saved['anchors']]
        sigmas=configure_video_schedule(pipe.scheduler,steps=30,grid='native',flow_shift=2.22)
        result['sigmas']=sigmas
        result['probe_sigma_indices']=[0,15,27]
        result['probe_sigmas']=[sigmas[i] for i in (0,15,27)]
        result['history_latent_sha256']=th(reference)
        result['initial_noise_sha256']=th(initial)
        def conditions(index,action):
            pair=saved['pairs'][index]
            prompt=pair['prompts'][action].to('cuda:0')
            layout,text=visible_inputs(packed,prompt,pair['stop'],390)
            return dict(full_packed=layout,prompt=text,anchor=anchors[index],audio=audio,
                chunk_frames=5,anchor_slot=1)
        def predict(state,index,action,sigma):
            return window_forward(model,state,history=reference[:,:,:index*5],
                                  sigma=sigma,index=index,history_chunks=5,**conditions(index,action))
        torch.cuda.reset_peak_memory_stats()
        generated={'A':[],'D':[]}; frames_by_action={'A':[],'D':[]}
        with torch.no_grad():
            # GPU Original identity and repeated-output floor. All paths use
            # the same physical future-free tensors and global coordinates.
            probe=initial[:,:,:5];sigma=sigmas[15]
            direct=predict(probe,0,'A',sigma)
            repeat=predict(probe,0,'A',sigma)
            assert torch.equal(direct,repeat),'T2 repeat changed output'
            with grounded_prefix():
                cached=chunk_forward(model,probe,index=0,cache=H3ChunkCache(5,'cpu'),sigma=sigma,
                    action_prefix_mode='own',action_feedback=True,**conditions(0,'A'))
            diff=(cached.float()-direct.float())
            result['diagnostic_forwards']+=3
            relative=float(diff.square().mean().sqrt()/direct.float().square().mean().sqrt().clamp_min(1e-12))
            result['preflight'].update(repeat_max_abs=0.,first_chunk_T1_T2_max_abs=float(diff.abs().max()),
                first_chunk_T1_T2_relative_rms=relative,
                note='Same mask predicate; one versus split SDPA query shapes can round differently in BF16.')
            if relative>0.02:raise RuntimeError('First-chunk identity differs by more than predeclared 2% RMS guard')
            del cached,direct,repeat,diff
            result['status']='sampling';save()
            for index,start in enumerate((0,5,10)):
                stop=min(start+5,12);end_rgb=(17,34,39)[index];start_rgb=(0,17,34)[index]
                predicted={};probes=[]
                for action in 'AD':
                    pipe.load_models_to_device(['dit']);sync();tick=time.perf_counter()
                    current=initial[:,:,start:stop].clone()
                    initial_hash=th(current);history_hash=th(reference[:,:,:start])
                    cond=conditions(index,action)
                    for j,t in enumerate(pipe.scheduler.timesteps):
                        if action=='A' and j in (0,15,27):
                            state=current.detach().cpu().clone()
                        vp=predict(current,index,action,float(t)/1000)
                        result['denoiser_forwards']+=1
                        if action=='A' and j in (0,15,27):
                            probes.append(dict(index=index,step=j,sigma=float(t)/1000,state=state,
                                velocity_A=vp.detach().cpu(),history_sha256=history_hash,
                                state_sha256=th(state),condition_action='A',
                                state_kind='actual T2 A solver state from fixed reference history'))
                        current=pipe.scheduler.step(vp,t,current)
                        if not torch.isfinite(current).all():raise FloatingPointError('Nonfinite local endpoint')
                        if j%10==0:
                            save();print(f'[T2 ref={args.history}] chunk{index} action{action} {j+1}/30',flush=True)
                    sync();sample_seconds=time.perf_counter()-tick
                    assert th(reference[:,:,:start])==history_hash
                    target=out/f'chunk{index}_{action}.pt';torch.save(current.cpu(),target)
                    generated[action].append(current.cpu());predicted[action]=current
                    record=dict(chunk=index,action=action,latent_range=[start,stop],rgb_range=[start_rgb,end_rgb],
                        initial_noise_sha256=initial_hash,history_sha256=history_hash,endpoint_sha256=th(current),
                        prompt_sha256=th(cond['prompt']),anchor_sha256=th(cond['anchor']),
                        visible_action_frames=len(cond['full_packed']['action_text_rows']),
                        sampling_seconds=sample_seconds,GPU_peak_MiB=torch.cuda.max_memory_allocated()/2**20)
                    result['records'].append(record);save()
                # The local A/D endpoints differ. The probes above all refer
                # to A's exact solver states; subsequent geometry must fork
                # BOTH actions on each saved state, never subtract D's path.
                torch.save(probes,out/f'chunk{index}_solver_states.pt')
                pipe.load_models_to_device(['video_vae']);sync();dtick=time.perf_counter()
                for action in 'AD':
                    known=torch.cat([reference[:,:,:start],predicted[action]],dim=2)
                    decoded=pipe.video_vae.decode_video(known,dtype=pipe.torch_dtype,
                        process_image=False,tiled=True,tile_size=256,tile_overlap=64)
                    allframes=pipe.vae_output_to_video(decoded,min_value=0,max_value=1)
                    assert len(allframes)==end_rgb
                    currentframes=allframes[start_rgb:end_rgb]
                    path=out/f'chunk{index}_{action}.mp4';write_video(currentframes,path)
                    frames_by_action[action].extend(currentframes)
                    row=next(r for r in result['records'] if r['chunk']==index and r['action']==action)
                    row['flow']=flow_metrics(path)
                    gray=np.stack([np.asarray(im.convert('L'),dtype=np.float32) for im in currentframes])
                    row['frame_gray_MAD']=float(np.abs(np.diff(gray,axis=0)).mean())
                    row['video_sha256']=sha(path)
                    del decoded,allframes
                sync();result.setdefault('decode_pair_seconds',[]).append(time.perf_counter()-dtick)
                save();print(f'[decoded] ref={args.history} chunk{index} A/D local RGB',flush=True)
                del predicted
            for action in 'AD':
                assert len(frames_by_action[action])==39
                write_video(frames_by_action[action],out/f'{action}_local39.mp4')
            grid=[]
            for frame,(left,right) in enumerate(zip(frames_by_action['A'],frames_by_action['D'])):
                image=Image.new('RGB',(left.width*2,left.height+40),'black')
                image.paste(left,(0,40));image.paste(right,(left.width,40))
                draw=ImageDraw.Draw(image);chunk=0 if frame<17 else (1 if frame<34 else 2)
                draw.text((8,10),f'T2 Original-prefix | fixed ref {args.history} | action A | chunk {chunk} | 30 steps',fill='white')
                draw.text((left.width+8,10),f'action D | same history/noise | LOCAL ORACLE FORKS, not free rollout',fill='white')
                grid.append(image)
            write_video(grid,out/'AD_local_forks.mp4')
            for action in 'AD':
                width,height=208,120;sheet=Image.new('RGB',(width*7,(height+18)*6),'white')
                draw=ImageDraw.Draw(sheet)
                for i,frame in enumerate(frames_by_action[action]):
                    x=(i%7)*width;y=(i//7)*(height+18)
                    draw.text((x+2,y+2),f'frame{i}',fill='black');sheet.paste(frame.resize((width,height)),(x,y+18))
                sheet.save(out/f'{action}_all39.jpg')
            assert all(p._version==version for p,version in versions),'Frozen weights changed'
        result.update(status='complete_pending_visual_review',parameter_versions_unchanged=True,
            GPU_peak_MiB=torch.cuda.max_memory_allocated()/2**20,
            history_latents_unchanged=th(reference)==result['history_latent_sha256'],
            latent_history_MiB=reference.numel()*reference.element_size()/2**20,
            local_action_pairs=[dict(chunk=i,A=next(r['flow']['horizontal_flow_px']['mean'] for r in result['records'] if r['chunk']==i and r['action']=='A'),
                D=next(r['flow']['horizontal_flow_px']['mean'] for r in result['records'] if r['chunk']==i and r['action']=='D')) for i in range(3)],
            future_RGB_revision_policy='Only newly available RGB appended; oracle history resets marked at frames17/34',
            stage1_accepted=False)
    except BaseException as exc:
        result.update(status='failed',error=repr(exc));raise
    finally:save()


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--gpu',type=int,required=True)
    ap.add_argument('--history',choices=['A','D'],required=True);args=ap.parse_args()
    free=int(subprocess.check_output(['nvidia-smi',f'--id={args.gpu}','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))
    if free<40000:raise RuntimeError('Positive control requires >=40000MiB free')
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu),ABOT_VRAM_RESERVE_GIB='18',HF_HUB_OFFLINE='1',
        TRANSFORMERS_OFFLINE='1',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',TOKENIZERS_PARALLELISM='false')
    sys.path[:0]=[str(RT/'code'),str(RT/'code/abot'),str(RT/'code/causal'),str(RT/'DiffSynth-Studio-h3-v2')]
    main(args)
