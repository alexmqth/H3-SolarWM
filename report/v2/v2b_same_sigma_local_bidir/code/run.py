"""C: second5 from own first12 history; B: first7. Original weights, 30 steps.

C forks clean/sigma-noised history protocols from the SAME saved own endpoint.
Only current actions differ between A/D. No reference/GT history reset, no
training, no persistent hidden KV, and no claim of full124 rollout acceptance.
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
import xml.etree.ElementTree as ET

BASE=Path(__file__).resolve().parent;RT=BASE/'runtime'
ROOT=BASE.parents[2]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(args):
    import numpy as np
    from PIL import Image,ImageDraw,ImageFont
    import torch
    import infer as abot
    from causal.h3_precision import configure_precision
    from causal.local_topology import visible_inputs
    from causal.anyflow_sampling import configure_video_schedule
    from benchmark import write_video
    from evaluate_action_control import evaluate as flow_metrics
    from interval_forward import interval_forward
    torch.set_num_threads(4);torch.manual_seed(13)
    assert json.loads((BASE/'vae_audit.json').read_text())['passed']
    suite=ET.parse(BASE/'cpu_tests.xml').getroot().find('testsuite')
    assert int(suite.attrib['tests'])==4 and not int(suite.attrib['failures']) and not int(suite.attrib['errors'])
    protocol=json.loads((BASE/'protocol.json').read_text())
    for rel,h in protocol['inputs_sha256'].items():assert sha(ROOT.parent/rel)==h,rel
    for rel,h in json.loads((BASE/'source_coarse/runtime_manifest.json').read_text()).items():assert sha(RT/rel)==h,rel
    launch=json.loads((BASE/'launch.json').read_text())
    for name,h in launch['sources_sha256'].items():assert sha(BASE/name)==h,name
    data={a:torch.load(BASE/f'source_coarse/inputs/parking_{a}.pt',map_location='cpu',weights_only=True) for a in 'AD'}
    for field in ('initial_noise','audio_noise','anchor'):
        assert torch.equal(data['A'][field],data['D'][field]),field
    old=json.loads((BASE/'source_coarse/coarse_A/evaluation.json').read_text())
    c=args.case.startswith('C_');ref=args.case[-1] if c else 'A'
    start,stop=(12,17) if c else (0,7)
    r0,r1=(39,56) if c else (0,22)
    modes=['clean','sigma_noised'] if c else ['clean']
    out=BASE/args.case;out.mkdir(exist_ok=False)
    stat=Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')',1)[1].split()
    result=dict(status='loading',scope=__doc__,at=datetime.now().astimezone().isoformat(),
        case=args.case,gpu=args.gpu,pid=os.getpid(),start_ticks=stat[19],
        source_sha256=sha(__file__),interval_sha256=sha(BASE/'interval_forward.py'),
        protocol_sha256=sha(BASE/'protocol.json'),launch_sha256=sha(BASE/'launch.json'),
        VAE_audit_sha256=sha(BASE/'vae_audit.json'),weights='Original H3 + released action LoRA',
        history_kind='own_generated_first12' if c else 'none',history_action=ref if c else None,
        latent_range=[start,stop],RGB_range=[r0,r1],steps_per_chunk=30,
        CPU_KV_MiB=0,clean_commits=0,optimizer_updates=0,
        denoiser_forwards=0,diagnostic_forwards=0,records=[],quality_accepted=False,
        topology='T2 recompute all visible history/prefix; explicit intervals; no hidden KV',
        timing_scope='Single shared-host sampling; C startup reused, timing separately sourced; no end-to-end speedup claim')
    began=time.perf_counter()
    def save():
        result['wall_seconds']=time.perf_counter()-began
        p=out/'evaluation.tmp.json';p.write_text(json.dumps(result,indent=2)+'\n');p.replace(out/'evaluation.json')
    def th(x):
        x=x.detach().cpu().contiguous()
        return hashlib.sha256(str((tuple(x.shape),str(x.dtype))).encode()+x.view(torch.uint8).numpy().tobytes()).hexdigest()
    def move(x):
        if torch.is_tensor(x):return x.to('cuda:0')
        if isinstance(x,dict):return {k:move(v) for k,v in x.items()}
        return x
    def sheet(frames,path,first=0):
        im=Image.new('RGB',(208*7,138*math.ceil(len(frames)/7)),'white');draw=ImageDraw.Draw(im)
        for j,f in enumerate(frames):
            x=j%7*208;y=j//7*138;draw.text((x+2,y+2),f'RGB {first+j}',fill='black')
            im.paste(f.resize((208,120)),(x,y+18))
        im.save(path)
    save()
    try:
        pipe=abot.load_pipeline('cuda:0')
        released=RT/'checkpoints/H3-World/step-10000.safetensors'
        pipe.load_lora(pipe.dit,state_dict=abot.load_checkpoint_lora(released),hotload=True)
        model=pipe.dit.eval().requires_grad_(False)
        result['released_LoRA_sha256']=sha(released)
        result['precision']=configure_precision(model,'h3_fp32',
            native_transformer_dir=RT/'DiffSynth-Studio-h3-v2/models/MiniMax/MiniMax-H3/FL2VA/transformer')
        pipe.load_models_to_device(['dit']);torch.cuda.synchronize()
        result['load_seconds']=time.perf_counter()-began
        versions=[(p,p._version) for p in model.parameters()]
        initial=data['A']['initial_noise'].to('cuda:0',torch.float32)
        if c:
            hp=BASE/f'source_coarse/coarse_A/window0_{ref}.pt'
            history=torch.load(hp,map_location='cpu',weights_only=True).to('cuda:0',torch.float32)
            source=next(x for x in old['records'] if x['window']==0 and x['action']==ref)
            assert th(history)==source['endpoint_sha256']
            result['reused_startup']=dict(file=str(hp),file_sha256=sha(hp),
                endpoint_sha256=th(history),sampling_seconds=source['sampling_seconds'],
                noisy_forwards=30,RGB_frames=39,source_evaluation_sha256=sha(BASE/'source_coarse/coarse_A/evaluation.json'))
        else:history=initial[:,:,:0].clone()
        assert history.shape[2]==start
        history_hash=th(history);noise_hash=th(initial)
        result['history_sha256']=history_hash;result['noise_sha256']=noise_hash
        result['history_latents_MiB']=history.numel()*history.element_size()/2**20
        audio=data['A']['audio_noise'].to('cuda:0');anchor=data['A']['anchor'].to('cuda:0')
        packed=move(data['A']['packed']);spans=packed['action_text_spans_local']
        result['anchor_sha256']=th(anchor);result['audio_sha256']=th(audio)
        sigmas=configure_video_schedule(pipe.scheduler,steps=30,grid='native',flow_shift=2.22)
        result['sigmas']=sigmas
        def conditions(action,lo=start,hi=stop,base_ref=ref):
            text=data[base_ref]['pairs'][0]['prompts'][base_ref].clone()
            donor=data[action]['pairs'][0]['prompts'][action]
            for a,b in spans[lo:hi]:text[a:b]=donor[a:b]
            layout,text=visible_inputs(packed,text.to('cuda:0'),hi,390)
            return dict(full_packed=layout,prompt=text,anchor=anchor,audio=audio)
        with torch.no_grad():
            # Frozen-source first-window replay, validates reused C checkpoint/config.
            probe=torch.load(BASE/'source_coarse/coarse_A/window0_solver_states.pt',map_location='cpu',weights_only=True)[1]
            replay=interval_forward(model,probe['state'].to('cuda:0'),start=0,
                history=initial[:,:,:0],history_noise=initial[:,:,:0],mode='clean',
                sigma=probe['sigma'],**conditions('A',0,12,'A'))
            reference=probe['velocity_A'].to('cuda:0');diff=(replay-reference).float()
            rr=float(diff.square().mean().sqrt()/reference.float().square().mean().sqrt().clamp_min(1e-12))
            result['diagnostic_forwards']+=1
            result['first12_replay']=dict(max_abs=float(diff.abs().max()),relative_rms=rr)
            assert rr<1e-6,('Saved first12 replay mismatch',rr)
            del replay,reference,diff
            torch.cuda.reset_peak_memory_stats();predictions={};result['status']='sampling';save()
            for mode in modes:
                for action in 'AD':
                    current=initial[:,:,start:stop].clone();cond=conditions(action)
                    torch.cuda.synchronize();tick=time.perf_counter()
                    probe_delta=None;probe_seconds=0.
                    for j,t in enumerate(pipe.scheduler.timesteps):
                        vp=interval_forward(model,current,start=start,history=history,
                            history_noise=initial[:,:,:start],mode=mode,sigma=float(t)/1000,**cond)
                        result['denoiser_forwards']+=1
                        if action=='A' and j==15:
                            # Same noisy state/history, not A-path minus D-path.
                            torch.cuda.synchronize();probe_tick=time.perf_counter()
                            other=interval_forward(model,current,start=start,history=history,
                                history_noise=initial[:,:,:start],mode=mode,sigma=float(t)/1000,**conditions('D'))
                            result['diagnostic_forwards']+=1
                            delta=(vp-other).float()
                            probe_delta=dict(sigma=float(t)/1000,state_sha256=th(current),
                                delta_rms=float(delta.square().mean().sqrt()),
                                A_velocity_rms=float(vp.float().square().mean().sqrt()),
                                note='Same-state action sensitivity, not teacher agreement or quality')
                            del other,delta
                            torch.cuda.synchronize();probe_seconds+=time.perf_counter()-probe_tick
                        current=pipe.scheduler.step(vp,t,current)
                        if not torch.isfinite(current).all():raise FloatingPointError('Nonfinite endpoint')
                        if j%5==0:save();print(f'[{args.case}] {mode} {action} {j+1}/30',flush=True)
                    torch.cuda.synchronize()
                    row=dict(mode=mode,action=action,sampling_with_probe_seconds=time.perf_counter()-tick,
                        diagnostic_probe_seconds=probe_seconds,
                        sampling_seconds=time.perf_counter()-tick-probe_seconds,
                        noisy_forwards=30,additional_same_state_probe_forwards=int(action=='A'),
                        endpoint_sha256=th(current),current_noise_sha256=th(initial[:,:,start:stop]),
                        history_sha256=history_hash,prompt_sha256=th(cond['prompt']),
                        positions_sha256=th(cond['full_packed']['img_position_ids']),
                        GPU_peak_MiB=torch.cuda.max_memory_allocated()/2**20,
                        same_state_action_probe=probe_delta)
                    predictions[mode,action]=current
                    torch.save(current.cpu(),out/f'{mode}_{action}_endpoint.pt')
                    assert th(history)==history_hash and th(initial)==noise_hash
                    result['records'].append(row);save()
            assert result['denoiser_forwards']==(120 if c else 60)
            pipe.load_models_to_device(['video_vae']);torch.cuda.synchronize();tick=time.perf_counter()
            def decode(z):
                rgb=pipe.video_vae.decode_video(z,dtype=pipe.torch_dtype,process_image=False,
                    tiled=True,tile_size=256,tile_overlap=64)
                frames=pipe.vae_output_to_video(rgb,min_value=0,max_value=1)
                del rgb
                return frames
            prior=decode(history) if c else []
            assert len(prior)==r0
            if c:
                write_video(prior,out/'startup_reused.mp4');sheet(prior,out/'startup_all_frames.jpg')
            allframes={}
            for row in result['records']:
                mode,action=row['mode'],row['action']
                frames=decode(torch.cat([history,predictions[mode,action]],2));assert len(frames)==r1
                active=frames[r0:];assert len(active)==r1-r0
                path=out/f'{mode}_{action}_current.mp4';write_video(active,path)
                row['video_sha256']=sha(path);row['flow']=flow_metrics(path)
                gray=np.stack([np.asarray(f.convert('L'),dtype=np.float32) for f in active])
                row['frame_gray_MAD']=float(np.abs(np.diff(gray,axis=0)).mean())
                if c:
                    row['boundary_gray_MAD']=float(np.abs(gray[0]-np.asarray(prior[-1].convert('L'),dtype=np.float32)).mean())
                    before=np.stack([np.asarray(f,dtype=np.float32) for f in prior[-5:]])
                    after=np.stack([np.asarray(f,dtype=np.float32) for f in frames[r0-5:r0]])
                    row['previous5_RGB_revision_MAD_not_applied']=float(np.abs(after-before).mean())
                sheet(active,out/f'{mode}_{action}_all_current.jpg',r0)
                display=prior+active;write_video(display,out/f'{mode}_{action}_rollout.mp4')
                allframes[mode,action]=display
                row['rollout_video_sha256']=sha(out/f'{mode}_{action}_rollout.mp4')
                save()
            font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
            def grid_video(first,path):
                grid=[]
                for j in range(first,r1):
                    im=Image.new('RGB',(832*len(modes),1040),'black');draw=ImageDraw.Draw(im)
                    for ai,action in enumerate('AD'):
                        for mi,mode in enumerate(modes):
                            x,y=mi*832,ai*520;im.paste(allframes[mode,action][j],(x,y+40))
                            stage=f'PAST {ref}' if j<r0 else f'CURRENT {action}'
                            label=f'{args.case} | {mode} | {stage} | RGB {j} | 30 steps/chunk'
                            draw.text((x+8,y+10),label,font=font,fill='white')
                    grid.append(im)
                write_video(grid,path)
            grid_video(0,out/'comparison_rollout.mp4')
            if c:grid_video(r0-8,out/'comparison_context.mp4')
            result['decode_render_evaluate_seconds']=time.perf_counter()-tick
            assert th(history)==history_hash and th(initial)==noise_hash
            assert all(p._version==v for p,v in versions)
        result.update(status='complete_pending_visual_review',history_and_noise_unchanged=True,
            frozen_parameter_versions_unchanged=True,GPU_peak_MiB=torch.cuda.max_memory_allocated()/2**20,
            display_policy='Append current RGB to immutable prior prefix; later re-decoding never revises emitted RGB')
    except BaseException as exc:
        result.update(status='failed',error=repr(exc));raise
    finally:save()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu',type=int,required=True);p.add_argument('--case',choices=['C_A','C_D','B_first'],required=True)
    a=p.parse_args()
    free=int(subprocess.check_output(['nvidia-smi',f'--id={a.gpu}','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))
    if free<40000:raise RuntimeError('Need idle GPU with >=40000 MiB free')
    os.environ.update(CUDA_VISIBLE_DEVICES=str(a.gpu),ABOT_VRAM_RESERVE_GIB='18',HF_HUB_OFFLINE='1',
        TRANSFORMERS_OFFLINE='1',DIFFSYNTH_SKIP_DOWNLOAD='True',TOKENIZERS_PARALLELISM='false',
        PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    sys.path[:0]=[str(RT/'code'),str(RT/'code/abot'),str(RT/'code/causal'),str(RT/'DiffSynth-Studio-h3-v2')]
    main(a)
