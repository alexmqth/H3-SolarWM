"""Conditional N-history local videos; 60 noisy forwards per reference/window."""
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

BASE=Path(__file__).resolve().parent;RT=BASE/'runtime';OLD=BASE/'source_coarse'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_terminal(r):
    proc=Path(f'/proc/{r["pid"]}/stat')
    if proc.exists():
        s=proc.read_text().rsplit(')',1)[1].split()
        assert s[19]!=r['start_ticks'] or s[0] in ('Z','X'),'Prerequisite process still active'


def main(args):
    import av
    import numpy as np
    from PIL import Image,ImageDraw,ImageFont
    import torch
    import infer as abot
    from causal.h3_precision import configure_precision
    from causal.local_topology import visible_inputs
    from causal.anyflow_sampling import configure_video_schedule
    from history_conditioning import history_forward
    from benchmark import write_video
    from evaluate_action_control import evaluate as flow_metrics
    torch.set_num_threads(4);torch.manual_seed(13)
    gate=json.loads((BASE/'probe_gate.json').read_text());assert gate['video_window1_launch_allowed']
    for ref in 'AD':
        path=BASE/f'probe_{ref}/probe.json';assert sha(path)==gate['probe_receipts'][ref]
        verify_terminal(json.loads(path.read_text()))
    if args.window==2:
        prior=json.loads((BASE/'window1_review.json').read_text())
        assert prior['both_histories_action_structure_and_adherence_pass'], 'Window2 requires manual window1 gate'
    protocol=json.loads((BASE/'protocol.json').read_text())
    for rel,expected in protocol['sources'].items():assert sha(BASE/rel)==expected,rel
    for rel,expected in json.loads((BASE/'runtime_manifest.json').read_text()).items():assert sha(RT/rel)==expected,rel
    launch=json.loads((BASE/'video_launch.json').read_text());assert sha(__file__)==launch['video_source_sha256']
    inp=OLD/f'inputs/parking_{args.history}.pt'
    prep=json.loads((OLD/'inputs/preparation.json').read_text());assert sha(inp)==prep['files'][inp.name]
    data=torch.load(inp,map_location='cpu',weights_only=True)
    old=json.loads((OLD/f'coarse_{args.history}/evaluation.json').read_text())
    out=BASE/f'window{args.window}_{args.history}';out.mkdir(exist_ok=False)
    ticks=Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')',1)[1].split()[19]
    i=args.window;start,stop=i*12,(i+1)*12;r0,r1=data['rgb_bounds'][i:i+2]
    result=dict(status='loading',at=datetime.now().astimezone().isoformat(),pid=os.getpid(),start_ticks=ticks,
                gpu=args.gpu,scope=__doc__,history_reference=args.history,history_kind=data['history_kind'],window=i,
                latent_range=[start,stop],RGB_range=[r0,r1],steps=30,history_mode='N',
                source_sha256=sha(__file__),protocol_sha256=sha(BASE/'protocol.json'),
                video_launch_sha256=sha(BASE/'video_launch.json'),input_sha256=sha(inp),probe_gate_sha256=sha(BASE/'probe_gate.json'),
                denoiser_forwards=0,optimizer_updates=0,CPU_KV_MiB=0,records=[],stage1_accepted=False,
                timing_scope='Single shared-host run, two action forks + evaluation/VAE; not an end-to-end speedup benchmark')
    began=time.perf_counter()
    def save():
        result['wall_seconds']=time.perf_counter()-began
        p=out/'evaluation.tmp.json';p.write_text(json.dumps(result,indent=2)+'\n');p.replace(out/'evaluation.json')
    def th(value):
        x=value.detach().cpu().contiguous()
        return hashlib.sha256(str((tuple(x.shape),str(x.dtype))).encode()+x.view(torch.uint8).numpy().tobytes()).hexdigest()
    def move(value):
        if torch.is_tensor(value):return value.to('cuda:0')
        if isinstance(value,dict):return {k:move(v) for k,v in value.items()}
        return value
    def sheet(frames,path):
        im=Image.new('RGB',(208*7,138*math.ceil(len(frames)/7)),'white');draw=ImageDraw.Draw(im)
        for j,f in enumerate(frames):
            x=j%7*208;y=j//7*138;draw.text((x+2,y+2),f'frame{r0+j}',fill='black');im.paste(f.resize((208,120)),(x,y+18))
        im.save(path)
    def decoded(path):
        with av.open(str(path)) as c:return [Image.fromarray(f.to_ndarray(format='rgb24')) for f in c.decode(video=0)]
    save()
    try:
        pipe=abot.load_pipeline('cuda:0');released=RT/'checkpoints/H3-World/step-10000.safetensors'
        pipe.load_lora(pipe.dit,state_dict=abot.load_checkpoint_lora(released),hotload=True)
        model=pipe.dit.requires_grad_(False).eval()
        result['precision']=configure_precision(model,'h3_fp32',native_transformer_dir=RT/'DiffSynth-Studio-h3-v2/models/MiniMax/MiniMax-H3/FL2VA/transformer')
        result['released_LoRA_sha256']=sha(released)
        pipe.load_models_to_device(['dit']);torch.cuda.synchronize();result['load_seconds']=time.perf_counter()-began
        history=data['reference_latents'].to('cuda:0',torch.float32);noise=data['initial_noise'].to('cuda:0',torch.float32)
        audio=data['audio_noise'].to('cuda:0');anchor=data['anchor'].to('cuda:0');packed=move(data['packed'])
        history_hash,noise_hash=th(history),th(noise)
        versions=[(p,p._version) for p in model.parameters()]
        sigmas=configure_video_schedule(pipe.scheduler,steps=30,grid='native',flow_shift=2.22)
        assert sigmas==old['sigmas']
        result.update(sigmas=sigmas,history_sha256=th(history[:,:,:start]),full_history_sha256=history_hash,
                      history_noise_sha256=th(noise[:,:,:start]),current_initial_noise_sha256=th(noise[:,:,start:stop]),
                      anchor_sha256=th(anchor),audio_sha256=th(audio))
        torch.cuda.reset_peak_memory_stats();predictions={}
        with torch.no_grad():
            result['status']='sampling';save()
            for a in 'AD':
                layout,text=visible_inputs(packed,data['pairs'][i]['prompts'][a].to('cuda:0'),stop,390)
                source=next(x for x in old['records'] if x['window']==i and x['action']==a)
                assert th(text)==source['prompt_sha256'] and th(layout['img_position_ids'])==source['layout_positions_sha256']
                assert result['history_sha256']==source['history_sha256'] and result['current_initial_noise_sha256']==source['initial_noise_sha256']
                current=noise[:,:,start:stop].clone();torch.cuda.synchronize();tick=time.perf_counter()
                for j,t in enumerate(pipe.scheduler.timesteps):
                    velocity=history_forward(model,current,history=history[:,:,:start],history_noise=noise[:,:,:start],
                        mode='N',sigma=float(t)/1000,index=i,history_chunks=5,chunk_frames=12,anchor_slot=0,
                        full_packed=layout,prompt=text,anchor=anchor,audio=audio)
                    result['denoiser_forwards']+=1
                    assert result['denoiser_forwards']<=60
                    current=pipe.scheduler.step(velocity,t,current)
                    if not torch.isfinite(current).all():raise FloatingPointError('Nonfinite endpoint')
                    if j%5==0:save();print(f'[ref{args.history}] window{i} N {a} {j+1}/30',flush=True)
                torch.cuda.synchronize()
                predictions[a]=current
                torch.save(current.cpu(),out/f'{a}_endpoint.pt')
                result['records'].append(dict(action=a,sampling_seconds=time.perf_counter()-tick,
                    endpoint_sha256=th(current),prompt_sha256=th(text),layout_positions_sha256=th(layout['img_position_ids'])))
                assert th(history)==history_hash and th(noise)==noise_hash
                save()
            pipe.load_models_to_device(['video_vae']);torch.cuda.synchronize();tick=time.perf_counter()
            def decode_prefix(z):
                rgb=pipe.video_vae.decode_video(z,dtype=pipe.torch_dtype,process_image=False,tiled=True,tile_size=256,tile_overlap=64)
                return pipe.vae_output_to_video(rgb,min_value=0,max_value=1)
            prior=decode_prefix(history[:,:,:start]);assert len(prior)==r0
            prior[-1].save(out/'history_last_visible.png')
            result['history_last_visible_sha256']=sha(out/'history_last_visible.png')
            clips={}
            for a in 'AD':
                visible=decode_prefix(torch.cat([history[:,:,:start],predictions[a]],2));assert len(visible)==r1
                current_frames=visible[r0:];path=out/f'{a}.mp4';write_video(current_frames,path)
                row=next(x for x in result['records'] if x['action']==a)
                row.update(flow=flow_metrics(path),video_sha256=sha(path),frames=len(current_frames),
                           frame_gray_MAD=float(np.abs(np.diff(np.stack([np.asarray(f.convert('L'),dtype=np.float32) for f in current_frames]),axis=0)).mean()))
                sheet(current_frames,out/f'{a}_all_frames.jpg')
                clips['N'+a]=decoded(path)
                control=OLD/f'coarse_{args.history}/window{i}_{a}.mp4'
                source=next(x for x in old['records'] if x['window']==i and x['action']==a)
                assert sha(control)==source['video_sha256']
                clips['C'+a]=decoded(control)
                ref=np.asarray(prior[-1].convert('L'),dtype=np.float32)
                row['boundary_MP4_gray_MAD']={mode:float(np.abs(np.asarray(clips[mode+a][0].convert('L'),dtype=np.float32)-ref).mean()) for mode in 'CN'}
                row['boundary_definition']='Decoded MP4 first current frame versus same lossless visible-prefix last RGB; C/N same method, includes any history reset'
                save()
            font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17)
            grid=[];context=prior[-8:]
            for j in range(len(context)+r1-r0):
                im=Image.new('RGB',(1664,1040),'black');draw=ImageDraw.Draw(im)
                for ri,a in enumerate('AD'):
                    for ci,mode in enumerate('CN'):
                        frame=context[j] if j<len(context) else clips[mode+a][j-len(context)]
                        x,y=ci*832,ri*520;im.paste(frame,(x,y+40))
                        label='KNOWN HISTORY' if j<len(context) else f'CURRENT {a} | RGB{r0+j-len(context)}'
                        draw.text((x+8,y+10),f'{mode} {"clean" if mode=="C" else "same-sigma"} | ref{args.history} | {label}',font=font,fill='white')
                grid.append(im)
            write_video(grid,out/'CN_AD_context.mp4')
            result['decode_evaluate_seconds']=time.perf_counter()-tick
            assert result['denoiser_forwards']==60 and th(history)==history_hash and th(noise)==noise_hash
            assert all(p._version==v for p,v in versions)
        result.update(status='complete_pending_visual_review',history_noise_unchanged=True,parameter_versions_unchanged=True,
                      GPU_peak_MiB=torch.cuda.max_memory_allocated()/2**20,
                      interpretation='Single local fork from Original-generated fixed history; not GT, free rollout or action acceptance. Context grid prepends8 previously visible RGB frames.')
    except BaseException as exc:
        result.update(status='failed',error=repr(exc));raise
    finally:save()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--gpu',type=int,required=True)
    p.add_argument('--history',choices=['A','D'],required=True);p.add_argument('--window',type=int,choices=[1,2],required=True);a=p.parse_args()
    free=int(subprocess.check_output(['nvidia-smi',f'--id={a.gpu}','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))
    if free<40000:raise RuntimeError('Need idle GPU with>=40000MiB free')
    os.environ.update(CUDA_VISIBLE_DEVICES=str(a.gpu),ABOT_VRAM_RESERVE_GIB='18',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',TOKENIZERS_PARALLELISM='false')
    sys.path[:0]=[str(RT/'code'),str(RT/'code/abot'),str(RT/'code/causal'),str(RT/'DiffSynth-Studio-h3-v2')]
    main(a)
