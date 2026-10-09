"""Controlled reference calibration: restore Original single RGB anchor.

Same window12, Original weights, native text time, h3_fp32, fixed audio/noise, global positions,
30-step solver. The sole conditioning change is removing the duplicate
image-anchor slot. This is reference calibration, not an anchor trick or
an accepted causal rollout. Requires completed native-text-time window12 dual-anchor control.
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


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(args):
    import numpy as np
    import torch
    from PIL import Image,ImageDraw
    import infer as abot
    from causal.h3_precision import configure_precision
    from causal.local_topology import visible_inputs,window_forward
    from native_prefix import native_prefix_variant
    window_forward=native_prefix_variant(window_forward)
    from causal.anyflow_sampling import configure_video_schedule
    from benchmark import write_video
    from evaluate_action_control import evaluate as flow
    torch.set_num_threads(4);torch.manual_seed(13)
    partial=json.loads((BASE/'native_prefix12_calibration/evaluation.json').read_text())
    assert partial['status']=='complete_pending_visual_review','Need complete matched window12 control'
    first=[r for r in partial['records'] if 'flow' in r]
    assert len(first)==2,'Need completed paired first-window evidence'
    values={r['action']:r['flow']['horizontal_flow_px']['mean'] for r in first}
    assert not (values['A']>0 and values['D']<0),'Calibration triggered by weak local positive control only'
    for rel,h in json.loads((BASE/'runtime_manifest.json').read_text()).items():assert sha(RT/rel)==h
    prepared={a:torch.load(BASE/f'inputs/parking_{a}.pt',map_location='cpu',weights_only=True) for a in 'AD'}
    assert torch.equal(prepared['A']['initial_noise'],prepared['D']['initial_noise'])
    out=BASE/'native_single_anchor12_calibration';out.mkdir(exist_ok=False)
    stat=Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')',1)[1].split()
    result=dict(status='loading',pid=os.getpid(),start_ticks=stat[19],gpu=args.gpu,
        started_at=datetime.now().astimezone().isoformat(),scope=__doc__,
        source_sha256=sha(__file__),runtime_manifest_sha256=sha(BASE/'runtime_manifest.json'),
        trigger_first_chunk_flows=values,trigger_scope='complete window12 dual-anchor A/D control',
        variable='duplicate image-anchor slot removed; same 12-latent current window',
        optimizer_updates=0,control_granularity_RGB_frames=39,steps_per_window=30,
        anchor='Original single image condition',audio='same fixed audio noise and native time',
        position_contract='same global positions; duplicate anchor rows removed physically',prefix_time='native1-sigma, unchanged relative to matched dual-anchor control',prefix_variant_sha256=sha(BASE/'native_prefix.py'),
        records=[],model_forwards=0,stage1_accepted=False)
    began=time.perf_counter()
    def save():
        result['wall_seconds']=time.perf_counter()-began
        tmp=out/'evaluation.tmp.json';tmp.write_text(json.dumps(result,indent=2)+'\n');tmp.replace(out/'evaluation.json')
    def th(x):
        x=x.detach().cpu().contiguous()
        return hashlib.sha256(str((tuple(x.shape),str(x.dtype))).encode()+x.view(torch.uint8).numpy().tobytes()).hexdigest()
    save()
    try:
        pipe=abot.load_pipeline('cuda:0')
        pipe.load_lora(pipe.dit,state_dict=abot.load_checkpoint_lora(RT/'checkpoints/H3-World/step-10000.safetensors'),hotload=True)
        model=pipe.dit.requires_grad_(False).eval()
        configure_precision(model,'h3_fp32',native_transformer_dir=RT/'DiffSynth-Studio-h3-v2/models/MiniMax/MiniMax-H3/FL2VA/transformer')
        pipe.load_models_to_device(['dit'])
        versions=[(p,p._version) for p in model.parameters()]
        result['load_seconds']=time.perf_counter()-began
        result['sigmas']=configure_video_schedule(pipe.scheduler,steps=30,grid='native',flow_shift=2.22)
        data=prepared['A'];initial=data['initial_noise'].to('cuda:0',torch.float32)
        audio=data['audio_noise'].to('cuda:0');anchor=data['anchors'][0][:390].to('cuda:0')
        source=BASE.parents[2]/'outputs/2026-10-02-03/action_A_teacher_39/conditioning.pt'
        original=torch.load(source,map_location='cpu',weights_only=True)
        result['single_anchor_layout_source_sha256']=sha(source)
        assert torch.equal(original['anchor'],data['anchors'][0][:390])
        packed={k:v.to('cuda:0') if torch.is_tensor(v) else v for k,v in original['packed'].items()}
        results={};torch.cuda.reset_peak_memory_stats();result['status']='sampling';save()
        with torch.no_grad():
            for action in 'AD':
                pipe.load_models_to_device(['dit']);torch.cuda.synchronize();tick=time.perf_counter()
                # For each source the reference action is constant across
                # all 12 frames, including pair0's matching branch.
                prompt=prepared[action]['pairs'][0]['prompts'][action].to('cuda:0')
                layout,text=visible_inputs(packed,prompt,12,390)
                current=initial.clone()
                for j,t in enumerate(pipe.scheduler.timesteps):
                    velocity=window_forward(model,current,history=initial[:,:,:0],
                        full_packed=layout,prompt=text,anchor=anchor,audio=audio,sigma=float(t)/1000,
                        index=0,chunk_frames=12,history_chunks=5,anchor_slot=1)
                    current=pipe.scheduler.step(velocity,t,current);result['model_forwards']+=1
                    assert torch.isfinite(current).all()
                    if j%10==0:save();print(f'[native-single-anchor12] {action} {j+1}/30',flush=True)
                torch.cuda.synchronize();sample_seconds=time.perf_counter()-tick
                torch.save(current.cpu(),out/f'{action}_latents.pt')
                pipe.load_models_to_device(['video_vae']);tick=time.perf_counter()
                decoded=pipe.video_vae.decode_video(current,dtype=pipe.torch_dtype,
                    process_image=False,tiled=True,tile_size=256,tile_overlap=64)
                frames=pipe.vae_output_to_video(decoded,min_value=0,max_value=1)
                assert len(frames)==39
                path=out/f'{action}.mp4';write_video(frames,path);results[action]=frames
                intervals=[]
                for start,stop in ((0,17),(17,34),(34,39)):
                    part=out/f'{action}_{start}_{stop}.mp4';write_video(frames[start:stop],part)
                    intervals.append(dict(range=[start,stop],flow=flow(part)))
                result['records'].append(dict(action=action,noise_sha256=th(initial),prompt_sha256=th(text),
                    anchor_sha256=th(anchor),endpoint_sha256=th(current),sampling_seconds=sample_seconds,
                    decode_encode_seconds=time.perf_counter()-tick,flow=flow(path),intervals=intervals,
                    video_sha256=sha(path)))
                sheet=Image.new('RGB',(208*7,138*6),'white');draw=ImageDraw.Draw(sheet)
                for i,f in enumerate(frames):
                    x=i%7*208;y=i//7*138;draw.text((x+2,y+2),f'frame{i}',fill='black')
                    sheet.paste(f.resize((208,120)),(x,y+18))
                sheet.save(out/f'{action}_all39.jpg');save()
                del decoded,current
            grid=[]
            for left,right in zip(results['A'],results['D']):
                image=Image.new('RGB',(1664,520),'black');image.paste(left,(0,40));image.paste(right,(832,40))
                draw=ImageDraw.Draw(image)
                draw.text((8,10),'Original single RGB window12 / 39f | held A | 30 steps | calibration only',fill='white')
                draw.text((840,10),'held D | same initial state/noise | NOT five-latent causal rollout',fill='white')
                grid.append(image)
            write_video(grid,out/'AD_window12.mp4')
            assert all(p._version==v for p,v in versions)
        result.update(status='complete_pending_visual_review',parameter_versions_unchanged=True,
            GPU_peak_MiB=torch.cuda.max_memory_allocated()/2**20,
            caveat='A larger known-action window changes interaction latency/granularity. First17 RGB may depend on the rest of this current window; no claim of five-latent online causality.')
    except BaseException as exc:
        result.update(status='failed',error=repr(exc));raise
    finally:save()


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--gpu',type=int,required=True);args=ap.parse_args()
    free=int(subprocess.check_output(['nvidia-smi',f'--id={args.gpu}','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))
    if free<40000:raise RuntimeError('Need >=40000MiB free')
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu),ABOT_VRAM_RESERVE_GIB='18',HF_HUB_OFFLINE='1',
        TRANSFORMERS_OFFLINE='1',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',TOKENIZERS_PARALLELISM='false')
    sys.path[:0]=[str(RT/'code'),str(RT/'code/abot'),str(RT/'code/causal'),str(RT/'DiffSynth-Studio-h3-v2')]
    main(args)
