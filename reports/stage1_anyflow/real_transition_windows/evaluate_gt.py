"""Frozen Original T2/N,30steps,held-out REAL GT history24; no optimizer."""
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
    import av
    from PIL import Image,ImageDraw,ImageFont
    import infer as abot
    from causal.benchmark import write_video
    from causal.h3_precision import configure_precision
    from causal.local_topology import visible_inputs
    from causal.local_transition import transition_forward
    from causal.anyflow_sampling import configure_video_schedule
    torch.set_num_threads(4);torch.manual_seed(13)
    spec=json.loads((BASE/'gt_eval_protocol.json').read_text())
    assert sha(__file__)==spec['source_sha256']
    for rel,expected in json.loads((BASE/'runtime_manifest.json').read_text()).items():assert sha(RT/rel)==expected,rel
    encpath=BASE/'encoded_validation/encoding.json';enc=json.loads(encpath.read_text())
    assert enc['status']=='complete' and sha(encpath)==spec['encoding_receipt_sha256']
    out=BASE/'gt_history_baseline';out.mkdir(exist_ok=False)
    began=time.perf_counter()
    result=dict(status='loading',at=datetime.now().astimezone().isoformat(),pid=os.getpid(),
                start_ticks=Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')',1)[1].split()[19],gpu=args.gpu,
                protocol_sha256=sha(BASE/'gt_eval_protocol.json'),source_sha256=sha(__file__),records=[],
                denoiser_forwards=0,optimizer_updates=0,CPU_hidden_KV_MiB=0,
                scope='Observed actual joint controls, real GT history24, current12. No counterfactual truth, free rollout or parking A/D sign claim.')
    def save():
        result['wall_seconds']=time.perf_counter()-began
        p=out/'evaluation.tmp.json';p.write_text(json.dumps(result,indent=2)+'\n');p.replace(out/'evaluation.json')
    def move(x):
        if torch.is_tensor(x):return x.to('cuda:0')
        if isinstance(x,dict):return {k:move(v) for k,v in x.items()}
        return x
    def th(x):return hashlib.sha256(x.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
    save()
    try:
        pipe=abot.load_pipeline('cuda:0')
        released=RT/'checkpoints/H3-World/step-10000.safetensors'
        pipe.load_lora(pipe.dit,state_dict=abot.load_checkpoint_lora(released),hotload=True)
        model=pipe.dit.requires_grad_(False).eval()
        result['precision']=configure_precision(model,'h3_fp32',native_transformer_dir=RT/'DiffSynth-Studio-h3-v2/models/MiniMax/MiniMax-H3/FL2VA/transformer')
        result['released_LoRA_sha256']=sha(released)
        versions=[(p,p._version) for p in model.parameters()]
        sigmas=configure_video_schedule(pipe.scheduler,steps=30,grid='native',flow_shift=2.22)
        result['sigmas']=sigmas;torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            for record in enc['clips']:
                assert sha(record['encoded_path'])==record['sha256']
                d=torch.load(record['encoded_path'],map_location='cpu',weights_only=True)
                assert d['format']=='h3world_native_single_I0_real_windows_v2'
                sid=record['clip_id'];folder=out/sid;folder.mkdir()
                history=d['safe_prefixes'][24].to('cuda:0',torch.float32)
                noise=d['initial_noise'].to('cuda:0',torch.float32);history_hash=th(history);noise_hash=th(noise)
                packed,prompt=visible_inputs(move(d['positive']['packed']),d['positive']['prompt_embeds'].to('cuda:0'),36,390)
                anchor=d['anchor'].to('cuda:0');audio=d['audio_noise'].to('cuda:0')
                pipe.load_models_to_device(['dit']);torch.cuda.synchronize();tick=time.perf_counter()
                current=noise[:,:,24:36].clone();result['status']='sampling';save()
                for j,t in enumerate(pipe.scheduler.timesteps):
                    velocity=transition_forward(model,current,history=history,history_noise=noise[:,:,:24],
                        full_packed=packed,prompt=prompt,anchor=anchor,audio=audio,sigma=float(t)/1000,index=2)
                    current=pipe.scheduler.step(velocity,t,current);result['denoiser_forwards']+=1
                    if not torch.isfinite(current).all():raise FloatingPointError('Nonfinite endpoint')
                    if j%5==0:save();print(f'{sid} {j+1}/30',flush=True)
                torch.cuda.synchronize();sampling=time.perf_counter()-tick
                torch.save(current.cpu(),folder/'endpoint.pt')
                assert th(history)==history_hash and th(noise)==noise_hash
                pipe.load_models_to_device(['video_vae']);torch.cuda.synchronize();tick=time.perf_counter()
                def decode(z):
                    rgb=pipe.video_vae.decode_video(z,dtype=pipe.torch_dtype,process_image=False,tiled=True,tile_size=256,tile_overlap=64)
                    return pipe.vae_output_to_video(rgb,min_value=0,max_value=1)
                before=decode(history);assert len(before)==81
                visible=decode(torch.cat((history,current),2));assert len(visible)==120
                frames=visible[81:];assert len(frames)==39
                with np.load(d['source']['path'],allow_pickle=False) as raw:
                    rgb=raw['rgb'];gt=[Image.fromarray(x) for x in rgb[81:120]]
                    context=[Image.fromarray(x) for x in rgb[73:81]]
                font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
                grid=[]
                for j in range(8+39):
                    left=context[j] if j<8 else gt[j-8]
                    right=context[j] if j<8 else frames[j-8]
                    im=Image.new('RGB',(1664,528),'black');draw=ImageDraw.Draw(im)
                    im.paste(left,(0,48));im.paste(right,(832,48))
                    draw.text((8,8),'Observed GT | same actual joint controls',font=font,fill='white')
                    label='KNOWN GT CONTEXT' if j<8 else f'Original T2/N | 30steps | {sampling:.1f}s | RGB{81+j-8}'
                    draw.text((840,8),label,font=font,fill='white');grid.append(im)
                write_video(frames,folder/'current.mp4');write_video(grid,folder/'GT_vs_local.mp4')
                sheet=Image.new('RGB',(208*7,138*6),'white');draw=ImageDraw.Draw(sheet)
                for j,im in enumerate(frames):
                    x=j%7*208;y=j//7*138;draw.text((x+2,y+2),f'RGB{81+j}',fill='black');sheet.paste(im.resize((208,120)),(x,y+18))
                sheet.save(folder/'all_current_frames.jpg',quality=94)
                gray=np.stack([np.asarray(im.convert('L'),dtype=np.float32) for im in frames])
                prior=np.asarray(before[-1].convert('L'),dtype=np.float32)
                gtprior=np.asarray(context[-1].convert('L'),dtype=np.float32)
                with av.open(str(folder/'GT_vs_local.mp4')) as c:
                    stream=c.streams.video[0];count=sum(1 for _ in c.decode(video=0))
                    assert count==47 and stream.codec_context.name=='h264' and stream.codec_context.pix_fmt=='yuv420p'
                result['records'].append(dict(clip_id=sid,history_kind='real_GT_safe_prefix',history_latents=24,current_latents=12,
                    RGB_start=81,RGB_stop=120,conditioning_keys=d['action_script'][24:36],
                    sampling_seconds=sampling,decode_and_artifact_seconds=time.perf_counter()-tick,frames=39,
                    frame_gray_MAD=float(np.abs(np.diff(gray,axis=0)).mean()),boundary_gray_MAD_vs_decoded_GT=float(np.abs(gray[0]-prior).mean()),
                    boundary_gray_MAD_vs_raw_GT=float(np.abs(gray[0]-gtprior).mean()),
                    history_and_noise_unchanged=True,video=str(folder/'GT_vs_local.mp4'),video_sha256=sha(folder/'GT_vs_local.mp4')))
                save();del d,history,noise,anchor,audio,current
            assert result['denoiser_forwards']==60 and all(p._version==v for p,v in versions)
        result.update(status='complete_pending_visual_review',frozen_parameters_unchanged=True,GPU_peak_MiB=torch.cuda.max_memory_allocated()/2**20)
    except BaseException as exc:
        result.update(status='failed',error=repr(exc));raise
    finally:save()


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--gpu',type=int,required=True);args=ap.parse_args()
    free=int(subprocess.check_output(['nvidia-smi',f'--id={args.gpu}','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))
    if free<40000:raise RuntimeError('Need idle GPU with >=40000MiB free')
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu),ABOT_VRAM_RESERVE_GIB='18',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    sys.path[:0]=[str(RT/'code'),str(RT/'code/abot'),str(RT/'DiffSynth-Studio-h3-v2')]
    main(args)
