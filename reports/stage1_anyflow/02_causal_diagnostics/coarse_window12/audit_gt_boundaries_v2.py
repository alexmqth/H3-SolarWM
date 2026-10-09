"""VAE-only future-RGB intervention for coarse12 GT-history boundaries.

This audits future training/evaluation data semantics. Current coarse jobs
use known Original-generated latent histories, not encoded GT prefixes.
No DiT, optimizer, new downloads, or claim that VAE causes velocity failure.
"""
import argparse
from datetime import datetime
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[3];RT=BASE/'runtime'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(args):
    import av
    import cv2
    import numpy as np
    import torch
    from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Pipeline,ModelConfig
    torch.set_num_threads(4);torch.manual_seed(13)
    out=BASE/'gt_boundary_audit_v2';out.mkdir(exist_ok=False)
    ticks=Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')',1)[1].split()[19]
    result=dict(status='preparing',pid=os.getpid(),start_ticks=ticks,gpu=args.gpu,
        at=datetime.now().astimezone().isoformat(),scope=__doc__,source_sha256=sha(__file__),
        runtime_manifest_sha256=sha(BASE/'runtime_manifest.json'),records=[],sources=[],
        denoiser_forwards=0,optimizer_updates=0,new_download_bytes=0,
        source_selection_note='First attempt stopped before model load: D_1750 extends beyond 1800-frame source. Use A_1030 from the SAME validation episode for an RGB-dependency audit, not an action comparison.',
        prior_attempt_sha256=sha(BASE/'gt_boundary_audit/audit.json'))
    began=time.perf_counter()
    def save():
        result['wall_seconds']=time.perf_counter()-began
        tmp=out/'audit.tmp.json';tmp.write_text(json.dumps(result,indent=2)+'\n');tmp.replace(out/'audit.json')
    def th(x):
        x=x.detach().cpu().contiguous()
        return hashlib.sha256(str((tuple(x.shape),str(x.dtype))).encode()+x.view(torch.uint8).numpy().tobytes()).hexdigest()
    save()
    try:
        manifest=json.loads((BASE/'runtime_manifest.json').read_text())
        for rel,h in manifest.items():assert sha(RT/rel)==h
        prep=json.loads((ROOT/'H3-World/data/abot_bridge/preparation.json').read_text())
        ids=['118eb5d8b75e1b8ac23a4e9ae77af9a9_A_1140','dfec8ed3237860eba14d67c089ecd041_A_1030']
        samples=[]
        for ident in ids:
            row=next(r for r in prep['clips'] if r['clip_id']==ident);assert row['split']=='validation'
            sid=row['sample_id'];path=ROOT/f'H3-World/data/abot_bridge/raw/data/{sid[:2]}/{sid}/video.mp4'
            assert sha(path)==row['source_video_sha256']
            wanted=[row['src_start']+(j*30)//24 for j in range(124)];wanted_set=set(wanted);frames={}
            with av.open(str(path)) as container:
                stream=container.streams.video[0];assert stream.average_rate==30
                assert stream.frames > wanted[-1], (ident, stream.frames, wanted[-1])
                container.seek(int(Fraction(row['src_start'],30)/stream.time_base),stream=stream,backward=True)
                for frame in container.decode(video=0):
                    assert frame.pts is not None
                    index=round(frame.pts*stream.time_base*30)
                    if index>wanted[-1]:break
                    if index in wanted_set:
                        rgb=frame.to_ndarray(format='rgb24')
                        frames[index]=cv2.resize(rgb,(832,480),interpolation=cv2.INTER_AREA)
            assert set(frames)==wanted_set,(ident,len(frames))
            video=np.stack([frames[i] for i in wanted]);del frames
            samples.append((ident,torch.from_numpy(video).permute(3,0,1,2).unsqueeze(0).float().div(255)))
            result['sources'].append(dict(clip_id=ident,source_video=str(path),source_sha256=row['source_video_sha256'],
                source_indices=wanted,RGB_sha256=th(samples[-1][1]),frames=124,resolution=[832,480]))
            save()
        cfg=dict(offload_dtype=torch.bfloat16,offload_device='cpu',onload_dtype=torch.bfloat16,onload_device='cpu',
            preparing_dtype=torch.bfloat16,preparing_device='cuda:0',computation_dtype=torch.bfloat16,computation_device='cuda:0')
        pipe=MiniMaxH3Pipeline.from_pretrained(torch_dtype=torch.bfloat16,device='cuda:0',model_configs=[ModelConfig(
            model_id='MiniMax/MiniMax-H3',origin_file_pattern='FL2VA/video_vae/source/model.safetensors',**cfg)],vram_limit=32.)
        assert pipe.dit is None and pipe.text_encoder is None
        pipe.load_models_to_device(['video_vae']);vae=pipe.video_vae.eval().requires_grad_(False)
        torch.cuda.reset_peak_memory_stats()
        result['vae_configuration']={k:getattr(vae,k) for k in ['clip_length','tokens_chunk_size','token_drop','token_overlap','frame_overlap']}
        result['status']='auditing';save()
        with torch.no_grad():
            for ident,cpu_rgb in samples:
                rgb=cpu_rgb.to('cuda:0')
                def encode(x):return vae.encode_video(x,dtype=torch.bfloat16,process_image=False,tiled=True,tile_size=256,tile_overlap=64).cpu()
                baseline=encode(rgb);assert baseline.shape[2]==37
                for rgb_stop,latent_stop in [(17,5),(34,10),(39,12),(81,24)]:
                    changed=rgb.clone();changed[:,:,rgb_stop:]=1-changed[:,:,rgb_stop:]
                    alternate=encode(changed);del changed
                    diff=(baseline[:,:,:latent_stop].float()-alternate[:,:,:latent_stop].float()).abs()
                    rec=dict(clip_id=ident,rgb_stop=rgb_stop,latent_stop=latent_stop,
                        intervention='Invert RGB strictly after visible prefix',historical_max_abs=float(diff.max()),
                        historical_mean_abs=float(diff.mean()),per_latent_mean_abs=diff.mean((0,1,3,4)).tolist(),
                        full_baseline_latents_sha256=th(baseline),altered_prefix_sha256=th(alternate[:,:,:latent_stop]))
                    # Need enough encoder output tokens despite token_drop.
                    # All added RGB is derived ONLY from last observed RGB.
                    needed=math.ceil((latent_stop+vae.token_drop)/vae.tokens_chunk_size)*vae.clip_length
                    known=rgb[:,:,:rgb_stop]
                    padded=torch.cat([known,known[:,:,-1:].repeat(1,1,max(0,needed-rgb_stop),1,1)],dim=2)
                    safe=encode(padded)[:,:,:latent_stop];del padded
                    assert safe.shape[2]==latent_stop
                    delta=(safe.float()-baseline[:,:,:latent_stop].float()).abs()
                    rec.update(past_only_encoder_input_frames=needed,safe_prefix_sha256=th(safe),
                        safe_vs_full_max_abs=float(delta.max()),safe_vs_full_mean_abs=float(delta.mean()),
                        safe_policy='Observed RGB only; deterministic last-observed-frame extension to satisfy VAE token_drop')
                    result['records'].append(rec);save()
                    print(f'[GT-boundary] {ident} RGB{rgb_stop}/latent{latent_stop} future_effect={rec["historical_max_abs"]:.6f}',flush=True)
                    if latent_stop in (5,10):assert rec['historical_max_abs']==0.,'Aligned control unexpectedly uses future RGB'
                del rgb
        result.update(status='complete',GPU_peak_MiB=torch.cuda.max_memory_allocated()/2**20,
            interpretation='Nonzero coarse-boundary future effect would invalidate slicing a full-GT encoding as causal history. Past-only extension is a proposed causal encoding contract, not a quality-validated model fix.')
    except BaseException as exc:
        result.update(status='failed',error=repr(exc));raise
    finally:save()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--gpu',type=int,required=True);a=p.parse_args()
    free=int(subprocess.check_output(['nvidia-smi',f'--id={a.gpu}','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))
    if free<40000:raise RuntimeError('Need idle GPU with>=40000MiB free')
    os.environ.update(CUDA_VISIBLE_DEVICES=str(a.gpu),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',DIFFSYNTH_SKIP_DOWNLOAD='True',
        DIFFSYNTH_MODEL_BASE_PATH=str(RT/'DiffSynth-Studio-h3-v2/models'),PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    sys.path[:0]=[str(RT/'code'),str(RT/'DiffSynth-Studio-h3-v2')]
    main(a)
