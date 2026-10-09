"""Freeze parking A/D inputs and audit real VAE temporal dependencies.

Only the released VAE is loaded. No DiT training, no text encoding/download.
Parking histories are Original-generated, explicitly not dataset GT.
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

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(args):
    import av
    import numpy as np
    import torch
    from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Pipeline, ModelConfig
    from diffsynth.models.minimax_h3_dit import patchify_video
    from causal.h3_cached import expand_packed_two_anchors
    torch.set_num_threads(4)
    torch.manual_seed(13)
    out=BASE/'inputs'
    out.mkdir(exist_ok=False)
    result=dict(status='loading',pid=os.getpid(),gpu=args.gpu,
        started_at=datetime.now().astimezone().isoformat(),histories={},encoder_audit=[],
        decoder_audit=[],files={},source_sha256={str(Path(__file__)):sha(__file__)},
        scope=__doc__)
    tick=time.perf_counter()
    def save():
        result['wall_seconds']=time.perf_counter()-tick
        tmp=out/'preparation.tmp.json';tmp.write_text(json.dumps(result,indent=2)+'\n')
        tmp.replace(out/'preparation.json')
    save()
    def th(x):
        x=x.detach().cpu().contiguous()
        return hashlib.sha256(str((tuple(x.shape),str(x.dtype))).encode()+x.view(torch.uint8).numpy().tobytes()).hexdigest()
    try:
        vram=dict(offload_dtype=torch.bfloat16,offload_device='cpu',onload_dtype=torch.bfloat16,
            onload_device='cpu',preparing_dtype=torch.bfloat16,preparing_device='cuda:0',
            computation_dtype=torch.bfloat16,computation_device='cuda:0')
        pipe=MiniMaxH3Pipeline.from_pretrained(torch_dtype=torch.bfloat16,device='cuda:0',
            model_configs=[ModelConfig(model_id='MiniMax/MiniMax-H3',
                origin_file_pattern='FL2VA/video_vae/source/model.safetensors',**vram)],
            vram_limit=32.)
        assert pipe.dit is None and pipe.text_encoder is None
        pipe.load_models_to_device(['video_vae'])
        vae=pipe.video_vae.eval().requires_grad_(False)
        result['vae_configuration']={k:getattr(vae,k) for k in
            ('clip_length','tokens_chunk_size','token_drop','token_overlap','frame_overlap','frame_pre_padding')}
        source=ROOT/'outputs/2026-10-02-03'
        conds={a:torch.load(source/f'action_{a}_teacher_39/conditioning.pt',map_location='cpu',weights_only=True) for a in 'AD'}
        for field in ('initial_noise','audio_noise','anchor'):
            assert torch.equal(conds['A'][field],conds['D'][field]),field
        for key,left in conds['A']['packed'].items():
            right=conds['D']['packed'][key]
            assert torch.equal(left,right) if torch.is_tensor(left) else left==right,key
        spans=conds['A']['packed']['action_text_spans_local']
        assert torch.equal(conds['A']['prompt_embeds'][:spans[0][0]],conds['D']['prompt_embeds'][:spans[0][0]])
        # A/D vocabulary has identical pre-encoded sentence lengths. The
        # global layout is frozen before selecting any future action values.
        assert len(set(hi-lo for lo,hi in spans))==1
        template=expand_packed_two_anchors(conds['A']['packed'],frame_rows=390)
        def decode(z):
            return vae.decode_video(z.to('cuda:0'),dtype=torch.bfloat16,
                process_image=False,tiled=True,tile_size=256,tile_overlap=64)
        def image_anchor(rgb):
            z=vae.encode_video(rgb[:,:,-1:].to('cuda:0').clamp(0,1),dtype=torch.bfloat16,
                process_image=True,tiled=True,tile_size=256,tile_overlap=64)
            return patchify_video(z).contiguous().cpu()
        with torch.no_grad():
            for ref in 'AD':
                path=source/f'action_{ref}_teacher_39/baseline_latents.pt'
                z=torch.load(path,map_location='cpu',weights_only=True)
                rgb=decode(z).cpu()
                assert rgb.shape[2]==39
                anchors=[torch.cat([conds[ref]['anchor']]*2)]
                decoded_prefix={}
                for stop,frames in [(5,17),(10,34)]:
                    part=decode(z[:,:,:stop]).cpu()
                    assert part.shape[2]==frames
                    decoded_prefix[stop]=part
                    anchors.append(torch.cat([conds[ref]['anchor'],image_anchor(part)]))
                    diff=(part.float()-rgb[:,:,:frames].float()).abs()
                    result['decoder_audit'].append(dict(reference=ref,latent_stop=stop,frames=frames,
                        prefix_vs_full_max_abs=float(diff.max()),prefix_vs_full_mean_abs=float(diff.mean()),
                        per_frame_mean_abs=diff.mean((0,1,3,4)).tolist(),
                        interpretation='Full decode may revise earlier RGB. Prefix decode uses no unavailable tokens.'))
                # Explicit intervention on future latent content, same total
                # length/codec graph, confirms temporal dependency.
                changed=z.clone();changed[:,:,5:]+=torch.randn_like(changed[:,:,5:])*.5
                future_rgb=decode(changed).cpu()
                delta=(future_rgb[:,:,:17].float()-rgb[:,:,:17].float()).abs()
                result['decoder_audit'].append(dict(reference=ref,intervention='future_latent_frames_5_plus',
                    earlier_rgb_max_abs=float(delta.max()),earlier_rgb_mean_abs=float(delta.mean())))
                pairs=[]
                for chunk,start in enumerate((0,5,10)):
                    stop=min(start+5,12);prompts={}
                    for action in 'AD':
                        prompt=conds[ref]['prompt_embeds'].clone()
                        for lo,hi in spans[start:stop]:prompt[lo:hi]=conds[action]['prompt_embeds'][lo:hi]
                        prompts[action]=prompt
                    allowed=torch.zeros(len(prompt),dtype=torch.bool)
                    for lo,hi in spans[start:stop]:allowed[lo:hi]=True
                    assert torch.equal(prompts['A'][~allowed],prompts['D'][~allowed])
                    assert not torch.equal(prompts['A'][allowed],prompts['D'][allowed])
                    pairs.append(dict(index=chunk,start=start,stop=stop,prompts=prompts))
                state=dict(format='h3_e1_parking_visible_local_v1',history_reference=ref,
                    history_kind='Original_generated_reference_not_GT',reference_latents=z,
                    initial_noise=conds[ref]['initial_noise'],audio_noise=conds[ref]['audio_noise'],
                    packed=template,anchors=anchors,pairs=pairs,
                    position_contract='Fixed 12-frame global layout from equal-length A/D vocabulary; independent of runtime future action choices',
                    reference_prefix_rgb=decoded_prefix,reference_full_rgb=rgb)
                target=out/f'parking_{ref}.pt';torch.save(state,target)
                result['files'][target.name]=sha(target)
                result['histories'][ref]=dict(source_latents=str(path),source_sha256=sha(path),
                    history_sha256=th(z),noise_sha256=th(state['initial_noise']),
                    audio_sha256=th(state['audio_noise']),anchor_sha256=[th(a) for a in anchors],
                    future_actions_not_required_by_runtime=True)
                save();print(f'[prepared] history {ref}, prefix-only RGB anchors',flush=True)
            # Real data encoding: perturb RGB strictly after each chunk's
            # observed support. Compare historical latent prefix, not targets.
            manifest=ROOT/'data/abot_bridge/encoded_manifest.json'
            clips=['118eb5d8b75e1b8ac23a4e9ae77af9a9_A_1140','dfec8ed3237860eba14d67c089ecd041_D_1750']
            data=json.loads(manifest.read_text())
            for clip in clips:
                row=next(x for x in data['clips'] if x['clip_id']==clip)
                assert sha(row['encoded_file'])==row['sha256']
                record=torch.load(row['encoded_file'],map_location='cpu',weights_only=True)
                path=Path(record['source']['video'])
                assert sha(path)==record['source']['video_sha256']
                with av.open(str(path)) as container:
                    frames=np.stack([f.to_ndarray(format='rgb24') for f in container.decode(video=0)])
                assert len(frames)==39 and frames.shape[1:3]==(480,832)
                real=torch.from_numpy(frames).permute(3,0,1,2).unsqueeze(0).float().div(255).to('cuda:0')
                def encode(x):return vae.encode_video(x,dtype=torch.bfloat16,process_image=False,
                    tiled=True,tile_size=256,tile_overlap=64).cpu()
                baseline=encode(real)
                for rgb_stop,latent_stop in ((17,5),(34,10)):
                    changed=real.clone();changed[:,:,rgb_stop:]=1-changed[:,:,rgb_stop:]
                    changed=encode(changed)
                    diff=(baseline[:,:,:latent_stop].float()-changed[:,:,:latent_stop].float()).abs()
                    result['encoder_audit'].append(dict(clip=clip,rgb_stop=rgb_stop,latent_stop=latent_stop,
                        historical_latent_max_abs=float(diff.max()),historical_latent_mean_abs=float(diff.mean()),
                        source_sha256=record['source']['video_sha256']))
                    if float(diff.max())!=0:raise RuntimeError('GT history encoder depends on future RGB')
                save();print(f'[encoder-audit] {clip} no historical effect under future RGB intervention',flush=True)
        result.update(status='complete',GPU_peak_MiB=torch.cuda.max_memory_allocated()/2**20,
            decoder_policy='Decode available prefix after each chunk, append only new RGB; never revise emitted frames',
            encoder_future_checks_passed=True)
    except BaseException as exc:
        result.update(status='failed',error=repr(exc));raise
    finally:save()


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--gpu',type=int,required=True)
    args=ap.parse_args()
    free=int(subprocess.check_output(['nvidia-smi',f'--id={args.gpu}','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))
    if free<34000:raise RuntimeError('VAE preparation requires an idle GPU with 34GB free')
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        DIFFSYNTH_SKIP_DOWNLOAD='True',DIFFSYNTH_MODEL_BASE_PATH=str(ROOT/'DiffSynth-Studio-h3-v2/models'),
        PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    sys.path[:0]=[str(ROOT/'code'),str(ROOT/'DiffSynth-Studio-h3-v2')]
    main(args)
