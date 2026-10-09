"""Encode REAL ABot RGB/action clips with the same H3 image/text/VAE paths.

Loads only text encoder + video VAE; no DiT or teacher generation. RGB dual
history anchors use decode-prefix/re-encode, matching current rollout semantics.
"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import time
import field as f


def cpu_tree(x):
    if torch.is_tensor(x):return x.detach().cpu()
    if isinstance(x,dict):return {k:cpu_tree(v) for k,v in x.items()}
    if isinstance(x,list):return [cpu_tree(v) for v in x]
    return x


def main(args):
    import av
    import numpy as np
    from PIL import Image
    import abot_action as A
    import action_script as S
    from diffsynth.pipelines.minimax_h3_audio_video import (MiniMaxH3Pipeline,ModelConfig,
        MiniMaxH3Unit_NoiseInitializer,MiniMaxH3Unit_KeyframeEncoder,
        MiniMaxH3Unit_PromptEmbedder,MiniMaxH3Unit_PackedSequenceBuilder)
    from causal.h3_cached import last_frame_image_anchor,expand_packed_two_anchors
    data=f.ROOT/'data/abot_bridge';path=data/'encoding.json';destination=data/'encoded'
    destination.mkdir(exist_ok=True)
    if path.exists():raise FileExistsError(path)
    prep=json.loads((data/'preparation.json').read_text());assert prep['status']=='complete'
    clips=prep['clips'];result=dict(status='preflight',pid=os.getpid(),gpu=args.gpu,clips=[],
        started_at=datetime.now().astimezone().isoformat(),source_kind='real_ABot_episode',
        source_manifest_sha256=f.p.sha(data/'clips.jsonl'),model_scope='video VAE and text encoder only; no DiT',
        anchor_protocol='global_retimed_rgb_prefix_last_image_dual_v2',seed=13)
    tick=time.perf_counter()
    def save():
        result['wall_seconds']=time.perf_counter()-tick
        tmp=path.with_suffix('.tmp.json');tmp.write_text(json.dumps(result,indent=2)+'\n');tmp.replace(path)
    save()
    try:
        free=int(subprocess.check_output(['nvidia-smi','-i',str(args.gpu),'--query-gpu=memory.free',
            '--format=csv,noheader,nounits'],text=True).strip())
        if free<24000:result.update(status='not_started_gpu_busy',free_MiB=free);return
        torch.set_num_threads(4);torch.manual_seed(13)
        config=dict(offload_dtype=torch.bfloat16,offload_device='cpu',onload_dtype=torch.bfloat16,
            onload_device='cpu',preparing_dtype=torch.bfloat16,preparing_device='cuda:0',
            computation_dtype=torch.bfloat16,computation_device='cuda:0')
        result['status']='loading';save()
        pipe=MiniMaxH3Pipeline.from_pretrained(torch_dtype=torch.bfloat16,device='cuda:0',
            model_configs=[ModelConfig(model_id='MiniMax/MiniMax-H3',origin_file_pattern=pattern,**config)
                for pattern in ('FL2VA/text_encoder/model*.safetensors','FL2VA/video_vae/source/model.safetensors')],
            processor_config=ModelConfig(model_id='MiniMax/MiniMax-H3',origin_file_pattern='FL2VA/processor/'),vram_limit=20.)
        assert pipe.dit is None and pipe.audio_vae is None
        result.update(status='encoding',gpu_free_start_MiB=free);save()
        with torch.no_grad():
            for index,clip in enumerate(clips):
                target=destination/(clip['clip_id']+'.pt')
                if target.exists():raise FileExistsError(target)
                for field,hashfield in [('video','video_sha256'),('action','action_sha256'),('first_frame','first_frame_sha256')]:
                    assert f.p.sha(clip[field])==clip[hashfield]
                with av.open(clip['video']) as container:
                    frames=[x.to_image().convert('RGB') for x in container.decode(video=0)]
                assert len(frames)==39
                matrix=np.load(clip['action']);assert matrix.shape==(39,17)
                pooled=A.bin_to_latent(matrix,12);keys=S.keys9(pooled);script=S.annotate(pooled)
                assert len(script)==12
                # Independent reconstruction of the key max-pooling interval.
                for k,(lo,hi) in enumerate(A.frame_spans(12)):
                    assert np.array_equal(pooled[k,:A.NUM_KEYS],matrix[lo:hi,:A.NUM_KEYS].max(axis=0))
                noise=MiniMaxH3Unit_NoiseInitializer().process(pipe,seed=13,num_frames=39,height=480,width=832,rand_device='cpu')
                pipe.load_models_to_device(['video_vae'])
                rgb=pipe.preprocess_video(frames,torch_dtype=torch.float32,min_value=0,device='cuda:0')
                clean=pipe.video_vae.encode_video(rgb,dtype=pipe.torch_dtype,process_image=False,
                    tiled=True,tile_size=256,tile_overlap=64).to(torch.bfloat16)
                assert clean.shape==(1,24,12,30,52) and torch.isfinite(clean).all()
                first=MiniMaxH3Unit_KeyframeEncoder().process(pipe,keyframes=[frames[0]],keyframe_indices=[0],
                    video_latents=noise['video_latents'],rand_device='cpu',seed=13,height=480,width=832)['keyframe_cond_anchor']
                anchors=[torch.cat((first,first))]
                for chunk in (1,2):
                    tail=last_frame_image_anchor(pipe.video_vae,clean[:,:,:chunk*5],dtype=pipe.torch_dtype)
                    anchors.append(torch.cat((first,tail)))
                reconstruction=None
                if index in (0,16):
                    decoded=pipe.video_vae.decode_video(clean,dtype=pipe.torch_dtype,process_image=False,
                        tiled=True,tile_size=256,tile_overlap=64).float().clamp(0,1)
                    assert decoded.shape==rgb.shape and torch.isfinite(decoded).all()
                    mse=float((decoded-rgb).square().mean());reconstruction=dict(MSE=mse,PSNR_dB=-10*np.log10(max(mse,1e-20)))
                    sheet=Image.new('RGB',(3*416,2*260),(245,245,245))
                    for col,frame_index in enumerate((0,19,38)):
                        sheet.paste(frames[frame_index].resize((416,240)),(col*416,0))
                        array=(decoded[0,:,frame_index].permute(1,2,0).cpu().numpy()*255).round().astype(np.uint8)
                        sheet.paste(Image.fromarray(array).resize((416,240)),(col*416,260))
                    sheet.save(data/f'vae_reconstruction_{clip["split"]}.jpg')
                    del decoded
                encoded=MiniMaxH3Unit_PromptEmbedder().process(pipe,prompt=clip['prompt'],keyframes=[frames[0]],
                    height=480,width=832,action_script=script)
                packed=MiniMaxH3Unit_PackedSequenceBuilder()._build_packed_fl2va(
                    encoded['prompt_embeds'].shape[0],12,30,52,65,[0],
                    action_text_spans=encoded['action_text_spans'])
                conditions=dict(prompt_embeds=encoded['prompt_embeds'],packed=packed,anchor=first,
                    initial_noise=noise['video_latents'],audio_noise=noise['audio_latents'])
                tensors=[clean,encoded['prompt_embeds'],*anchors,noise['video_latents'],noise['audio_latents']]
                assert all(torch.isfinite(x).all() for x in tensors)
                artifact=dict(format='h3world_real_abot_causal_v1',source=clip,
                    source_kind='real_ABot_episode',clean_latents=clean,conditioning=conditions,
                    causal_packed=expand_packed_two_anchors(packed,frame_rows=390),
                    anchors=anchors,action_cond=torch.from_numpy(keys),action_script=script,
                    latent_RGB_spans=[list(x) for x in A.frame_spans(12)],seed=13,
                    anchor_protocol=result['anchor_protocol'])
                torch.save(cpu_tree(artifact),target)
                result['clips'].append(dict(clip_id=clip['clip_id'],sample_id=clip['sample_id'],split=clip['split'],
                    encoded_file=str(target),sha256=f.p.sha(target),clean_shape=list(clean.shape),
                    prompt_rows=encoded['prompt_embeds'].shape[0],action_sentences=script,
                    source_frame_action_alignment=True,finite=True,reconstruction=reconstruction))
                save();print(json.dumps(dict(index=index+1,total=len(clips),clip=clip['clip_id'])),flush=True)
                del clean,conditions,anchors,artifact,rgb,encoded,noise,first,tensors
        train={x['sample_id'] for x in result['clips'] if x['split']=='train'}
        val={x['sample_id'] for x in result['clips'] if x['split']=='validation'}
        assert train.isdisjoint(val)
        manifest=dict(format='h3world_real_abot_manifest_v1',source_kind='real_ABot_episode',
            anchor_protocol=result['anchor_protocol'],chunk_frames=5,history_chunks=5,
            clips=result['clips'],episode_split_disjoint=True)
        (data/'encoded_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        result.update(status='complete',gpu_allocated_peak_MiB=torch.cuda.max_memory_allocated()/2**20,
            episode_split_disjoint=True)
    except BaseException as exc:
        result.update(status='failed',error=repr(exc));raise
    finally:save()


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--gpu',type=int,required=True)
    args=ap.parse_args();f.setup(args.gpu)
    os.environ['DIFFSYNTH_MODEL_BASE_PATH']=str(f.p.RUNTIME/'DiffSynth-Studio-h3-v2/models')
    os.environ['TOKENIZERS_PARALLELISM']='false'
    import torch
    main(args)
