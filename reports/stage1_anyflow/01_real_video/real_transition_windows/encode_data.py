"""Encode audited real transitions: only frozen video VAE and text encoder."""
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


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''): h.update(b)
    return h.hexdigest()


def main(args):
    import numpy as np
    from PIL import Image,ImageDraw
    import torch
    from causal.real_transition_data import A,S,bounded_keys9
    from causal.position_contract import initial_action_text_reference,fixed_action_position_origin
    from causal.local_topology import visible_inputs
    from diffsynth.pipelines.minimax_h3_audio_video import (MiniMaxH3Pipeline,ModelConfig,
        MiniMaxH3Unit_NoiseInitializer,MiniMaxH3Unit_KeyframeEncoder,
        MiniMaxH3Unit_PromptEmbedder,MiniMaxH3Unit_PackedSequenceBuilder)
    torch.set_num_threads(4);torch.manual_seed(13)
    protocol=json.loads((BASE/'protocol.json').read_text())
    assert sha(BASE/'preparation.json')==protocol['preparation_sha256']
    assert json.loads((BASE/'GT_review.json').read_text())['encode_allowed']
    assert sha(__file__)==protocol['sources']['encode_data.py']
    for rel,value in json.loads((BASE/'runtime_manifest.json').read_text()).items():assert sha(RT/rel)==value,rel
    prep=json.loads((BASE/'preparation.json').read_text());clips=[c for c in prep['clips'] if c['split']==args.split]
    dest=BASE/f'encoded_{args.split}';dest.mkdir(exist_ok=False)
    result=dict(status='loading',at=datetime.now().astimezone().isoformat(),pid=os.getpid(),
                start_ticks=Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')',1)[1].split()[19],
                gpu=args.gpu,split=args.split,clips=[],protocol_sha256=sha(BASE/'protocol.json'),
                source_sha256=sha(__file__),model_scope='frozen text encoder + video VAE only; no DiT',optimizer_updates=0)
    begin=time.perf_counter()
    def save():
        result['wall_seconds']=time.perf_counter()-begin
        t=dest/'encoding.tmp.json';t.write_text(json.dumps(result,indent=2)+'\n');t.replace(dest/'encoding.json')
    def cpu(x):
        if torch.is_tensor(x):return x.detach().cpu()
        if isinstance(x,dict):return {k:cpu(v) for k,v in x.items()}
        if isinstance(x,list):return [cpu(v) for v in x]
        return x
    save()
    try:
        config=dict(offload_dtype=torch.bfloat16,offload_device='cpu',onload_dtype=torch.bfloat16,onload_device='cpu',
                    preparing_dtype=torch.bfloat16,preparing_device='cuda:0',computation_dtype=torch.bfloat16,computation_device='cuda:0')
        pipe=MiniMaxH3Pipeline.from_pretrained(torch_dtype=torch.bfloat16,device='cuda:0',
            model_configs=[ModelConfig(model_id='MiniMax/MiniMax-H3',origin_file_pattern=pat,**config)
                           for pat in ('FL2VA/text_encoder/model*.safetensors','FL2VA/video_vae/source/model.safetensors')],
            processor_config=ModelConfig(model_id='MiniMax/MiniMax-H3',origin_file_pattern='FL2VA/processor/'),vram_limit=20.)
        assert pipe.dit is None and pipe.audio_vae is None
        pipe.video_vae.requires_grad_(False).eval();pipe.text_encoder.requires_grad_(False).eval()
        torch.cuda.reset_peak_memory_stats();result['status']='encoding';save()
        with torch.no_grad():
            for index,row in enumerate(clips):
                assert sha(row['path'])==row['sha256']
                with np.load(row['path'],allow_pickle=False) as source:
                    frames=[Image.fromarray(x) for x in source['rgb']]
                    keys=bounded_keys9(A.bin_to_latent(source['raw_actions'],37))
                    np.testing.assert_array_equal(keys,source['keys9'])
                assert len(frames)==124 and S.annotate_from_keys9(keys)==row['action_script']
                noise=MiniMaxH3Unit_NoiseInitializer().process(pipe,seed=13,num_frames=124,height=480,width=832,rand_device='cpu')
                pipe.load_models_to_device(['video_vae'])
                rgb=pipe.preprocess_video(frames,torch_dtype=torch.float32,min_value=0,device='cuda:0')
                def encode(x):
                    return pipe.video_vae.encode_video(x,dtype=pipe.torch_dtype,process_image=False,
                        tiled=True,tile_size=256,tile_overlap=64).to(torch.bfloat16)
                clean=encode(rgb);assert clean.shape==(1,24,37,30,52) and torch.isfinite(clean).all()
                checks=[];safe_prefixes={}
                for stop,cutoff in ((12,39),(24,81),(36,120)):
                    needed=math.ceil((stop+pipe.video_vae.token_drop)/pipe.video_vae.tokens_chunk_size)*pipe.video_vae.clip_length
                    known=rgb[:,:,:cutoff]
                    padded=torch.cat((known,known[:,:,-1:].repeat(1,1,max(0,needed-cutoff),1,1)),2)
                    safe=encode(padded)[:,:,:stop];del padded
                    assert safe.shape[2]==stop
                    error=float((safe.float()-clean[:,:,:stop].float()).abs().max())
                    assert error==0.,f'Full encoding is not prefix-safe: {stop}: {error}'
                    rec=dict(latent_stop=stop,RGB_stop=cutoff,observed_only_padded_frames=needed,safe_vs_full_max_abs=error)
                    safe_prefixes[stop]=safe.cpu()
                    if index==0:
                        changed=rgb.clone();changed[:,:,cutoff:]=1-changed[:,:,cutoff:]
                        future=encode(changed)
                        rec['future_RGB_inversion_max_abs']=float((future[:,:,:stop].float()-safe.float()).abs().max())
                        assert rec['future_RGB_inversion_max_abs']==0.
                        del changed,future
                    checks.append(rec);del safe
                first=MiniMaxH3Unit_KeyframeEncoder().process(pipe,keyframes=[frames[0]],keyframe_indices=[0],
                    video_latents=noise['video_latents'],rand_device='cpu',seed=13,height=480,width=832)['keyframe_cond_anchor']
                recon=None
                if index==0:
                    decoded=pipe.video_vae.decode_video(clean,dtype=pipe.torch_dtype,process_image=False,
                        tiled=True,tile_size=256,tile_overlap=64).float().clamp(0,1)
                    assert decoded.shape==rgb.shape
                    mse=float((decoded-rgb).square().mean());recon=dict(MSE=mse,PSNR_dB=-10*math.log10(max(mse,1e-20)))
                    sheet=Image.new('RGB',(4*416,2*264),'white');draw=ImageDraw.Draw(sheet)
                    for j,fid in enumerate((0,39,81,119)):
                        draw.text((j*416+5,5),f'GT {fid}',fill='black');sheet.paste(frames[fid].resize((416,240)),(j*416,24))
                        arr=(decoded[0,:,fid].permute(1,2,0).cpu().numpy()*255).round().astype(np.uint8)
                        draw.text((j*416+5,269),f'VAE {fid}',fill='black');sheet.paste(Image.fromarray(arr).resize((416,240)),(j*416,288))
                    sheet.save(dest/'vae_reconstruction.jpg',quality=92);del decoded
                del rgb
                emb=MiniMaxH3Unit_PromptEmbedder()
                positive=emb.process(pipe,prompt=row['prompt'],keyframes=[frames[0]],height=480,width=832,action_script=row['action_script'])
                def pack(prompt,spans):
                    return MiniMaxH3Unit_PackedSequenceBuilder()._build_packed_fl2va(len(prompt),37,30,52,
                        noise['audio_latents'].shape[-1],[0],action_text_spans=spans)
                full=pack(positive['prompt_embeds'],positive['action_text_spans'])
                reference=initial_action_text_reference(full)
                head=positive['prompt_embeds'][:positive['action_text_spans'][0][0]]
                fixed=fixed_action_position_origin(full,reference_text_length=reference)
                # Head/image encoded once; every clause is independently encoded.
                negatives={}
                for start,script in row['negative_scripts'].items():
                    pieces,spans=emb._encode_action_script(pipe,script,len(head),pipe.torch_dtype)
                    prompt=torch.cat((head,*pieces))
                    layout=fixed_action_position_origin(pack(prompt,spans),reference_text_length=reference)
                    for j,(a,b) in enumerate(spans):
                        if not int(start)<=j<int(start)+12:
                            lo,hi=positive['action_text_spans'][j]
                            assert torch.equal(prompt[a:b],positive['prompt_embeds'][lo:hi])
                    negatives[start]=dict(prompt_embeds=prompt,packed=layout)
                # A future-clause length intervention cannot change kept positions.
                position_checks=[]
                for stop in (12,24,36):
                    orig,t=visible_inputs(fixed,positive['prompt_embeds'],stop,390)
                    pieces=[head];spans=[];cursor=len(head)
                    for j,(lo,hi) in enumerate(positive['action_text_spans']):
                        piece=positive['prompt_embeds'][lo:hi]
                        if j>=stop:piece=piece.repeat(2,1)+1
                        pieces.append(piece);spans.append((cursor,cursor+len(piece)));cursor+=len(piece)
                    changed_prompt=torch.cat(pieces)
                    changed=fixed_action_position_origin(pack(changed_prompt,spans),reference_text_length=reference)
                    lp,pt=visible_inputs(changed,changed_prompt,stop,390)
                    assert torch.equal(pt,t)
                    assert torch.equal(orig['img_position_ids'],lp['img_position_ids'])
                    position_checks.append(dict(stop=stop,future_length_and_content_visible_input_identity=True))
                artifact=dict(format='h3world_native_single_I0_real_windows_v2',source=row,source_kind='real_ABot_episode',
                    clean_latents=clean,safe_prefixes=safe_prefixes,initial_noise=noise['video_latents'],audio_noise=noise['audio_latents'],
                    anchor=first,positive=dict(prompt_embeds=positive['prompt_embeds'],packed=fixed),negatives=negatives,
                    action_position_reference=reference,keys9=torch.from_numpy(keys),action_script=row['action_script'],
                    chunk_frames=12,history_chunks=5,history_protocol='N_fixed_sigma_noise_and_native_time',seed=13)
                file=dest/(row['clip_id']+'.pt');torch.save(cpu(artifact),file)
                record=dict(clip_id=row['clip_id'],split=args.split,encoded_path=str(file),sha256=sha(file),
                            clean_shape=list(clean.shape),anchor_shape=list(first.shape),audio_shape=list(noise['audio_latents'].shape),
                            prefix_checks=checks,position_checks=position_checks,reconstruction=recon,
                            action_position_reference=reference,negative_windows=sorted(negatives),
                            prompt_rows=len(positive['prompt_embeds']))
                result['clips'].append(record);save()
                print(json.dumps(dict(index=index+1,total=len(clips),clip=row['clip_id'],prefix_max=max(x['safe_vs_full_max_abs'] for x in checks))),flush=True)
                del artifact,positive,negatives,clean,safe_prefixes,noise,first,head,fixed
        result.update(status='complete',GPU_peak_MiB=torch.cuda.max_memory_allocated()/2**20)
    except BaseException as exc:
        result.update(status='failed',error=repr(exc));raise
    finally:save()


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--gpu',type=int,required=True);ap.add_argument('--split',choices=('train','validation'),required=True)
    args=ap.parse_args()
    free=int(subprocess.check_output(['nvidia-smi',f'--id={args.gpu}','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))
    if free<40000:raise RuntimeError('Need idle GPU with >=40000MiB free')
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
                      DIFFSYNTH_MODEL_BASE_PATH=str(RT/'DiffSynth-Studio-h3-v2/models'),PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    sys.path[:0]=[str(RT/'code'),str(RT/'code/abot'),str(RT/'DiffSynth-Studio-h3-v2')]
    main(args)
