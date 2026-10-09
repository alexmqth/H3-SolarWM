"""Freeze full37 Original reference inputs and measure causal display bounds.

Only the VAE is loaded on GPU. Future action/video rows are deleted from
DiT inputs; reference histories are Original-generated, never called GT.
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
ROOT = BASE.parents[3]
RT = BASE/'runtime'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(args):
    import torch
    from causal.local_topology import visible_inputs
    from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Pipeline, ModelConfig
    torch.set_num_threads(4); torch.manual_seed(13)
    protocol = json.loads((BASE/'protocol.json').read_text())
    for row in protocol['sources'].values():
        for entry in row.values():
            assert sha(ROOT/entry['path']) == entry['sha256']
    old = ROOT/'H3-World/outputs/2026-10-09-04/stage1_local_topology'
    assert json.loads((old/'native_single_positive_review.json').read_text())['status'] == 'review_complete_no_go'
    assert json.loads((old/'native_single_anchor12_review.json').read_text())['window12_condition_usable']
    for rel, expected in json.loads((BASE/'runtime_manifest.json').read_text()).items():
        assert sha(RT/rel) == expected
    out = BASE/'inputs'; out.mkdir(exist_ok=False)
    ticks = Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')',1)[1].split()[19]
    result = dict(status='preparing', pid=os.getpid(), start_ticks=ticks, gpu=args.gpu,
        at=datetime.now().astimezone().isoformat(), scope=__doc__, source_sha256=sha(__file__),
        protocol_sha256=sha(BASE/'protocol.json'), files={}, references={}, layout_checks=[],
        decoder_bounds=[], optimizer_updates=0, history_kind='Original_generated_reference_not_GT')
    start_time = time.perf_counter()
    def save():
        result['wall_seconds'] = time.perf_counter()-start_time
        tmp=out/'preparation.tmp.json'; tmp.write_text(json.dumps(result,indent=2)+'\n')
        tmp.replace(out/'preparation.json')
    def th(x):
        x=x.detach().cpu().contiguous()
        return hashlib.sha256(str((tuple(x.shape),str(x.dtype))).encode()+x.view(torch.uint8).numpy().tobytes()).hexdigest()
    save()
    try:
        cond = {a:torch.load(ROOT/protocol['sources'][a]['conditioning.pt']['path'],
                            map_location='cpu',weights_only=True) for a in 'AD'}
        for field in ('initial_noise','audio_noise','anchor'):
            assert torch.equal(cond['A'][field],cond['D'][field]),field
        for key,value in cond['A']['packed'].items():
            other=cond['D']['packed'][key]
            assert torch.equal(value,other) if torch.is_tensor(value) else value==other,key
        spans=cond['A']['packed']['action_text_spans_local']
        assert len(spans)==37 and set(hi-lo for lo,hi in spans)=={10}
        assert torch.equal(cond['A']['prompt_embeds'][:spans[0][0]],cond['D']['prompt_embeds'][:spans[0][0]])
        assert cond['A']['initial_noise'].shape==(1,24,37,30,52)
        assert cond['A']['anchor'].shape==(390,96)
        prepared={}
        for ref in 'AD':
            z=torch.load(ROOT/protocol['sources'][ref]['baseline_latents.pt']['path'],map_location='cpu',weights_only=True)
            assert z.shape==cond[ref]['initial_noise'].shape and torch.isfinite(z).all()
            pairs=[]
            for i,(start,stop) in enumerate(((0,12),(12,24),(24,36))):
                prompts={}
                allowed=torch.zeros(len(cond[ref]['prompt_embeds']),dtype=torch.bool)
                for lo,hi in spans[start:stop]: allowed[lo:hi]=True
                for action in 'AD':
                    prompt=cond[ref]['prompt_embeds'].clone()
                    for lo,hi in spans[start:stop]: prompt[lo:hi]=cond[action]['prompt_embeds'][lo:hi]
                    prompts[action]=prompt
                    packed,text=visible_inputs(cond[ref]['packed'],prompt,stop,390)
                    future=prompt.clone()
                    for lo,hi in spans[stop:]: future[lo:hi]+=7
                    changed,other=visible_inputs(cond[ref]['packed'],future,stop,390)
                    assert torch.equal(text,other)
                    assert packed.keys()==changed.keys()
                    for k,v in packed.items():
                        assert torch.equal(v,changed[k]) if torch.is_tensor(v) else v==changed[k]
                    assert len(packed['action_text_rows'])==stop
                    assert packed['seq_len']==packed['action_video_start']+stop*390
                    assert torch.equal(packed['img_position_ids'][0,packed['img_pos'][:390]],
                        cond[ref]['packed']['img_position_ids'][0,cond[ref]['packed']['img_pos'][:390]])
                    result['layout_checks'].append(dict(reference=ref,window=i,action=action,
                        visible_latents=stop,seq_len=packed['seq_len'],text_rows=len(text),
                        future_content_effect=0,anchor_positions_unchanged=True,
                        prompt_sha256=th(text),layout_positions_sha256=th(packed['img_position_ids'])))
                assert torch.equal(prompts['A'][~allowed],prompts['D'][~allowed])
                assert not torch.equal(prompts['A'][allowed],prompts['D'][allowed])
                pairs.append(dict(index=i,start=start,stop=stop,prompts=prompts))
            prepared[ref]=dict(format='e1_coarse12_full37_v1',history_reference=ref,
                history_kind=result['history_kind'],reference_latents=z,initial_noise=cond[ref]['initial_noise'],
                audio_noise=cond[ref]['audio_noise'],anchor=cond[ref]['anchor'],packed=cond[ref]['packed'],pairs=pairs)
            result['references'][ref]=dict(latents_sha256=th(z),noise_sha256=th(cond[ref]['initial_noise']),
                audio_sha256=th(cond[ref]['audio_noise']),anchor_sha256=th(cond[ref]['anchor']))
        save()
        vram=dict(offload_dtype=torch.bfloat16,offload_device='cpu',onload_dtype=torch.bfloat16,
            onload_device='cpu',preparing_dtype=torch.bfloat16,preparing_device='cuda:0',
            computation_dtype=torch.bfloat16,computation_device='cuda:0')
        pipe=MiniMaxH3Pipeline.from_pretrained(torch_dtype=torch.bfloat16,device='cuda:0',
            model_configs=[ModelConfig(model_id='MiniMax/MiniMax-H3',
                origin_file_pattern='FL2VA/video_vae/source/model.safetensors',**vram)],vram_limit=32.)
        assert pipe.dit is None and pipe.text_encoder is None
        pipe.load_models_to_device(['video_vae'])
        pipe.video_vae.requires_grad_(False).eval(); torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            for ref,data in prepared.items():
                bounds=[0]
                for stop in (12,24,36):
                    z=data['reference_latents'][:,:,:stop].to('cuda:0')
                    rgb=pipe.video_vae.decode_video(z,dtype=torch.bfloat16,process_image=False,
                        tiled=True,tile_size=256,tile_overlap=64)
                    assert torch.isfinite(rgb).all() and rgb.shape[2]>bounds[-1]
                    bounds.append(int(rgb.shape[2]))
                    result['decoder_bounds'].append(dict(reference=ref,latent_stop=stop,
                        decoded_rgb_frames=int(rgb.shape[2]),prefix_rgb_sha256=th(rgb),
                        scope='Actual visible-prefix decode, no later latent passed to VAE'))
                    del rgb,z
                    save();print(f'[coarse12 prep] {ref} latent{stop} -> RGB{bounds[-1]}',flush=True)
                data['rgb_bounds']=bounds
                target=out/f'parking_{ref}.pt';torch.save(data,target)
                result['files'][target.name]=sha(target)
        assert prepared['A']['rgb_bounds']==prepared['D']['rgb_bounds']
        result.update(status='complete',rgb_bounds=prepared['A']['rgb_bounds'],
            GPU_peak_MiB=torch.cuda.max_memory_allocated()/2**20,
            decoder_policy='Decode available prefix at each window; append only new RGB; never revise emitted frames',
            future_action_video_policy='Physical visibility trimming on fixed action-content-independent full37 positions',
            caveat='Audio noise/layout fixed to full37 reference; this fixture differs from previous39f in RNG and global positions.')
    except BaseException as e:
        result.update(status='failed',error=repr(e));raise
    finally:save()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--gpu',type=int,required=True);a=p.parse_args()
    free=int(subprocess.check_output(['nvidia-smi',f'--id={a.gpu}','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))
    if free<34000:raise RuntimeError('Need idle GPU with>=34000MiB free')
    os.environ.update(CUDA_VISIBLE_DEVICES=str(a.gpu),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        DIFFSYNTH_SKIP_DOWNLOAD='True',DIFFSYNTH_MODEL_BASE_PATH=str(RT/'DiffSynth-Studio-h3-v2/models'),
        PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    sys.path[:0]=[str(RT/'code'),str(RT/'DiffSynth-Studio-h3-v2')]
    main(a)
