"""Encode only two independent action sentences; reuse all saved real context.

A/D get exactly equal layouts. Outside the CURRENT chunk, each original
action embedding is reused bit-for-bit. Historical KV must be rebuilt for
this diagnostic layout; never import KV from a different prompt layout.
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
MANIFEST=BASE.parents[2]/'data/abot_bridge/encoded_manifest.json'
CLIPS=['118eb5d8b75e1b8ac23a4e9ae77af9a9_A_1140','dfec8ed3237860eba14d67c089ecd041_D_1750']
LINES=['the man strafes left, camera follows him','the man strafes right, camera follows him']

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def assemble(state,action_rows,chunk):
    import torch
    from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Unit_PackedSequenceBuilder
    from causal.h3_cached import expand_packed_two_anchors
    spans=state['conditioning']['packed']['action_text_spans_local']
    original=state['conditioning']['prompt_embeds'];head=original[:int(spans[0][0])]
    lo,hi=chunk*5,min(chunk*5+5,12)
    prompts={};scripts={};layouts={}
    for act,row in zip('AD',action_rows):
        pieces=[head];new_spans=[];cursor=len(head)
        script=list(state['action_script'])
        for frame,(s,e) in enumerate(spans):
            part=row if lo<=frame<hi else original[int(s):int(e)]
            if lo<=frame<hi:script[frame]=LINES['AD'.index(act)]
            new_spans.append((cursor,cursor+len(part)));cursor+=len(part);pieces.append(part)
        prompt=torch.cat(pieces)
        packed=MiniMaxH3Unit_PackedSequenceBuilder()._build_packed_fl2va(
            len(prompt),12,30,52,65,[0],action_text_spans=new_spans)
        layouts[act]=expand_packed_two_anchors(packed,frame_rows=390)
        prompts[act]=prompt;scripts[act]=script
        assert torch.equal(prompt[:len(head)],head)
        for frame,(s,e) in enumerate(new_spans):
            if not lo<=frame<hi:
                old_s,old_e=spans[frame]
                assert torch.equal(prompt[s:e],original[int(old_s):int(old_e)])
    assert layouts['A'].keys()==layouts['D'].keys()
    for name,left in layouts['A'].items():
        right=layouts['D'][name]
        assert torch.equal(left,right) if torch.is_tensor(left) else left==right,name
    allowed=torch.zeros(len(prompts['A']),dtype=torch.bool)
    for s,e in layouts['A']['action_text_spans_local'][lo:hi]:allowed[int(s):int(e)]=True
    changed=(prompts['A']!=prompts['D']).any(dim=-1)
    assert not changed[~allowed].any() and changed[allowed].any()
    return dict(chunk=chunk,start=lo,stop=hi,prompts=prompts,packed=layouts['A'],scripts=scripts,
        changed_rows=int(changed.sum()),outside_current_bitwise_equal=True,
        historical_embeddings_original_values=True,paired_layout_exact=True,
        layout_note='Current action sentence length may differ from natural recording; A/D layouts equal. Rebuild KV using this layout.')

def main(args):
    import torch
    from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Pipeline,ModelConfig,MiniMaxH3Unit_PromptEmbedder
    torch.set_num_threads(4);torch.manual_seed(13)
    out=BASE/'counterfactual';out.mkdir(exist_ok=False)
    receipt=dict(status='loading',pid=os.getpid(),gpu=args.gpu,started_at=datetime.now().astimezone().isoformat(),
        source_manifest_sha256=sha(MANIFEST),model_scope='text encoder only; no DiT/VAE',
        intervention='only current chunk pure strafe A/D; history real combined actions unchanged',records=[])
    tick=time.perf_counter()
    def save():
        receipt['wall_seconds']=time.perf_counter()-tick
        tmp=out/'preparation.tmp.json';tmp.write_text(json.dumps(receipt,indent=2)+'\n');tmp.replace(out/'preparation.json')
    save()
    try:
        config=dict(offload_dtype=torch.bfloat16,offload_device='cpu',onload_dtype=torch.bfloat16,
            onload_device='cpu',preparing_dtype=torch.bfloat16,preparing_device='cuda:0',
            computation_dtype=torch.bfloat16,computation_device='cuda:0')
        pipe=MiniMaxH3Pipeline.from_pretrained(torch_dtype=torch.bfloat16,device='cuda:0',
            model_configs=[ModelConfig(model_id='MiniMax/MiniMax-H3',origin_file_pattern='FL2VA/text_encoder/model*.safetensors',**config)],
            processor_config=ModelConfig(model_id='MiniMax/MiniMax-H3',origin_file_pattern='FL2VA/processor/'),vram_limit=16.)
        assert pipe.dit is None and pipe.video_vae is None
        pipe.load_models_to_device(['text_encoder'])
        with torch.no_grad():rows,_=MiniMaxH3Unit_PromptEmbedder()._encode_action_script(pipe,LINES,0,torch.bfloat16)
        rows=[r.cpu() for r in rows];assert rows[0].shape==rows[1].shape
        assert all(torch.isfinite(r).all() for r in rows)
        torch.save(dict(lines=LINES,rows=rows),out/'action_sentences.pt')
        manifest=json.loads(MANIFEST.read_text())
        for clip in CLIPS:
            source=next(r for r in manifest['clips'] if r['clip_id']==clip)
            assert source['split']=='validation' and sha(source['encoded_file'])==source['sha256']
            state=torch.load(source['encoded_file'],map_location='cpu',weights_only=True)
            for chunk in (0,1,2):
                pair=assemble(state,rows,chunk)
                pair.update(format='h3world_real_current_action_counterfactual_v1',clip=clip,source_sha256=source['sha256'])
                path=out/f'{clip}_chunk{chunk}.pt';torch.save(pair,path)
                receipt['records'].append(dict(clip=clip,chunk=chunk,file=str(path),sha256=sha(path),
                    source_sha256=source['sha256'],paired_layout_exact=True,outside_current_bitwise_equal=True,
                    historical_embeddings_original_values=True,changed_rows=pair['changed_rows']))
                save()
        receipt.update(status='complete',action_sentence_rows=len(rows[0]),gpu_allocated_peak_MiB=torch.cuda.max_memory_allocated()/2**20)
    except BaseException as exc:receipt.update(status='failed',error=repr(exc));raise
    finally:save()

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--gpu',type=int,required=True);args=ap.parse_args()
    free=int(subprocess.check_output(['nvidia-smi',f'--id={args.gpu}','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
    if free<24000:raise RuntimeError(f'Need24000MiB free, got{free}')
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu),HF_HUB_OFFLINE='1',DIFFSYNTH_SKIP_DOWNLOAD='True',
        DIFFSYNTH_MODEL_BASE_PATH=str(RT/'DiffSynth-Studio-h3-v2/models'),TOKENIZERS_PARALLELISM='false')
    sys.path[:0]=[str(RT/'code'),str(RT/'DiffSynth-Studio-h3-v2')]
    main(args)
