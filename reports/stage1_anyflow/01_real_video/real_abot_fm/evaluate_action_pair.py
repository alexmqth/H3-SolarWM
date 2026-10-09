"""Pure A/D full39f response on one held-out first image, prompt and noise.

This is generated-history inference. The recorded real-video action sequence
is replaced explicitly; its GT latent history is never supplied to generation.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

BASE=Path(__file__).resolve().parent;RT=BASE/'runtime'
CLIP='118eb5d8b75e1b8ac23a4e9ae77af9a9_A_1140'

def pure_context(raw,action):
    import torch
    import action_script as S
    from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Unit_PackedSequenceBuilder
    from causal.h3_cached import expand_packed_two_anchors
    source=BASE/'counterfactual/action_sentences.pt'
    encoded=torch.load(source,map_location='cpu',weights_only=True)
    rows=encoded['rows'];assert rows[0].shape==rows[1].shape
    assert encoded['lines']==['the man strafes left, camera follows him','the man strafes right, camera follows him']
    cond=raw['conditioning'];headlen=int(cond['packed']['action_text_spans_local'][0][0]);head=cond['prompt_embeds'][:headlen]
    block=rows['AD'.index(action)];prompt=torch.cat([head]+[block]*12)
    spans=[(headlen+i*len(block),headlen+(i+1)*len(block)) for i in range(12)]
    packed=MiniMaxH3Unit_PackedSequenceBuilder()._build_packed_fl2va(len(prompt),12,30,52,65,[0],action_text_spans=spans)
    keys=torch.zeros(12,9,dtype=raw['action_cond'].dtype);keys[:,S.KEYS9.index(action)]=1
    assert torch.equal(prompt[:headlen],head) and torch.isfinite(prompt).all()
    state=dict(raw,conditioning=dict(cond,prompt_embeds=prompt,packed=packed),
        causal_packed=expand_packed_two_anchors(packed,frame_rows=390),action_cond=keys,
        action_script=[encoded['lines']['AD'.index(action)]]*12,
        intervention=f'PURE {action} current and future actions for complete generated trajectory; real GT history is not used',
        intervention_provenance=dict(action=action,action_embedding_file=str(source),
            action_embedding_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            first_image_and_scene_head_and_noises_preserved=True,real_recorded_actions_replaced=True))
    return state

def main(args):
    if args.history!='generated':raise ValueError('Pure full-action videos require generated history')
    args.clip=CLIP
    from evaluate_real_core import main as evaluate
    evaluate(args,lambda state:pure_context(state,args.action))

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--gpu',type=int,required=True)
    ap.add_argument('--action',choices=['A','D'],required=True)
    ap.add_argument('--mode',choices=['original','causal'],required=True)
    ap.add_argument('--history',choices=['generated'],default='generated')
    ap.add_argument('--steps',type=int,choices=[8,30],default=30)
    ap.add_argument('--checkpoint',type=Path);ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    free=int(subprocess.check_output(['nvidia-smi',f'--id={args.gpu}','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
    if free<34000:raise RuntimeError(f'Need34000MiB free, got{free}')
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu),ABOT_VRAM_RESERVE_GIB='18',HF_HUB_OFFLINE='1',
        TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    sys.path[:0]=[str(RT/'code'),str(RT/'code/abot'),str(RT/'code/causal'),str(RT/'DiffSynth-Studio-h3-v2')]
    main(args)
