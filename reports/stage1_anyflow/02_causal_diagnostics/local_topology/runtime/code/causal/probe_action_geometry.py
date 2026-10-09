#!/usr/bin/env python3
"""Compare frozen causal and original H3 A/D score geometry on one shared state."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code/abot')); sys.path.insert(0,str(ROOT/'code'))
import infer as abot
from causal.h3_cached import H3ChunkCache, chunk_forward, expand_packed_two_anchors
from causal.pretrained_lora import load_adapter
from causal.train_online_selfrollout import action_condition, full_teacher_forward, move_tree

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--generated-dir',type=Path,required=True)
    ap.add_argument('--conditioning',type=Path,nargs=2,required=True); ap.add_argument('--causal-adapter',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True); ap.add_argument('--device',default='cuda:0'); ap.add_argument('--sigma',type=float,default=.6); ap.add_argument('--chunk-frames',type=int,default=5)
    a=ap.parse_args(); a.out.parent.mkdir(parents=True,exist_ok=True); torch.set_num_threads(4)
    generated=torch.load(a.generated_dir/'cached_latents.pt',map_location='cpu',weights_only=True)
    cond=[move_tree(torch.load(p/'conditioning.pt',map_location='cpu',weights_only=True),a.device) for p in a.conditioning]
    pipe=abot.load_pipeline(a.device); pipe.load_lora(pipe.dit,state_dict=abot.load_checkpoint_lora(ROOT/'checkpoints/H3-World/step-10000.safetensors'),hotload=True); pipe.dit.requires_grad_(False).eval(); load_adapter(pipe.dit,a.causal_adapter,a.device); pipe.load_models_to_device(['dit'])
    generated=generated.to(a.device); rows=(generated.shape[-2]//2)*(generated.shape[-1]//2)
    dual=[expand_packed_two_anchors(c['packed'],frame_rows=rows) for c in cond]; actions=[action_condition(x,generated.shape[2],a.device,generated.dtype) for x in ('A','D')]
    anchor_dual=torch.cat((cond[0]['anchor'],cond[0]['anchor'].clone()),dim=0)
    history=generated[:,:,:a.chunk_frames].contiguous(); current=generated[:,:,a.chunk_frames:a.chunk_frames*2].contiguous(); cache=H3ChunkCache(5,'cpu')
    with torch.no_grad():
      # Commit the same generated A history used by both counterfactual branches.
      chunk_forward(pipe.dit,history,index=0,cache=cache,sigma=0.,commit=True,full_packed=dual[0],prompt=cond[0]['prompt_embeds'],anchor=anchor_dual,audio=cond[0]['audio_noise'],chunk_frames=a.chunk_frames,action_cond=actions[0][:a.chunk_frames],action_prefix_mode='causal',action_feedback=True)
      causal=[]
      for i in range(2):
        causal.append(chunk_forward(pipe.dit,current,index=1,cache=cache,sigma=a.sigma,commit=False,full_packed=dual[i],prompt=cond[i]['prompt_embeds'],anchor=anchor_dual,audio=cond[i]['audio_noise'],chunk_frames=a.chunk_frames,action_cond=actions[i][a.chunk_frames:a.chunk_frames*2],action_prefix_mode='causal',action_feedback=True).float())
      teacher=[]
      for i in range(2):
        teacher.append(full_teacher_forward(pipe.dit,current,history=history,full_packed=cond[i]['packed'],prompt=cond[i]['prompt_embeds'],anchor=cond[i]['anchor'],audio=cond[i]['audio_noise'],sigma=a.sigma,action_cond=actions[i]).float())
    dc=(causal[0]-causal[1]).flatten(1); dt=(teacher[0]-teacher[1]).flatten(1)
    result={'status':'complete','sigma':a.sigma,'state':'generated A history chunk 0 -> chunk 1','causal_delta_norm':float(torch.linalg.vector_norm(dc,dim=1).mean()),'teacher_delta_norm':float(torch.linalg.vector_norm(dt,dim=1).mean()),'causal_over_teacher_norm_ratio':float(torch.linalg.vector_norm(dc,dim=1).mean()/(torch.linalg.vector_norm(dt,dim=1).mean()+1e-8)),'causal_teacher_delta_cosine':float(F.cosine_similarity(dc,dt,dim=1).mean()),'causal_A_velocity_norms':[float(torch.linalg.vector_norm(x.flatten(1),dim=1).mean()) for x in causal],'teacher_A_velocity_norms':[float(torch.linalg.vector_norm(x.flatten(1),dim=1).mean()) for x in teacher]}
    a.out.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2))
if __name__=='__main__': main()
