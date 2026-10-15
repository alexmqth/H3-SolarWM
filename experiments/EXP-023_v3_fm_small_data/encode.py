"""EXP-023 source-RGB VAE/text encoding; pilot and rest are separate ledgers."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image

from common import (HERE, ROOT, FROZEN, ENCODED, PhaseLedger,
                    inputs, new_rows, raw_source_rgb, sha, write)


def setup():
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:4])
    os.environ.update(DIFFSYNTH_MODEL_BASE_PATH=str(FROZEN/"DiffSynth-Studio-h3-v2/models"),
                      DIFFSYNTH_SKIP_DOWNLOAD="True",TOKENIZERS_PARALLELISM="false")
    sys.path[:0]=[str(FROZEN/"DiffSynth-Studio-h3-v2"),str(FROZEN/"code"),
                  str(FROZEN/"code/abot"),str(FROZEN/"code/causal"),
                  str(ROOT/"submission/experiments/EXP-017_v3_fixed_action_layout")]


def save_tensor(path, value):
    import torch
    if path.exists():
        raise FileExistsError(path)
    tmp=path.with_suffix(path.suffix+".partial")
    torch.save(value.detach().cpu(),tmp)
    tmp.rename(path)
    return {"path":str(path.relative_to(ROOT)),"sha256":sha(path),
            "shape":list(value.shape),"dtype":str(value.dtype),"bytes":path.stat().st_size}


def process_one(pipe, ledger, row, *, pilot, caption_cache, anchor_noise, audio_noise, initial_noise, builder, unit, canonical_a):
    import torch
    import abot_action as A
    from diffsynth.pipelines.minimax_h3_audio_video import patchify_video
    from fixed_layout import candidate,action_ids,same_layout
    from causal.local_topology import visible_inputs

    clip=row["clip_id"]
    dest=ENCODED/clip
    if dest.exists():
        raise FileExistsError(f"encoded output exists: {dest}")
    dest.mkdir(parents=True)
    rgb=ledger.measure(f"{clip}_source_rgb",lambda: raw_source_rgb(row))
    assert rgb.shape==(56,480,832,3)
    source_rgb_sha=hashlib.sha256(rgb.tobytes()).hexdigest()
    frames=[Image.fromarray(x) for x in rgb]
    if row["episode_id"] not in caption_cache:
        ep=A.read_episode(str(ROOT/row["source_annotations"]))
        caption_cache[row["episode_id"]]=ep["caption"]["scene_static"]
    caption=caption_cache[row["episode_id"]]
    assert hashlib.sha256(caption.encode()).hexdigest()==row["scene_static_sha256"]
    image=pipe.preprocess_image(frames[0],torch_dtype=torch.float32,device="cuda:0",min_value=0,max_value=1)
    video=pipe.preprocess_video(frames,torch_dtype=torch.float32,device="cuda:0",min_value=0,max_value=1)
    assert torch.equal(image,video[:,:,0])
    pipe.load_models_to_device(["video_vae"])
    def vae_encode(tensor, name, process_image):
        ledger.reserve("vae_encode",f"{clip} {name}")
        with torch.no_grad():
            return ledger.measure(f"{clip}_{name}",lambda: pipe.video_vae.encode_video(
                tensor,dtype=torch.bfloat16,process_image=process_image,
                tile_size=256,tile_overlap=64))
    zi=vae_encode(image,"I0",True)
    zfull=vae_encode(video,"full17",False)
    zprefix=vae_encode(video[:,:,:39].contiguous(),"prefix12",False) if pilot else None
    assert zi.shape==(1,24,1,30,52) and zfull.shape==(1,24,17,30,52)
    assert torch.isfinite(zi).all() and torch.isfinite(zfull).all()
    if zprefix is not None:
        assert zprefix.shape==(1,24,12,30,52) and torch.equal(zfull[:,:,:12],zprefix)
    latents={"I0":save_tensor(dest/"I0.pt",zi),"full17":save_tensor(dest/"full17.pt",zfull)}
    if zprefix is not None:
        latents["prefix12"]=save_tensor(dest/"prefix12.pt",zprefix)
    nr=patchify_video(anchor_noise[:,:,:1].to("cuda:0"))
    ci=patchify_video(zi).to("cuda:0",torch.bfloat16)
    ts=torch.tensor(.999,device="cuda:0",dtype=torch.bfloat16)
    anchor=(ts*ci+(1-ts)*nr).contiguous().cpu()
    del image,video,zi,zfull,zprefix,ci,nr
    gc.collect();torch.cuda.empty_cache()

    # The text head is tied to this exact I0. Actual actions are independently
    # encoded, then placed on a canonical full37 semantic grid fixed by A.
    lines=list(row["action_script17"])+[canonical_a]*20
    with torch.no_grad():
        result=ledger.measure(f"{clip}_text",lambda: unit.process(
            pipe,caption,keyframes=[frames[0]],height=480,width=832,action_script=lines))
    prompt=result["prompt_embeds"].detach().cpu()
    spans=result["action_text_spans"]
    head=int(spans[0][0]);alen=len(action_ids(pipe.tokenizer,[canonical_a])[0])
    canonical_spans=[(head+i*alen,head+(i+1)*alen) for i in range(37)]
    canonical=builder._build_packed_fl2va(head+37*alen,37,30,52,207,[0],
                                            action_text_spans=canonical_spans)
    canonical["token_tags"][:head]=result["text_token_tags"][:head].cpu()
    packed,_=candidate(builder,canonical,action_ids(pipe.tokenizer,lines))
    assert packed["action_text_spans_local"]==spans
    assert torch.equal(result["text_token_tags"].cpu(),packed["token_tags"][packed["text_pos"]])
    assert len(prompt)==len(packed["text_pos"])
    seen={}
    for stop in (12,17):
        p,pr=visible_inputs(packed,prompt,stop,390)
        assert len(p["action_text_rows"])==stop
        assert int(p["seq_len"])==int(p["action_video_start"])+stop*390
        future=lines[:stop]+["A"]*(37-stop)
        alt,_=candidate(builder,canonical,action_ids(pipe.tokenizer,future))
        alt_prompt=torch.cat([prompt[:spans[stop-1][1]],
            torch.ones((len(alt["text_pos"])-spans[stop-1][1],prompt.shape[1]),dtype=prompt.dtype)])
        pa,pr_alt=visible_inputs(alt,alt_prompt,stop,390)
        assert same_layout(p,pa) and torch.equal(pr,pr_alt)
        seen[stop]=True
    fixture={"format":"exp023_source_rgb_v1","clip_id":clip,"episode_id":row["episode_id"],
             "packed":packed,"prompt":prompt,"anchor":anchor,
             "audio_noise":audio_noise,"initial_noise":initial_noise,
             "script":row["action_script17"],"caption":caption,
             "source_rgb_sha256":source_rgb_sha,"source_video_sha256":row["source_video_sha256"],
             "action_script_sha256":row["action_script_sha256"],
             "conditioning_contract":"source RGB / Single I0 / native full37 semantic layout / visible 12+5"}
    condition_path=dest/"condition.pt"
    if condition_path.exists():raise FileExistsError(condition_path)
    torch.save(fixture,condition_path)
    record={"clip_id":clip,"split":row["split"],"episode_id":row["episode_id"],
            "category":row["category"],"source_start":row["source_start"],
            "source_video_sha256":row["source_video_sha256"],
            "source_rgb_sha256":source_rgb_sha,"source_caption_sha256":row["scene_static_sha256"],
            "action_script_sha256":row["action_script_sha256"],
            "latents":latents,"condition":{"path":str(condition_path.relative_to(ROOT)),
              "sha256":sha(condition_path),"bytes":condition_path.stat().st_size},
            "future_action_isolation":seen,"pilot_prefix_exact":True if pilot else None,
            "source_rgb_direct":True,"old_h264_latents_reused":False,
            "prompt_rows":len(prompt),"head_rows":head,
            "native_action_spans":len(spans)}
    write(dest/"record.json",record)
    print("ENCODED",clip,"pilot",pilot,flush=True)
    del result,prompt,packed,fixture,frames,rgb
    gc.collect();torch.cuda.empty_cache()
    return record


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--stage",choices=("pilot","rest"),required=True)
    parser.add_argument("--gpu",type=int,required=True)
    args=parser.parse_args()
    setup()
    import torch
    from diffsynth.pipelines.minimax_h3_audio_video import (MiniMaxH3Pipeline,ModelConfig,
        MiniMaxH3Unit_PromptEmbedder)
    from fixed_layout import frozen_cpu_functions
    if os.environ.get("CUDA_VISIBLE_DEVICES")!=str(args.gpu):
        raise RuntimeError("GPU selector mismatch")
    if not (HERE/"CPU_PREFLIGHT.json").is_file():
        raise RuntimeError("CPU preflight required")
    torch.set_num_threads(4);torch.cuda.set_device(0);torch.cuda.reset_peak_memory_stats()
    rows=new_rows()[0 if args.stage=="pilot" else 1]
    if args.stage=="rest":
        previous=json.loads((HERE/"ENCODE_PILOT_BUDGET.json").read_text())
        if previous["status"]!="COMPLETE_PENDING_JUDGE":raise RuntimeError("pilot not complete")
        max_seconds=4200-previous["wall_seconds"]
    else:max_seconds=900
    ledger=PhaseLedger("encode_"+args.stage,args.gpu,max_seconds)
    try:
        dtype=torch.bfloat16
        cfg=dict(offload_dtype=dtype,offload_device="cpu",onload_dtype=dtype,onload_device="cpu",
                 preparing_dtype=dtype,preparing_device="cuda:0",computation_dtype=dtype,computation_device="cuda:0")
        pipe=ledger.measure("load_text_vae",lambda: MiniMaxH3Pipeline.from_pretrained(
            torch_dtype=dtype,device="cuda:0",model_configs=[
                ModelConfig(model_id="MiniMax/MiniMax-H3",origin_file_pattern="FL2VA/text_encoder/model*.safetensors",**cfg),
                ModelConfig(model_id="MiniMax/MiniMax-H3",origin_file_pattern="FL2VA/video_vae/source/model.safetensors",**cfg)],
            processor_config=ModelConfig(model_id="MiniMax/MiniMax-H3",origin_file_pattern="FL2VA/processor/"),vram_limit=27.))
        if pipe.dit is not None or pipe.text_encoder is None or pipe.video_vae is None:
            raise RuntimeError("encode process must load text and VAE only")
        hook=pipe.text_encoder.register_forward_pre_hook(lambda *a:ledger.reserve("text","text encoder forward"))
        baseline=torch.load(ROOT/"H3-World/outputs/EXP-019_v3_mixed_action_validation/E/b784d995.pt",
                            map_location="cpu",weights_only=True)
        canonical_a="the man strafes left, camera follows him"
        builder,_=frozen_cpu_functions();unit=MiniMaxH3Unit_PromptEmbedder()
        records=[];caption_cache={}
        for row in rows:
            records.append(process_one(pipe,ledger,row,pilot=args.stage=="pilot",
                caption_cache=caption_cache,anchor_noise=baseline["initial_noise"],
                audio_noise=baseline["audio_noise"],initial_noise=baseline["initial_noise"],
                builder=builder,unit=unit,canonical_a=canonical_a))
            ledger.save(last_clip=row["clip_id"])
        hook.remove()
        ledger.close("COMPLETE_PENDING_JUDGE")
        write(HERE/f"ENCODE_{args.stage.upper()}_RESULT.json",{
            "status":"COMPLETE_PENDING_JUDGE","stage":args.stage,
            "clips":[r["clip_id"] for r in records],"counts":ledger.counts,
            "wall_seconds":time.monotonic()-ledger.started,
            "source_rgb_direct":True,"new_model_train_calls":0})
        print("ENCODE COMPLETE",args.stage,len(records),ledger.counts,flush=True)
    except BaseException as exc:
        ledger.close("FAILED_STOPPED",repr(exc))
        print("ENCODE STOPPED",repr(exc),flush=True)
        raise


if __name__=="__main__":main()
