"""Prepared matched FM4 versus AF4 continuation runner. GPU needs Judge release.

C1 is EXP-006's FM8 endpoint and published RGB. Each candidate builds its own
C1 KV and then branches AA/AD through C2 and C3. FM4 uses native ordinary
velocity/scheduler updates; AF4 targets the next native sigma each map.
"""
from __future__ import annotations

from datetime import datetime
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
CFG=json.loads((HERE/"config.json").read_text())
OUT=ROOT/CFG["output_root"]
AF2=ROOT/"H3-World/outputs/EXP-007_v3_anyflow_af2"
FM8=ROOT/"H3-World/outputs/EXP-006_v3_fm8_full"
FROZEN=ROOT/"H3-World/outputs/2026-10-09-22/chunk_partition_cb"
EXP005=HERE.parent/"EXP-005_v3_sliding_window"
ROUTER=ROOT/"submission/reports/stage1_anyflow/02_causal_diagnostics/current_prefix_candidate"
CONTRACT=HERE.parent/"EXP-001_v2b_124"
EXP002=HERE.parent/"EXP-002_native_cached"
EXP006=HERE.parent/"EXP-006_v3_fm8_full"
MANIFEST=HERE/"source_manifest.json"
AFEXP=HERE.parent/"EXP-007_v3_anyflow"


def sha(path:Path|str)->str:
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(4<<20),b""):h.update(block)
    return h.hexdigest()


def atomic_json(path:Path,value)->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=Path(str(path)+".partial")
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n")
    os.replace(temp,path)


def paths()->None:
    sys.path[:0]=[str(FROZEN/"runtime/code"),str(FROZEN/"runtime/code/abot"),
                 str(FROZEN/"runtime/code/causal"),
                 str(FROZEN/"runtime/DiffSynth-Studio-h3-v2"),str(EXP005),
                 str(ROUTER),str(CONTRACT),str(EXP002),str(EXP006),str(AFEXP),str(HERE)]


def selected_checkpoint():
    if CFG["checkpoint"]!="EXP-007/v3-AF2-step32":raise ValueError("EXP009 must select frozen AF2 step32")
    result_path=AF2/"result.json"
    if not result_path.exists():return None
    result=json.loads(result_path.read_text())
    if result["status"] not in ("complete_32_pending_judge","budget_stopped_pending_judge"):
        return None
    last=result.get("last_complete_step")
    if last!=32:raise RuntimeError("EXP009 AF2 checkpoint must be step32")
    candidates=[row for row in result["checkpoint_paths"] if row["step"]==last]
    if len(candidates)!=1:raise RuntimeError("AF2 final complete step has no unique checkpoint")
    ckpt=ROOT/candidates[0]["path"]
    if not ckpt.is_dir():raise FileNotFoundError(ckpt)
    return result,ckpt,candidates[0]


def source_manifest(create:bool)->dict|None:
    selected=selected_checkpoint()
    if selected is None:return None
    result,checkpoint,receipt=selected
    sources={
        "runner":HERE/"run_fm4_af4.py","config":HERE/"config.json",
        "taskbook":HERE/"taskbook_v1.md",
        "student_entry":AFEXP/"interval_student.py",
        "ordinary_entry":EXP002/"interval_cached.py",
        "chunk_plan":EXP005/"chunk_plan.py",
        "position_sw":EXP005/"position_sw.py",
        "fm8_runner":EXP006/"run_fm8.py",
        "scheduler":FROZEN/"runtime/DiffSynth-Studio-h3-v2/diffsynth/diffusion/flow_match.py",
        "anyflow_sampling":FROZEN/"runtime/code/causal/anyflow_sampling.py",
        "cached_attention":FROZEN/"runtime/code/causal/h3_cached.py",
        "local_topology":FROZEN/"runtime/code/causal/local_topology.py",
        "precision":FROZEN/"runtime/code/causal/h3_precision.py",
        "qkv_adapter":FROZEN/"runtime/code/causal/pretrained_lora.py",
        "benchmark_writer":FROZEN/"runtime/code/causal/benchmark.py",
        "action_flow":FROZEN/"runtime/code/causal/evaluate_action_control.py",
        "rollout_contract":CONTRACT/"rollout_contract.py",
        "af2_result":AF2/"result.json","af2_qkv":checkpoint/"qkv.pt",
        "af2_target":checkpoint/"target_time.pt","af2_state":checkpoint/"trainer_state.pt",
        "fm8_first":FM8/"first12.pt","fm8_rgb39":FM8/"published_39.npy",
        "input_A":FROZEN/"source_coarse/inputs/parking_A.pt",
        "input_D":FROZEN/"source_coarse/inputs/parking_D.pt",
        "released_lora":FROZEN/"runtime/checkpoints/H3-World/step-10000.safetensors",
        "anyflow":FROZEN/"runtime/code/causal/anyflow.py",
        "prefix_feedback":ROUTER/"current_prefix.py",
    }
    entries={name:{"path":str(path.relative_to(ROOT)),"sha256":sha(path)}
             for name,path in sources.items()}
    for label,filename in (("af2_qkv","qkv.pt"),("af2_target","target_time.pt"),("af2_state","trainer_state.pt")):
        assert entries[label]["sha256"]==receipt["files"][filename]
    data=dict(task=CFG["task"],checkpoint_step=result["last_complete_step"],sources=entries)
    if MANIFEST.exists():
        if json.loads(MANIFEST.read_text())!=data:raise RuntimeError("EXP009 source changed after freeze")
    elif create:atomic_json(MANIFEST,data)
    return data


def checkpoint_cpu(manifest:dict):
    import torch
    def p(name):return ROOT/manifest["sources"][name]["path"]
    qkv=torch.load(p("af2_qkv"),map_location="cpu",weights_only=True)
    target=torch.load(p("af2_target"),map_location="cpu",weights_only=True)
    state=torch.load(p("af2_state"),map_location="cpu",weights_only=True)
    step=manifest["checkpoint_step"]
    assert qkv["format"]=="h3_causal_tail_qkv_v2" and qkv["rank"]==8
    assert qkv["block_indices"]==list(range(42,50))
    assert target["format"]=="h3world_anyflow_time_v1" and target["gate"]==.25
    assert target["velocity_convention"]=="noise-clean"
    assert state["task"]==qkv["metadata"]["task"]==target["metadata"]["task"]=="EXP-007/v3-AF2"
    assert state["step"]==qkv["metadata"]["step"]==target["metadata"]["step"]==step
    assert state["metadata"]==qkv["metadata"]==target["metadata"]
    assert state["qkv_sha256"]==manifest["sources"]["af2_qkv"]["sha256"]
    assert state["target_sha256"]==manifest["sources"]["af2_target"]["sha256"]
    assert state["metadata"]["protocol"]=="V3 strict causal/global/current-prefix/Single I0/12+5/sigma0 student KV"
    assert state["metadata"]["precision"]=="h3_fp32"
    return qkv,target,state


def preflight()->None:
    import torch
    import numpy as np
    from benchmark import write_video
    from evaluate_action_control import evaluate
    from run_fm8 import cache_signature,cache_info,tensor_sha,decode_video,contact_sheet
    from causal.local_topology import visible_inputs
    from causal.anyflow_sampling import configure_video_schedule
    from diffsynth.diffusion.flow_match import FlowMatchScheduler
    from rollout_contract import prompt_for_path
    assert CFG["partition"]==[12,5,5] and CFG["models"]==["fm4","af4"]
    assert (CFG["max_forwards"],CFG["max_sampling"],CFG["max_commits"],CFG["max_vae"])==(38,32,6,8)
    assert CFG["steps_per_chunk"]==4 and CFG["flow_shift"]==2.22
    selected=source_manifest(create=False)
    if selected is None:
        print(json.dumps(dict(task=CFG["task"],status="WAITING_FOR_AF2_CHECKPOINT",gpu_calls=0)))
        return
    checkpoint_cpu(selected)
    first=torch.load(FM8/"first12.pt",map_location="cpu",weights_only=True)
    assert first.shape==(1,24,12,30,52)
    assert np.load(FM8/"published_39.npy",mmap_mode="r").shape==(39,480,832,3)
    source={a:torch.load(FROZEN/f"source_coarse/inputs/parking_{a}.pt",map_location="cpu",weights_only=True) for a in "AD"}
    for key in ("initial_noise","audio_noise","anchor"):
        assert torch.equal(source["A"][key],source["D"][key]),key
    # AA/AD share every visible C1 token. In C2 only the corresponding action
    # spans may differ; future actions/video must be physically absent.
    layout=source["A"]["packed"]
    frame_rows=(first.shape[-2]//2)*(first.shape[-1]//2)
    prompts={}
    for stop in (12,17,22):
        for path in ("AA","AD"):
            prompt=prompt_for_path(source,path,stop)
            visible,trimmed=visible_inputs(layout,prompt,stop,frame_rows)
            assert len(visible["action_text_rows"])==stop
            assert visible["seq_len"]==int(visible["action_video_start"])+stop*frame_rows
            assert len(trimmed)==len(visible["text_pos"])
            prompts[path,stop]=trimmed
    assert torch.equal(prompts["AA",12],prompts["AD",12])
    assert not torch.equal(prompts["AA",17],prompts["AD",17])
    assert not torch.equal(prompts["AA",22],prompts["AD",22])
    scheduler=FlowMatchScheduler("MiniMax-H3")
    sigmas=configure_video_schedule(scheduler,steps=4,grid="native",flow_shift=2.22)
    frozen=json.loads((FM8/"chunk_0_12.json").read_text())["sigmas"]
    assert sigmas==frozen[::2] and len(sigmas)==5 and sigmas[0]==1 and sigmas[-1]==0
    assert [float(t)/1000 for t in scheduler.timesteps]==sigmas[:-1]
    from causal.anyflow import finite_map_step
    assert all(sigmas[i]>sigmas[i+1] for i in range(4))
    sample=torch.tensor([1.25],dtype=torch.float32)
    velocity=torch.tensor([-.75],dtype=torch.float32)
    scalar_update_max_abs_error=0.0
    for j,(sigma,next_sigma) in enumerate(zip(sigmas[:-1],sigmas[1:])):
        ordinary=scheduler.step(velocity,scheduler.timesteps[j],sample)
        finite=finite_map_step(sample,velocity,sigma,next_sigma)
        difference=float((ordinary-finite).abs().max())
        scalar_update_max_abs_error=max(scalar_update_max_abs_error,difference)
        assert difference<=2e-7  # FP32 arithmetic order may differ by one ULP.
    # Freeze only after every CPU assertion succeeds. An existing manifest is
    # checked for drift by source_manifest above.
    source_manifest(create=True)
    result=dict(task=CFG["task"],status="CPU_PASS_NO_GPU_AUTHORIZATION",
                checkpoint_step=selected["checkpoint_step"],native_sigmas=sigmas,
                ordinary_time=[float(t)/1000 for t in scheduler.timesteps],
                af_target_sigma=sigmas[1:],source_count=len(selected["sources"]),
                source_manifest_sha256=sha(MANIFEST),
                first12_sha256=sha(FM8/"first12.pt"),
                first39_rgb_file_sha256=sha(FM8/"published_39.npy"),
                action_spans_checked=[12,17,22],
                fm_vs_finite_scalar_update_max_abs_error=scalar_update_max_abs_error,
                gpu_calls=0)
    atomic_json(HERE/"cpu_preflight.json",result)
    print(json.dumps(result))


def run(model_kind:str,stage:str,path:str|None,gpu:int,ledger:dict,manifest:dict)->None:
    import numpy as np
    import torch
    from PIL import Image
    import infer as abot
    from benchmark import write_video
    from causal.anyflow import install_anyflow,finite_map_step
    from causal.pretrained_lora import install_adapters
    from causal.h3_cached import H3ChunkCache
    from causal.h3_precision import configure_precision
    from causal.local_topology import visible_inputs
    from causal.anyflow_sampling import configure_video_schedule
    from current_prefix import current_prefix_feedback
    from interval_student import interval_student
    from interval_cached import interval_cached
    from rollout_contract import prompt_for_path,rgb_sha,stitch_immutable,rgb_stop
    from evaluate_action_control import evaluate as flow_metrics
    from run_fm8 import cache_signature,cache_info,tensor_sha,decode_video,contact_sheet

    torch.set_num_threads(4)
    torch.manual_seed(CFG["seed"])
    start,stop,index={"prefill":(0,12,0),"second":(12,17,1),"third":(17,22,2)}[stage]
    assert model_kind in CFG["models"]
    model_out=OUT/model_kind
    directory=model_out if stage=="prefill" else model_out/path
    directory.mkdir(parents=True,exist_ok=True)
    row_path=directory/f"{stage}_{start}_{stop}.json"
    if row_path.exists():raise RuntimeError("EXP009 prior stage exists; no retry")
    began=time.perf_counter()
    row=dict(task=CFG["task"],status="loading",model=model_kind,stage=stage,path=path,
             interval=[start,stop],index=index,gpu=gpu,
             checkpoint_step=manifest["checkpoint_step"] if model_kind=="af4" else None,
             checkpoint_manifest_sha256=sha(MANIFEST),runner_sha256=sha(__file__),
             sampling_forwards=0,commit_forwards=0,vae_calls=0)
    atomic_json(row_path,row)
    def check_peak():
        peak=torch.cuda.max_memory_allocated()/2**30
        if peak>CFG["max_allocated_gib"]:
            raise MemoryError(f"EXP009 allocated VRAM cap exceeded: {peak:.3f} GiB")
    def reserve(kind:str):
        now=time.time()
        used=ledger["gpu_seconds"]+(now-ledger["active_start"])
        if used>=CFG["max_gpu_hours"]*3600 or now>=datetime.fromisoformat(CFG["deadline_hkt"]).timestamp():
            raise TimeoutError("EXP009 budget/deadline")
        if now-ledger["first_start"]>=CFG["max_elapsed_minutes"]*60:
            raise TimeoutError("EXP009 elapsed cap")
        if kind=="vae":
            if ledger["vae"]>=8:raise RuntimeError("VAE budget exhausted")
            ledger["vae"]+=1
        else:
            if ledger["forwards"]>=38:raise RuntimeError("forward budget exhausted")
            key="sampling" if kind=="sampling" else "commits"
            cap=32 if kind=="sampling" else 6
            if ledger[key]>=cap:raise RuntimeError(f"{kind} cap")
            ledger[key]+=1;ledger["forwards"]+=1
        atomic_json(OUT/"budget.json",ledger)
    def move(x):
        if torch.is_tensor(x):return x.to("cuda:0")
        if isinstance(x,dict):return {k:move(v) for k,v in x.items()}
        return x
    try:
        qkv_state,target_state,_=checkpoint_cpu(manifest)
        inputs={a:torch.load(FROZEN/f"source_coarse/inputs/parking_{a}.pt",map_location="cpu",weights_only=True) for a in "AD"}
        for key in ("initial_noise","audio_noise","anchor"):
            assert torch.equal(inputs["A"][key],inputs["D"][key])
        first=torch.load(FM8/"first12.pt",map_location="cpu",weights_only=True).to("cuda:0",torch.float32)
        row["fm8_first_sha256"]=sha(FM8/"first12.pt")
        row["first_tensor_sha256"]=tensor_sha(first)
        full=move(inputs["A"]["packed"])
        prompt=prompt_for_path(inputs,path or "AA",stop).to("cuda:0")
        layout,prompt=visible_inputs(full,prompt,stop,390)
        assert len(layout["action_text_rows"])==stop
        row["prompt_sha256"]=tensor_sha(prompt)
        anchor=inputs["A"]["anchor"].to("cuda:0")
        audio=inputs["A"]["audio_noise"].to("cuda:0")
        pipe=abot.load_pipeline("cuda:0")
        released=FROZEN/"runtime/checkpoints/H3-World/step-10000.safetensors"
        pipe.load_lora(pipe.dit,state_dict=abot.load_checkpoint_lora(released),hotload=True)
        model=pipe.dit.eval().requires_grad_(False)
        row["precision"]=configure_precision(model,"h3_fp32",
            native_transformer_dir=FROZEN/"runtime/DiffSynth-Studio-h3-v2/models/MiniMax/MiniMax-H3/FL2VA/transformer")
        pipe.load_models_to_device(["dit"])
        if model_kind=="af4":
            adapters,indices=install_adapters(model,rank=8,block_indices=tuple(range(42,50)),device="cuda:0")
            target=install_anyflow(model,device="cuda:0",gate=.25)
            with torch.no_grad():
                for adapter,a,b in zip(adapters,qkv_state["lora_A"],qkv_state["lora_B"]):
                    adapter.lora_A.copy_(a.to("cuda:0"));adapter.lora_B.copy_(b.to("cuda:0"))
            target.load_state_dict(target_state["weights"],strict=True)
        else:
            assert not hasattr(model,"anyflow_conditioner")
        row["model_condition"]=("original_H3_released_action_LoRA_only" if model_kind=="fm4"
                                else "AF2_step32_QKV_and_target_time")
        model.requires_grad_(False)
        cache=H3ChunkCache(5,"cpu") if stage=="prefill" else None
        if stage=="prefill":
            current=first
            published=None;history=None
        else:
            cache_file=(model_out/"cache_through12.pt" if stage=="second" else model_out/path/"cache_through17.pt")
            parent_row_path=(model_out/"prefill_0_12.json" if stage=="second" else model_out/path/"second_12_17.json")
            parent_row=json.loads(parent_row_path.read_text())
            assert parent_row["status"]=="complete_pending_visual_review"
            assert parent_row["model"]==model_kind
            assert parent_row["checkpoint_manifest_sha256"]==sha(MANIFEST)
            assert parent_row["checkpoint_step"]==(manifest["checkpoint_step"] if model_kind=="af4" else None)
            assert sha(cache_file)==parent_row["cache_after_sha256"]
            cache=torch.load(cache_file,map_location="cpu",weights_only=False)
            row["source_cache_sha256"]=sha(cache_file)
            row["cache_before"]=cache_info(cache,index)
            if stage=="second":
                history=first
                published=np.load(FM8/"published_39.npy")
            else:
                c2_path=model_out/path/"chunk_12_17.pt"
                assert sha(c2_path)==parent_row["endpoint_sha256"]
                c2=torch.load(c2_path,map_location="cpu",weights_only=True).to("cuda:0",torch.float32)
                history=torch.cat((first,c2),dim=2)
                published_file=model_out/path/"published_56.npy"
                assert sha(published_file)==parent_row["published_file_sha256"]
                published=np.load(published_file)
            assert published.shape==(rgb_stop(start),480,832,3)
            row["prior_RGB_sha256"]=rgb_sha(published)
            row["history_tensor_sha256"]=tensor_sha(history)
            current=inputs["A"]["initial_noise"][:,:,start:stop].to("cuda:0",torch.float32).clone()
            row["noise_sha256"]=tensor_sha(current)
        row["model_ready_seconds"]=time.perf_counter()-began
        history_hash=None if history is None else tensor_sha(history)
        current_noise_hash=None if stage=="prefill" else tensor_sha(current)
        sigmas=configure_video_schedule(pipe.scheduler,steps=4,grid="native",flow_shift=2.22)
        assert sigmas==json.loads((FM8/"chunk_0_12.json").read_text())["sigmas"][::2]
        row["sigmas"]=sigmas
        row["sampler"]="native_FM_scheduler_step" if model_kind=="fm4" else "finite_map_next_sigma"
        signature=cache_signature(cache)
        with current_prefix_feedback(),torch.no_grad():
            if stage=="prefill":
                reserve("commit")
                tick=time.perf_counter()
                if model_kind=="af4":
                    _=interval_student(model,current,start=0,index=0,cache=cache,
                        full_packed=layout,prompt=prompt,anchor=anchor,audio=audio,
                        sigma=0,target_sigma=0,commit=True)
                else:
                    _=interval_cached(model,current,start=0,index=0,cache=cache,
                        full_packed=layout,prompt=prompt,anchor=anchor,audio=audio,
                        sigma=0,commit=True)
                torch.cuda.synchronize()
                check_peak()
                row["commit_seconds"]=time.perf_counter()-tick;row["commit_forwards"]=1
            else:
                tick=time.perf_counter()
                for j,(sigma,next_sigma) in enumerate(zip(sigmas[:-1],sigmas[1:])):
                    reserve("sampling")
                    if model_kind=="af4":
                        velocity=interval_student(model,current,start=start,index=index,cache=cache,
                            full_packed=layout,prompt=prompt,anchor=anchor,audio=audio,
                            sigma=sigma,target_sigma=next_sigma)
                        current=finite_map_step(current,velocity,sigma,next_sigma)
                    else:
                        velocity=interval_cached(model,current,start=start,index=index,cache=cache,
                            full_packed=layout,prompt=prompt,anchor=anchor,audio=audio,sigma=sigma)
                        timestep=pipe.scheduler.timesteps[j]
                        assert float(timestep)/1000==sigma
                        current=pipe.scheduler.step(velocity,timestep,current)
                    if not torch.isfinite(current).all():raise FloatingPointError("nonfinite EXP009 latent")
                    check_peak()
                    row["step_completed"]=j+1;atomic_json(row_path,row)
                    print(f"EXP009 {model_kind} {stage} {path} {j+1}/4",flush=True)
                torch.cuda.synchronize()
                row["sampling_seconds"]=time.perf_counter()-tick;row["sampling_forwards"]=4
                assert cache_signature(cache)==signature
                assert tensor_sha(history)==history_hash
                assert tensor_sha(inputs["A"]["initial_noise"][:,:,start:stop].to("cuda:0",torch.float32))==current_noise_hash
                row["history_cache_read_unchanged"]=True
                endpoint=directory/f"chunk_{start}_{stop}.pt"
                temp=Path(str(endpoint)+".partial")
                torch.save(current.detach().cpu(),temp);os.replace(temp,endpoint)
                row["endpoint_sha256"]=sha(endpoint)
                if stage=="second":
                    reserve("commit")
                    tick=time.perf_counter()
                    if model_kind=="af4":
                        _=interval_student(model,current,start=start,index=index,cache=cache,
                            full_packed=layout,prompt=prompt,anchor=anchor,audio=audio,
                            sigma=0,target_sigma=0,commit=True)
                    else:
                        _=interval_cached(model,current,start=start,index=index,cache=cache,
                            full_packed=layout,prompt=prompt,anchor=anchor,audio=audio,
                            sigma=0,commit=True)
                    torch.cuda.synchronize()
                    check_peak()
                    row["commit_seconds"]=time.perf_counter()-tick;row["commit_forwards"]=1
        if stage in ("prefill","second"):
            row["cache_after"]=cache_info(cache,index+1)
            cache_out=directory/("cache_through12.pt" if stage=="prefill" else "cache_through17.pt")
            temp=Path(str(cache_out)+".partial")
            torch.save(cache,temp);os.replace(temp,cache_out)
            row["cache_after_sha256"]=sha(cache_out)
        if stage!="prefill":
            assert sha(cache_file)==row["source_cache_sha256"]
            row["source_cache_file_unchanged"]=True
        if stage!="prefill":
            pipe.load_models_to_device(["video_vae"])
            reserve("vae")
            tick=time.perf_counter()
            decoded=decode_video(pipe,torch.cat((history,current),dim=2))
            check_peak()
            row["vae_calls"]=1
            assert len(decoded)==rgb_stop(stop)
            appended=stitch_immutable(published,decoded,start,stop)
            assert rgb_sha(appended[:len(published)])==row["prior_RGB_sha256"]
            row["old_RGB_unchanged"]=True
            row["decode_seconds"]=time.perf_counter()-tick
            published_file=directory/f"published_{rgb_stop(stop)}.npy"
            np.save(published_file,appended)
            row["published_file_sha256"]=sha(published_file)
            row["published_RGB_sha256"]=rgb_sha(appended)
            video=directory/f"rollout_{rgb_stop(stop)}.mp4"
            write_video([Image.fromarray(f) for f in appended],video)
            active=directory/f"chunk_{start}_{stop}.mp4"
            new=appended[rgb_stop(start):]
            write_video([Image.fromarray(f) for f in new],active)
            contact_sheet(new,directory/f"chunk_{start}_{stop}_all_frames.jpg")
            row["video_sha256"]=sha(video)
            row["flow"]=flow_metrics(active)
            gray=np.stack([np.asarray(Image.fromarray(f).convert("L"),dtype=np.float32)
                           for f in appended[rgb_stop(start)-1:]])
            row["boundary_gray_MAD"]=float(np.abs(gray[1]-gray[0]).mean())
            row["inside_gray_MAD"]=float(np.abs(np.diff(gray[1:],axis=0)).mean())
        row["peak_allocated_gib"]=torch.cuda.max_memory_allocated()/2**30
        row["peak_reserved_gib"]=torch.cuda.max_memory_reserved()/2**30
        row["peak_cpu_rss_mib"]=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
        row["seconds"]=time.perf_counter()-began
        row["status"]="complete_pending_visual_review"
        atomic_json(row_path,row)
    except BaseException as exc:
        row["status"]="failed";row["error"]=repr(exc)
        row["seconds"]=time.perf_counter()-began
        atomic_json(row_path,row)
        raise


def main()->None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight",action="store_true")
    parser.add_argument("--model",choices=CFG["models"])
    parser.add_argument("--stage",choices=["prefill","second","third"])
    parser.add_argument("--path",choices=["AA","AD"])
    parser.add_argument("--gpu",type=int,default=CFG["gpu"])
    args=parser.parse_args()
    paths()
    if args.preflight:preflight();return
    if args.model is None or args.stage is None or (args.stage=="prefill")!=(args.path is None):
        parser.error("prefill has no path; second/third require AA/AD")
    manifest=source_manifest(create=False)
    if manifest is None or not MANIFEST.exists():raise RuntimeError("EXP009 CPU preflight/frozen checkpoint required")
    authorization=HERE/"GPU_AUTHORIZATION.json"
    if not authorization.exists():raise RuntimeError("EXP009 Judge GPU authorization marker absent")
    permit=json.loads(authorization.read_text())
    if not (permit.get("approved") and permit.get("checkpoint_step")==manifest["checkpoint_step"]
            and permit.get("source_manifest_sha256")==sha(MANIFEST)
            and args.model in permit.get("allowed_models",[])
            and args.stage in permit.get("allowed_stages",[])
            and args.gpu==permit.get("gpu")):
        raise RuntimeError("EXP009 authorization/source/stage mismatch")
    if args.stage=="third":
        c3_marker=HERE/"C3_GPU_AUTHORIZATION.json"
        if not c3_marker.exists():raise RuntimeError("EXP009 C3 requires separate Judge release")
        c3=json.loads(c3_marker.read_text())
        if not (c3.get("approved") and c3.get("source_manifest_sha256")==sha(MANIFEST)
                and args.model in c3.get("allowed_models",[])
                and args.gpu==c3.get("gpu")):
            raise RuntimeError("EXP009 C3 authorization mismatch")
    free=int(subprocess.check_output(["nvidia-smi",f"--id={args.gpu}",
        "--query-gpu=memory.free","--format=csv,noheader,nounits"],text=True).strip())
    if free<44000:raise RuntimeError(f"GPU{args.gpu} not idle enough: {free}MiB")
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/"worker.lock").open("a+") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        budget=OUT/"budget.json"
        ledger=json.loads(budget.read_text()) if budget.exists() else dict(
            task=CFG["task"],first_start=time.time(),gpu_seconds=0.,forwards=0,
            sampling=0,commits=0,vae=0,runs=[])
        started=time.time();ledger["active_start"]=started
        atomic_json(budget,ledger)
        remaining=min(CFG["max_gpu_hours"]*3600-ledger["gpu_seconds"],
                      CFG["max_elapsed_minutes"]*60-(started-ledger["first_start"]),
                      datetime.fromisoformat(CFG["deadline_hkt"]).timestamp()-started)
        if remaining<=0:raise TimeoutError("EXP009 budget exhausted")
        def alarm(_signum,_frame):raise TimeoutError("EXP009 absolute alarm")
        signal.signal(signal.SIGALRM,alarm)
        signal.setitimer(signal.ITIMER_REAL,remaining)
        os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu),ABOT_VRAM_RESERVE_GIB="18",
            HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",DIFFSYNTH_SKIP_DOWNLOAD="True",
            TOKENIZERS_PARALLELISM="false",PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
        status="failed"
        try:
            run(args.model,args.stage,args.path,args.gpu,ledger,manifest)
            status="complete_pending_visual_review"
        finally:
            signal.setitimer(signal.ITIMER_REAL,0)
            ledger["gpu_seconds"]+=time.time()-started
            ledger.pop("active_start",None)
            ledger["runs"].append(dict(model=args.model,stage=args.stage,path=args.path,gpu=args.gpu,
                                       start=started,end=time.time(),status=status))
            atomic_json(budget,ledger)


if __name__=="__main__":main()
