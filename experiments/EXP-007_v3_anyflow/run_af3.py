"""Prepared AF3 8NFE continuation runner. GPU stages need Judge AF3 release.

C1 is EXP-006's FM8 endpoint and published RGB. AF3 builds its own C1 KV,
then branches AA/AD through C2 and C3. Each 8NFE map targets the next native
sigma. CPU --preflight may return WAITING_FOR_AF2 until final training exists.
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
CFG=json.loads((HERE/"config_v3_af3.json").read_text())
OUT=ROOT/CFG["output_root"]
AF2=ROOT/"H3-World/outputs/EXP-007_v3_anyflow_af2"
FM8=ROOT/"H3-World/outputs/EXP-006_v3_fm8_full"
FROZEN=ROOT/"H3-World/outputs/2026-10-09-22/chunk_partition_cb"
EXP005=HERE.parent/"EXP-005_v3_sliding_window"
ROUTER=ROOT/"submission/reports/stage1_anyflow/02_causal_diagnostics/current_prefix_candidate"
CONTRACT=HERE.parent/"EXP-001_v2b_124"
EXP002=HERE.parent/"EXP-002_native_cached"
EXP006=HERE.parent/"EXP-006_v3_fm8_full"
MANIFEST=HERE/"source_manifest_v3_af3.json"


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
                 str(ROUTER),str(CONTRACT),str(EXP002),str(EXP006),str(HERE)]


def selected_checkpoint():
    if CFG["checkpoint"]!="auto_final_af2":raise ValueError("AF3 must select AF2 final checkpoint")
    result_path=AF2/"result.json"
    if not result_path.exists():return None
    result=json.loads(result_path.read_text())
    if result["status"] not in ("complete_32_pending_judge","budget_stopped_pending_judge"):
        return None
    last=result.get("last_complete_step")
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
        "runner":HERE/"run_af3.py","config":HERE/"config_v3_af3.json",
        "student_entry":HERE/"interval_student.py",
        "fm8_helpers":EXP006/"run_fm8.py",
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
        if json.loads(MANIFEST.read_text())!=data:raise RuntimeError("AF3 source changed after freeze")
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
    from benchmark import write_video
    from evaluate_action_control import evaluate
    from run_fm8 import cache_signature,cache_info,tensor_sha,decode_video,contact_sheet
    from causal.anyflow_sampling import configure_video_schedule
    from diffsynth.diffusion.flow_match import FlowMatchScheduler
    assert CFG["partition"]==[12,5,5]
    assert (CFG["max_forwards"],CFG["max_sampling"],CFG["max_commits"],CFG["max_vae"])==(35,32,3,4)
    selected=source_manifest(create=True)
    if selected is None:
        print(json.dumps(dict(task=CFG["task"],status="WAITING_FOR_AF2_CHECKPOINT",gpu_calls=0)))
        return
    checkpoint_cpu(selected)
    first=torch.load(FM8/"first12.pt",map_location="cpu",weights_only=True)
    assert first.shape==(1,24,12,30,52)
    import numpy as np
    assert np.load(FM8/"published_39.npy",mmap_mode="r").shape==(39,480,832,3)
    source={a:torch.load(FROZEN/f"source_coarse/inputs/parking_{a}.pt",map_location="cpu",weights_only=True) for a in "AD"}
    for key in ("initial_noise","audio_noise","anchor"):
        assert torch.equal(source["A"][key],source["D"][key]),key
    sigmas=configure_video_schedule(FlowMatchScheduler("MiniMax-H3"),steps=8,grid="native",flow_shift=2.22)
    frozen=json.loads((FM8/"chunk_0_12.json").read_text())["sigmas"]
    assert sigmas==frozen and len(sigmas)==9 and sigmas[0]==1 and sigmas[-1]==0
    print(json.dumps(dict(task=CFG["task"],status="CPU_PASS",checkpoint_step=selected["checkpoint_step"],
                          native_sigmas=sigmas,source_count=len(selected["sources"]),gpu_calls=0)))


def run(stage:str,path:str|None,gpu:int,ledger:dict,manifest:dict)->None:
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
    from rollout_contract import prompt_for_path,rgb_sha,stitch_immutable,rgb_stop
    from evaluate_action_control import evaluate as flow_metrics
    from run_fm8 import cache_signature,cache_info,tensor_sha,decode_video,contact_sheet

    torch.set_num_threads(4)
    torch.manual_seed(CFG["seed"])
    start,stop,index={"prefill":(0,12,0),"second":(12,17,1),"third":(17,22,2)}[stage]
    directory=OUT if stage=="prefill" else OUT/path
    directory.mkdir(parents=True,exist_ok=True)
    row_path=directory/f"{stage}_{start}_{stop}.json"
    if row_path.exists():raise RuntimeError("AF3 prior stage exists; no retry")
    began=time.perf_counter()
    row=dict(task=CFG["task"],status="loading",stage=stage,path=path,
             interval=[start,stop],index=index,gpu=gpu,checkpoint_step=manifest["checkpoint_step"],
             checkpoint_manifest_sha256=sha(MANIFEST),runner_sha256=sha(__file__),
             sampling_forwards=0,commit_forwards=0,vae_calls=0)
    atomic_json(row_path,row)
    def check_peak():
        peak=torch.cuda.max_memory_allocated()/2**30
        if peak>CFG["max_allocated_gib"]:
            raise MemoryError(f"AF3 allocated VRAM cap exceeded: {peak:.3f} GiB")
    def reserve(kind:str):
        now=time.time()
        used=ledger["gpu_seconds"]+(now-ledger["active_start"])
        if used>=CFG["max_gpu_hours"]*3600 or now>=datetime.fromisoformat(CFG["deadline_hkt"]).timestamp():
            raise TimeoutError("AF3 budget/deadline")
        if now-ledger["first_start"]>=CFG["max_elapsed_minutes"]*60:
            raise TimeoutError("AF3 elapsed cap")
        if kind=="vae":
            if ledger["vae"]>=4:raise RuntimeError("VAE budget exhausted")
            ledger["vae"]+=1
        else:
            if ledger["forwards"]>=35:raise RuntimeError("forward budget exhausted")
            key="sampling" if kind=="sampling" else "commits"
            cap=32 if kind=="sampling" else 3
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
        adapters,indices=install_adapters(model,rank=8,block_indices=tuple(range(42,50)),device="cuda:0")
        target=install_anyflow(model,device="cuda:0",gate=.25)
        with torch.no_grad():
            for adapter,a,b in zip(adapters,qkv_state["lora_A"],qkv_state["lora_B"]):
                adapter.lora_A.copy_(a.to("cuda:0"));adapter.lora_B.copy_(b.to("cuda:0"))
        target.load_state_dict(target_state["weights"],strict=True)
        model.requires_grad_(False)
        cache=H3ChunkCache(5,"cpu") if stage=="prefill" else None
        if stage=="prefill":
            current=first
            published=None;history=None
        else:
            cache_file=(OUT/"cache_through12.pt" if stage=="second" else OUT/path/"cache_through17.pt")
            parent_row_path=(OUT/"prefill_0_12.json" if stage=="second" else OUT/path/"second_12_17.json")
            parent_row=json.loads(parent_row_path.read_text())
            assert parent_row["status"]=="complete_pending_visual_review"
            assert parent_row["checkpoint_manifest_sha256"]==sha(MANIFEST)
            assert parent_row["checkpoint_step"]==manifest["checkpoint_step"]
            assert sha(cache_file)==parent_row["cache_after_sha256"]
            cache=torch.load(cache_file,map_location="cpu",weights_only=False)
            row["source_cache_sha256"]=sha(cache_file)
            row["cache_before"]=cache_info(cache,index)
            if stage=="second":
                history=first
                published=np.load(FM8/"published_39.npy")
            else:
                c2_path=OUT/path/"chunk_12_17.pt"
                assert sha(c2_path)==parent_row["endpoint_sha256"]
                c2=torch.load(c2_path,map_location="cpu",weights_only=True).to("cuda:0",torch.float32)
                history=torch.cat((first,c2),dim=2)
                published_file=OUT/path/"published_56.npy"
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
        sigmas=configure_video_schedule(pipe.scheduler,steps=8,grid="native",flow_shift=2.22)
        assert sigmas==json.loads((FM8/"chunk_0_12.json").read_text())["sigmas"]
        row["sigmas"]=sigmas
        signature=cache_signature(cache)
        with current_prefix_feedback(),torch.no_grad():
            if stage=="prefill":
                reserve("commit")
                tick=time.perf_counter()
                _=interval_student(model,current,start=0,index=0,cache=cache,
                    full_packed=layout,prompt=prompt,anchor=anchor,audio=audio,
                    sigma=0,target_sigma=0,commit=True)
                torch.cuda.synchronize()
                check_peak()
                row["commit_seconds"]=time.perf_counter()-tick;row["commit_forwards"]=1
            else:
                tick=time.perf_counter()
                for j,(sigma,next_sigma) in enumerate(zip(sigmas[:-1],sigmas[1:])):
                    reserve("sampling")
                    velocity=interval_student(model,current,start=start,index=index,cache=cache,
                        full_packed=layout,prompt=prompt,anchor=anchor,audio=audio,
                        sigma=sigma,target_sigma=next_sigma)
                    current=finite_map_step(current,velocity,sigma,next_sigma)
                    if not torch.isfinite(current).all():raise FloatingPointError("nonfinite AF3 latent")
                    check_peak()
                    row["step_completed"]=j+1;atomic_json(row_path,row)
                    print(f"AF3 {stage} {path} {j+1}/8",flush=True)
                torch.cuda.synchronize()
                row["sampling_seconds"]=time.perf_counter()-tick;row["sampling_forwards"]=8
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
                    _=interval_student(model,current,start=start,index=index,cache=cache,
                        full_packed=layout,prompt=prompt,anchor=anchor,audio=audio,
                        sigma=0,target_sigma=0,commit=True)
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
    parser.add_argument("--stage",choices=["prefill","second","third"])
    parser.add_argument("--path",choices=["AA","AD"])
    parser.add_argument("--gpu",type=int,default=CFG["gpu"])
    args=parser.parse_args()
    paths()
    if args.preflight:preflight();return
    if args.stage is None or (args.stage=="prefill")!=(args.path is None):
        parser.error("prefill has no path; second/third require AA/AD")
    manifest=source_manifest(create=False)
    if manifest is None or not MANIFEST.exists():raise RuntimeError("AF3 CPU preflight/frozen checkpoint required")
    # AF3 GPU release is a separate Judge decision after AF2; a small marker
    # prevents accidentally treating prepared code as authorization.
    authorization=HERE/"AF3_GPU_AUTHORIZATION.json"
    if not authorization.exists():raise RuntimeError("AF3 Judge GPU authorization marker absent")
    permit=json.loads(authorization.read_text())
    if not permit.get("approved") or permit.get("checkpoint_step")!=manifest["checkpoint_step"]:
        raise RuntimeError("AF3 authorization checkpoint mismatch")
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
        if remaining<=0:raise TimeoutError("AF3 budget exhausted")
        def alarm(_signum,_frame):raise TimeoutError("AF3 absolute alarm")
        signal.signal(signal.SIGALRM,alarm)
        signal.setitimer(signal.ITIMER_REAL,remaining)
        os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu),ABOT_VRAM_RESERVE_GIB="18",
            HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",DIFFSYNTH_SKIP_DOWNLOAD="True",
            TOKENIZERS_PARALLELISM="false",PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
        status="failed"
        try:
            run(args.stage,args.path,args.gpu,ledger,manifest)
            status="complete_pending_visual_review"
        finally:
            signal.setitimer(signal.ITIMER_REAL,0)
            ledger["gpu_seconds"]+=time.time()-started
            ledger.pop("active_start",None)
            ledger["runs"].append(dict(stage=args.stage,path=args.path,gpu=args.gpu,
                                       start=started,end=time.time(),status=status))
            atomic_json(budget,ledger)


if __name__=="__main__":main()
