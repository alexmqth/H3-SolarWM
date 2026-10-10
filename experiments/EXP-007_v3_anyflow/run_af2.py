"""EXP-007/v3 AF2: at most 31 restored finite-map updates, AA/AD alternating.

No video inference here. One current-weight clean C1 prefill precedes each batch.
All model/optimizer calls are reserved before execution and failures consume budget.
"""
from __future__ import annotations

from datetime import datetime
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CFG = json.loads((HERE / "config_v3_af2.json").read_text())
OUT = ROOT / CFG["output_root"]
AF1 = ROOT / "H3-World/outputs/EXP-007_v3_anyflow_af1_attempt2/step_01"
FROZEN = ROOT / "H3-World/outputs/2026-10-09-22/chunk_partition_cb"
EXP005 = HERE.parent / "EXP-005_v3_sliding_window"
ROUTER = ROOT / "submission/reports/stage1_anyflow/02_causal_diagnostics/current_prefix_candidate"
CONTRACT = HERE.parent / "EXP-001_v2b_124"
EXP002 = HERE.parent / "EXP-002_native_cached"
SOURCES = {
    "runner": HERE / "run_af2.py", "config": HERE / "config_v3_af2.json",
    "taskbook": HERE / "taskbook_v3.md", "student_entry": HERE / "interval_student.py",
    "af1_result": ROOT / "H3-World/outputs/EXP-007_v3_anyflow_af1_attempt2/result.json",
    "af1_qkv": AF1 / "qkv.pt", "af1_target": AF1 / "target_time.pt",
    "af1_state": AF1 / "trainer_state.pt",
    "first12": ROOT / "submission/experiments/11_causal_12_then5_selfhistory/states/first12_A.pt",
    "second_A": ROOT / "H3-World/outputs/EXP-002_native_cached/AA/chunk_12_17.pt",
    "second_D": ROOT / "H3-World/outputs/EXP-002_native_cached/AD/chunk_12_17.pt",
    "input_A": FROZEN / "source_coarse/inputs/parking_A.pt",
    "input_D": FROZEN / "source_coarse/inputs/parking_D.pt",
    "released_lora": FROZEN / "runtime/checkpoints/H3-World/step-10000.safetensors",
    "anyflow": FROZEN / "runtime/code/causal/anyflow.py",
    "qkv_adapter": FROZEN / "runtime/code/causal/pretrained_lora.py",
    "prefix_feedback": ROUTER / "current_prefix.py",
}


def sha(path: Path | str) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".partial")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(tmp, path)


def paths() -> None:
    sys.path[:0] = [str(FROZEN / "runtime/code"), str(FROZEN / "runtime/code/abot"),
                    str(FROZEN / "runtime/DiffSynth-Studio-h3-v2"),
                    str(EXP005), str(ROUTER), str(CONTRACT), str(EXP002), str(HERE)]


def manifest() -> dict:
    rows = {name: {"path": str(path.relative_to(ROOT)), "sha256": sha(path)}
            for name, path in SOURCES.items()}
    af1 = json.loads(SOURCES["af1_result"].read_text())
    assert af1["status"] == "complete_pending_judge"
    assert af1["step1_checkpoint"]["qkv.pt"] == rows["af1_qkv"]["sha256"]
    assert af1["step1_checkpoint"]["target_time.pt"] == rows["af1_target"]["sha256"]
    assert af1["step1_checkpoint"]["trainer_state.pt"] == rows["af1_state"]["sha256"]
    frozen = HERE / "source_manifest_v3_af2.json"
    data = {"task": CFG["task"], "sources": rows}
    if frozen.exists():
        if json.loads(frozen.read_text()) != data:
            raise RuntimeError("AF2 source changed after freeze")
    else:
        atomic_json(frozen, data)
    return rows


def load_checkpoint_cpu():
    import torch
    state = torch.load(AF1 / "trainer_state.pt", map_location="cpu", weights_only=True)
    qkv = torch.load(AF1 / "qkv.pt", map_location="cpu", weights_only=True)
    target = torch.load(AF1 / "target_time.pt", map_location="cpu", weights_only=True)
    assert state["task"] == qkv["metadata"]["task"] == target["metadata"]["task"] == "EXP-007/v2"
    assert state["step"] == qkv["metadata"]["step"] == target["metadata"]["step"] == 1
    assert state["qkv_sha256"] == sha(AF1 / "qkv.pt")
    assert state["target_sha256"] == sha(AF1 / "target_time.pt")
    assert state["metadata"] == qkv["metadata"] == target["metadata"]
    assert state["metadata"]["protocol"] == "V3 strict causal/global/current-prefix/Single I0/12+5/sigma0 student KV"
    assert state["metadata"]["precision"] == "h3_fp32"
    assert qkv["format"] == "h3_causal_tail_qkv_v2" and qkv["rank"] == 8
    assert qkv["block_indices"] == list(range(42,50))
    assert target["format"] == "h3world_anyflow_time_v1" and target["gate"] == .25
    assert target["velocity_convention"] == "noise-clean"
    assert len(state["optimizer"]["state"]) == 20
    return state, qkv, target


def preflight() -> None:
    import torch
    assert (CFG["end_step"]-CFG["start_step"], CFG["max_forwards"], CFG["max_backward"], CFG["max_updates"]) == (31,527,124,31)
    manifest()
    state, qkv, target = load_checkpoint_cpu()
    first = torch.load(SOURCES["first12"], map_location="cpu", weights_only=True)
    a = torch.load(SOURCES["second_A"], map_location="cpu", weights_only=True)
    d = torch.load(SOURCES["second_D"], map_location="cpu", weights_only=True)
    assert first.shape == (1,24,12,30,52) and a.shape == d.shape == (1,24,5,30,52)
    assert torch.isfinite(first).all() and torch.isfinite(a).all() and torch.isfinite(d).all()
    assert not torch.equal(a,d)
    print(json.dumps({"status":"CPU_PASS","task":CFG["task"],"resume_step":state["step"],
                      "trainable_modules":len(qkv["block_indices"])+1,
                      "source_count":len(SOURCES),"A_D_endpoints_distinct":True}),flush=True)


def cpu_tree(value):
    import torch
    if torch.is_tensor(value): return value.detach().cpu()
    if isinstance(value, dict): return {k: cpu_tree(v) for k,v in value.items()}
    if isinstance(value, list): return [cpu_tree(v) for v in value]
    return value


def train(gpu: int, ledger: dict, sources: dict) -> None:
    import torch
    import infer as abot
    from causal.anyflow import install_anyflow, save_anyflow, logical_time_pairs, anyflow_sample_loss, adaptive_scale
    from causal.pretrained_lora import install_adapters, save_adapters
    from causal.h3_cached import H3ChunkCache
    from causal.h3_precision import configure_precision
    from causal.local_topology import visible_inputs
    from chunk_plan import cache_identity
    from current_prefix import current_prefix_feedback
    from interval_student import interval_student
    from rollout_contract import prompt_for_path

    torch.set_num_threads(4)
    device = "cuda:0"
    started = time.time()
    deadline = min(started + CFG["max_gpu_hours"]*3600,
                   datetime.fromisoformat(CFG["deadline_hkt"]).timestamp())
    result = dict(task=CFG["task"], status="loading", gpu=gpu, started=started,
                  source_manifest_sha256=sha(HERE / "source_manifest_v3_af2.json"),
                  runner_sha256=sha(__file__), updates=[], checkpoint_paths=[],
                  resume_checkpoint=str(AF1.relative_to(ROOT)), source=sources)
    atomic_json(OUT / "result.json", result)
    def reserve(kind: str) -> None:
        if time.time() >= deadline: raise TimeoutError("AF2 absolute time cap")
        key, cap = {"forward":("forwards",527),"backward":("backward",124),
                    "update":("updates",31)}[kind]
        if ledger[key] >= cap: raise RuntimeError(f"AF2 {kind} cap exhausted")
        ledger[key] += 1
        atomic_json(OUT / "budget.json", ledger)
    def move(x):
        if torch.is_tensor(x): return x.to(device)
        if isinstance(x,dict): return {k:move(v) for k,v in x.items()}
        return x
    try:
        restore, qkv_state, target_state = load_checkpoint_cpu()
        inputs = {key: torch.load(SOURCES[f"input_{key}"],map_location="cpu",weights_only=True)
                  for key in "AD"}
        for key in ("initial_noise","audio_noise","anchor"):
            assert torch.equal(inputs["A"][key],inputs["D"][key]),key
        first = torch.load(SOURCES["first12"],map_location="cpu",weights_only=True).to(device,torch.float32)
        second = {key:torch.load(SOURCES[f"second_{key}"],map_location="cpu",weights_only=True).to(device,torch.float32)
                  for key in "AD"}
        full = move(inputs["A"]["packed"])
        prompts={}
        for action in "AD":
            prompt=prompt_for_path(inputs,"A"+action,17).to(device)
            layout,prompt=visible_inputs(full,prompt,17,390)
            assert len(layout["action_text_rows"])==17
            prompts[action]=(layout,prompt)
        first_prompt=prompt_for_path(inputs,"AA",12).to(device)
        first_layout,first_prompt=visible_inputs(full,first_prompt,12,390)
        anchor=inputs["A"]["anchor"].to(device)
        audio=inputs["A"]["audio_noise"].to(device)
        pipe=abot.load_pipeline(device)
        pipe.load_lora(pipe.dit,state_dict=abot.load_checkpoint_lora(SOURCES["released_lora"]),hotload=True)
        model=pipe.dit.eval().requires_grad_(False)
        result["precision"]=configure_precision(model,"h3_fp32",
            native_transformer_dir=FROZEN/"runtime/DiffSynth-Studio-h3-v2/models/MiniMax/MiniMax-H3/FL2VA/transformer")
        pipe.load_models_to_device(["dit"])
        base_versions=[(p,p._version) for p in model.parameters()]
        adapters,indices=install_adapters(model,rank=8,block_indices=tuple(range(42,50)),device=device)
        target=install_anyflow(model,device=device,gate=.25)
        for adapter,a,b in zip(adapters,qkv_state["lora_A"],qkv_state["lora_B"]):
            with torch.no_grad():
                adapter.lora_A.copy_(a.to(device));adapter.lora_B.copy_(b.to(device))
        target.load_state_dict(target_state["weights"],strict=True)
        params=[p for a in adapters for p in (a.lora_A,a.lora_B)]+list(target.parameters())
        assert {id(p) for p in model.parameters() if p.requires_grad}=={id(p) for p in params}
        optimizer=torch.optim.AdamW(params,lr=CFG["lr"],betas=(.9,.95),weight_decay=.01)
        optimizer.load_state_dict(restore["optimizer"])
        generator=torch.Generator().manual_seed(CFG["seed"])
        # Restore after all module/optimizer setup; no new random draw intervenes.
        generator.set_state(restore["logical_rng_state"])
        torch.set_rng_state(restore["cpu_rng_state"])
        torch.cuda.set_rng_state(restore["cuda_rng_state"],device)
        result["resume_verified"]={"step":1,"qkv_sha256":sha(AF1/"qkv.pt"),
            "target_sha256":sha(AF1/"target_time.pt"),"state_sha256":sha(AF1/"trainer_state.pt"),
            "optimizer_entries":len(restore["optimizer"]["state"]),
            "logical_rng_restored":True,"cpu_cuda_rng_restored":True}
        atomic_json(OUT/"result.json",result)
        def forward(cache,index,sample,sigma,target_sigma,layout,prompt,*,commit=False,gradient=False):
            reserve("forward")
            output=interval_student(model,sample,start=0 if index==0 else 12,index=index,
                cache=cache,full_packed=layout,prompt=prompt,anchor=anchor,audio=audio,
                sigma=sigma,target_sigma=target_sigma,commit=commit,
                use_gradient_checkpointing=gradient,use_gradient_checkpointing_offload=gradient)
            if not torch.isfinite(output).all():raise FloatingPointError("nonfinite AF2 velocity")
            if torch.cuda.max_memory_allocated()/2**30>CFG["max_allocated_gib"]:
                raise MemoryError("AF2 allocated VRAM cap")
            return output
        def save_checkpoint(step:int):
            directory=OUT/f"step_{step:02d}"
            directory.mkdir(exist_ok=False)
            metadata=dict(task=CFG["task"],step=step,parent_step=1,
                config_sha256=sha(HERE/"config_v3_af2.json"),
                manifest_sha256=sha(HERE/"source_manifest_v3_af2.json"),
                protocol=restore["metadata"]["protocol"],precision="h3_fp32",
                velocity_convention="noise-clean",source_kind="generated V3 FM30 C1/C2 AA/AD")
            save_adapters(directory/"qkv.pt",adapters,indices,metadata)
            save_anyflow(directory/"target_time.pt",target,metadata)
            state=dict(task=CFG["task"],step=step,metadata=metadata,
                optimizer=cpu_tree(optimizer.state_dict()),logical_rng_state=generator.get_state(),
                cpu_rng_state=torch.get_rng_state(),cuda_rng_state=torch.cuda.get_rng_state(device),
                qkv_sha256=sha(directory/"qkv.pt"),target_sha256=sha(directory/"target_time.pt"))
            torch.save(state,directory/"trainer_state.pt")
            result["checkpoint_paths"].append(dict(step=step,path=str(directory.relative_to(ROOT)),
                files={p.name:sha(p) for p in directory.iterdir() if p.is_file()}))
            atomic_json(OUT/"result.json",result)
        last_step=1
        with current_prefix_feedback():
            for step in range(2,33):
                # Leave room for one entire update and final checkpoint. Never start a partial update knowingly.
                elapsed=time.time()-started
                recent=result["updates"][-1]["seconds"] if result["updates"] else 180.
                reserve_seconds=max(180.,recent*1.5)+30.
                if deadline-time.time()<reserve_seconds:
                    result["budget_boundary_stop"]={"before_step":step,"remaining_seconds":deadline-time.time()}
                    if last_step not in (1,8,32):save_checkpoint(last_step)
                    break
                action="D" if step%2==0 else "A"
                clean=second[action]
                layout,prompt=prompts[action]
                tick=time.perf_counter()
                student_cache=H3ChunkCache(5,"cpu")
                with torch.no_grad():
                    forward(student_cache,0,first,0,0,first_layout,first_prompt,commit=True)
                cache_before=cache_identity(student_cache)
                assert student_cache.commits==50
                optimizer.zero_grad(set_to_none=True)
                pairs=logical_time_pairs(generator,shift=CFG["shift"],batch_size=4)
                diffusion=[]; samples=[]
                for i in range(4):
                    noise=torch.randn(clean.shape,generator=generator,dtype=torch.float32).to(device)
                    t,r=float(pairs.t[i]),float(pairs.r[i])
                    def velocity(x,tau,rho):
                        return forward(student_cache,1,x,tau,rho,layout,prompt,
                                       gradient=torch.is_grad_enabled())
                    raw,weight,item=anyflow_sample_loss(velocity,clean,noise,t,r,
                        shift=CFG["shift"],epsilon=CFG["epsilon"],preserve_fp32_inputs=True)
                    is_diff=bool(pairs.is_diffusion[i])
                    scale=adaptive_scale(raw,is_diffusion=is_diff,diffusion_losses=diffusion)
                    if is_diff:diffusion.append(float(raw.detach()))
                    loss=raw*weight*scale/4
                    if not torch.isfinite(loss):raise FloatingPointError("nonfinite AF2 loss")
                    reserve("backward");loss.backward()
                    item.update(sample_type="diffusion" if is_diff else
                                "endpoint" if int(pairs.sample_type[i])==1 else "flow_map",
                                adaptive_scale=float(scale),weighted_loss=float(loss.detach())*4)
                    samples.append(item)
                    del raw,loss,noise
                assert cache_identity(student_cache)==cache_before
                assert all(not entry.key.requires_grad and not entry.value.requires_grad
                           for entries in student_cache.layers.values() for entry in entries)
                grads=[p.grad for p in params]
                assert all(g is not None and torch.isfinite(g).all() for g in grads)
                target_count=len(list(target.parameters()))
                qkv_norm=float(torch.linalg.vector_norm(torch.stack([g.float().norm() for g in grads[:-target_count]])))
                target_norm=float(torch.linalg.vector_norm(torch.stack([g.float().norm() for g in grads[-target_count:]])))
                if qkv_norm<=0 or target_norm<=0:raise AssertionError("missing AF2 target/QKV gradient")
                clip_norm=float(torch.nn.utils.clip_grad_norm_(params,1.0))
                reserve("update");optimizer.step()
                student_cache.clear()
                assert all(p._version==version for p,version in base_versions)
                last_step=step
                row=dict(step=step,action=action,samples=samples,qkv_grad_norm=qkv_norm,
                         target_grad_norm=target_norm,grad_clip_pre_norm=clip_norm,
                         history_cache_read_unchanged=True,history_cache_rebuilt_this_step=True,
                         seconds=time.perf_counter()-tick,forwards=ledger["forwards"],
                         backward=ledger["backward"],updates=ledger["updates"],
                         peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
                result["updates"].append(row)
                result["last_complete_step"]=step
                atomic_json(OUT/"result.json",result)
                print(f"AF2 completed step {step}/32 action={action} seconds={row['seconds']:.2f}",flush=True)
                if step in (8,32):save_checkpoint(step)
        if last_step==32:
            result["status"]="complete_32_pending_judge"
        else:
            result["status"]="budget_stopped_pending_judge"
        result["peak_allocated_gib"]=torch.cuda.max_memory_allocated()/2**30
        result["peak_reserved_gib"]=torch.cuda.max_memory_reserved()/2**30
        result["seconds"]=time.time()-started
        atomic_json(OUT/"result.json",result)
    except BaseException as exc:
        result["status"]="failed";result["error"]=repr(exc)
        result["seconds"]=time.time()-started
        atomic_json(OUT/"result.json",result)
        raise


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight",action="store_true")
    parser.add_argument("--gpu",type=int,default=CFG["gpu"])
    args=parser.parse_args()
    paths();sources=manifest()
    if args.preflight:preflight();return
    free=int(subprocess.check_output(["nvidia-smi",f"--id={args.gpu}",
        "--query-gpu=memory.free","--format=csv,noheader,nounits"],text=True).strip())
    if free<44000:raise RuntimeError(f"GPU{args.gpu} not idle enough: {free}MiB")
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/"worker.lock").open("a+") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (OUT/"budget.json").exists():raise RuntimeError("AF2 prior attempt exists; no automatic retry")
        ledger=dict(task=CFG["task"],gpu=args.gpu,start=time.time(),forwards=0,
                    backward=0,updates=0,vae=0)
        atomic_json(OUT/"budget.json",ledger)
        deadline=min(ledger["start"]+CFG["max_gpu_hours"]*3600,
                     datetime.fromisoformat(CFG["deadline_hkt"]).timestamp())
        def alarm(_signum,_frame):raise TimeoutError("AF2 absolute deadline")
        signal.signal(signal.SIGALRM,alarm)
        signal.setitimer(signal.ITIMER_REAL,deadline-time.time())
        os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu),ABOT_VRAM_RESERVE_GIB="18",
            HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",DIFFSYNTH_SKIP_DOWNLOAD="True",
            TOKENIZERS_PARALLELISM="false",PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
        try:train(args.gpu,ledger,sources)
        finally:
            signal.setitimer(signal.ITIMER_REAL,0)
            ledger["end"]=time.time();ledger["gpu_seconds"]=ledger["end"]-ledger["start"]
            atomic_json(OUT/"budget.json",ledger)


if __name__=="__main__":main()
