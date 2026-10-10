"""Seven-cycle DMD continuation from the frozen EXP-008 pilot checkpoint.

CPU preflight is available. GPU execution requires a separate Judge TRAIN
authorization marker and never retries an existing output ledger.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
EXP=HERE.parent
ROOT=EXP.parents[2]
AFEXP=EXP.parent/"EXP-007_v3_anyflow"
PILOT=ROOT/"H3-World/outputs/EXP-008_v3_dmd_pilot"
PILOT_CODE=EXP/"dmd_cpu"
CFG=json.loads((HERE/"config_v2.json").read_text())
OUT=ROOT/CFG["output_root"]
AF2=ROOT/"H3-World/outputs/EXP-007_v3_anyflow_af2"
AF3=ROOT/"H3-World/outputs/EXP-007_v3_anyflow_af3"
FM8=ROOT/"H3-World/outputs/EXP-006_v3_fm8_full"
FROZEN=ROOT/"H3-World/outputs/2026-10-09-22/chunk_partition_cb"
SW=EXP.parent/"EXP-005_v3_sliding_window"
PREFIX=ROOT/"submission/reports/stage1_anyflow/02_causal_diagnostics/current_prefix_candidate"
CONTRACT=EXP.parent/"EXP-001_v2b_124"
MANIFEST=HERE/"source_manifest_v2.json"


def sha(path:Path|str)->str:
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(4<<20),b""):h.update(block)
    return h.hexdigest()


def atomic_json(path:Path,obj)->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=Path(str(path)+".partial")
    temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+"\n")
    os.replace(temp,path)


def setup_paths()->None:
    sys.path[:0]=[str(FROZEN/"runtime/code"),str(FROZEN/"runtime/code/abot"),
                 str(FROZEN/"runtime/DiffSynth-Studio-h3-v2"),str(SW),str(PREFIX),
                 str(CONTRACT),str(AFEXP)]


def final_checkpoint():
    result=AF2/"result.json"
    if not result.exists():return None
    data=json.loads(result.read_text())
    if data["status"] not in ("complete_32_pending_judge","budget_stopped_pending_judge"):
        return None
    step=data["last_complete_step"]
    rows=[x for x in data["checkpoint_paths"] if x["step"]==step]
    if len(rows)!=1:raise RuntimeError("final AF2 checkpoint is not unique")
    folder=ROOT/rows[0]["path"]
    return data,step,folder,rows[0]


def af3_ready(step:int)->bool:
    budget=AF3/"budget.json"
    if not budget.exists():return False
    b=json.loads(budget.read_text())
    if (b.get("forwards"),b.get("vae"))!=(35,4):return False
    for path in ("AA","AD"):
        row=AF3/path/"third_17_22.json"
        if not row.exists():return False
        data=json.loads(row.read_text())
        if data.get("status")!="complete_pending_visual_review" or data.get("checkpoint_step")!=step:
            return False
    return True


def manifest(create:bool):
    selected=final_checkpoint()
    if selected is None:return None
    _,step,folder,receipt=selected
    if not af3_ready(step):return None
    result_path=PILOT/"result.json"
    budget_path=PILOT/"budget.json"
    if not result_path.exists() or not budget_path.exists():return None
    pilot_result=json.loads(result_path.read_text())
    pilot_budget=json.loads(budget_path.read_text())
    if pilot_result.get("status")!="complete_pending_judge" or pilot_result.get("af2_step")!=step:
        return None
    if (pilot_budget.get("forwards"),pilot_budget.get("backward"),pilot_budget.get("updates"),pilot_budget.get("vae"))!=(17,3,3,0):
        return None
    parent=PILOT/"cycle_01"
    for file,digest in pilot_result["checkpoint_files"].items():
        if sha(parent/file)!=digest:raise RuntimeError(f"pilot checkpoint hash mismatch: {file}")
    paths={
        "runner":HERE/"run_v2.py","config":HERE/"config_v2.json",
        "taskbook":EXP/"taskbook_v2.md","pilot_runner":PILOT_CODE/"run_pilot.py",
        "pilot_result":result_path,"pilot_budget":budget_path,
        "pilot_student_qkv":parent/"student_qkv.pt",
        "pilot_student_target":parent/"student_target.pt",
        "pilot_fake_qkv":parent/"fake_qkv.pt",
        "pilot_state":parent/"trainer_state.pt",
        "dmd_math":ROOT/"H3-World/code/causal/dmd.py",
        "runtime_dmd_math":FROZEN/"runtime/code/causal/dmd.py",
        "actual_dmd_import":FROZEN/"runtime/code/causal/dmd.py",
        "anyflow":FROZEN/"runtime/code/causal/anyflow.py",
        "qkv_adapter":FROZEN/"runtime/code/causal/pretrained_lora.py",
        "precision":FROZEN/"runtime/code/causal/h3_precision.py",
        "rollout_contract":CONTRACT/"rollout_contract.py",
        "student_entry":AFEXP/"interval_student.py",
        "af2_qkv":folder/"qkv.pt","af2_target":folder/"target_time.pt",
        "af2_state":folder/"trainer_state.pt",
        "af2_result":AF2/"result.json",
        "af3_AA":AF3/"AA/third_17_22.json",
        "af3_AD":AF3/"AD/third_17_22.json",
        "fm8_first":FM8/"first12.pt",
        "input_A":FROZEN/"source_coarse/inputs/parking_A.pt",
        "input_D":FROZEN/"source_coarse/inputs/parking_D.pt",
        "released_lora":FROZEN/"runtime/checkpoints/H3-World/step-10000.safetensors",
        "prefix_feedback":PREFIX/"current_prefix.py",
    }
    sources={k:{"path":str(p.relative_to(ROOT)),"sha256":sha(p)} for k,p in paths.items()}
    for key,file in (("af2_qkv","qkv.pt"),("af2_target","target_time.pt"),("af2_state","trainer_state.pt")):
        assert sources[key]["sha256"]==receipt["files"][file]
    data=dict(task=CFG["task"],af2_step=step,parent_cycle=1,sources=sources)
    if MANIFEST.exists():
        if json.loads(MANIFEST.read_text())!=data:raise RuntimeError("DMD continuation sources changed")
    elif create:atomic_json(MANIFEST,data)
    return data


def checkpoint_cpu(data):
    import torch
    def p(k):return ROOT/data["sources"][k]["path"]
    q=torch.load(p("pilot_student_qkv"),map_location="cpu",weights_only=True)
    t=torch.load(p("pilot_student_target"),map_location="cpu",weights_only=True)
    f=torch.load(p("pilot_fake_qkv"),map_location="cpu",weights_only=True)
    s=torch.load(p("pilot_state"),map_location="cpu",weights_only=True)
    step=data["af2_step"]
    metadata=s["metadata"]
    assert metadata==q["metadata"]==t["metadata"]==f["metadata"]
    assert metadata["task"]=="EXP-008/v1-DMD-PILOT" and metadata["parent_af2_step"]==step
    assert metadata["protocol"]=="V3 strict causal/global/current-prefix/Single I0/12+5/sigma0 own-role KV"
    for state in (q,f):
        assert state["format"]=="h3_causal_tail_qkv_v2"
        assert state["block_indices"]==list(range(42,50))
        assert state["rank"]==CFG["rank"]==8 and state["scale"]==1/CFG["rank"]
    assert t["format"]=="h3world_anyflow_time_v1" and t["gate"]==.25
    assert t["velocity_convention"]=="noise-clean"
    for key,expected in (("student_optimizer",1),("fake_optimizer",2)):
        assert s[key]["state"] and {int(v["step"]) for v in s[key]["state"].values()}=={expected}
    assert torch.is_tensor(s["cpu_rng"]) and len(s["cuda_rng"])==3
    assert torch.is_tensor(s["train_noise_rng"]) and torch.is_tensor(s["score_noise_rng"])
    return q,t,f,s


def cpu_preflight():
    import torch
    import infer as abot
    from causal.anyflow import install_anyflow,save_anyflow,finite_map_step
    from causal.pretrained_lora import install_adapters,save_adapters
    from causal.h3_cached import H3ChunkCache
    from causal.h3_precision import configure_precision
    from causal.local_topology import visible_inputs
    from causal.anyflow_sampling import configure_video_schedule
    from diffsynth.diffusion.flow_match import FlowMatchScheduler
    from causal.dmd import score_sample,critic_flow_loss,distribution_matching_loss
    import causal.dmd as dmd_module
    from current_prefix import current_prefix_feedback
    from interval_student import interval_student
    from rollout_contract import prompt_for_path
    from chunk_plan import cache_identity
    assert (CFG["max_forwards"],CFG["max_backward"],CFG["max_updates"],CFG["max_vae"])==(99,14,14,0)
    assert (CFG["first_cycle"],CFG["last_cycle"])==(2,8)
    assert len(set(CFG["physical_gpu_ids"]))==3
    assert (CFG["training_noise_seed"],CFG["score_noise_seed"])==(170008,170108)
    data=manifest(create=True)
    if data is None:
        print(json.dumps(dict(task=CFG["task"],status="WAITING_FOR_PILOT_AND_AF3",gpu_calls=0)))
        return
    _,_,_,parent=checkpoint_cpu(data)
    # Exercise PyTorch optimizer restoration with the exact saved parameter
    # ordering and shapes, without loading a 33B model or touching CUDA.
    for key,expected in (("student_optimizer",1),("fake_optimizer",2)):
        state=parent[key]
        ids=state["param_groups"][0]["params"]
        params=[torch.nn.Parameter(torch.zeros_like(state["state"][i]["exp_avg"])) for i in ids]
        opt=torch.optim.AdamW(params,lr=CFG["student_lr"] if key.startswith("student") else CFG["fake_lr"])
        opt.load_state_dict(state)
        assert len(opt.state)==len(ids)
        assert {int(v["step"]) for v in opt.state.values()}=={expected}
        del opt,params
    train_a=torch.Generator();train_a.set_state(parent["train_noise_rng"])
    train_b=torch.Generator();train_b.set_state(parent["train_noise_rng"])
    assert torch.equal(torch.randn((1,24,5,30,52),generator=train_a),
                       torch.randn((1,24,5,30,52),generator=train_b))
    assert ["AD" if c%2==0 else "AA" for c in range(2,9)]==["AD","AA","AD","AA","AD","AA","AD"]
    actual_dmd=Path(dmd_module.__file__).resolve()
    expected=ROOT/data["sources"]["actual_dmd_import"]["path"]
    assert actual_dmd==expected.resolve()
    assert sha(actual_dmd)==data["sources"]["actual_dmd_import"]["sha256"]
    x=torch.load(FM8/"first12.pt",map_location="cpu",weights_only=True)
    assert x.shape==(1,24,12,30,52)
    print(json.dumps(dict(task=CFG["task"],status="CPU_PASS_NO_GPU_AUTHORIZATION",
                          af2_step=data["af2_step"],parent_cycle=data["parent_cycle"],sources=len(data["sources"]),
                          actual_dmd_import=str(actual_dmd.relative_to(ROOT)),gpu_calls=0)))


def cpu_tree(x):
    import torch
    if torch.is_tensor(x):return x.detach().cpu()
    if isinstance(x,dict):return {k:cpu_tree(v) for k,v in x.items()}
    if isinstance(x,list):return [cpu_tree(v) for v in x]
    return x


def pilot(physical:list[int],ledger:dict,data:dict):
    import torch
    import infer as abot
    from causal.anyflow import install_anyflow,save_anyflow,finite_map_step
    from causal.pretrained_lora import install_adapters,save_adapters
    from causal.h3_cached import H3ChunkCache
    from causal.h3_precision import configure_precision
    from causal.local_topology import visible_inputs
    from causal.anyflow_sampling import configure_video_schedule
    from diffsynth.diffusion.flow_match import FlowMatchScheduler
    from causal.dmd import score_sample,critic_flow_loss,distribution_matching_loss
    import causal.dmd as dmd_module
    from current_prefix import current_prefix_feedback
    from interval_student import interval_student
    from rollout_contract import prompt_for_path
    from chunk_plan import cache_identity

    torch.set_num_threads(4)
    torch.manual_seed(CFG["training_noise_seed"])
    torch.cuda.manual_seed_all(CFG["training_noise_seed"])
    actual_dmd=Path(dmd_module.__file__).resolve()
    expected_dmd=ROOT/data["sources"]["actual_dmd_import"]["path"]
    if actual_dmd!=expected_dmd.resolve() or sha(actual_dmd)!=data["sources"]["actual_dmd_import"]["sha256"]:
        raise RuntimeError("actual imported causal.dmd differs from frozen source")
    started=time.time()
    deadline=min(started+CFG["max_wall_minutes"]*60,
                 datetime.fromisoformat(CFG["deadline_hkt"]).timestamp())
    result=dict(task=CFG["task"],status="loading",physical_gpu_ids=physical,
                af2_step=data["af2_step"],parent_cycle=1,last_complete_cycle=1,
                source_manifest_sha256=sha(MANIFEST),runner_sha256=sha(__file__),
                cycles=[],checkpoint_paths=[])
    result["actual_dmd_import"]={"path":str(actual_dmd.relative_to(ROOT)),"sha256":sha(actual_dmd)}
    atomic_json(OUT/"result.json",result)
    devices={"teacher":"cuda:0","fake":"cuda:1","student":"cuda:2"}
    def limits():
        elapsed=time.time()-ledger["start"]
        if time.time()>=deadline or 3*elapsed>=CFG["max_three_gpu_hours"]*3600:
            raise TimeoutError("DMD three-GPU conservative budget")
        for i in range(3):
            peak=torch.cuda.max_memory_allocated(i)/2**30
            if peak>CFG["max_allocated_gib_per_gpu"]:
                raise MemoryError(f"DMD role GPU{i} allocated {peak:.3f}GiB")
    def reserve(kind,role,detail):
        limits()
        key,cap={"forward":("forwards",99),"backward":("backward",14),
                 "update":("updates",14)}[kind]
        if ledger[key]>=cap:raise RuntimeError(f"DMD {kind} cap")
        ledger[key]+=1
        entry=dict(kind=kind,role=role,detail=detail,ordinal=ledger[key],time=time.time())
        ledger["events"].append(entry)
        atomic_json(OUT/"budget.json",ledger)
        return entry
    def move(x,dev):
        if torch.is_tensor(x):return x.to(dev)
        if isinstance(x,dict):return {k:move(v,dev) for k,v in x.items()}
        return x
    try:
        q_state,t_state,f_state,parent_state=checkpoint_cpu(data)
        inputs={a:torch.load(FROZEN/f"source_coarse/inputs/parking_{a}.pt",map_location="cpu",weights_only=True) for a in "AD"}
        for key in ("initial_noise","audio_noise","anchor"):
            assert torch.equal(inputs["A"][key],inputs["D"][key])
        first_cpu=torch.load(FM8/"first12.pt",map_location="cpu",weights_only=True).float()
        gen=torch.Generator().manual_seed(CFG["training_noise_seed"])
        score_gen=torch.Generator().manual_seed(CFG["score_noise_seed"])
        roles={}
        for role in ("teacher","fake","student"):
            dev=devices[role]
            with torch.cuda.device(dev):
                pipe=abot.load_pipeline(dev)
                pipe.load_lora(pipe.dit,state_dict=abot.load_checkpoint_lora(FROZEN/"runtime/checkpoints/H3-World/step-10000.safetensors"),hotload=True)
                model=pipe.dit.eval().requires_grad_(False)
                precision=configure_precision(model,"h3_fp32",
                    native_transformer_dir=FROZEN/"runtime/DiffSynth-Studio-h3-v2/models/MiniMax/MiniMax-H3/FL2VA/transformer")
                pipe.load_models_to_device(["dit"])
                base_versions=[(p,p._version) for p in model.parameters()]
                adapters=None;target=None
                if role!="teacher":
                    adapters,indices=install_adapters(model,rank=8,block_indices=tuple(range(42,50)),device=dev)
                    if role=="student":
                        target=install_anyflow(model,device=dev,gate=.25)
                    state=q_state if role=="student" else f_state
                    with torch.no_grad():
                        for a,aa,bb in zip(adapters,state["lora_A"],state["lora_B"]):
                            a.lora_A.copy_(aa.to(dev));a.lora_B.copy_(bb.to(dev))
                    if role=="student":
                        target.load_state_dict(t_state["weights"],strict=True)
                first=first_cpu.to(dev)
                full=move(inputs["A"]["packed"],dev)
                text12=prompt_for_path(inputs,"AA",12).to(dev)
                packed12,text12=visible_inputs(full,text12,12,390)
                current={}
                for path in ("AA","AD"):
                    text17=prompt_for_path(inputs,path,17).to(dev)
                    packed17,text17=visible_inputs(full,text17,17,390)
                    current[path]=(packed17,text17)
                cache=H3ChunkCache(5,"cpu")
                roles[role]=dict(model=model,pipe=pipe,device=dev,first=first,
                    packed={0:packed12,1:{p:current[p][0] for p in current}},
                    prompt={0:text12,1:{p:current[p][1] for p in current}},
                    anchor=inputs["A"]["anchor"].to(dev),audio=inputs["A"]["audio_noise"].to(dev),
                    cache=cache,adapters=adapters,target=target,base_versions=base_versions,
                    precision=precision)
                limits()
        result["role_model_loaded"]={role:{"device":roles[role]["device"],
            "precision":roles[role]["precision"]["profile"],
            "has_target":roles[role]["target"] is not None} for role in roles}
        atomic_json(OUT/"result.json",result)
        student=roles["student"]
        fake=roles["fake"]
        student_params=[p for a in student["adapters"] for p in (a.lora_A,a.lora_B)]+list(student["target"].parameters())
        fake_params=[p for a in fake["adapters"] for p in (a.lora_A,a.lora_B)]
        student_opt=torch.optim.AdamW(student_params,lr=CFG["student_lr"],betas=(.9,.95),weight_decay=.01)
        fake_opt=torch.optim.AdamW(fake_params,lr=CFG["fake_lr"],betas=(.9,.95),weight_decay=.01)
        student_opt.load_state_dict(parent_state["student_optimizer"])
        fake_opt.load_state_dict(parent_state["fake_optimizer"])
        for opt,expected in ((student_opt,1),(fake_opt,2)):
            assert {int(v["step"]) for v in opt.state.values()}=={expected}
            assert all(t.device==next(iter(opt.param_groups[0]["params"])).device
                       for state in opt.state.values() for key,t in state.items()
                       if key!="step" and torch.is_tensor(t))
        # No stochastic model/module/optimizer installation follows this restoration.
        torch.set_rng_state(parent_state["cpu_rng"])
        for i,state in enumerate(parent_state["cuda_rng"]):torch.cuda.set_rng_state(state,device=i)
        gen.set_state(parent_state["train_noise_rng"])
        score_gen.set_state(parent_state["score_noise_rng"])
        result["restored_rng"]={"cpu_sha256":hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
            "cuda_sha256":[hashlib.sha256(torch.cuda.get_rng_state(i).cpu().numpy().tobytes()).hexdigest()
                           for i in range(3)],
            "train_noise_sha256":hashlib.sha256(gen.get_state().numpy().tobytes()).hexdigest(),
            "score_noise_sha256":hashlib.sha256(score_gen.get_state().numpy().tobytes()).hexdigest()}
        for name,actual in (("cpu_rng",result["restored_rng"]["cpu_sha256"]),
                            ("train_noise_rng",result["restored_rng"]["train_noise_sha256"]),
                            ("score_noise_rng",result["restored_rng"]["score_noise_sha256"])):
            assert actual==hashlib.sha256(parent_state[name].numpy().tobytes()).hexdigest()
        for i,actual in enumerate(result["restored_rng"]["cuda_sha256"]):
            assert actual==hashlib.sha256(parent_state["cuda_rng"][i].cpu().numpy().tobytes()).hexdigest()
        atomic_json(OUT/"result.json",result)

        def call(role,index,x,sigma,r=None,*,commit=False,grad=False,path=None,cycle=None):
            obj=roles[role];dev=obj["device"]
            reserve("forward",role,dict(cycle=cycle,path=path,index=index,sigma=float(sigma),
                                        r=r,commit=commit,grad=grad))
            packed=obj["packed"][0] if index==0 else obj["packed"][1][path]
            prompt=obj["prompt"][0] if index==0 else obj["prompt"][1][path]
            with torch.cuda.device(dev):
                y=interval_student(obj["model"],x,start=0 if index==0 else 12,index=index,
                    cache=obj["cache"],full_packed=packed,prompt=prompt,
                    anchor=obj["anchor"],audio=obj["audio"],sigma=sigma,target_sigma=r,
                    commit=commit,use_gradient_checkpointing=grad,
                    use_gradient_checkpointing_offload=grad)
                if not torch.isfinite(y).all():raise FloatingPointError(f"DMD {role} nonfinite velocity")
                limits()
                return y
        def prefill(role,cycle):
            obj=roles[role]
            obj["cache"].clear();obj["cache"]=H3ChunkCache(5,"cpu")
            with torch.no_grad():
                call(role,0,obj["first"],0,0 if role=="student" else None,
                     commit=True,cycle=cycle)
            assert obj["cache"].commits==50
            return cache_identity(obj["cache"])
        def checkpoint(cycle):
            directory=OUT/f"cycle_{cycle:02d}"
            directory.mkdir(parents=True,exist_ok=False)
            metadata=dict(task=CFG["task"],cycle=cycle,parent_pilot_result_sha256=data["sources"]["pilot_result"]["sha256"],
                source_manifest_sha256=sha(MANIFEST),
                protocol="V3 strict causal/global/current-prefix/Single I0/12+5/sigma0 own-role KV")
            save_adapters(directory/"student_qkv.pt",student["adapters"],list(range(42,50)),metadata)
            save_anyflow(directory/"student_target.pt",student["target"],metadata)
            save_adapters(directory/"fake_qkv.pt",fake["adapters"],list(range(42,50)),metadata)
            state=dict(metadata=metadata,student_optimizer=cpu_tree(student_opt.state_dict()),
                fake_optimizer=cpu_tree(fake_opt.state_dict()),cpu_rng=torch.get_rng_state(),
                cuda_rng=[torch.cuda.get_rng_state(i) for i in range(3)],
                train_noise_rng=gen.get_state(),score_noise_rng=score_gen.get_state())
            torch.save(state,directory/"trainer_state.pt")
            files={f.name:sha(f) for f in directory.iterdir() if f.is_file()}
            receipt=dict(cycle=cycle,path=str(directory.relative_to(ROOT)),files=files,
                student_optimizer_steps=sorted({int(v["step"]) for v in student_opt.state.values()}),
                fake_optimizer_steps=sorted({int(v["step"]) for v in fake_opt.state.values()}))
            assert receipt["student_optimizer_steps"]==[cycle]
            assert receipt["fake_optimizer_steps"]==[cycle+1]
            result["checkpoint_paths"].append(receipt)
            atomic_json(OUT/"result.json",result)
            return receipt
        sigmas=configure_video_schedule(FlowMatchScheduler("MiniMax-H3"),steps=8,grid="native",flow_shift=2.22)
        assert sigmas==json.loads((FM8/"chunk_0_12.json").read_text())["sigmas"]
        with current_prefix_feedback():
            teacher_signature=prefill("teacher",1)
            previous_cycle_seconds=None
            for cycle in range(CFG["first_cycle"],CFG["last_cycle"]+1):
                limits()
                # Leave enough wall time for one complete cycle and a checkpoint.
                remaining=min(deadline-time.time(),CFG["max_three_gpu_hours"]*1200-(time.time()-ledger["start"]))
                required=max(240.0,1.25*previous_cycle_seconds) if previous_cycle_seconds else 240.0
                if remaining<required or ledger["forwards"]+14>CFG["max_forwards"]:
                    result["stop_reason"]=f"insufficient remaining budget before cycle {cycle}: {remaining:.1f}s"
                    break
                began_cycle=time.time()
                path="AD" if cycle%2==0 else "AA"
                action=path[-1]
                row=dict(cycle=cycle,path=path,action=action,status="running",forwards_before=ledger["forwards"])
                result["active_cycle"]=row
                atomic_json(OUT/"result.json",result)
                student_opt.zero_grad(set_to_none=True)
                fake_opt.zero_grad(set_to_none=True)
                student_signature=prefill("student",cycle)
                fake_signature=prefill("fake",cycle)
                assert cache_identity(roles["teacher"]["cache"])==teacher_signature
                assert len({roles[r]["cache"].layers[0][0].key.data_ptr() for r in roles})==3
                c2_noise_cpu=torch.randn(inputs["A"]["initial_noise"][:,:,12:17].shape,
                                         generator=gen,dtype=torch.float32)
                row["training_noise_sha256"]=hashlib.sha256(c2_noise_cpu.numpy().tobytes()).hexdigest()
                x=c2_noise_cpu.to(student["device"])
                velocities=[]
                for sigma,r in zip(sigmas[:-1],sigmas[1:]):
                    v=call("student",1,x,sigma,r,grad=True,path=path,cycle=cycle)
                    v.retain_grad();velocities.append(v)
                    x=finite_map_step(x,v,sigma,r)
                endpoint=x
                assert endpoint.requires_grad and cache_identity(student["cache"])==student_signature
                sigma=CFG["score_sigma"]
                fake_weights_before=[p.detach().clone() for p in fake_params]
                noise=torch.randn(endpoint.shape,generator=score_gen,dtype=torch.float32).to(fake["device"])
                z=score_sample(endpoint.detach().to(fake["device"]),noise,sigma)
                pred=call("fake",1,z,sigma,grad=True,path=path,cycle=cycle)
                fake_loss=critic_flow_loss(pred,endpoint.detach().to(fake["device"]),noise)
                if not torch.isfinite(fake_loss):raise FloatingPointError("nonfinite fake FM loss")
                reserve("backward","fake",dict(cycle=cycle,path=path))
                with torch.cuda.device(fake["device"]):fake_loss.backward()
                limits()
                assert cache_identity(fake["cache"])==fake_signature
                assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in fake_params)
                assert all(p.grad is None for p in student_params)
                fake_grad_norm=float(torch.linalg.vector_norm(torch.stack([p.grad.float().norm() for p in fake_params])))
                assert fake_grad_norm>0
                with torch.cuda.device(fake["device"]):
                    fake_clip=float(torch.nn.utils.clip_grad_norm_(fake_params,CFG["grad_clip_norm"]))
                reserve("update","fake",dict(cycle=cycle,path=path))
                with torch.cuda.device(fake["device"]):fake_opt.step()
                limits()
                fake_changed=any(not torch.equal(p,b) for p,b in zip(fake_params,fake_weights_before))
                assert fake_changed and cache_identity(fake["cache"])==fake_signature
                fake_signature=prefill("fake",cycle)
                fake_opt.zero_grad(set_to_none=True)
                score_noise=torch.randn(endpoint.shape,generator=score_gen,dtype=torch.float32).to(student["device"])
                z_student=score_sample(endpoint,score_noise,sigma)
                with torch.no_grad():
                    real=call("teacher",1,z_student.to(roles["teacher"]["device"]),sigma,
                              path=path,cycle=cycle)
                    fake_v=call("fake",1,z_student.to(fake["device"]),sigma,path=path,cycle=cycle)
                dmd_loss,direction,stats=distribution_matching_loss(endpoint,z_student,sigma,
                    real_velocity=real.detach().to(student["device"]),
                    fake_velocity=fake_v.detach().to(student["device"]),normalize=True)
                assert torch.isfinite(dmd_loss) and direction.norm()>0
                student_weights_before=[p.detach().clone() for p in student_params]
                reserve("backward","student",dict(cycle=cycle,path=path,maps=8))
                with torch.cuda.device(student["device"]):dmd_loss.backward()
                limits()
                velocity_grad_norms=[float(v.grad.float().norm()) for v in velocities]
                assert len(velocity_grad_norms)==8 and all(n>0 for n in velocity_grad_norms)
                assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in student_params)
                assert all(p.grad is None for p in fake_params)
                target_count=len(list(student["target"].parameters()))
                target_grad=float(torch.linalg.vector_norm(torch.stack(
                    [p.grad.float().norm() for p in student_params[-target_count:]])))
                qkv_grad=float(torch.linalg.vector_norm(torch.stack(
                    [p.grad.float().norm() for p in student_params[:-target_count]])))
                assert target_grad>0 and qkv_grad>0
                with torch.cuda.device(student["device"]):
                    student_clip=float(torch.nn.utils.clip_grad_norm_(student_params,CFG["grad_clip_norm"]))
                reserve("update","student",dict(cycle=cycle,path=path,maps=8))
                with torch.cuda.device(student["device"]):student_opt.step()
                limits()
                qkv_changed=any(not torch.equal(p,b) for p,b in zip(
                    student_params[:-target_count],student_weights_before[:-target_count]))
                target_changed=any(not torch.equal(p,b) for p,b in zip(
                    student_params[-target_count:],student_weights_before[-target_count:]))
                assert qkv_changed and target_changed
                assert {int(v["step"]) for v in student_opt.state.values()}=={cycle}
                assert {int(v["step"]) for v in fake_opt.state.values()}=={cycle+1}
                assert cache_identity(roles["teacher"]["cache"])==teacher_signature
                assert cache_identity(fake["cache"])==fake_signature
                assert cache_identity(student["cache"])==student_signature
                assert all(p._version==version for obj in roles.values() for p,version in obj["base_versions"])
                student["cache"].clear();student["cache"]=H3ChunkCache(5,"cpu")
                row.update(status="complete",fake_loss=float(fake_loss.detach()),
                    dmd_loss=float(dmd_loss.detach()),direction_stats=stats,
                    fake_grad_norm=fake_grad_norm,student_target_grad_norm=target_grad,
                    student_qkv_grad_norm=qkv_grad,velocity_grad_norms=velocity_grad_norms,
                    fake_clip_pre_norm=fake_clip,student_clip_pre_norm=student_clip,
                    fake_changed=fake_changed,student_qkv_changed=qkv_changed,
                    student_target_changed=target_changed,forwards=ledger["forwards"]-row["forwards_before"],
                    teacher_cache_reused_unchanged=True,student_cache_rebuilt_and_read_unchanged=True,
                    fake_cache_rebuilt_after_update_and_read_unchanged=True,base_weights_frozen=True,
                    peak_allocated_gib_by_role={role:torch.cuda.max_memory_allocated(i)/2**30
                        for i,role in enumerate(("teacher","fake","student"))},
                    seconds=time.time()-began_cycle)
                assert row["forwards"]==14
                result["cycles"].append(row)
                result["last_complete_cycle"]=cycle
                result.pop("active_cycle",None)
                atomic_json(OUT/"result.json",result)
                previous_cycle_seconds=row["seconds"]
                student_opt.zero_grad(set_to_none=True)
                del velocities,x,endpoint,v,pred,z,noise,score_noise,z_student,real,fake_v,dmd_loss,direction,c2_noise_cpu
                if cycle in (4,8):checkpoint(cycle)
                limits()
            last=result["last_complete_cycle"]
            if last>1 and last not in [r["cycle"] for r in result["checkpoint_paths"]]:checkpoint(last)
        result["status"]=("complete_8_pending_judge" if result["last_complete_cycle"]==8
                          else "budget_stopped_pending_judge")
        result["wall_seconds"]=time.time()-started
        result["conservative_three_gpu_hours"]=3*result["wall_seconds"]/3600
        result["peak_allocated_gib_by_role"]={role:torch.cuda.max_memory_allocated(i)/2**30
                                              for i,role in enumerate(("teacher","fake","student"))}
        atomic_json(OUT/"result.json",result)
    except BaseException as exc:
        result["status"]="failed";result["error"]=repr(exc)
        result["wall_seconds"]=time.time()-started
        atomic_json(OUT/"result.json",result)
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight",action="store_true")
    args=parser.parse_args()
    setup_paths()
    if args.preflight:cpu_preflight();return
    data=manifest(create=False)
    if data is None or not MANIFEST.exists():raise RuntimeError("pilot checkpoint and AF3 evidence required")
    marker=HERE/"TRAIN_GPU_AUTHORIZATION.json"
    if not marker.exists():raise RuntimeError("Judge DMD continuation GPU authorization absent")
    permit=json.loads(marker.read_text())
    if not (permit.get("approved") and permit.get("af2_step")==data["af2_step"]
            and permit.get("parent_cycle")==1 and permit.get("last_cycle")==8
            and permit.get("source_manifest_sha256")==sha(MANIFEST)):
        raise RuntimeError("DMD continuation authorization/source mismatch")
    physical=CFG["physical_gpu_ids"]
    if permit.get("physical_gpu_ids")!=physical:raise RuntimeError("DMD continuation GPU assignment mismatch")
    if len(set(physical))!=3:raise ValueError("DMD requires three distinct devices")
    free=[int(subprocess.check_output(["nvidia-smi",f"--id={gpu}",
        "--query-gpu=memory.free","--format=csv,noheader,nounits"],text=True).strip())
        for gpu in physical]
    if any(value<44000 for value in free):raise RuntimeError(f"DMD continuation GPU not idle: {free}")
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/"worker.lock").open("a+") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (OUT/"budget.json").exists():raise RuntimeError("DMD continuation prior attempt exists; no automatic retry")
        ledger=dict(task=CFG["task"],physical_gpu_ids=physical,start=time.time(),
                    forwards=0,backward=0,updates=0,vae=0,events=[])
        atomic_json(OUT/"budget.json",ledger)
        deadline=min(ledger["start"]+CFG["max_wall_minutes"]*60,
                     datetime.fromisoformat(CFG["deadline_hkt"]).timestamp())
        def alarm(_signum,_frame):raise TimeoutError("DMD pilot absolute wall cap")
        signal.signal(signal.SIGALRM,alarm)
        signal.setitimer(signal.ITIMER_REAL,deadline-time.time())
        os.environ.update(CUDA_VISIBLE_DEVICES=",".join(map(str,physical)),ABOT_VRAM_RESERVE_GIB="18",
            HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",DIFFSYNTH_SKIP_DOWNLOAD="True",
            TOKENIZERS_PARALLELISM="false",PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
        try:pilot(physical,ledger,data)
        finally:
            signal.setitimer(signal.ITIMER_REAL,0)
            ledger["end"]=time.time();ledger["wall_seconds"]=ledger["end"]-ledger["start"]
            ledger["conservative_three_gpu_hours"]=3*ledger["wall_seconds"]/3600
            atomic_json(OUT/"budget.json",ledger)


if __name__=="__main__":main()
