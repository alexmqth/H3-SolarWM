"""Prepared real-H3 DMD one-cycle pilot; no GPU permission is implied.

Three independently loaded roles occupy three devices. Student keeps the
actual eight-map graph while fake receives only detached on-policy endpoints.
GPU execution additionally requires AF3 video evidence and Judge's separate
authorization marker; without them only CPU WAITING preflight is available.
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
CFG=json.loads((HERE/"config_pilot_prep.json").read_text())
OUT=ROOT/CFG["output_root"]
AF2=ROOT/"H3-World/outputs/EXP-007_v3_anyflow_af2"
AF3=ROOT/"H3-World/outputs/EXP-007_v3_anyflow_af3"
FM8=ROOT/"H3-World/outputs/EXP-006_v3_fm8_full"
FROZEN=ROOT/"H3-World/outputs/2026-10-09-22/chunk_partition_cb"
SW=EXP.parent/"EXP-005_v3_sliding_window"
PREFIX=ROOT/"submission/reports/stage1_anyflow/02_causal_diagnostics/current_prefix_candidate"
CONTRACT=EXP.parent/"EXP-001_v2b_124"
MANIFEST=HERE/"source_manifest_pilot.json"


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
    paths={
        "runner":HERE/"run_pilot.py","config":HERE/"config_pilot_prep.json",
        "taskbook":EXP/"taskbook_v1.md","draft":HERE/"GPU_PILOT_DRAFT.md",
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
    data=dict(task=CFG["task"],af2_step=step,sources=sources)
    if MANIFEST.exists():
        if json.loads(MANIFEST.read_text())!=data:raise RuntimeError("DMD pilot sources changed")
    elif create:atomic_json(MANIFEST,data)
    return data


def checkpoint_cpu(data):
    import torch
    def p(k):return ROOT/data["sources"][k]["path"]
    q=torch.load(p("af2_qkv"),map_location="cpu",weights_only=True)
    t=torch.load(p("af2_target"),map_location="cpu",weights_only=True)
    s=torch.load(p("af2_state"),map_location="cpu",weights_only=True)
    step=data["af2_step"]
    assert s["task"]==q["metadata"]["task"]==t["metadata"]["task"]=="EXP-007/v3-AF2"
    assert s["step"]==q["metadata"]["step"]==t["metadata"]["step"]==step
    assert q["format"]=="h3_causal_tail_qkv_v2" and q["block_indices"]==list(range(42,50))
    assert q["rank"]==CFG["rank"]==8 and q["scale"]==1/CFG["rank"]
    assert t["format"]=="h3world_anyflow_time_v1" and t["gate"]==.25
    assert t["velocity_convention"]=="noise-clean"
    assert s["qkv_sha256"]==data["sources"]["af2_qkv"]["sha256"]
    assert s["target_sha256"]==data["sources"]["af2_target"]["sha256"]
    assert s["metadata"]==q["metadata"]==t["metadata"]
    assert s["metadata"]["protocol"]=="V3 strict causal/global/current-prefix/Single I0/12+5/sigma0 student KV"
    assert s["metadata"]["precision"]=="h3_fp32"
    return q,t,s


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
    assert (CFG["max_forwards"],CFG["max_backward"],CFG["max_updates"],CFG["max_vae"])==(17,3,3,0)
    assert len(set(CFG["physical_gpu_ids"]))==3
    assert CFG["training_noise_seed"]==170008 and CFG["action_path"]=="AA"
    data=manifest(create=True)
    if data is None:
        print(json.dumps(dict(task=CFG["task"],status="WAITING_FOR_AF2_AND_AF3",gpu_calls=0)))
        return
    checkpoint_cpu(data)
    actual_dmd=Path(dmd_module.__file__).resolve()
    expected=ROOT/data["sources"]["actual_dmd_import"]["path"]
    assert actual_dmd==expected.resolve()
    assert sha(actual_dmd)==data["sources"]["actual_dmd_import"]["sha256"]
    x=torch.load(FM8/"first12.pt",map_location="cpu",weights_only=True)
    assert x.shape==(1,24,12,30,52)
    print(json.dumps(dict(task=CFG["task"],status="CPU_PASS_NO_GPU_AUTHORIZATION",
                          af2_step=data["af2_step"],sources=len(data["sources"]),
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
    initial_rng=dict(cpu_sha256=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
        cuda_sha256=[hashlib.sha256(torch.cuda.get_rng_state(i).cpu().numpy().tobytes()).hexdigest()
                     for i in range(3)])
    started=time.time()
    deadline=min(started+CFG["max_wall_minutes"]*60,
                 datetime.fromisoformat(CFG["deadline_hkt"]).timestamp())
    result=dict(task=CFG["task"],status="loading",physical_gpu_ids=physical,
                af2_step=data["af2_step"],source_manifest_sha256=sha(MANIFEST),
                runner_sha256=sha(__file__),calls=[],backward_roles=[],update_roles=[])
    result["rng_initial"]={"seed":CFG["training_noise_seed"],**initial_rng}
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
        key,cap={"forward":("forwards",17),"backward":("backward",3),
                 "update":("updates",3)}[kind]
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
        q_state,t_state,_=checkpoint_cpu(data)
        inputs={a:torch.load(FROZEN/f"source_coarse/inputs/parking_{a}.pt",map_location="cpu",weights_only=True) for a in "AD"}
        for key in ("initial_noise","audio_noise","anchor"):
            assert torch.equal(inputs["A"][key],inputs["D"][key])
        first_cpu=torch.load(FM8/"first12.pt",map_location="cpu",weights_only=True).float()
        gen=torch.Generator().manual_seed(CFG["training_noise_seed"])
        c2_noise_cpu=torch.randn(inputs["A"]["initial_noise"][:,:,12:17].shape,generator=gen,dtype=torch.float32)
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
                        with torch.no_grad():
                            for a,aa,bb in zip(adapters,q_state["lora_A"],q_state["lora_B"]):
                                a.lora_A.copy_(aa.to(dev));a.lora_B.copy_(bb.to(dev))
                        target.load_state_dict(t_state["weights"],strict=True)
                first=first_cpu.to(dev)
                full=move(inputs["A"]["packed"],dev)
                text12=prompt_for_path(inputs,"AA",12).to(dev)
                text17=prompt_for_path(inputs,"AA",17).to(dev)
                packed12,text12=visible_inputs(full,text12,12,390)
                packed17,text17=visible_inputs(full,text17,17,390)
                cache=H3ChunkCache(5,"cpu")
                roles[role]=dict(model=model,pipe=pipe,device=dev,first=first,
                    packed={0:packed12,1:packed17},prompt={0:text12,1:text17},
                    anchor=inputs["A"]["anchor"].to(dev),audio=inputs["A"]["audio_noise"].to(dev),
                    cache=cache,adapters=adapters,target=target,base_versions=base_versions,
                    precision=precision)
                limits()
        result["role_model_loaded"]={role:{"device":roles[role]["device"],
            "precision":roles[role]["precision"]["profile"],
            "has_target":roles[role]["target"] is not None} for role in roles}
        atomic_json(OUT/"result.json",result)
        def call(role,index,x,sigma,r=None,*,commit=False,grad=False):
            obj=roles[role];dev=obj["device"]
            reserve("forward",role,dict(index=index,sigma=float(sigma),r=r,commit=commit,grad=grad))
            with torch.cuda.device(dev):
                y=interval_student(obj["model"],x,start=0 if index==0 else 12,index=index,
                    cache=obj["cache"],full_packed=obj["packed"][index],prompt=obj["prompt"][index],
                    anchor=obj["anchor"],audio=obj["audio"],sigma=sigma,target_sigma=r,
                    commit=commit,use_gradient_checkpointing=grad,
                    use_gradient_checkpointing_offload=grad)
                if not torch.isfinite(y).all():raise FloatingPointError(f"DMD {role} nonfinite velocity")
                limits()
                return y
        def prefill(role):
            obj=roles[role]
            obj["cache"].clear()
            obj["cache"]=H3ChunkCache(5,"cpu")
            with torch.no_grad():
                call(role,0,obj["first"],0,0 if role=="student" else None,commit=True)
            assert obj["cache"].commits==50
        with current_prefix_feedback():
            for role in ("teacher","fake","student"):prefill(role)
            cache_signatures={role:cache_identity(roles[role]["cache"]) for role in roles}
            assert len({roles[r]["cache"].layers[0][0].key.data_ptr() for r in roles})==3
            student=roles["student"]
            student_params=[p for a in student["adapters"] for p in (a.lora_A,a.lora_B)]+list(student["target"].parameters())
            fake=roles["fake"]
            fake_params=[p for a in fake["adapters"] for p in (a.lora_A,a.lora_B)]
            student_opt=torch.optim.AdamW(student_params,lr=CFG["student_lr"],betas=(.9,.95),weight_decay=.01)
            fake_opt=torch.optim.AdamW(fake_params,lr=CFG["fake_lr"],betas=(.9,.95),weight_decay=.01)
            sigmas=configure_video_schedule(FlowMatchScheduler("MiniMax-H3"),steps=8,grid="native",flow_shift=2.22)
            frozen=json.loads((FM8/"chunk_0_12.json").read_text())["sigmas"]
            assert sigmas==frozen
            x=c2_noise_cpu.to(student["device"])
            velocities=[]
            for sigma,r in zip(sigmas[:-1],sigmas[1:]):
                v=call("student",1,x,sigma,r,grad=True)
                v.retain_grad();velocities.append(v)
                x=finite_map_step(x,v,sigma,r)
            endpoint=x
            assert endpoint.requires_grad and cache_identity(student["cache"])==cache_signatures["student"]
            result["student_eight_map_graph_live"]=True
            atomic_json(OUT/"result.json",result)
            sigma=CFG["score_sigma"]
            for warmup in (1,2):
                fake_opt.zero_grad(set_to_none=True)
                fake_cache_before=cache_identity(fake["cache"])
                fake_weights_before=[p.detach().clone() for p in fake_params]
                noise=torch.randn(endpoint.shape,generator=score_gen,dtype=torch.float32).to(fake["device"])
                z=score_sample(endpoint.detach().to(fake["device"]),noise,sigma)
                pred=call("fake",1,z,sigma,grad=True)
                loss=critic_flow_loss(pred,endpoint.detach().to(fake["device"]),noise)
                if not torch.isfinite(loss):raise FloatingPointError("nonfinite fake FM loss")
                reserve("backward","fake",warmup)
                with torch.cuda.device(fake["device"]):
                    loss.backward()
                limits()
                assert cache_identity(fake["cache"])==fake_cache_before
                assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in fake_params)
                assert all(p.grad is None for p in student_params)
                fake_grad_norm=float(torch.linalg.vector_norm(torch.stack(
                    [p.grad.float().norm() for p in fake_params])))
                assert fake_grad_norm>0
                with torch.cuda.device(fake["device"]):
                    fake_clip_pre_norm=float(torch.nn.utils.clip_grad_norm_(fake_params,CFG["grad_clip_norm"]))
                reserve("update","fake",warmup)
                with torch.cuda.device(fake["device"]):
                    fake_opt.step()
                limits()
                fake_changed=any(not torch.equal(p,b) for p,b in zip(fake_params,fake_weights_before))
                assert fake_changed and cache_identity(fake["cache"])==fake_cache_before
                # Updating fake invalidates its own C1 raw KV, not teacher/student KV.
                prefill("fake")
                result.setdefault("fake_warmups",[]).append(dict(index=warmup,loss=float(loss.detach()),
                    cache_rebuilt=True,cache_read_unchanged=True,grad_norm=fake_grad_norm,
                    clip_pre_norm=fake_clip_pre_norm,parameters_changed=fake_changed))
                atomic_json(OUT/"result.json",result)
            fake_opt.zero_grad(set_to_none=True)
            score_noise=torch.randn(endpoint.shape,generator=score_gen,dtype=torch.float32).to(student["device"])
            z_student=score_sample(endpoint,score_noise,sigma)
            z_teacher=z_student.to(roles["teacher"]["device"])
            z_fake=z_student.to(fake["device"])
            with torch.no_grad():
                real=call("teacher",1,z_teacher,sigma)
                fake_v=call("fake",1,z_fake,sigma)
            dmd_loss,direction,stats=distribution_matching_loss(endpoint,z_student,sigma,
                real_velocity=real.detach().to(student["device"]),
                fake_velocity=fake_v.detach().to(student["device"]),normalize=True)
            assert torch.isfinite(dmd_loss) and direction.norm()>0
            student_opt.zero_grad(set_to_none=True)
            student_weights_before=[p.detach().clone() for p in student_params]
            reserve("backward","student","DMD full 8-map")
            with torch.cuda.device(student["device"]):
                dmd_loss.backward()
            limits()
            velocity_grad_norms=[float(v.grad.float().norm()) for v in velocities]
            assert all(n>0 for n in velocity_grad_norms)
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in student_params)
            assert all(p.grad is None for p in fake_params)
            with torch.cuda.device(student["device"]):
                student_clip_pre_norm=float(torch.nn.utils.clip_grad_norm_(student_params,CFG["grad_clip_norm"]))
            reserve("update","student","DMD full 8-map")
            with torch.cuda.device(student["device"]):
                student_opt.step()
            limits()
            target_count=len(list(student["target"].parameters()))
            qkv_changed=any(not torch.equal(p,b) for p,b in zip(
                student_params[:-target_count],student_weights_before[:-target_count]))
            target_changed=any(not torch.equal(p,b) for p,b in zip(
                student_params[-target_count:],student_weights_before[-target_count:]))
            assert qkv_changed and target_changed
            assert cache_identity(roles["teacher"]["cache"])==cache_signatures["teacher"]
            assert cache_identity(student["cache"])==cache_signatures["student"]
            assert all(p._version==version for role in roles.values() for p,version in role["base_versions"])
            result.update(dmd_loss=float(dmd_loss.detach()),direction_stats=stats,
                velocity_grad_norms=velocity_grad_norms,
                student_clip_pre_norm=student_clip_pre_norm,
                student_qkv_parameters_changed=qkv_changed,
                student_target_parameters_changed=target_changed,
                student_target_grad_norm=float(torch.linalg.vector_norm(torch.stack(
                    [p.grad.float().norm() for p in student["target"].parameters()]))),
                student_qkv_grad_norm=float(torch.linalg.vector_norm(torch.stack(
                    [p.grad.float().norm() for p in student_params[:-len(list(student["target"].parameters()))]]))))
            assert result["student_target_grad_norm"]>0 and result["student_qkv_grad_norm"]>0
            OUT.mkdir(parents=True,exist_ok=True)
            ckpt=OUT/"cycle_01";ckpt.mkdir(exist_ok=False)
            metadata=dict(task=CFG["task"],parent_af2_step=data["af2_step"],
                source_manifest_sha256=sha(MANIFEST),protocol="V3 strict causal/global/current-prefix/Single I0/12+5/sigma0 own-role KV")
            save_adapters(ckpt/"student_qkv.pt",student["adapters"],list(range(42,50)),metadata)
            save_anyflow(ckpt/"student_target.pt",student["target"],metadata)
            save_adapters(ckpt/"fake_qkv.pt",fake["adapters"],list(range(42,50)),metadata)
            state=dict(metadata=metadata,student_optimizer=cpu_tree(student_opt.state_dict()),
                fake_optimizer=cpu_tree(fake_opt.state_dict()),cpu_rng=torch.get_rng_state(),
                cuda_rng=[torch.cuda.get_rng_state(i) for i in range(3)],
                train_noise_rng=gen.get_state(),score_noise_rng=score_gen.get_state())
            torch.save(state,ckpt/"trainer_state.pt")
            result["checkpoint_files"]={p.name:sha(p) for p in ckpt.iterdir() if p.is_file()}
            limits()
        result["status"]="complete_pending_judge"
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
    if data is None or not MANIFEST.exists():raise RuntimeError("AF2 final and AF3 video evidence required")
    marker=HERE/"DMD_GPU_AUTHORIZATION.json"
    if not marker.exists():raise RuntimeError("Judge DMD GPU authorization absent")
    permit=json.loads(marker.read_text())
    if not permit.get("approved") or permit.get("af2_step")!=data["af2_step"]:
        raise RuntimeError("DMD authorization checkpoint mismatch")
    physical=CFG["physical_gpu_ids"]
    if len(set(physical))!=3:raise ValueError("DMD requires three distinct devices")
    free=[int(subprocess.check_output(["nvidia-smi",f"--id={gpu}",
        "--query-gpu=memory.free","--format=csv,noheader,nounits"],text=True).strip())
        for gpu in physical]
    if any(value<44000 for value in free):raise RuntimeError(f"DMD GPU not idle: {free}")
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/"worker.lock").open("a+") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (OUT/"budget.json").exists():raise RuntimeError("DMD pilot prior attempt exists; no automatic retry")
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
