"""EXP-023 frozen 8-NFE baseline / 128-update FM fit / matched final evaluation.

Run in order: --phase pre, --phase train, --phase post. No implicit retries.
The route context deliberately encloses the entire differentiable forward and
backward; a first-step hook gates optimizer.step on real H3 recomputation.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from common import HERE, ROOT, ENCODED, inputs, sha, write, rss_gib

EXP019 = ROOT / "submission/experiments/EXP-019_v3_mixed_action_validation"
sys.path.insert(0, str(EXP019))


def verify_run_sources():
    path=HERE/"TRAIN_SOURCE_MANIFEST.json"
    if not path.exists(): raise RuntimeError("frozen train source manifest required")
    manifest=json.loads(path.read_text())
    for relative,expected in manifest.items():
        if sha(ROOT/relative)!=expected:
            raise RuntimeError(f"training source changed: {relative}")
    return sha(path)


def cpu_tree(obj):
    import torch
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: cpu_tree(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [cpu_tree(x) for x in obj]
    return obj


class Budget:
    ORDER = ("encode_pilot", "encode_rest", "eval_pre", "train", "eval_post")
    CAPS = {"dit": 440, "backward": 128, "update": 128,
            "vae_encode": 70, "vae_decode": 12, "text": 600}

    def __init__(self, phase, gpu):
        import torch
        self.phase, self.gpu = phase, gpu
        self.path = HERE / f"{phase.upper()}_BUDGET.json"
        if self.path.exists():
            raise FileExistsError(self.path)
        self.previous = []
        for p in self.ORDER[:self.ORDER.index(phase)]:
            item = json.loads((HERE / f"{p.upper()}_BUDGET.json").read_text())
            if item["status"] != "COMPLETE_PENDING_JUDGE":
                raise RuntimeError(f"previous phase {p}: {item['status']}")
            self.previous.append(item)
        self.started = time.monotonic()
        self.wall_cap = {"eval_pre": 1800, "train": 6000, "eval_post": 1800}[phase]
        self.counts = {k: 0 for k in self.CAPS}
        self.events = []
        self.status = "RUNNING"
        self.save()

    def totals(self):
        return {k: self.counts[k] + sum(x["counts"][k] for x in self.previous)
                for k in self.CAPS}

    def save(self, **extra):
        import torch
        elapsed = time.monotonic() - self.started
        total_seconds = elapsed + sum(x["wall_seconds"] for x in self.previous)
        value = dict(task="EXP-023/v1", phase=self.phase, status=self.status,
                     gpu=self.gpu, wall_seconds=elapsed, gpu_seconds_total=total_seconds,
                     counts=self.counts.copy(), cumulative_counts=self.totals(),
                     limits=dict(phase_seconds=self.wall_cap, total_seconds=13800,
                                 allocated_gib=44, **self.CAPS),
                     peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                     peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                     rss_gib=rss_gib(), events=self.events, **extra)
        write(self.path, value)
        write(HERE / "CUMULATIVE_BUDGET.json", dict(task="EXP-023/v1",
              active_phase=self.phase, active_status=self.status,
              gpu_process_seconds_sum=total_seconds, counts=self.totals(),
              phase_seconds={x["phase"]: x["wall_seconds"] for x in self.previous}
                            | {self.phase: elapsed}))

    def reserve(self, kind, note):
        elapsed = time.monotonic()-self.started
        if elapsed >= self.wall_cap-30:
            raise TimeoutError(f"{self.phase} wall budget")
        if elapsed + sum(x["wall_seconds"] for x in self.previous) >= 13800-30:
            raise TimeoutError("task GPU-seconds budget")
        if self.totals()[kind] >= self.CAPS[kind]:
            raise RuntimeError(f"{kind} cap")
        self.counts[kind] += 1
        self.events.append(dict(kind=kind, note=note, elapsed=elapsed))
        self.save()

    def measure(self, name, fn):
        import torch
        torch.cuda.synchronize()
        start = time.monotonic()
        out = fn()
        torch.cuda.synchronize()
        duration = time.monotonic()-start
        if torch.cuda.max_memory_allocated()/2**30 > 44:
            raise MemoryError("allocated exceeds 44 GiB")
        self.events.append(dict(kind="timing", name=name, seconds=duration,
                                elapsed=time.monotonic()-self.started, rss_gib=rss_gib()))
        self.save()
        print("TIMING", name, round(duration, 3), flush=True)
        return out

    def close(self, status, error=None):
        self.status = status
        self.save(error=error)


def load_clip(clip, device="cuda:0"):
    import torch
    from causal.local_topology import visible_inputs
    record = json.loads((ENCODED/clip/"record.json").read_text())
    for item in ("condition",):
        assert sha(ROOT/record[item]["path"]) == record[item]["sha256"]
    for item in ("I0", "full17"):
        assert sha(ROOT/record["latents"][item]["path"]) == record["latents"][item]["sha256"]
    f = torch.load(ENCODED/clip/"condition.pt", map_location="cpu", weights_only=True)
    z = torch.load(ENCODED/clip/"full17.pt", map_location="cpu", weights_only=True).float().to(device)
    assert z.shape == (1,24,17,30,52) and f["clip_id"] == clip
    packed = {k: v.to(device) if torch.is_tensor(v) else v for k,v in f["packed"].items()}
    prompt = f["prompt"].to(device)
    visible = {stop: visible_inputs(packed, prompt, stop, 390) for stop in (12,17)}
    return dict(f=f, z=z, vis=visible, anchor=f["anchor"].to(device),
                audio=f["audio_noise"].to(device), record=record)


def setup_model():
    import torch
    import run as prior
    from causal.pretrained_lora import install_adapters
    prior.setup()
    torch.manual_seed(13)
    torch.cuda.manual_seed_all(13)
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats()
    pipe, model, precision = prior.model()
    adapters, indices = install_adapters(model, rank=8,
              block_indices=tuple(range(len(model.blocks)-8, len(model.blocks))), device="cuda:0")
    params = [p for a in adapters for p in (a.lora_A, a.lora_B)]
    assert indices == list(range(42,50)) and sum(p.numel() for p in params) == 1720320
    assert all(not p.requires_grad for n,p in model.named_parameters()
                   if not (n.endswith("lora_A") or n.endswith("lora_B")))
    return pipe, model, adapters, params, precision


def state_dict(adapters, optimizer, step, order_gen=None, noise_gen=None, order=None, pos=0):
    import torch
    state = dict(adapter=[cpu_tree({"A":a.lora_A,"B":a.lora_B}) for a in adapters],
                 optimizer=cpu_tree(optimizer.state_dict()), step=step,
                 cpu_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(),
                 metadata=dict(task="EXP-023/v1", objective="ordinary_fm",
                               source="34 direct-source RGB clips", train_clips=32,
                               partition=[12,5], action="own/current-prefix+feedback",
                               precision="h3_fp32", adapter="tail8_qkv_rank8",
                               checkpoint_context_addendum_sha256=sha(HERE/"ADDENDUM_02_CHECKPOINT_CONTEXT.md"),
                               checkpoint_context_cpu_sha256=sha(HERE/"GRADIENT_PROTOCOL_CPU.json")))
    if order_gen is not None:
        state.update(order_rng=order_gen.get_state(), noise_rng=noise_gen.get_state(),
                     order=order, order_pos=pos)
    return state


def save_checkpoint(path, state):
    import torch
    if path.exists(): raise FileExistsError(path)
    tmp = path.with_suffix(".partial")
    torch.save(state, tmp)
    tmp.rename(path)
    write(path.with_suffix(".json"), dict(path=str(path.relative_to(ROOT)),
          sha256=sha(path), bytes=path.stat().st_size, step=state["step"],
          metadata=state["metadata"]))


def load_checkpoint(path, adapters, optimizer=None):
    import torch
    state = torch.load(path, map_location="cpu", weights_only=True)
    assert state["metadata"]["task"] == "EXP-023/v1"
    with torch.no_grad():
        for a,s in zip(adapters, state["adapter"]):
            a.lora_A.copy_(s["A"].to(a.lora_A.device))
            a.lora_B.copy_(s["B"].to(a.lora_B.device))
    if optimizer is not None: optimizer.load_state_dict(state["optimizer"])
    return state


def velocity(model, data, cache, index, x, sigma, budget, note, *, grad=False):
    import torch
    from interval_train import interval_train
    from current_prefix import current_prefix_feedback
    stop, start = (12,0) if index == 0 else (17,12)
    packed,prompt = data["vis"][stop]
    budget.reserve("dit", note)
    def call():
        with (torch.enable_grad() if grad else torch.no_grad()), current_prefix_feedback():
            return interval_train(model, x, start=start, index=index, cache=cache,
                      full_packed=packed, prompt=prompt, anchor=data["anchor"],
                      audio=data["audio"], sigma=sigma, commit=(index==0 and sigma==0),
                      checkpoint=grad, offload=grad)
    return budget.measure(note, call)


def make_cache(model, data, history, budget, note):
    from causal.h3_cached import H3ChunkCache
    cache = H3ChunkCache(5,"cpu")
    velocity(model,data,cache,0,history,0.,budget,note)
    assert cache.commits == 50 and len(cache.layers) == 50
    return cache


def probe(model, data, budget, label):
    import torch
    from current_prefix import current_prefix_feedback
    from interval_train import interval_train
    from causal.h3_cached import H3ChunkCache
    cache = make_cache(model,data,data["z"][:,:,:12],budget,label+"_commit")
    packed,prompt=data["vis"][17]
    clean=data["z"][:,:,12:17]
    noise=data["f"]["initial_noise"][:,:,12:17].float().to("cuda:0")
    result=[]
    for sigma in (.25,.75):
        z=(1-sigma)*clean+sigma*noise
        budget.reserve("dit",f"{label}_sigma{sigma}")
        def one():
            with torch.no_grad(),current_prefix_feedback():
                return interval_train(model,z,start=12,index=1,cache=cache,
                      full_packed=packed,prompt=prompt,anchor=data["anchor"],
                      audio=data["audio"],sigma=sigma)
        y=budget.measure(f"{label}_sigma{sigma}",one)
        target=noise-clean
        result.append(dict(sigma=sigma, fm_mse=float(torch.nn.functional.mse_loss(y.float(),target)),
                           velocity_rms=float(y.float().square().mean().sqrt())))
    return result


def frozen_entry_regression(model,budget):
    """Zero-adapter compare against the EXP-022 b784d995 frozen entry."""
    import torch
    from causal.h3_cached import H3ChunkCache
    from causal.local_topology import visible_inputs
    from current_prefix import current_prefix_feedback
    from interval_cached import interval_cached
    from interval_train import interval_train
    source=json.loads((HERE/"CPU_PREFLIGHT.json").read_text())["source_manifest_sha256"]
    fixture_path=ROOT/"H3-World/outputs/EXP-019_v3_mixed_action_validation/E/b784d995.pt"
    assert sha(fixture_path)=="0347bbdd9c7860bef593f8861eca495fe91a0708cfdbee59f45409640eaae9cc"
    f=torch.load(fixture_path,map_location="cpu",weights_only=True)
    z=torch.load(ROOT/"H3-World/outputs/EXP-016_v3_real56_vae/b784d995/full17.pt",
                 map_location="cpu",weights_only=True).float().to("cuda:0")
    packed={k:v.to("cuda:0") if torch.is_tensor(v) else v for k,v in f["packed"].items()}
    prompt=f["prompt"].to("cuda:0")
    vis={stop:visible_inputs(packed,prompt,stop,390) for stop in (12,17)}
    anchor=f["anchor"].to("cuda:0");audio=f["audio_noise"].to("cuda:0")
    noise=f["initial_noise"][:,:,12:17].float().to("cuda:0")
    state=.5*z[:,:,12:17]+.5*noise
    outputs=[];caches=[]
    for name,fn in (("frozen",interval_cached),("exp023",interval_train)):
        cache=H3ChunkCache(5,"cpu")
        for idx,sigma,x in ((0,0.,z[:,:,:12]),(1,.5,state)):
            stop,start=(12,0) if idx==0 else (17,12)
            p,pr=vis[stop]
            note=f"entry_{name}_C{idx+1}"
            budget.reserve("dit",note)
            def call():
                with torch.no_grad(),current_prefix_feedback():
                    return fn(model,x,start=start,index=idx,cache=cache,
                              full_packed=p,prompt=pr,anchor=anchor,audio=audio,
                              sigma=sigma,commit=(idx==0))
            y=budget.measure(note,call)
            if idx==1: outputs.append(y.detach().cpu())
        caches.append(cache)
    exact=bool(torch.equal(*outputs))
    max_abs=float((outputs[0].float()-outputs[1].float()).abs().max())
    cache_equal=all(torch.equal(getattr(caches[0].history(layer,1)[0],field),
                                getattr(caches[1].history(layer,1)[0],field))
                    for layer in range(50) for field in ("key","value","rope"))
    result=dict(status="PASS" if exact and cache_equal else "FAILED",
                old_fixture="H264-derived EXP-016 regression only; not EXP-023 train data",
                fixture_sha256=sha(fixture_path),cpu_preflight_source_sha256=source,
                velocity_exact=exact,velocity_max_abs=max_abs,
                cache_all_150_tensors_exact=cache_equal,top_level_dit_forwards=4)
    write(HERE/"FROZEN_ENTRY_REGRESSION.json",result)
    if result["status"]!="PASS": raise RuntimeError("frozen entry regression failed")
    return result


def cache_signature(cache):
    import torch
    entries=[]
    for layer in sorted(cache.layers):
        for entry in cache.layers[layer]:
            entries.append((layer,id(entry.key),id(entry.value),id(entry.rope),
                            tuple(entry.key.shape),tuple(entry.value.shape),
                            float(entry.key.flatten()[:8].float().sum()),
                            float(entry.value.flatten()[:8].float().sum())))
    return (cache.commits,cache.nbytes,entries)


def train_one(model, data, cache, clean, noise, sigma, budget, step, optimizer, params,
              *, gate=False):
    import torch
    import current_prefix as route
    from causal.h3_cached import ChunkAttention
    from current_prefix import current_prefix_feedback
    from interval_train import interval_train
    z=(1-sigma)*clean+sigma*noise
    target=noise-clean
    packed,prompt=data["vis"][17]
    optimizer.zero_grad(set_to_none=True)
    phase="forward"; trace=[]
    cache_before=cache_signature(cache) if gate else None
    original_route=route.current_prefix_attend
    def audited(self,*args,**kwargs):
        assert ChunkAttention.attend.__module__ == "current_prefix"
        assert self.action_prefix_mode == "own" and self.action_feedback
        assert self.index == 1 and self.frame_start == 12
        assert self.action_rows is not None and len(self.action_rows)==17
        assert int(self.action_rows[12,1])>int(self.action_rows[12,0])
        assert int(self.action_rows[0,0]) > 0  # common prefix exists
        history_rows = sum(e.key.shape[0] for e in self.cache.history(kwargs["layer"],1))
        prefix_mask, video_mask = self.masks(5*390,history_rows,args[0].device)
        assert bool(video_mask[0,0])  # common condition remains visible
        assert bool(video_mask[0,int(self.action_rows[12,0])])  # own current action
        assert not bool(video_mask[0,int(self.action_rows[11,0])])  # no direct past action
        assert bool(prefix_mask[0,0])
        trace.append(dict(phase=phase,layer=int(kwargs["layer"]),
                          route="current_prefix", feedback=True,
                          common_to_current_video=True,
                          current_action_span=True,
                          past_action_direct_read=False))
        return original_route(self,*args,**kwargs)
    if gate: route.current_prefix_attend=audited
    try:
        budget.reserve("dit",f"train_step{step}_C2")
        # A single owner holds the patch through non-reentrant recomputation.
        with current_prefix_feedback():
            def forward():
                return interval_train(model,z,start=12,index=1,cache=cache,
                         full_packed=packed,prompt=prompt,anchor=data["anchor"],
                         audio=data["audio"],sigma=sigma,checkpoint=True,offload=True)
            y=budget.measure(f"step{step}_forward",forward)
            loss=torch.nn.functional.mse_loss(y.float(),target.float())
            if not torch.isfinite(loss): raise FloatingPointError("nonfinite FM loss")
            phase="backward"
            budget.reserve("backward",f"train_step{step}")
            budget.measure(f"step{step}_backward",loss.backward)
        if gate:
            fw=[x["layer"] for x in trace if x["phase"]=="forward"]
            bw=[x["layer"] for x in trace if x["phase"]=="backward"]
            report=dict(status="PASS" if fw and bw else "FAILED",forward_layers=fw,
                        backward_recomputed_layers=bw,all_routes_current_prefix=all(
                            x["route"]=="current_prefix" for x in trace),
                        cache_unchanged=cache_before==cache_signature(cache),
                        current_prefix_source_sha256=sha(ROOT/"submission/reports/stage1_anyflow/02_causal_diagnostics/current_prefix_candidate/current_prefix.py"),
                        attention_semantics="own action; action feedback; common prefix reads current video",
                        first_step_counted_in_training_budget=True)
            write(HERE/"GRADIENT_PROTOCOL_CHECK.json",report)
            if report["status"]!="PASS" or not report["cache_unchanged"]:
                raise RuntimeError("real H3 backward route/cache gate failed")
    finally:
        if gate: route.current_prefix_attend=original_route
    grads=[p.grad for p in params]
    if any(g is None or not torch.isfinite(g).all() for g in grads):
        raise FloatingPointError("missing/nonfinite adapter grad")
    grad_l2=float(torch.sqrt(sum(g.float().square().sum() for g in grads)))
    if grad_l2<=0: raise FloatingPointError("zero adapter grad")
    if any(p.grad is not None for n,p in model.named_parameters()
           if not (n.endswith("lora_A") or n.endswith("lora_B"))):
        raise RuntimeError("frozen base/released Action LoRA received gradient")
    if gate:
        report=json.loads((HERE/"GRADIENT_PROTOCOL_CHECK.json").read_text())
        report.update(grad_l2=grad_l2,base_gradient_absent=True)
        write(HERE/"GRADIENT_PROTOCOL_CHECK.json",report)
    budget.reserve("update",f"train_step{step}")
    budget.measure(f"step{step}_optimizer",optimizer.step)
    return dict(step=step,sigma=sigma,loss=float(loss.detach()),grad_l2=grad_l2,
                clip_id=data["f"]["clip_id"],cpu_kv_bytes=cache.nbytes,
                peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                rss_gib=rss_gib())


def train(model,adapters,params,budget):
    import torch
    optimizer=torch.optim.AdamW(params,lr=1e-4,betas=(.9,.95),weight_decay=.01)
    zero=load_checkpoint(HERE/"checkpoint_000.pt",adapters,optimizer)
    assert zero["step"]==0 and all(bool(torch.count_nonzero(a.lora_B)==0) for a in adapters)
    _,selection,_,_,_,_=inputs()
    ids=selection["train_clip_ids"]
    order_gen=torch.Generator(device="cpu").manual_seed(13)
    noise_gen=torch.Generator(device="cpu").manual_seed(130023)
    order=[]
    for _ in range(4): order += [ids[i] for i in torch.randperm(32,generator=order_gen).tolist()]
    assert len(order)==128 and all(order.count(x)==4 for x in ids)
    rows=[]
    for pos,clip in enumerate(order):
        step=pos+1
        data=load_clip(clip)
        clean=data["z"][:,:,12:17]
        noise=torch.randn(clean.shape,generator=noise_gen,dtype=torch.float32).to("cuda:0")
        sigma=float(torch.rand((),generator=noise_gen))
        cache=make_cache(model,data,data["z"][:,:,:12],budget,f"step{step}_GT_C1_commit")
        row=train_one(model,data,cache,clean,noise,sigma,budget,step,optimizer,params,gate=(step==1))
        rows.append(row)
        write(HERE/"TRAIN_UPDATES.json",rows)
        print("UPDATE",row,flush=True)
        if step in (8,32,64,128):
            save_checkpoint(HERE/f"checkpoint_{step:03d}.pt",
                 state_dict(adapters,optimizer,step,order_gen,noise_gen,order,pos+1))
        del data,clean,noise,cache
        gc.collect()
        if step==8:
            assert all(x["grad_l2"]>0 and x["loss"]==x["loss"] for x in rows)
    return rows


def persist_rgb(frames,path):
    import numpy as np
    # Called immediately after VAE decode, before identity serialization.
    # Any later CPU failure must leave a recoverable 56-frame array.
    raw=path.with_suffix(".rgb.npy")
    if raw.exists():
        assert np.array_equal(np.load(raw),frames)
    else:
        raw_tmp=raw.with_suffix(".npy.partial")
        if raw_tmp.exists():raise FileExistsError(raw_tmp)
        with raw_tmp.open("wb") as stream: np.save(stream,np.ascontiguousarray(frames))
        raw_tmp.replace(raw)
    return raw


def tensor_sha(value):
    import torch
    x=value.detach().cpu().contiguous()
    digest=hashlib.sha256(str((tuple(x.shape),str(x.dtype))).encode())
    digest.update(x.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def video_identity(data,role,phase,history):
    return dict(clip_id=data["f"]["clip_id"],role=role,history=history,
       checkpoint=(0 if phase=="eval_pre" else 128),nfe_per_chunk=8,seed=13,
       condition_sha256=data["record"]["condition"]["sha256"],
       source_video_sha256=data["record"]["source_video_sha256"],
       action_script_sha256=data["record"]["action_script_sha256"],
       initial_noise_sha256=tensor_sha(data["f"]["initial_noise"]))


def save_video(frames,path):
    import av
    import imageio_ffmpeg
    import numpy as np
    if path.exists(): raise FileExistsError(path)
    executable=Path(imageio_ffmpeg.get_ffmpeg_exe()).resolve()
    assert executable.is_file()
    raw=persist_rgb(frames,path)
    h,w=frames.shape[1:3]
    temp=path.with_suffix(".partial.mp4")
    if temp.exists():raise FileExistsError(temp)
    command=[str(executable),"-v","error","-f","rawvideo","-pix_fmt","rgb24",
             "-s",f"{w}x{h}","-r","24","-i","pipe:0","-an","-c:v","libx264",
             "-threads","2","-crf","18","-pix_fmt","yuv420p",str(temp)]
    p=subprocess.Popen(command,stdin=subprocess.PIPE)
    p.communicate(np.ascontiguousarray(frames).tobytes())
    if p.returncode: raise RuntimeError(f"ffmpeg failed: {temp}")
    with av.open(str(temp)) as container:
        stream=container.streams.video[0]
        assert abs(float(stream.average_rate)-24)<.001
        count=0
        for decoded in container.decode(video=0):
            assert (decoded.width,decoded.height)==(w,h)
            count+=1
    assert count==len(frames),(count,len(frames))
    temp.replace(path)
    version=subprocess.check_output([str(executable),"-version"],text=True).splitlines()[0]
    write(path.with_suffix(".media.json"),dict(ffmpeg_path=str(executable),
          ffmpeg_sha256=sha(executable),ffmpeg_version=version,
          rgb_path=str(raw.relative_to(ROOT)),rgb_sha256=sha(raw),
          mp4_sha256=sha(path),frames=count,width=w,height=h,fps=24))


def sample_chunk(pipe,model,data,cache,index,budget,label):
    import torch
    from causal.anyflow_sampling import configure_video_schedule
    pipe.load_models_to_device(["dit"])
    configure_video_schedule(pipe.scheduler,steps=8,grid="native",flow_shift=2.22)
    start,stop=(0,12) if index==0 else (12,17)
    x=data["f"]["initial_noise"][:,:,start:stop].float().to("cuda:0").clone()
    tick=time.monotonic()
    with torch.no_grad():
        for n,t in enumerate(pipe.scheduler.timesteps):
            # Here sigma=0 means a denoising step, not a commit; only C1
            # after sampling is clean-committed through make_cache().
            packed,prompt=data["vis"][stop]
            from interval_train import interval_train
            from current_prefix import current_prefix_feedback
            budget.reserve("dit",f"{label}_C{index+1}_NFE{n+1}")
            def call():
                with current_prefix_feedback():
                    return interval_train(model,x,start=start,index=index,cache=cache,
                           full_packed=packed,prompt=prompt,anchor=data["anchor"],
                           audio=data["audio"],sigma=float(t)/1000)
            y=budget.measure(f"{label}_C{index+1}_NFE{n+1}",call)
            x=pipe.scheduler.step(y,t,x)
            if not torch.isfinite(x).all(): raise FloatingPointError("sample nonfinite")
    return x,time.monotonic()-tick


def decode(pipe,latents,budget,label):
    import numpy as np
    import torch
    pipe.load_models_to_device(["video_vae"])
    budget.reserve("vae_decode",label)
    def call():
        with torch.no_grad():
            rgb=pipe.video_vae.decode_video(latents,dtype=pipe.torch_dtype,
                 process_image=False,tiled=True,tile_size=256,tile_overlap=64)
            return np.stack([np.asarray(im.convert("RGB"),dtype=np.uint8)
                 for im in pipe.vae_output_to_video(rgb,min_value=0,max_value=1)])
    frames=budget.measure(label+"_decode",call)
    assert frames.shape==(56,480,832,3)
    return frames


def evaluate(pipe,model,budget,phase):
    import numpy as np
    import torch
    from causal.h3_cached import H3ChunkCache
    from common import raw_source_rgb
    _,selection,_,train,dev,roles=inputs()
    role_order=("train_constant","train_switch","dev_constant","dev_switch")
    outdir=HERE/("eval_before" if phase=="eval_pre" else "eval_after")
    if outdir.exists(): raise FileExistsError(outdir)
    outdir.mkdir()
    results=[]
    for role in role_order:
        clip=roles[role]["clip_id"]
        data=load_clip(clip)
        source=raw_source_rgb((train if role.startswith("train") else dev)[clip])
        history=data["z"][:,:,:12]
        cache=make_cache(model,data,history,budget,role+"_GT_C1_commit")
        x,seconds=sample_chunk(pipe,model,data,cache,1,budget,role+"_GT")
        frames=decode(pipe,torch.cat([history,x],dim=2),budget,role+"_GT")
        frames[:39]=source[:39]
        path=outdir/f"{role}_GT39_generated17.mp4"
        persist_rgb(frames,path)
        write(path.with_suffix(".identity.json"),video_identity(data,role,phase,
              "GT first39 displayed and committed"))
        save_video(frames,path)
        row=dict(role=role,clip_id=clip,history="GT first39 displayed and committed",
                 nfe_per_chunk=8,video=str(path.relative_to(ROOT)),sha256=sha(path),
                 sampling_seconds=seconds,boundary_mad=float(np.abs(
                     frames[39].astype(float)-frames[38].astype(float)).mean()),
                 frame_mad=float(np.abs(np.diff(frames.astype(float),axis=0)).mean()),
                 cpu_kv_bytes=cache.nbytes)
        results.append(row)
        del data,history,cache,x,frames;gc.collect()
        write(outdir/"results.json",results)
    for role in ("dev_constant","dev_switch"):
        clip=roles[role]["clip_id"]
        data=load_clip(clip)
        empty=H3ChunkCache(5,"cpu")
        c1,s1=sample_chunk(pipe,model,data,empty,0,budget,role+"_SELF")
        cache=make_cache(model,data,c1,budget,role+"_SELF_C1_commit")
        c2,s2=sample_chunk(pipe,model,data,cache,1,budget,role+"_SELF")
        frames=decode(pipe,torch.cat([c1,c2],dim=2),budget,role+"_SELF")
        path=outdir/f"{role}_self56.mp4"
        persist_rgb(frames,path)
        write(path.with_suffix(".identity.json"),video_identity(data,role,phase,
              "self-generated C1 and C2"))
        save_video(frames,path)
        row=dict(role=role,clip_id=clip,history="self-generated C1 and C2",
                 nfe_per_chunk=8,video=str(path.relative_to(ROOT)),sha256=sha(path),
                 sampling_c1_seconds=s1,sampling_c2_seconds=s2,
                 boundary_mad=float(np.abs(frames[39].astype(float)-frames[38].astype(float)).mean()),
                 frame_mad=float(np.abs(np.diff(frames.astype(float),axis=0)).mean()),
                 cpu_kv_bytes=cache.nbytes)
        results.append(row)
        del data,c1,c2,cache,frames;gc.collect()
        write(outdir/"results.json",results)
    probes={}
    for role in role_order:
        data=load_clip(roles[role]["clip_id"])
        probes[role]=probe(model,data,budget,role+"_PROBE")
        del data;gc.collect()
        write(outdir/"probes.json",probes)
    return results,probes


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--phase",choices=("pre","train","post"),required=True)
    parser.add_argument("--gpu",type=int,required=True)
    args=parser.parse_args()
    phase={"pre":"eval_pre","train":"train","post":"eval_post"}[args.phase]
    if os.environ.get("CUDA_VISIBLE_DEVICES")!=str(args.gpu):
        raise RuntimeError("GPU selector mismatch")
    import run as prior
    prior.setup()
    import torch
    torch.set_num_threads(4)
    if not (HERE/"GRADIENT_PROTOCOL_CPU.json").exists():
        raise RuntimeError("CPU real-attention gradient preflight required")
    if args.phase!="pre" and not (HERE/"checkpoint_000.pt").exists():
        raise RuntimeError("zero checkpoint missing")
    run_source_sha=verify_run_sources()
    budget=Budget(phase,args.gpu)
    try:
        pipe,model,adapters,params,precision=budget.measure("load_model",setup_model)
        budget.save(precision=precision,train_source_manifest_sha256=run_source_sha)
        optimizer=torch.optim.AdamW(params,lr=1e-4,betas=(.9,.95),weight_decay=.01)
        if args.phase=="pre":
            assert all(bool(torch.count_nonzero(a.lora_B)==0) for a in adapters)
            save_checkpoint(HERE/"checkpoint_000.pt",state_dict(adapters,optimizer,0))
            frozen_entry_regression(model,budget)
            result=evaluate(pipe,model,budget,phase)
        elif args.phase=="train":
            del optimizer
            result=train(model,adapters,params,budget)
        else:
            final=load_checkpoint(HERE/"checkpoint_128.pt",adapters)
            assert final["step"]==128
            result=evaluate(pipe,model,budget,phase)
        write(HERE/f"{phase.upper()}_RESULT.json",dict(status="COMPLETE_PENDING_JUDGE",
              phase=phase,counts=budget.counts,wall_seconds=time.monotonic()-budget.started,
              result_path="eval results/probes" if args.phase!="train" else "TRAIN_UPDATES.json"))
        budget.close("COMPLETE_PENDING_JUDGE")
        print("COMPLETE",phase,budget.counts,flush=True)
    except BaseException as error:
        budget.close("FAILED_STOPPED",repr(error))
        print("STOPPED",phase,repr(error),flush=True)
        raise


if __name__=="__main__":
    main()
