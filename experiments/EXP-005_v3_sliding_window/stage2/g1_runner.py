"""EXP-005/v2 G1: certified 47-latent SW-G C7/C8, A-continue vs D-switch.

Separate authorization is mandatory. No GT reset, training, Local RoPE or C9.
"""
from __future__ import annotations

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

from g0_runner import (ROOT, FROZEN, PREV, EXP003, EXP001, ROUTER, EXP,
                       sha, atomic_json)

HERE = Path(__file__).resolve().parent
CONFIG = HERE / "g1_config.json"
AUTH = HERE / "g1_authorization.json"
FIXTURE = ROOT / "H3-World/outputs/EXP-005_v3_sliding_window/inputs/long47.pt"
FIXTURE_META = FIXTURE.with_suffix(".json")
G0_OUT = ROOT / "H3-World/outputs/EXP-005_v3_sliding_window/G0"
G0_REVIEW = HERE / "g0_judge_review.json"
SOURCES = {
    "long_fixture": FIXTURE,
    "long_fixture_metadata": FIXTURE_META,
    "cache_through32": PREV / "cache_through32.pt",
    "C6_endpoint": PREV / "chunk_32_37.pt",
    "prior_RGB_124": PREV / "published_124.npy",
    "baseline_C6_metrics": EXP003 / "artifacts/metrics/AA_chunk_32_37.json",
    "released_lora": FROZEN / "runtime/checkpoints/H3-World/step-10000.safetensors",
    "interval_sw": EXP / "interval_sw.py",
    "interval_stage2": HERE / "interval_stage2.py",
    "chunk_plan": EXP / "chunk_plan.py",
    "position_sw": EXP / "position_sw.py",
    "router": ROUTER / "current_prefix.py",
    "raw_KV": FROZEN / "runtime/code/causal/h3_cached.py",
    "infer": FROZEN / "runtime/code/abot/infer.py",
    "h3_precision": FROZEN / "runtime/code/causal/h3_precision.py",
    "base_source_manifest": EXP001 / "source_manifest_v3.json",
    "g0_runner": HERE / "g0_runner.py",
    "g0_manifest": HERE / "g0_source_manifest.json",
    "g0_result": G0_OUT / "result.json",
    "g0_budget": G0_OUT / "budget.json",
    "g0_review": G0_REVIEW,
    "long_fixture_review": HERE / "long_fixture_review.json",
    "H3_dit": FROZEN / "runtime/DiffSynth-Studio-h3-v2/diffsynth/models/minimax_h3_dit.py",
    "H3_pipeline": FROZEN / "runtime/DiffSynth-Studio-h3-v2/diffsynth/pipelines/minimax_h3_audio_video.py",
    "runner": HERE / "g1_runner.py",
    "config": CONFIG,
}
for i,path in enumerate((
    ROOT / "submission/experiments/11_causal_12_then5_selfhistory/states/first12_A.pt",
    ROOT / "H3-World/outputs/EXP-002_native_cached/AA/chunk_12_17.pt",
    ROOT / "H3-World/outputs/EXP-002_native_cached/AA/chunk_17_22.pt",
    PREV / "chunk_22_27.pt", PREV / "chunk_27_32.pt"), start=1):
    SOURCES[f"C{i}_endpoint"] = path


def preflight():
    import torch
    cfg=json.loads(CONFIG.read_text())
    assert cfg["max_sampling"]==120 and cfg["max_commit"]==3
    assert cfg["max_forwards"]==123 and cfg["max_vae"]==4
    hashes={name:dict(path=str(path),sha256=sha(path)) for name,path in SOURCES.items()}
    long=json.loads(FIXTURE_META.read_text())
    fixture_review=json.loads(SOURCES["long_fixture_review"].read_text())
    baseline=json.loads(SOURCES["baseline_C6_metrics"].read_text())
    assert hashes["long_fixture"]["sha256"]==long["fixture_sha256"]
    assert long["certified_cpu"] is True and long["latent_stop"]==47
    assert fixture_review["decision"]=="accepted_for_G1_input_protocol"
    assert fixture_review["fixture_sha256"]==hashes["long_fixture"]["sha256"]
    assert hashes["cache_through32"]["sha256"]==baseline["cache_file_sha256_before"]
    assert hashes["C6_endpoint"]["sha256"]==baseline["endpoint_sha256"]
    assert hashes["prior_RGB_124"]["sha256"]==baseline["published_file_sha256"]
    assert hashes["released_lora"]["sha256"]==baseline["source_checkpoint_sha256"]
    assert hashes["router"]["sha256"]==baseline["accepted_router_sha256"]
    g0=json.loads(SOURCES["g0_result"].read_text())
    g0_budget=json.loads(SOURCES["g0_budget"].read_text())
    review=json.loads(G0_REVIEW.read_text())
    assert g0["status"]=="complete_pending_judge_review"
    assert g0["old107_unchanged"] and g0["cache_read_unchanged"]
    assert g0["endpoint_vs_baseline"]["allclose_1e-5"]
    assert g0_budget["status"]=="complete_pending_judge_review"
    assert (g0_budget["total_forwards"],g0_budget["vae"])==(34,1)
    assert review.get("approved") is True and review.get("stage")=="G0"
    assert review.get("verdict")=="PASS" and review.get("approved_next_stage")=="G1"
    assert review.get("endpoint_equal_independently_verified") is True
    assert review.get("all124_RGB_equal_independently_verified") is True
    assert (review.get("total_forwards"),review.get("vae"))==(34,1)
    pieces=[torch.load(SOURCES[f"C{i}_endpoint"],map_location="cpu",weights_only=True)
            for i in range(1,6)]
    history=torch.cat(pieces,dim=2).contiguous()
    tensor_sha=hashlib.sha256(str((tuple(history.shape),str(history.dtype))).encode()+
                              history.view(torch.uint8).numpy().tobytes()).hexdigest()
    assert tensor_sha==baseline["history_tensor_sha256"]
    manifest=dict(task="EXP-005/v2",stage="G1",sources=hashes,
                  source_history_tensor_sha256=tensor_sha,
                  core_first_start=g0_budget["first_start"],
                  long_fixture_certification_sha256=hashes["long_fixture_metadata"]["sha256"],
                  created_utc=time.time())
    atomic_json(HERE/"g1_source_manifest.json",manifest)
    return manifest


def verify_inputs():
    path=HERE/"g1_source_manifest.json"
    if not path.exists(): raise RuntimeError("G1 preflight missing")
    manifest=json.loads(path.read_text())
    if set(manifest["sources"])!=set(SOURCES): raise RuntimeError("G1 source set changed")
    for name,file in SOURCES.items():
        if manifest["sources"][name]!=dict(path=str(file),sha256=sha(file)):
            raise RuntimeError(f"G1 frozen source changed: {name}")
    return manifest


def verify_authorization():
    if not AUTH.exists(): raise RuntimeError("G1 Judge authorization missing")
    approval=json.loads(AUTH.read_text())
    expected=dict(approved=True,stage="G1",task="EXP-005/v2",
                  runner_sha256=sha(Path(__file__)),config_sha256=sha(CONFIG),
                  manifest_sha256=sha(HERE/"g1_source_manifest.json"),
                  max_forwards=123,max_vae=4,max_gpu_seconds=2520)
    for key,value in expected.items():
        if approval.get(key)!=value: raise RuntimeError(f"G1 authorization mismatch: {key}")
    return approval


class Budget:
    def __init__(self,cfg,root):
        self.cfg,self.root=cfg,root
        self.start=time.time()
        self.core_start=json.loads(SOURCES["g0_budget"].read_text())["first_start"]
        self.ledger=dict(task="EXP-005/v2",stage="G1",gpu=cfg["gpu"],
                         first_start=self.start,sampling=0,commit=0,
                         core_first_start=self.core_start,
                         total_forwards=0,vae=0,status="running")
        self.write()

    def write(self):
        self.ledger["elapsed_seconds"]=time.time()-self.start
        atomic_json(self.root/"budget.json",self.ledger)

    def check(self):
        if time.time()-self.start>=self.cfg["max_gpu_seconds"]:
            raise TimeoutError("G1 GPU/wall deadline")
        if time.time()-self.core_start>=10800:
            raise TimeoutError("EXP-005 core three-hour elapsed deadline")

    def reserve(self,kind):
        self.check()
        if kind not in ("sampling","commit"): raise ValueError(kind)
        if (self.ledger[kind]>=self.cfg[f"max_{kind}"] or
            self.ledger["total_forwards"]>=self.cfg["max_forwards"]):
            raise RuntimeError("G1 forward budget exhausted")
        self.ledger[kind]+=1;self.ledger["total_forwards"]+=1;self.write()

    def reserve_vae(self):
        self.check()
        if self.ledger["vae"]>=self.cfg["max_vae"]:
            raise RuntimeError("G1 VAE budget exhausted")
        self.ledger["vae"]+=1;self.write()


def rgb_stop(stop):
    if stop<12 or (stop-12)%5: raise ValueError(stop)
    return 39+17*((stop-12)//5)


def tensor_sha(value):
    value=value.detach().cpu().contiguous()
    return hashlib.sha256(str((tuple(value.shape),str(value.dtype))).encode()+
                          value.view(__import__("torch").uint8).numpy().tobytes()).hexdigest()


def rss_mib():
    pages=int(Path("/proc/self/statm").read_text().split()[1])
    return pages*os.sysconf("SC_PAGE_SIZE")/2**20


def run(cfg,budget):
    import numpy as np
    import torch
    from PIL import Image
    import infer as abot
    from benchmark import write_video
    from evaluate_action_control import evaluate as flow_metrics
    from causal.h3_cached import H3ChunkCache
    from causal.h3_precision import configure_precision
    from causal.local_topology import visible_inputs
    from causal.anyflow_sampling import configure_video_schedule
    from current_prefix import current_prefix_feedback
    from chunk_plan import PLAN,cache_identity,validate_cache
    from interval_sw import interval_sw
    from interval_stage2 import interval_stage2
    from rollout_contract import rgb_sha

    torch.set_num_threads(4);torch.manual_seed(cfg["seed"])
    device="cuda:0";out=budget.root
    fixture=torch.load(FIXTURE,map_location="cpu",weights_only=True)
    assert fixture["metadata"]["certified_cpu"] and fixture["metadata"]["latent_stop"]==47
    pipe=abot.load_pipeline(device)
    pipe.load_lora(pipe.dit,state_dict=abot.load_checkpoint_lora(SOURCES["released_lora"]),hotload=True)
    model=pipe.dit.eval().requires_grad_(False)
    precision=configure_precision(model,"h3_fp32",native_transformer_dir=
        FROZEN/"runtime/DiffSynth-Studio-h3-v2/models/MiniMax/MiniMax-H3/FL2VA/transformer")
    pipe.load_models_to_device(["dit"]);torch.cuda.synchronize()
    versions=[(p,p._version) for p in model.parameters()]

    def move(x):
        if torch.is_tensor(x):return x.to(device)
        if isinstance(x,dict):return {k:move(v) for k,v in x.items()}
        return x

    full=move(fixture["packed"])
    prompts={x:fixture["prompts"][x].to(device) for x in "AD"}
    frozen_reference=move(fixture["frozen_reference"])
    frozen_reference_prompt=fixture["frozen_reference_prompt"].to(device)
    anchor=fixture["anchor"].to(device);audio=fixture["audio_noise"].to(device)
    initial=fixture["initial_noise"].to(device,torch.float32)
    assert initial.shape==(1,24,47,30,52)
    c6=torch.load(SOURCES["C6_endpoint"],map_location="cpu",weights_only=True).to(device,torch.float32)
    pieces=[torch.load(SOURCES[f"C{i}_endpoint"],map_location="cpu",weights_only=True)
            for i in range(1,6)]
    history37=torch.cat([*pieces,c6.cpu()],dim=2).to(device,torch.float32)
    assert history37.shape==(1,24,37,30,52)
    rgb124=np.load(SOURCES["prior_RGB_124"])
    assert rgb124.shape==(124,480,832,3)
    cache=torch.load(SOURCES["cache_through32"],map_location="cpu",weights_only=False)
    assert isinstance(cache,H3ChunkCache) and cache.storage_device=="cpu"
    assert validate_cache(cache,plan=PLAN,index=5,frame_rows=390,expected_layers=50)["ancestors"]==[0,1,2,3,4]
    sigmas=configure_video_schedule(pipe.scheduler,steps=30,grid="native",flow_shift=2.22)
    baseline=json.loads(SOURCES["baseline_C6_metrics"].read_text())
    assert sigmas==baseline["sigmas"]
    result=dict(task="EXP-005/v2",stage="G1",status="running",
                source_manifest_sha256=sha(HERE/"g1_source_manifest.json"),
                authorization_sha256=sha(AUTH),runner_sha256=sha(Path(__file__)),
                config_sha256=sha(CONFIG),precision=precision,sigmas=sigmas,
                environment=dict(torch=torch.__version__,torch_cuda=torch.version.cuda,
                    gpu_name=torch.cuda.get_device_name(0),
                    sdpa_flash_enabled=torch.backends.cuda.flash_sdp_enabled(),
                    sdpa_math_enabled=torch.backends.cuda.math_sdp_enabled(),
                    sdpa_mem_efficient_enabled=torch.backends.cuda.mem_efficient_sdp_enabled(),
                    sdpa_cudnn_enabled=torch.backends.cuda.cudnn_sdp_enabled()),
                chunks=[],cache_transfer_note="CPU KV transfer is included in sampling_seconds; not separately timed")
    atomic_json(out/"result.json",result)

    def peak_check():
        value=torch.cuda.max_memory_allocated()/2**30
        if value>cfg["max_allocated_gib"]:raise MemoryError("G1 allocated peak >44GiB")
        return value

    def forward(current,index,cache,layout,prompt,sigma,kind,commit=False):
        budget.reserve(kind)
        with torch.no_grad(),current_prefix_feedback():
            output=interval_stage2(model,current,start=PLAN.span(index)[0],index=index,
                cache=cache,full_packed=layout,prompt=prompt,anchor=anchor,audio=audio,
                sigma=sigma,commit=commit,mode="global",expected_layers=50,
                frozen_reference=frozen_reference,
                frozen_reference_prompt=frozen_reference_prompt,
                certified_long_fixture=True)
        peak_check()
        return output

    # EXP-003 saved C6 endpoint without a final commit. Commit exactly once;
    # this triggers the first real eviction C1 -> retained C2..C6.
    c6_layout,c6_prompt=visible_inputs(full,prompts["A"],37,390)
    assert torch.equal(c6_layout["img_position_ids"],frozen_reference["img_position_ids"])
    commit_tick=time.perf_counter()
    budget.reserve("commit")
    with torch.no_grad(),current_prefix_feedback():
        _=interval_sw(model,c6,start=32,index=5,cache=cache,full_packed=c6_layout,
                      prompt=c6_prompt,anchor=anchor,audio=audio,sigma=0,commit=True)
    peak_check()
    result["shared_C6_commit_seconds"]=time.perf_counter()-commit_tick
    shared=validate_cache(cache,plan=PLAN,index=6,frame_rows=390,expected_layers=50)
    assert shared["ancestors"]==[1,2,3,4,5]
    result["shared_C6_commit_cache"]=shared
    shared_signature=cache_identity(cache)
    cache_file=out/"cache_through37.pt"
    torch.save(cache,cache_file)
    result["shared_C6_cache_sha256"]=sha(cache_file)
    atomic_json(out/"result.json",result)

    for action in "AD":
        path=out/action;path.mkdir(exist_ok=True)
        branch_cache=H3ChunkCache(max_history=5,storage_device="cpu",
            layers={layer:list(entries) for layer,entries in cache.layers.items()},
            peak_bytes=cache.peak_bytes,commits=cache.commits)
        assert cache_identity(cache)==shared_signature
        history=history37
        published=rgb124
        for index in (6,7):
            chunk_tick=time.perf_counter()
            start,stop=PLAN.span(index)
            assert (start,stop) in ((37,42),(42,47))
            row=dict(action=action,index=index,interval=[start,stop],
                     RGB_interval=[rgb_stop(start),rgb_stop(stop)],status="running",
                     sampling_forwards=0,commit_forwards=0,vae_calls=0,
                     cache_before=validate_cache(branch_cache,plan=PLAN,index=index,
                                                  frame_rows=390,expected_layers=50))
            row_path=path/f"chunk_{start}_{stop}.json"
            atomic_json(row_path,row)
            layout,prompt=visible_inputs(full,prompts[action],stop,390)
            assert len(layout["action_text_rows"])==stop
            current=initial[:,:,start:stop].clone()
            row.update(noise_sha256=tensor_sha(current),prompt_sha256=tensor_sha(prompt),
                       position_sha256=tensor_sha(layout["img_position_ids"]),
                       history_sha256=tensor_sha(history),
                       cpu_rss_mib_before=rss_mib())
            signature=cache_identity(branch_cache)
            tick=time.perf_counter()
            for j,t in enumerate(pipe.scheduler.timesteps):
                velocity=forward(current,index,branch_cache,layout,prompt,
                    float(t)/1000,"sampling")
                current=pipe.scheduler.step(velocity,t,current)
                if not torch.isfinite(current).all():raise FloatingPointError("G1 nonfinite latent")
                row["sampling_forwards"]=j+1
                if (j+1)%5==0:
                    row["peak_allocated_gib"]=peak_check()
                    atomic_json(row_path,row)
                del velocity
            torch.cuda.synchronize()
            row["sampling_seconds"]=time.perf_counter()-tick
            row["cache_read_unchanged"]=cache_identity(branch_cache)==signature
            if not row["cache_read_unchanged"]:raise RuntimeError("G1 sampling mutated cache")
            endpoint=path/f"chunk_{start}_{stop}.pt"
            torch.save(current.detach().cpu(),endpoint)
            row["endpoint_sha256"]=sha(endpoint)
            if index==6:
                tick=time.perf_counter()
                _=forward(current,index,branch_cache,layout,prompt,0,"commit",commit=True)
                row["commit_forwards"]=1
                row["commit_seconds"]=time.perf_counter()-tick
                row["cache_after_commit"]=validate_cache(branch_cache,plan=PLAN,index=7,
                    frame_rows=390,expected_layers=50)
                assert row["cache_after_commit"]["ancestors"]==[2,3,4,5,6]
                branch_cache_file=path/"cache_through42.pt"
                torch.save(branch_cache,branch_cache_file)
                row["cache_after_commit_sha256"]=sha(branch_cache_file)
            else:
                row["commit_forwards"]=0
            budget.reserve_vae()
            pipe.load_models_to_device(["video_vae"]);torch.cuda.synchronize()
            tick=time.perf_counter()
            with torch.no_grad():
                decoded=pipe.video_vae.decode_video(torch.cat((history,current),dim=2),
                    dtype=pipe.torch_dtype,process_image=False,tiled=True,
                    tile_size=256,tile_overlap=64)
                frames=pipe.vae_output_to_video(decoded,min_value=0,max_value=1)
            rgb=np.stack([np.asarray(frame.convert("RGB"),dtype=np.uint8) for frame in frames])
            assert len(rgb)==rgb_stop(stop)
            row["vae_seconds"]=time.perf_counter()-tick
            row["vae_calls"]=1
            extended=np.concatenate((published,rgb[rgb_stop(start):]),axis=0)
            assert np.array_equal(extended[:rgb_stop(start)],published)
            row["old_RGB_unchanged"]=True
            np.save(path/f"published_{rgb_stop(stop)}.npy",extended)
            write_video([Image.fromarray(frame) for frame in extended],
                        path/f"rollout_{rgb_stop(stop)}.mp4")
            chunk_video=path/f"chunk_{start}_{stop}.mp4"
            new_frames=extended[rgb_stop(start):]
            write_video([Image.fromarray(frame) for frame in new_frames],chunk_video)
            sheet=Image.new("RGB",(4*416,5*240),color=(0,0,0))
            for frame_index,frame in enumerate(new_frames):
                image=Image.fromarray(frame).resize((416,240))
                sheet.paste(image,((frame_index%4)*416,(frame_index//4)*240))
            sheet.save(path/f"chunk_{start}_{stop}_all_new_frames.jpg",quality=90)
            gray=np.stack([np.asarray(Image.fromarray(frame).convert("L"),dtype=np.float32)
                           for frame in extended[rgb_stop(start)-1:]])
            row["boundary_gray_MAD"]=float(np.abs(gray[1]-gray[0]).mean())
            row["inside_gray_MAD"]=float(np.abs(np.diff(gray[1:],axis=0)).mean())
            row["flow"]=flow_metrics(chunk_video)
            row["published_RGB_sha256"]=rgb_sha(extended)
            row["peak_allocated_gib"]=peak_check()
            row["peak_reserved_gib"]=torch.cuda.max_memory_reserved()/2**30
            row["cpu_rss_mib_after"]=rss_mib()
            row["cpu_peak_rss_mib"]=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
            row["chunk_wall_seconds"]=time.perf_counter()-chunk_tick
            row["status"]="complete_pending_judge_review"
            atomic_json(row_path,row)
            result["chunks"].append(row)
            atomic_json(out/"result.json",result)
            if index==6:
                history=torch.cat((history,current),dim=2)
                published=extended
                pipe.load_models_to_device(["dit"]);torch.cuda.synchronize();peak_check()
            elif action=="A":
                pipe.load_models_to_device(["dit"]);torch.cuda.synchronize();peak_check()
        assert cache_identity(cache)==shared_signature
        del branch_cache
    result["shared_cache_unchanged"]=cache_identity(cache)==shared_signature
    result["frozen_parameters_unchanged"]=all(p._version==v for p,v in versions)
    if not result["shared_cache_unchanged"] or not result["frozen_parameters_unchanged"]:
        raise RuntimeError("G1 frozen state changed")
    result["peak_allocated_gib"]=peak_check()
    result["peak_reserved_gib"]=torch.cuda.max_memory_reserved()/2**30
    result["cpu_peak_rss_mib"]=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    result["status"]="complete_pending_judge_review"
    atomic_json(out/"result.json",result)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight",action="store_true")
    parser.add_argument("--run",action="store_true")
    args=parser.parse_args()
    if args.preflight==args.run:parser.error("choose exactly one mode")
    if args.preflight:
        manifest=preflight()
        print(f"G1 CPU preflight: {len(manifest['sources'])} sources verified",flush=True)
        return
    cfg=json.loads(CONFIG.read_text());out=Path(cfg["output_root"])
    out.mkdir(parents=True,exist_ok=True)
    with (out/"worker.lock").open("a+") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (out/"budget.json").exists():raise RuntimeError("G1 already attempted; no retry")
        free=int(subprocess.check_output(["nvidia-smi",f"--id={cfg['gpu']}",
            "--query-gpu=memory.free","--format=csv,noheader,nounits"],text=True).strip())
        if free<44000:raise RuntimeError(f"GPU0 not idle: {free} MiB free")
        verify_inputs();verify_authorization()
        os.environ.update(CUDA_VISIBLE_DEVICES="0",ABOT_VRAM_RESERVE_GIB="18",
            HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",
            DIFFSYNTH_SKIP_DOWNLOAD="True",TOKENIZERS_PARALLELISM="false",
            PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
        sys.path[:0]=[str(FROZEN/"runtime/code"),str(FROZEN/"runtime/code/abot"),
            str(FROZEN/"runtime/code/causal"),str(FROZEN/"runtime/DiffSynth-Studio-h3-v2"),
            str(EXP003),str(EXP001),str(ROUTER),str(EXP),str(HERE)]
        budget=Budget(cfg,out)
        budget.check()
        def timeout_handler(_signum,_frame):raise TimeoutError("G1 absolute 0.70GPUh deadline")
        signal.signal(signal.SIGALRM,timeout_handler)
        signal.setitimer(signal.ITIMER_REAL,
            min(cfg["max_gpu_seconds"],10800-(time.time()-budget.core_start)))
        try:run(cfg,budget)
        except BaseException as exc:
            budget.ledger.update(status="failed",error=repr(exc));budget.write()
            result=out/"result.json"
            if result.exists():
                data=json.loads(result.read_text());data.update(status="failed",error=repr(exc))
                atomic_json(result,data)
            raise
        else:
            budget.ledger["status"]="complete_pending_judge_review";budget.write()
        finally:signal.setitimer(signal.ITIMER_REAL,0)


if __name__=="__main__":main()
