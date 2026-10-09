"""EXP-002 controlled native-condition persistent-KV rollout. One path/chunk per call.

Run AA second first to commit shared first12 once, then AD second. Inspect both
videos before explicitly running either third chunk. No training or GT reset.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CFG = json.loads((HERE / "config.json").read_text())
OUT = Path(CFG["output_root"])
FROZEN = ROOT / "H3-World/outputs/2026-10-09-22/chunk_partition_cb"
SOURCE = ROOT / "submission/experiments/11_causal_12_then5_selfhistory/states/first12_A.pt"
OLD = ROOT / "submission/experiments/EXP-001_v2b_124"
CANDIDATE = ROOT / "submission/reports/stage1_anyflow/02_causal_diagnostics/current_prefix_candidate"


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def th(x):
    import torch
    x = x.detach().cpu().contiguous()
    return hashlib.sha256(str((tuple(x.shape), str(x.dtype))).encode() +
                          x.view(torch.uint8).numpy().tobytes()).hexdigest()


def save_json(path, obj):
    tmp = Path(str(path) + ".partial")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def check_sources():
    assert sha(SOURCE) == "242a1db06bc3423fe21207af5d2ccf59c9f27f07c9d88f22ce9f77921f6b0eea"
    manifest = json.loads((OLD / "source_manifest_v3.json").read_text())
    assert sha(FROZEN / "runtime/checkpoints/H3-World/step-10000.safetensors") == manifest["released_lora_sha256"]
    proto = json.loads((FROZEN / "protocol.json").read_text())
    for rel, expected in proto["inputs_sha256"].items():
        assert sha(ROOT / rel) == expected, rel
    assert sha(HERE / "config.json")


def reserve_call(ledger, kind):
    elapsed = time.time() - ledger["first_start"]
    if elapsed > CFG["max_elapsed_hours"] * 3600:
        raise TimeoutError("four-hour EXP-002 deadline")
    if ledger["gpu_seconds"] + time.time() - ledger["active_start"] > CFG["max_gpu_hours"] * 3600:
        raise TimeoutError("three GPU-hour EXP-002 budget")
    if ledger["total_forwards"] >= CFG["max_forwards"]:
        raise RuntimeError("forward budget exhausted")
    if kind == "sampling" and ledger["sampling"] >= CFG["max_sampling"]:
        raise RuntimeError("sampling budget exhausted")
    if kind == "commit" and ledger["prefill_commit"] >= CFG["max_prefill_commit"]:
        raise RuntimeError("prefill/commit budget exhausted")
    if kind == "diagnostic" and ledger["diagnostic"] >= 4:
        raise RuntimeError("diagnostic budget exhausted")
    ledger["total_forwards"] += 1  # count attempts, including failed calls
    ledger[{"sampling":"sampling", "commit":"prefill_commit", "diagnostic":"diagnostic"}[kind]] += 1
    save_json(OUT / "budget.json", ledger)


def cache_info(cache):
    return {"nbytes": cache.nbytes, "peak_bytes": cache.peak_bytes,
            "layer_count": len(cache.layers), "entries_per_layer":
            {str(k): [(e.index, int(e.key.shape[0])) for e in v]
             for k, v in cache.layers.items()}, "commit_calls_per_layer": cache.commits}


def decode_video(pipe, z):
    import numpy as np
    with __import__("torch").no_grad():
        rgb = pipe.video_vae.decode_video(z, dtype=pipe.torch_dtype,
            process_image=False, tiled=True, tile_size=256, tile_overlap=64)
        frames = pipe.vae_output_to_video(rgb, min_value=0, max_value=1)
    return np.stack([np.asarray(im.convert("RGB"), dtype=np.uint8) for im in frames])


def first39():
    import av
    import numpy as np
    source = FROZEN / "C_A/sigma_noised_A_rollout.mp4"
    with av.open(str(source)) as container:
        frames = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
    assert len(frames) == 56
    return np.stack(frames[:39]), {"source": str(source), "sha256": sha(source),
                                   "note": "first39 decoded from existing V2b file; its H.264 pixels are frozen"}


def run(args, ledger):
    import numpy as np
    import torch
    from PIL import Image
    import infer as abot
    from benchmark import write_video
    from causal.h3_cached import H3ChunkCache
    from causal.h3_precision import configure_precision
    from causal.local_topology import visible_inputs
    from causal.anyflow_sampling import configure_video_schedule
    from current_prefix import current_prefix_feedback
    from evaluate_action_control import evaluate as flow_metrics
    from interval_cached import interval_cached
    from rollout_contract import prompt_for_path, rgb_sha, stitch_immutable, rgb_stop

    torch.set_num_threads(4)
    torch.manual_seed(13)
    path_dir = OUT / args.path
    path_dir.mkdir(exist_ok=True)
    start, stop, index = (12, 17, 1) if args.stage == "second" else (17, 22, 2)
    row_path = path_dir / f"chunk_{start}_{stop}.json"
    assert not row_path.exists(), "prior attempt exists; inspect before any retry"
    row = dict(task="EXP-002/v1", stage=args.stage, path=args.path,
        interval=[start, stop], gpu=args.gpu, status="loading", seed=13,
        forward_count_start=ledger["total_forwards"], config_sha256=sha(HERE/"config.json"),
        runner_sha256=sha(__file__), interval_sha256=sha(HERE/"interval_cached.py"),
        current_prefix_sha256=sha(CANDIDATE/"current_prefix.py"),
        first12_file_sha256=sha(SOURCE), optimizer_updates=0, vae_decodes=0)
    save_json(row_path, row)
    began = time.perf_counter()
    try:
        data = {a: torch.load(FROZEN / f"source_coarse/inputs/parking_{a}.pt",
                              map_location="cpu", weights_only=True) for a in "AD"}
        for key in ("initial_noise", "audio_noise", "anchor"):
            assert torch.equal(data["A"][key], data["D"][key]), key
        assert torch.equal(data["A"]["packed"]["img_position_ids"],
                           data["D"]["packed"]["img_position_ids"])
        pipe = abot.load_pipeline("cuda:0")
        released = FROZEN / "runtime/checkpoints/H3-World/step-10000.safetensors"
        pipe.load_lora(pipe.dit, state_dict=abot.load_checkpoint_lora(released), hotload=True)
        model = pipe.dit.eval().requires_grad_(False)
        row["precision"] = configure_precision(model, "h3_fp32",
            native_transformer_dir=FROZEN / "runtime/DiffSynth-Studio-h3-v2/models/MiniMax/MiniMax-H3/FL2VA/transformer")
        pipe.load_models_to_device(["dit"])
        torch.cuda.synchronize()
        row["load_seconds"] = time.perf_counter() - began
        row["released_lora_sha256"] = sha(released)
        frozen_versions = [(p, p._version) for p in model.parameters()]
        def move(x):
            if torch.is_tensor(x): return x.to("cuda:0")
            if isinstance(x, dict): return {k: move(v) for k,v in x.items()}
            return x
        initial = data["A"]["initial_noise"].to("cuda:0", torch.float32)
        first = torch.load(SOURCE, map_location="cpu", weights_only=True).to("cuda:0", torch.float32)
        assert first.shape == (1,24,12,30,52) and initial.shape[2] == 37
        audio = data["A"]["audio_noise"].to("cuda:0")
        anchor = data["A"]["anchor"].to("cuda:0")
        full = move(data["A"]["packed"])
        shared_cache = OUT / "first12_A_cache.pt"
        if args.path == "AA" and args.stage == "second" and not shared_cache.exists():
            cache = H3ChunkCache(max_history=5, storage_device="cpu")
            prior_prompt = data["A"]["pairs"][0]["prompts"]["A"].to("cuda:0")
            prior_layout, prior_prompt = visible_inputs(full, prior_prompt, 12, 390)
            probe = torch.load(FROZEN / "source_coarse/coarse_A/window0_solver_states.pt",
                               map_location="cpu", weights_only=True)[1]
            reserve_call(ledger, "diagnostic")
            with current_prefix_feedback(), torch.no_grad():
                observed = interval_cached(model, probe["state"].to("cuda:0"),
                    start=0, index=0, cache=cache, full_packed=prior_layout,
                    prompt=prior_prompt, anchor=anchor, audio=audio,
                    sigma=probe["sigma"])
            reference = probe["velocity_A"].to("cuda:0")
            delta = (observed-reference).float()
            row["first12_identity_relative_RMS"] = float(
                delta.square().mean().sqrt()/reference.float().square().mean().sqrt().clamp_min(1e-12))
            row["first12_identity_max_abs"] = float(delta.abs().max())
            assert row["first12_identity_relative_RMS"] < 1e-5, "first12 candidate mismatches Original"
            assert not cache.layers, "diagnostic forward must not mutate cache"
            del probe, observed, reference, delta
            reserve_call(ledger, "commit")
            with current_prefix_feedback(), torch.no_grad():
                _ = interval_cached(model, first, start=0, index=0, cache=cache,
                    full_packed=prior_layout, prompt=prior_prompt, anchor=anchor,
                    audio=audio, sigma=0, commit=True)
            assert len(cache.layers) > 0 and all(len(v)==1 for v in cache.layers.values())
            tmp = Path(str(shared_cache)+".partial")
            torch.save(cache, tmp); os.replace(tmp, shared_cache)
            row["first12_prefill_seconds"] = time.perf_counter()-began-row["load_seconds"]
        else:
            assert shared_cache.exists(), "AA second must prefill common history first"
            cache = torch.load(shared_cache, map_location="cpu", weights_only=False)
        assert isinstance(cache, H3ChunkCache)
        row["first12_cache_sha256"] = sha(shared_cache)
        row["first12_cache"] = cache_info(cache)
        if args.stage == "third":
            second_file = path_dir / "chunk_12_17.pt"
            cache_file = path_dir / "cache_through17.pt"
            assert second_file.exists() and cache_file.exists()
            second = torch.load(second_file, map_location="cpu", weights_only=True).to("cuda:0", torch.float32)
            cache = torch.load(cache_file, map_location="cpu", weights_only=False)
            row["second_endpoint_sha256"] = sha(second_file)
            row["cache_through17_sha256"] = sha(cache_file)
            history = torch.cat((first, second), dim=2)
        else:
            history = first
        history_hash = th(history)
        prompt = prompt_for_path(data, args.path, stop).to("cuda:0")
        layout, prompt = visible_inputs(full, prompt, stop, 390)
        assert len(layout["action_text_rows"]) == stop
        row.update(history_tensor_sha256=history_hash, initial_noise_sha256=th(initial),
            prompt_sha256=th(prompt), position_sha256=th(layout["img_position_ids"]),
            anchor_sha256=th(anchor), audio_sha256=th(audio))
        sigmas = configure_video_schedule(pipe.scheduler, steps=30, grid="native", flow_shift=2.22)
        row["sigmas"] = sigmas
        current = initial[:, :, start:stop].clone()
        current_noise_hash = th(current)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize(); tick = time.perf_counter()
        row["status"] = "sampling"; save_json(row_path, row)
        with current_prefix_feedback(), torch.no_grad():
            for j,t in enumerate(pipe.scheduler.timesteps):
                reserve_call(ledger, "sampling")
                velocity = interval_cached(model, current, start=start, index=index,
                    cache=cache, full_packed=layout, prompt=prompt, anchor=anchor,
                    audio=audio, sigma=float(t)/1000)
                current = pipe.scheduler.step(velocity, t, current)
                if not torch.isfinite(current).all(): raise FloatingPointError("nonfinite latent")
                if torch.cuda.max_memory_allocated()/2**30 > CFG["max_allocated_gib"]:
                    raise MemoryError("allocated memory exceeds 44 GiB")
                if (j+1)%5 == 0:
                    row["sampling_completed"] = j+1; save_json(row_path,row)
                    print(f"{args.path} {start}:{stop} {j+1}/30",flush=True)
            torch.cuda.synchronize(); row["sampling_seconds"] = time.perf_counter()-tick
            assert th(history)==history_hash and th(initial[:,:,start:stop])==current_noise_hash
            endpoint = path_dir / f"chunk_{start}_{stop}.pt"
            tmp = Path(str(endpoint)+".partial")
            torch.save(current.detach().cpu(),tmp); os.replace(tmp,endpoint)
            row["endpoint_sha256"] = sha(endpoint)
            row["endpoint_tensor_sha256"] = th(current)
            row["cache_read_unchanged"] = (sha(shared_cache)==row["first12_cache_sha256"])
            if args.stage == "second":
                reserve_call(ledger,"commit")
                _ = interval_cached(model,current,start=start,index=index,cache=cache,
                    full_packed=layout,prompt=prompt,anchor=anchor,audio=audio,sigma=0,commit=True)
                assert all(len(v)==2 for v in cache.layers.values())
                cache_file = path_dir / "cache_through17.pt"
                tmp = Path(str(cache_file)+".partial")
                torch.save(cache,tmp); os.replace(tmp,cache_file)
                row["cache_through17_sha256"] = sha(cache_file)
                row["cache_through17"] = cache_info(cache)
        del velocity
        pipe.load_models_to_device(["video_vae"])
        torch.cuda.synchronize(); decode_tick = time.perf_counter()
        if args.stage == "second":
            published, prior_info = first39()
            row["first39"] = prior_info
        else:
            published_file = path_dir / "published_56.npy"
            published = np.load(published_file)
            assert sha(published_file) == json.loads((path_dir/"chunk_12_17.json").read_text())["published_file_sha256"]
        assert len(published)==rgb_stop(start)
        prior_rgb_hash=rgb_sha(published)
        assert ledger["vae_decodes"] < CFG["max_decode"]
        decoded=decode_video(pipe,torch.cat((history,current),dim=2)); row["vae_decodes"]+=1
        assert len(decoded)==rgb_stop(stop)
        row["past5_redecode_MAD"] = float(np.abs(
            decoded[rgb_stop(start)-5:rgb_stop(start)].astype(np.float32)-
            published[-5:].astype(np.float32)).mean())
        appended=stitch_immutable(published,decoded,start,stop)
        assert rgb_sha(appended[:len(published)])==prior_rgb_hash
        publish_file=path_dir/f"published_{rgb_stop(stop)}.npy"
        np.save(publish_file,appended)
        row["published_file_sha256"]=sha(publish_file)
        row["published_RGB_sha256"]=rgb_sha(appended)
        row["prior_RGB_sha256"]=prior_rgb_hash
        row["decode_seconds"]=time.perf_counter()-decode_tick
        video_file=path_dir/f"rollout_{rgb_stop(stop)}.mp4"
        active_file=path_dir/f"chunk_{start}_{stop}.mp4"
        write_video([Image.fromarray(f) for f in appended],video_file)
        write_video([Image.fromarray(f) for f in appended[rgb_stop(start):]],active_file)
        row["video_sha256"]=sha(video_file)
        row["active_video_sha256"]=sha(active_file)
        row["flow"]=flow_metrics(active_file)
        gray=np.stack([np.asarray(Image.fromarray(f).convert("L"),dtype=np.float32)
                       for f in appended[-18:]])
        row["boundary_gray_MAD"]=float(np.abs(gray[1]-gray[0]).mean())
        row["inside_gray_MAD"]=float(np.abs(np.diff(gray[1:],axis=0)).mean())
        row["GPU_peak_allocated_MiB"]=torch.cuda.max_memory_allocated()/2**20
        row["GPU_peak_reserved_MiB"]=torch.cuda.max_memory_reserved()/2**20
        row["frozen_parameter_versions_unchanged"] = all(p._version==v for p,v in frozen_versions)
        assert row["frozen_parameter_versions_unchanged"]
        row["wall_seconds"]=time.perf_counter()-began
        row["forward_count_end"]=ledger["total_forwards"]
        row["status"]="complete_pending_visual_review"
        save_json(row_path,row)
        print(json.dumps({"path":args.path,"stage":args.stage,"video":str(video_file),
              "flow":row["flow"]["horizontal_flow_px"]["mean"],"status":row["status"]}),flush=True)
    except BaseException as exc:
        row.update(status="failed",error=repr(exc),wall_seconds=time.perf_counter()-began,
                   forward_count_end=ledger["total_forwards"])
        save_json(row_path,row)
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",type=Path,required=True)
    p.add_argument("--stage",choices=["second","third"],required=True)
    p.add_argument("--path",choices=["AA","AD"],required=True)
    p.add_argument("--gpu",type=int,required=True)
    args=p.parse_args()
    assert args.config.resolve()==(HERE/"config.json").resolve()
    check_sources()
    assert not (OUT/args.path/f"chunk_{12 if args.stage=='second' else 17}_{17 if args.stage=='second' else 22}.json").exists()
    free=int(subprocess.check_output(["nvidia-smi",f"--id={args.gpu}",
        "--query-gpu=memory.free","--format=csv,noheader,nounits"],text=True).strip())
    assert free>=44000, ("GPU not idle",args.gpu,free)
    cutoff=datetime.fromisoformat(CFG["gpu_cutoff_hkt"]).timestamp()
    # One GPU process at a time; the user permits up to 8/3 project-wide.
    assert 0 <= args.gpu < 8
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/"worker.lock").open("a+") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        ledger_file=OUT/"budget.json"
        ledger=json.loads(ledger_file.read_text()) if ledger_file.exists() else dict(
            first_start=time.time(),gpu_seconds=0,total_forwards=0,sampling=0,
            prefill_commit=0,diagnostic=0,vae_decodes=0,runs=[])
        assert time.time()-ledger["first_start"]<CFG["max_elapsed_hours"]*3600
        assert ledger["gpu_seconds"]<CFG["max_gpu_hours"]*3600
        save_json(ledger_file,ledger)
        t=time.time()
        ledger["active_start"] = t
        os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu),ABOT_VRAM_RESERVE_GIB="18",
            HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",DIFFSYNTH_SKIP_DOWNLOAD="True",
            TOKENIZERS_PARALLELISM="false",PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
        sys.path[:0]=[str(FROZEN/"runtime/code"),str(FROZEN/"runtime/code/abot"),
            str(FROZEN/"runtime/code/causal"),str(FROZEN/"runtime/DiffSynth-Studio-h3-v2"),
            str(OLD),str(CANDIDATE)]
        try:
            run(args,ledger)
            status="complete"
        except BaseException:
            status="failed"; raise
        finally:
            ledger["gpu_seconds"] += time.time()-t
            ledger["vae_decodes"] += json.loads((OUT/args.path/f"chunk_{12 if args.stage=='second' else 17}_{17 if args.stage=='second' else 22}.json").read_text()).get("vae_decodes",0)
            ledger.pop("active_start",None)
            ledger["runs"].append(dict(path=args.path,stage=args.stage,gpu=args.gpu,
                start=t,end=time.time(),status=status))
            save_json(ledger_file,ledger)


if __name__=="__main__": main()
