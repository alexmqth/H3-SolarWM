"""Fresh full-124 native 12+5 rollout for V2b, V3 FM30/FM8 and V4 cycle8.

This imports the frozen interval implementations and never loads historical
endpoints or K/V. One process owns all six chunks. No training or retry logic.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
FROZEN = ROOT / "H3-World/outputs/2026-10-09-22/chunk_partition_cb"
EXP = ROOT / "submission/experiments"
sys.path[:0] = [str(FROZEN / "runtime/code"), str(FROZEN / "runtime/code/abot"),
    str(FROZEN / "runtime/code/causal"), str(FROZEN / "runtime/DiffSynth-Studio-h3-v2"),
    str(EXP / "EXP-001_v2b_124"), str(EXP / "EXP-005_v3_sliding_window"),
    str(EXP / "EXP-007_v3_anyflow"), str(EXP / "EXP-008_v3_dmd/dmd_eval"),
    str(ROOT / "submission/reports/stage1_anyflow/02_causal_diagnostics/current_prefix_candidate")]


def tensor_sha(x):
    h = hashlib.sha256()
    y = x.detach().cpu().contiguous()
    h.update(str((tuple(y.shape), str(y.dtype))).encode())
    h.update(y.view(-1).view(__import__("torch").uint8).numpy().tobytes())
    return h.hexdigest()


def atomic(path, obj):
    temp = path.with_suffix(path.suffix + ".partial")
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")
    os.replace(temp, path)


def run(row: str, out: Path):
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
    from interval_forward import interval_forward
    from interval_sw import interval_sw
    from interval_student import interval_student
    from causal.anyflow import install_anyflow, finite_map_step
    from causal.pretrained_lora import install_adapters
    from rollout_contract import prompt_for_path, rgb_stop, stitch_immutable
    from chunk_plan import PLAN

    torch.set_num_threads(4)
    torch.manual_seed(13)
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats(0)
    began = time.perf_counter()
    state = {"row": row, "status": "loading", "partition": [12,5,5,5,5,5],
        "seed": 13, "flow_shift": 2.22, "sampling_nfe_total": 0,
        "commit_forward_total": 0, "prefill_forward_total": 0,
        "vae_decode_count": 0, "image_encode_count": None, "text_encode_count": None,
        "condition_encoder_call_count_note": "unit_runner executed fully; individual text/image unit calls not instrumented",
        "chunks": [], "fresh_first_chunk": True, "historical_endpoint_reads": 0,
        "historical_kv_reads": 0, "precision_profile": "h3_fp32"}
    metrics_path = out / "native_metrics.json"
    atomic(metrics_path, state)
    try:
        # Frozen full37 fixture is used only as a read-only protocol witness.
        # Each row actually encodes its own text and I0 and creates its own
        # seed-13 video/audio noise, as in the V0/V1/V2a benchmark.
        fixture = FROZEN / "source_coarse/inputs/parking_A.pt"
        data = {"A": torch.load(fixture, map_location="cpu", weights_only=True)}
        assert data["A"]["initial_noise"].shape == (1,24,37,30,52)
        assert data["A"]["pairs"][0]["prompts"]["A"].shape[0] == len(data["A"]["packed"]["text_pos"])
        state["fixture_sha256"] = hashlib.sha256(fixture.read_bytes()).hexdigest()
        pipe = abot.load_pipeline("cuda:0")
        lora = FROZEN / "runtime/checkpoints/H3-World/step-10000.safetensors"
        pipe.load_lora(pipe.dit, state_dict=abot.load_checkpoint_lora(lora), hotload=True)
        model = pipe.dit.eval().requires_grad_(False)
        torch.cuda.synchronize()
        state["model_load_seconds"] = time.perf_counter()-began
        # Native A script and the same I0/prompt used by frozen benchmark.
        import action_script as action_script
        keys=np.zeros((37,len(action_script.KEYS9)),dtype=np.int64)
        keys[:,action_script.KEYS9.index("A")]=1
        shared=dict(cfg_scale=1.,height=480,width=832,num_frames=124,seed=13,rand_device="cpu",
            keyframes=[abot.load_first_frame(ROOT/"H3-World/examples/first_frame.png")],
            keyframe_indices=[0],imgvid_cond_noise_aug=.999,audio_cond_noise_aug=1.)
        pos=dict(prompt="A man in a yellow floral shirt stands in a dim, multi-level concrete parking garage.",
            action_script=action_script.annotate_from_keys9(keys))
        neg={}
        condition_tick=time.perf_counter()
        with torch.no_grad():
            for unit in pipe.units:
                shared,pos,neg=pipe.unit_runner(unit,pipe,shared,pos,neg)
        torch.cuda.synchronize()
        state["conditioning_seconds"]=time.perf_counter()-condition_tick
        initial=shared["video_latents"].to("cuda:0",torch.float32)
        audio=shared["audio_latents"].to("cuda:0")
        anchor=shared["keyframe_cond_anchor"].to("cuda:0")
        full=pos["packed"]
        full_prompt=pos["prompt_embeds"]
        assert torch.equal(shared["video_latents"].cpu(),data["A"]["initial_noise"])
        assert torch.equal(shared["audio_latents"].cpu(),data["A"]["audio_noise"])
        assert torch.equal(shared["keyframe_cond_anchor"].cpu(),data["A"]["anchor"])
        assert torch.equal(full["img_position_ids"].cpu(),data["A"]["packed"]["img_position_ids"])
        assert torch.equal(full_prompt.cpu(),data["A"]["pairs"][0]["prompts"]["A"])
        state["condition_matches_frozen_fixture"] = True
        state["video_noise_sha256"] = hashlib.sha256(shared["video_latents"].detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
        state["audio_noise_sha256"] = hashlib.sha256(shared["audio_latents"].detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
        state["anchor_sha256"] = tensor_sha(anchor)
        tick=time.perf_counter()
        state["precision"] = configure_precision(model, "h3_fp32",
            native_transformer_dir=FROZEN / "runtime/DiffSynth-Studio-h3-v2/models/MiniMax/MiniMax-H3/FL2VA/transformer")
        pipe.load_models_to_device(["dit"])
        torch.cuda.synchronize()
        state["dit_prepare_seconds"] = time.perf_counter()-tick
        if row == "v4":
            train = ROOT / "H3-World/outputs/EXP-008_v3_dmd_train_v2"
            result = json.loads((train / "result.json").read_text())
            assert result["last_complete_cycle"] == 8
            receipt = next(x for x in result["checkpoint_paths"] if x["cycle"] == 8)
            ckpt = ROOT / receipt["path"]
            qkv = torch.load(ckpt / "student_qkv.pt", map_location="cpu", weights_only=True)
            target_state = torch.load(ckpt / "student_target.pt", map_location="cpu", weights_only=True)
            assert qkv["rank"] == 8 and qkv["block_indices"] == list(range(42,50))
            adapters, _ = install_adapters(model, rank=8, block_indices=tuple(range(42,50)), device="cuda:0")
            target = install_anyflow(model, device="cuda:0", gate=.25)
            with torch.no_grad():
                for adapter, a, b in zip(adapters, qkv["lora_A"], qkv["lora_B"]):
                    adapter.lora_A.copy_(a.to("cuda:0")); adapter.lora_B.copy_(b.to("cuda:0"))
            target.load_state_dict(target_state["weights"], strict=True)
            model.requires_grad_(False)
            state["checkpoint"] = str(ckpt)
        torch.cuda.synchronize()
        state["load_seconds"] = state["model_load_seconds"]+state["dit_prepare_seconds"]
        steps = 30 if row in ("v2b","v3_30") else 8
        sigmas = configure_video_schedule(pipe.scheduler, steps=steps, grid="native", flow_shift=2.22)
        assert len(sigmas) == steps+1 and sigmas[0] == 1 and sigmas[-1] == 0
        state["sigmas"] = sigmas
        state["sampling_steps_per_chunk"] = steps
        cache = None if row == "v2b" else H3ChunkCache(max_history=5,storage_device="cpu")
        done = []
        published = None
        torch.cuda.synchronize()
        ready = time.perf_counter()
        state["model_and_condition_ready_seconds"] = ready - began
        sampling_total = 0.; commit_total = 0.; decode_total = 0.
        for index in range(6):
            start, stop = PLAN.span(index)
            assert stop <= 37
            layout, prompt = visible_inputs(full, full_prompt, stop, 390)
            assert len(layout["action_text_rows"]) == stop
            current = initial[:,:,start:stop].clone()
            chunk = {"index": index, "latent_interval": [start,stop],
                     "rgb_interval": [0 if index==0 else rgb_stop(start),rgb_stop(stop)],
                     "noise_sha256": tensor_sha(current), "sampling_nfe": 0,
                     "commit_forward": 0, "cache_before_bytes": 0 if cache is None else cache.nbytes}
            atomic(metrics_path, {**state,"status":"sampling","current_chunk":chunk})
            tick = time.perf_counter()
            with torch.no_grad(), current_prefix_feedback():
                for j,(sigma,next_sigma) in enumerate(zip(sigmas[:-1],sigmas[1:])):
                    if row == "v2b":
                        history = torch.cat(done,dim=2) if done else initial[:,:,:0]
                        velocity = interval_forward(model,current,start=start,history=history,
                            history_noise=initial[:,:,:start],mode="sigma_noised",
                            full_packed=layout,prompt=prompt,anchor=anchor,audio=audio,sigma=sigma)
                    elif row == "v4":
                        velocity = interval_student(model,current,start=start,index=index,cache=cache,
                            full_packed=layout,prompt=prompt,anchor=anchor,audio=audio,
                            sigma=sigma,target_sigma=next_sigma)
                    else:
                        velocity = interval_sw(model,current,start=start,index=index,cache=cache,
                            full_packed=layout,prompt=prompt,anchor=anchor,audio=audio,sigma=sigma,mode="global")
                    state["sampling_nfe_total"] += 1
                    chunk["sampling_nfe"] += 1
                    current = (finite_map_step(current,velocity,sigma,next_sigma) if row=="v4"
                               else pipe.scheduler.step(velocity,pipe.scheduler.timesteps[j],current))
                    if not torch.isfinite(current).all():
                        raise FloatingPointError(f"nonfinite endpoint, chunk={index}, step={j}")
                    if torch.cuda.max_memory_allocated() > 44 * 2**30:
                        raise MemoryError("GPU allocated over 44GiB")
                    if (j+1)%5 == 0 or j+1==steps:
                        chunk["last_step"] = j+1
                        atomic(metrics_path,{**state,"status":"sampling","current_chunk":chunk})
                        print(f"[{row}] chunk {index+1}/6 step {j+1}/{steps}",flush=True)
            torch.cuda.synchronize()
            chunk["sampling_seconds"] = time.perf_counter()-tick
            sampling_total += chunk["sampling_seconds"]
            done.append(current)
            if cache is not None and index < 5:
                tick=time.perf_counter()
                with torch.no_grad(), current_prefix_feedback():
                    if row == "v4":
                        _ = interval_student(model,current,start=start,index=index,cache=cache,
                            full_packed=layout,prompt=prompt,anchor=anchor,audio=audio,
                            sigma=0,target_sigma=0,commit=True)
                    else:
                        _ = interval_sw(model,current,start=start,index=index,cache=cache,
                            full_packed=layout,prompt=prompt,anchor=anchor,audio=audio,
                            sigma=0,commit=True,mode="global")
                torch.cuda.synchronize()
                chunk["commit_seconds"] = time.perf_counter()-tick
                commit_total += chunk["commit_seconds"]
                state["commit_forward_total"] += 1
                chunk["commit_forward"] = 1
                chunk["cache_after_bytes"] = cache.nbytes
                state["cpu_kv_peak_gib"] = max(state.get("cpu_kv_peak_gib",0),cache.peak_bytes/2**30)
            pipe.load_models_to_device(["video_vae"])
            tick=time.perf_counter()
            with torch.no_grad():
                latent_prefix = torch.cat(done,dim=2)
                decoded=pipe.video_vae.decode_video(latent_prefix,dtype=pipe.torch_dtype,
                    process_image=False,tiled=True,tile_size=256,tile_overlap=64)
                frames=pipe.vae_output_to_video(decoded,min_value=0,max_value=1)
            rgb=np.stack([np.asarray(im.convert("RGB"),dtype=np.uint8) for im in frames])
            assert len(rgb)==rgb_stop(stop)
            published=rgb if published is None else stitch_immutable(published,rgb,start,stop)
            assert len(published)==rgb_stop(stop)
            state["vae_decode_count"] += 1
            torch.cuda.synchronize()
            chunk["decode_seconds"] = time.perf_counter()-tick
            decode_total += chunk["decode_seconds"]
            state["chunks"].append(chunk)
            state["peak_allocated_gib"] = torch.cuda.max_memory_allocated()/2**30
            state["peak_reserved_gib"] = torch.cuda.max_memory_reserved()/2**30
            atomic(metrics_path,{**state,"status":"chunk_complete"})
            if index < 5:
                pipe.load_models_to_device(["dit"])
        assert published is not None and published.shape==(124,480,832,3)
        state["sampling_loop_inclusive_seconds"] = sampling_total
        state["commit_seconds"] = commit_total
        state["decode_seconds"] = decode_total
        tick=time.perf_counter()
        write_video([Image.fromarray(f) for f in published],out/"full_124.mp4")
        state["encode_seconds"] = time.perf_counter()-tick
        state["ready_to_video_seconds"] = time.perf_counter()-ready
        state["peak_allocated_gib"] = torch.cuda.max_memory_allocated()/2**30
        state["peak_reserved_gib"] = torch.cuda.max_memory_reserved()/2**30
        state["status"] = "complete"
        state["process_to_final_file_seconds"] = time.perf_counter()-began
        atomic(metrics_path,state)
    except BaseException as exc:
        state["status"]="failed"
        state["error"]=repr(exc)
        state["process_to_failure_seconds"]=time.perf_counter()-began
        if __import__("torch").cuda.is_initialized():
            state["peak_allocated_gib"]=__import__("torch").cuda.max_memory_allocated()/2**30
        atomic(metrics_path,state)
        raise


if __name__ == "__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--row",choices=["v2b","v3_30","v3_8","v4"],required=True)
    ap.add_argument("--out",type=Path,required=True)
    a=ap.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    run(a.row,a.out)
