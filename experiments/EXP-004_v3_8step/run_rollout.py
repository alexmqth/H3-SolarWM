"""EXP-004: V3 original weights, native FM/Euler, eight steps on new chunks.

Reuse EXP-002 first12 and committed raw KV. One path/chunk per GPU process.
No prefill, diagnostic, training, AnyFlow, or baseline inference.
"""
from __future__ import annotations

import argparse
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
PREV = ROOT / "H3-World/outputs/EXP-002_native_cached"
PREV_CODE = ROOT / "submission/experiments/EXP-002_native_cached"
FROZEN = ROOT / "H3-World/outputs/2026-10-09-22/chunk_partition_cb"
FIRST = ROOT / "submission/experiments/11_causal_12_then5_selfhistory/states/first12_A.pt"
OLD = ROOT / "submission/experiments/EXP-001_v2b_124"
ROUTER = ROOT / "submission/reports/stage1_anyflow/02_causal_diagnostics/current_prefix_candidate"


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def tensor_sha(t):
    import torch
    t = t.detach().cpu().contiguous()
    return hashlib.sha256(str((tuple(t.shape), str(t.dtype))).encode() +
                          t.view(torch.uint8).numpy().tobytes()).hexdigest()


def atomic_json(path, value):
    tmp = Path(str(path) + ".partial")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def cache_info(cache):
    return dict(nbytes=cache.nbytes, layer_count=len(cache.layers),
                entries_per_layer={str(k): [(e.index, int(e.key.shape[0])) for e in v]
                                   for k, v in cache.layers.items()},
                commit_calls_per_layer=cache.commits)


def cache_identity(cache):
    return (cache.commits, tuple((layer, tuple(
        (e.index, id(e), tuple((id(t), t.data_ptr(), t._version, tuple(t.shape))
                                  for t in (e.key, e.value, e.rope))) for e in entries))
        for layer, entries in sorted(cache.layers.items())))


def reserve(ledger, kind):
    if time.time() - ledger["first_start"] > CFG["max_elapsed_minutes"] * 60:
        raise TimeoutError("EXP-004 elapsed limit")
    if ledger["gpu_seconds"] + time.time() - ledger["active_start"] > CFG["max_gpu_hours"] * 3600:
        raise TimeoutError("EXP-004 GPU-hour limit")
    if ledger["total_forwards"] >= CFG["max_forwards"]:
        raise RuntimeError("forward budget exhausted")
    bound = CFG["max_sampling" if kind == "sampling" else "max_commits"]
    if ledger[kind + "s" if kind == "commit" else kind] >= bound:
        raise RuntimeError(f"{kind} budget exhausted")
    ledger["total_forwards"] += 1
    ledger[kind + "s" if kind == "commit" else kind] += 1
    atomic_json(OUT / "budget.json", ledger)


def reserve_decode(ledger):
    if time.time() - ledger["first_start"] > CFG["max_elapsed_minutes"] * 60:
        raise TimeoutError("EXP-004 elapsed limit before VAE")
    if ledger["gpu_seconds"] + time.time() - ledger["active_start"] > CFG["max_gpu_hours"] * 3600:
        raise TimeoutError("EXP-004 GPU-hour limit before VAE")
    if ledger["vae_decodes"] >= CFG["max_decode"]:
        raise RuntimeError("VAE budget exhausted")
    ledger["vae_decodes"] += 1  # count attempts before call
    atomic_json(OUT / "budget.json", ledger)


def decode_video(pipe, z):
    import numpy as np
    import torch
    with torch.no_grad():
        rgb = pipe.video_vae.decode_video(z, dtype=pipe.torch_dtype,
            process_image=False, tiled=True, tile_size=256, tile_overlap=64)
        frames = pipe.vae_output_to_video(rgb, min_value=0, max_value=1)
    return np.stack([np.asarray(im.convert("RGB"), dtype=np.uint8) for im in frames])


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
    assert not row_path.exists(), "prior attempt exists; no automatic retry"
    began = time.perf_counter()
    row = dict(task="EXP-004/v1", path=args.path, stage=args.stage,
        interval=[start, stop], index=index, gpu=args.gpu, status="loading",
        seed=13, optimizer_updates=0, sampling_forwards=0, commit_forwards=0,
        vae_decodes=0, forward_count_start=ledger["total_forwards"],
        runner_sha256=sha(__file__), config_sha256=sha(HERE / "config.json"),
        interval_sha256=sha(PREV_CODE / "interval_cached.py"),
        router_sha256=sha(ROUTER / "current_prefix.py"))
    atomic_json(row_path, row)
    try:
        data = {a: torch.load(FROZEN / f"source_coarse/inputs/parking_{a}.pt",
                              map_location="cpu", weights_only=True) for a in "AD"}
        for key in ("initial_noise", "audio_noise", "anchor"):
            assert torch.equal(data["A"][key], data["D"][key]), key
        assert torch.equal(data["A"]["packed"]["img_position_ids"],
                           data["D"]["packed"]["img_position_ids"])
        pipe = abot.load_pipeline("cuda:0")
        released = FROZEN / "runtime/checkpoints/H3-World/step-10000.safetensors"
        assert sha(released) == "ddd9187b920b1e52c2d090f4e264fd83d8d433efc2a5b159e58883aeaf96e526"
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
            if isinstance(x, dict): return {k: move(v) for k, v in x.items()}
            return x
        initial = data["A"]["initial_noise"].to("cuda:0", torch.float32)
        first = torch.load(FIRST, map_location="cpu", weights_only=True).to("cuda:0", torch.float32)
        audio = data["A"]["audio_noise"].to("cuda:0")
        anchor = data["A"]["anchor"].to("cuda:0")
        full = move(data["A"]["packed"])
        assert first.shape == (1, 24, 12, 30, 52) and initial.shape[2] == 37
        shared = PREV / "first12_A_cache.pt"
        cache_file = shared if args.stage == "second" else path_dir / "cache_through17.pt"
        if args.stage == "third":
            assert (path_dir / "chunk_12_17.json").exists()
            second_file = path_dir / "chunk_12_17.pt"
            second = torch.load(second_file, map_location="cpu", weights_only=True).to("cuda:0", torch.float32)
            history = torch.cat((first, second), dim=2)
            row["second_endpoint_sha256"] = sha(second_file)
            assert row["second_endpoint_sha256"] == json.loads(
                (path_dir / "chunk_12_17.json").read_text())["endpoint_sha256"]
        else:
            history = first
        cache = torch.load(shared, map_location="cpu", weights_only=False)
        assert isinstance(cache, H3ChunkCache) and cache.storage_device == "cpu" and cache.max_history == 5
        assert len(cache.layers) == 50 and all([(e.index, int(e.key.shape[0])) for e in entries] == [(0, 4680)]
                                                for entries in cache.layers.values())
        row["first12_cache_sha256"] = sha(shared)
        assert row["first12_cache_sha256"] == json.loads((PREV / "AA/chunk_12_17.json").read_text())[
            "first12_cache_sha256"]
        row["first12_cache"] = cache_info(cache)
        row["history_tensor_sha256"] = tensor_sha(history)
        row["initial_noise_sha256"] = tensor_sha(initial)
        row["anchor_sha256"] = tensor_sha(anchor)
        row["audio_sha256"] = tensor_sha(audio)
        if args.stage == "third":
            assert not cache_file.exists(), "prior commit attempt exists"
            prompt2 = prompt_for_path(data, args.path, 17).to("cuda:0")
            layout2, prompt2 = visible_inputs(full, prompt2, 17, 390)
            before = cache_identity(cache)
            reserve(ledger, "commit")
            tick = time.perf_counter()
            with current_prefix_feedback(), torch.no_grad():
                _ = interval_cached(model, second, start=12, index=1, cache=cache,
                    full_packed=layout2, prompt=prompt2, anchor=anchor, audio=audio,
                    sigma=0, commit=True)
            torch.cuda.synchronize()
            row["second_clean_commit_seconds"] = time.perf_counter() - tick
            row["commit_forwards"] = 1
            assert cache_identity(cache) != before
            assert len(cache.layers) == 50 and all(len(v) == 2 for v in cache.layers.values())
            tmp = Path(str(cache_file) + ".partial")
            torch.save(cache, tmp); os.replace(tmp, cache_file)
            row["cache_through17_sha256"] = sha(cache_file)
            row["cache_after_commit"] = cache_info(cache)
        row["cache_file_sha256"] = sha(cache_file)
        prompt = prompt_for_path(data, args.path, stop).to("cuda:0")
        layout, prompt = visible_inputs(full, prompt, stop, 390)
        assert len(layout["action_text_rows"]) == stop
        row["prompt_sha256"] = tensor_sha(prompt)
        row["position_sha256"] = tensor_sha(layout["img_position_ids"])
        sigmas = configure_video_schedule(pipe.scheduler, steps=8, grid="native", flow_shift=2.22)
        assert len(sigmas) == 9 and sigmas[0] == 1 and sigmas[-1] == 0
        assert all(sigmas[i] > sigmas[i + 1] for i in range(8))
        row["sigmas"] = sigmas
        current = initial[:, :, start:stop].clone()
        noise_sha = tensor_sha(current)
        history_sha = tensor_sha(history)
        read_signature = cache_identity(cache)
        row["status"] = "sampling"; atomic_json(row_path, row)
        torch.cuda.synchronize(); tick = time.perf_counter()
        with current_prefix_feedback(), torch.no_grad():
            for j, t in enumerate(pipe.scheduler.timesteps):
                reserve(ledger, "sampling")
                velocity = interval_cached(model, current, start=start, index=index,
                    cache=cache, full_packed=layout, prompt=prompt, anchor=anchor,
                    audio=audio, sigma=float(t) / 1000)
                current = pipe.scheduler.step(velocity, t, current)
                row["sampling_forwards"] += 1
                if not torch.isfinite(current).all(): raise FloatingPointError("nonfinite latent")
                if torch.cuda.max_memory_allocated() / 2**30 > CFG["max_allocated_gib"]:
                    raise MemoryError("allocated memory exceeds 44 GiB")
                row["step_completed"] = j + 1; atomic_json(row_path, row)
                print(f"{args.path} [{start},{stop}) {j+1}/8", flush=True)
        torch.cuda.synchronize()
        row["sampling_seconds"] = time.perf_counter() - tick
        row["memory_cache_read_unchanged"] = cache_identity(cache) == read_signature
        assert row["memory_cache_read_unchanged"]
        assert tensor_sha(history) == history_sha and tensor_sha(initial[:, :, start:stop]) == noise_sha
        endpoint = path_dir / f"chunk_{start}_{stop}.pt"
        tmp = Path(str(endpoint) + ".partial")
        torch.save(current.detach().cpu(), tmp); os.replace(tmp, endpoint)
        row["endpoint_sha256"] = sha(endpoint)
        row["endpoint_tensor_sha256"] = tensor_sha(current)
        row["cache_file_unchanged_during_sampling"] = sha(cache_file) == row["cache_file_sha256"]
        assert row["cache_file_unchanged_during_sampling"]
        row["GPU_peak_allocated_MiB"] = torch.cuda.max_memory_allocated() / 2**20
        del velocity
        pipe.load_models_to_device(["video_vae"])
        torch.cuda.synchronize(); tick = time.perf_counter()
        if args.stage == "second":
            prior_file = PREV / "AA/published_56.npy"
            prior_row = json.loads((PREV / "AA/chunk_12_17.json").read_text())
            assert sha(prior_file) == prior_row["published_file_sha256"]
            published = np.load(prior_file)[:39].copy()
            row["first39_source"] = str(prior_file)
            row["first39_source_sha256"] = sha(prior_file)
        else:
            prior_file = path_dir / "published_56.npy"
            prior_row = json.loads((path_dir / "chunk_12_17.json").read_text())
            assert sha(prior_file) == prior_row["published_file_sha256"]
            published = np.load(prior_file)
        assert published.shape == (rgb_stop(start), 480, 832, 3)
        if args.stage == "second":
            assert np.array_equal(published, np.load(PREV / "AD/published_56.npy")[:39])
        prior_hash = rgb_sha(published)
        reserve_decode(ledger)
        decoded = decode_video(pipe, torch.cat((history, current), dim=2))
        row["vae_decodes"] = 1
        assert len(decoded) == rgb_stop(stop)
        row["decode_seconds"] = time.perf_counter() - tick
        row["past5_redecode_MAD"] = float(np.abs(
            decoded[rgb_stop(start)-5:rgb_stop(start)].astype(np.float32) -
            published[-5:].astype(np.float32)).mean())
        appended = stitch_immutable(published, decoded, start, stop)
        assert rgb_sha(appended[:len(published)]) == prior_hash
        publish_file = path_dir / f"published_{rgb_stop(stop)}.npy"
        np.save(publish_file, appended)
        row["published_file_sha256"] = sha(publish_file)
        row["published_RGB_sha256"] = rgb_sha(appended)
        row["prior_RGB_sha256"] = prior_hash
        video_file = path_dir / f"rollout_{rgb_stop(stop)}.mp4"
        active_file = path_dir / f"chunk_{start}_{stop}.mp4"
        write_video([Image.fromarray(f) for f in appended], video_file)
        write_video([Image.fromarray(f) for f in appended[rgb_stop(start):]], active_file)
        row["video_sha256"] = sha(video_file)
        row["active_video_sha256"] = sha(active_file)
        row["flow"] = flow_metrics(active_file)
        gray = np.stack([np.asarray(Image.fromarray(f).convert("L"), dtype=np.float32)
                         for f in appended[-18:]])
        row["boundary_gray_MAD"] = float(np.abs(gray[1] - gray[0]).mean())
        row["inside_gray_MAD"] = float(np.abs(np.diff(gray[1:], axis=0)).mean())
        row["GPU_peak_allocated_MiB"] = torch.cuda.max_memory_allocated() / 2**20
        row["GPU_peak_reserved_MiB"] = torch.cuda.max_memory_reserved() / 2**20
        assert row["GPU_peak_allocated_MiB"] <= CFG["max_allocated_gib"] * 1024
        row["frozen_parameter_versions_unchanged"] = all(p._version == v for p, v in frozen_versions)
        assert row["frozen_parameter_versions_unchanged"]
        row["wall_seconds"] = time.perf_counter() - began
        row["forward_count_end"] = ledger["total_forwards"]
        row["status"] = "complete_pending_visual_review"
        atomic_json(row_path, row)
        print(json.dumps({"path": args.path, "stage": args.stage,
              "video": str(video_file), "flow": row["flow"]["horizontal_flow_px"]["mean"],
              "status": row["status"]}), flush=True)
    except BaseException as exc:
        row.update(status="failed", error=repr(exc), wall_seconds=time.perf_counter()-began,
                   forward_count_end=ledger["total_forwards"])
        atomic_json(row_path, row)
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=["second", "third"], required=True)
    p.add_argument("--path", choices=["AA", "AD"], required=True)
    p.add_argument("--gpu", type=int, required=True)
    args = p.parse_args()
    assert sha(FIRST) == "242a1db06bc3423fe21207af5d2ccf59c9f27f07c9d88f22ce9f77921f6b0eea"
    assert sha(PREV_CODE / "interval_cached.py") == json.loads(
        (PREV / "AA/chunk_12_17.json").read_text())["interval_sha256"]
    free = int(subprocess.check_output(["nvidia-smi", f"--id={args.gpu}",
        "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True).strip())
    assert free >= 44000, ("GPU not idle", args.gpu, free)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "worker.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ledger_file = OUT / "budget.json"
        ledger = json.loads(ledger_file.read_text()) if ledger_file.exists() else dict(
            first_start=time.time(), gpu_seconds=0, total_forwards=0,
            sampling=0, commits=0, vae_decodes=0, runs=[])
        assert time.time() - ledger["first_start"] < CFG["max_elapsed_minutes"] * 60
        assert ledger["gpu_seconds"] < CFG["max_gpu_hours"] * 3600
        t = time.time(); ledger["active_start"] = t
        atomic_json(ledger_file, ledger)
        os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu), ABOT_VRAM_RESERVE_GIB="18",
            HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
            DIFFSYNTH_SKIP_DOWNLOAD="True", TOKENIZERS_PARALLELISM="false",
            PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
        sys.path[:0] = [str(FROZEN / "runtime/code"), str(FROZEN / "runtime/code/abot"),
            str(FROZEN / "runtime/code/causal"), str(FROZEN / "runtime/DiffSynth-Studio-h3-v2"),
            str(OLD), str(ROUTER), str(PREV_CODE)]
        status = "failed"
        try:
            run(args, ledger)
            status = "complete"
        finally:
            ledger["gpu_seconds"] += time.time() - t
            ledger.pop("active_start", None)
            ledger["runs"].append(dict(path=args.path, stage=args.stage, gpu=args.gpu,
                start=t, end=time.time(), status=status))
            atomic_json(ledger_file, ledger)


if __name__ == "__main__": main()
