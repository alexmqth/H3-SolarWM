"""EXP-006: native ordinary-FM8 V3 rollout from a newly generated first chunk.

Run --preflight first, then --stage first. Inspect all 39 RGB frames before
running AA/AD second and third stages. Every model call is metered before use.
No old 30-step first chunk, new weights, AnyFlow, or training are used.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
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

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CFG = json.loads((HERE / "config.json").read_text())
OUT = ROOT / CFG["output_root"]
FROZEN = ROOT / "H3-World/outputs/2026-10-09-22/chunk_partition_cb"
ACCEPTED = ROOT / "submission/experiments/EXP-002_native_cached"
ROUTER = ROOT / "submission/reports/stage1_anyflow/02_causal_diagnostics/current_prefix_candidate"
CONTRACT = ROOT / "submission/experiments/EXP-001_v2b_124"
MANIFEST = HERE / "source_manifest.json"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def tensor_sha(tensor) -> str:
    import torch
    x = tensor.detach().cpu().contiguous()
    return hashlib.sha256(str((tuple(x.shape), str(x.dtype))).encode() +
                          x.view(torch.uint8).numpy().tobytes()).hexdigest()


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(str(path) + ".partial")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(temp, path)


def verify_manifest() -> dict:
    manifest = json.loads(MANIFEST.read_text())
    assert manifest["task"] == "EXP-006/v1"
    for name, entry in manifest["sources"].items():
        path = ROOT / entry["path"]
        actual = sha(path)
        if actual != entry["sha256"]:
            raise RuntimeError(f"source changed: {name}: {path}")
    return manifest


def setup_paths() -> None:
    sys.path[:0] = [str(FROZEN / "runtime/code"), str(FROZEN / "runtime/code/abot"),
                    str(FROZEN / "runtime/code/causal"),
                    str(FROZEN / "runtime/DiffSynth-Studio-h3-v2"),
                    str(CONTRACT), str(ROUTER), str(ACCEPTED)]


def check_budget(ledger: dict) -> None:
    now = time.time()
    if now - ledger["first_start"] >= CFG["max_elapsed_minutes"] * 60:
        raise TimeoutError("EXP-006 absolute elapsed cap")
    if now >= datetime.fromisoformat(CFG["deadline_hkt"]).timestamp():
        raise TimeoutError("EXP-006 09:00 HKT cutoff")
    used = ledger["gpu_seconds"] + (now - ledger["active_start"])
    if used >= CFG["max_gpu_hours"] * 3600:
        raise TimeoutError("EXP-006 GPU-hour cap")


def reserve(ledger: dict, kind: str) -> None:
    check_budget(ledger)
    if kind == "vae":
        if ledger["vae_decodes"] >= CFG["max_decode"]:
            raise RuntimeError("VAE budget exhausted")
        ledger["vae_decodes"] += 1
    else:
        if ledger["total_forwards"] >= CFG["max_forwards"]:
            raise RuntimeError("full-forward budget exhausted")
        field = "sampling" if kind == "sampling" else "commits"
        cap = CFG["max_sampling"] if kind == "sampling" else CFG["max_commits"]
        if ledger[field] >= cap:
            raise RuntimeError(f"{kind} budget exhausted")
        ledger[field] += 1
        ledger["total_forwards"] += 1
    atomic_json(OUT / "budget.json", ledger)  # Count even a failed call.


def cache_info(cache, index: int) -> dict:
    assert len(cache.layers) == 50
    entries = [[(e.index, int(e.key.shape[0])) for e in row]
               for row in cache.layers.values()]
    assert all(row == entries[0] for row in entries)
    assert [i for i, _ in entries[0]] == list(range(index))
    return dict(index=index, layer_count=50, entries=entries[0],
                nbytes=cache.nbytes, commits=cache.commits)


def cache_signature(cache):
    return (cache.commits, tuple((layer, tuple(
        (e.index, id(e), tuple((id(t), t.data_ptr(), t._version, tuple(t.shape))
                            for t in (e.key, e.value, e.rope))) for e in entries))
        for layer, entries in sorted(cache.layers.items())))


def decode_video(pipe, latents):
    import numpy as np
    import torch
    with torch.no_grad():
        rgb = pipe.video_vae.decode_video(latents, dtype=pipe.torch_dtype,
            process_image=False, tiled=True, tile_size=256, tile_overlap=64)
        images = pipe.vae_output_to_video(rgb, min_value=0, max_value=1)
    return np.stack([np.asarray(im.convert("RGB"), dtype=np.uint8) for im in images])


def contact_sheet(frames, path: Path, *, columns: int = 5) -> None:
    from PIL import Image, ImageDraw
    width, height = 208, 120
    rows = (len(frames) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * width, rows * (height + 20)), "black")
    draw = ImageDraw.Draw(sheet)
    for i, frame in enumerate(frames):
        x, y = (i % columns) * width, (i // columns) * (height + 20)
        sheet.paste(Image.fromarray(frame).resize((width, height)), (x, y))
        draw.text((x + 4, y + height + 2), str(i), fill="white")
    sheet.save(path, quality=90)


def preflight() -> None:
    import torch
    from causal.h3_cached import H3ChunkCache, ChunkAttention
    from causal.local_topology import visible_inputs
    from causal.anyflow_sampling import configure_video_schedule
    from diffsynth.diffusion.flow_match import FlowMatchScheduler
    from rollout_contract import prompt_for_path

    assert CFG["partition"] == [12, 5, 5]
    assert (CFG["max_sampling"], CFG["max_commits"], CFG["max_forwards"], CFG["max_decode"]) == (40, 3, 43, 5)
    assert CFG["steps"] == 8 and CFG["flow_shift"] == 2.22
    cache = H3ChunkCache(max_history=5, storage_device="cpu")
    assert cache.history(0, 0) == [] and not cache.layers
    data = {a: torch.load(FROZEN / f"source_coarse/inputs/parking_{a}.pt",
                          map_location="cpu", weights_only=True) for a in "AD"}
    for key in ("initial_noise", "audio_noise", "anchor"):
        assert torch.equal(data["A"][key], data["D"][key]), key
    assert data["A"]["initial_noise"].shape == (1, 24, 37, 30, 52)
    assert torch.equal(data["A"]["packed"]["img_position_ids"],
                       data["D"]["packed"]["img_position_ids"])
    for stop in (12, 17, 22):
        for path in ("AA", "AD"):
            prompt = prompt_for_path(data, path, stop)
            layout, visible = visible_inputs(data["A"]["packed"], prompt, stop, 390)
            assert len(layout["action_text_rows"]) == stop
            assert visible.shape[0] < prompt.shape[0] or stop == 22
            assert int(layout["seq_len"]) < int(data["A"]["packed"]["seq_len"])
            # A current video row reads its own action and no future action.
            frame_rows = 390
            attn = ChunkAttention(cache, 0, int(layout["action_video_start"]),
                                  frame_rows, 0, layout["action_text_rows"], "own", True)
            _, mask = attn.masks(frame_rows, 0, "cpu")
            own = layout["action_text_rows"][0].tolist()
            assert bool(mask[0, own[0]:own[1]].all())
            if stop > 1:
                future = layout["action_text_rows"][1].tolist()
                assert not bool(mask[0, future[0]:future[1]].any())
    assert torch.equal(data["A"]["initial_noise"][:, :, :12],
                       data["D"]["initial_noise"][:, :, :12])
    schedule = configure_video_schedule(FlowMatchScheduler("MiniMax-H3"),
        steps=CFG["steps"], grid="native", flow_shift=CFG["flow_shift"])
    frozen_schedule = json.loads((ROOT / "H3-World/outputs/EXP-004_v3_8step/AA/chunk_12_17.json").read_text())["sigmas"]
    assert schedule == frozen_schedule
    synthetic = dict(first_start=time.time(), active_start=time.time(), gpu_seconds=0,
                     sampling=40, commits=3, total_forwards=43, vae_decodes=5)
    for kind in ("sampling", "commit", "vae"):
        try:
            reserve(synthetic.copy(), kind)
        except RuntimeError:
            pass
        else:
            raise AssertionError(f"{kind} budget did not reject overrun")
    print(json.dumps({"task": "EXP-006/v1", "preflight": "pass",
                      "empty_index0_cache": True, "visible_stops": [12, 17, 22],
                      "same_noise": True, "own_action_feedback": True,
                      "native_8step_sigmas": schedule,
                      "budget_rejects_overrun": True}), flush=True)


def run(stage: str, path: str | None, gpu: int, ledger: dict, manifest: dict) -> None:
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
    torch.manual_seed(CFG["seed"])
    start, stop, index = {"first": (0, 12, 0), "second": (12, 17, 1),
                          "third": (17, 22, 2)}[stage]
    output_dir = OUT if stage == "first" else OUT / path
    output_dir.mkdir(parents=True, exist_ok=True)
    row_path = output_dir / f"chunk_{start}_{stop}.json"
    if row_path.exists():
        raise RuntimeError(f"prior attempt exists: {row_path}; no automatic retry")
    began = time.perf_counter()
    row = dict(task="EXP-006/v1", stage=stage, path=path, status="loading",
               interval=[start, stop], index=index, gpu=gpu, seed=CFG["seed"],
               runner_sha256=sha(__file__), config_sha256=sha(HERE / "config.json"),
               manifest_sha256=sha(MANIFEST),
               sampling_forwards=0, commit_forwards=0, vae_calls=0,
               forward_count_start=ledger["total_forwards"])
    atomic_json(row_path, row)
    try:
        data = {a: torch.load(FROZEN / f"source_coarse/inputs/parking_{a}.pt",
                              map_location="cpu", weights_only=True) for a in "AD"}
        for key in ("initial_noise", "audio_noise", "anchor"):
            assert torch.equal(data["A"][key], data["D"][key]), key
        pipe = abot.load_pipeline("cuda:0")
        released = FROZEN / "runtime/checkpoints/H3-World/step-10000.safetensors"
        assert sha(released) == manifest["sources"]["released_action_lora"]["sha256"]
        pipe.load_lora(pipe.dit, state_dict=abot.load_checkpoint_lora(released), hotload=True)
        model = pipe.dit.eval().requires_grad_(False)
        row["precision"] = configure_precision(model, CFG["precision"],
            native_transformer_dir=FROZEN / "runtime/DiffSynth-Studio-h3-v2/models/MiniMax/MiniMax-H3/FL2VA/transformer")
        pipe.load_models_to_device(["dit"])
        torch.cuda.synchronize()
        row["load_seconds"] = time.perf_counter() - began
        versions = [(p, p._version) for p in model.parameters()]
        def move(x):
            if torch.is_tensor(x): return x.to("cuda:0")
            if isinstance(x, dict): return {k: move(v) for k, v in x.items()}
            return x
        initial = data["A"]["initial_noise"].to("cuda:0", torch.float32)
        audio = data["A"]["audio_noise"].to("cuda:0")
        anchor = data["A"]["anchor"].to("cuda:0")
        full = move(data["A"]["packed"])
        row["initial_noise_sha256"] = tensor_sha(initial)
        row["audio_sha256"] = tensor_sha(audio)
        row["anchor_sha256"] = tensor_sha(anchor)
        row["source_input_sha256"] = sha(FROZEN / "source_coarse/inputs/parking_A.pt")
        if stage == "first":
            history = None
            published = None
            cache = H3ChunkCache(max_history=5, storage_device="cpu")
            assert not cache.layers and cache.history(0, 0) == []
        else:
            first_file = OUT / "first12.pt"
            first_row = json.loads((OUT / "chunk_0_12.json").read_text())
            assert first_row["status"] == "complete_pending_visual_review"
            assert sha(first_file) == first_row["endpoint_sha256"]
            first = torch.load(first_file, map_location="cpu", weights_only=True).to("cuda:0", torch.float32)
            if stage == "second":
                history = first
                cache_file = OUT / "cache_through12.pt"
                published_file = OUT / "published_39.npy"
            else:
                second_file = OUT / path / "chunk_12_17.pt"
                second_row = json.loads((OUT / path / "chunk_12_17.json").read_text())
                assert second_row["status"] == "complete_pending_visual_review"
                assert sha(second_file) == second_row["endpoint_sha256"]
                second = torch.load(second_file, map_location="cpu", weights_only=True).to("cuda:0", torch.float32)
                history = torch.cat((first, second), dim=2)
                cache_file = OUT / path / "cache_through17.pt"
                published_file = OUT / path / "published_56.npy"
            cache = torch.load(cache_file, map_location="cpu", weights_only=False)
            assert isinstance(cache, H3ChunkCache) and cache.storage_device == "cpu"
            row["cache_file_sha256"] = sha(cache_file)
            published = np.load(published_file)
            assert published.shape == (rgb_stop(start), 480, 832, 3)
            row["prior_RGB_sha256"] = rgb_sha(published)
            row["history_tensor_sha256"] = tensor_sha(history)
        if index:
            row["cache_before"] = cache_info(cache, index)
        prompt = prompt_for_path(data, path or "AA", stop).to("cuda:0")
        layout, prompt = visible_inputs(full, prompt, stop, 390)
        assert len(layout["action_text_rows"]) == stop
        row["prompt_sha256"] = tensor_sha(prompt)
        row["position_sha256"] = tensor_sha(layout["img_position_ids"])
        sigmas = configure_video_schedule(pipe.scheduler, steps=CFG["steps"],
                                          grid="native", flow_shift=CFG["flow_shift"])
        assert len(sigmas) == 9 and sigmas[0] == 1 and sigmas[-1] == 0
        assert all(sigmas[i] > sigmas[i+1] for i in range(8))
        row["sigmas"] = sigmas
        current = initial[:, :, start:stop].clone()
        row["current_noise_sha256"] = tensor_sha(current)
        history_hash = None if history is None else tensor_sha(history)
        signature = cache_signature(cache)
        row["status"] = "sampling"; atomic_json(row_path, row)
        torch.cuda.synchronize(); tick = time.perf_counter()
        with current_prefix_feedback(), torch.no_grad():
            for j, timestep in enumerate(pipe.scheduler.timesteps):
                reserve(ledger, "sampling")
                velocity = interval_cached(model, current, start=start, index=index,
                    cache=cache, full_packed=layout, prompt=prompt, anchor=anchor,
                    audio=audio, sigma=float(timestep) / 1000)
                current = pipe.scheduler.step(velocity, timestep, current)
                row["sampling_forwards"] += 1
                if not torch.isfinite(current).all():
                    raise FloatingPointError("nonfinite latent")
                if torch.cuda.max_memory_allocated() / 2**30 > CFG["max_allocated_gib"]:
                    raise MemoryError("allocated VRAM cap exceeded")
                row["step_completed"] = j + 1; atomic_json(row_path, row)
                print(f"{stage} {path or 'shared'} {j+1}/8", flush=True)
        torch.cuda.synchronize()
        row["sampling_seconds"] = time.perf_counter() - tick
        assert cache_signature(cache) == signature
        row["cache_read_unchanged"] = True
        assert tensor_sha(initial[:, :, start:stop]) == row["current_noise_sha256"]
        if history is not None: assert tensor_sha(history) == history_hash
        endpoint = output_dir / ("first12.pt" if stage == "first" else f"chunk_{start}_{stop}.pt")
        temp = Path(str(endpoint) + ".partial")
        torch.save(current.detach().cpu(), temp); os.replace(temp, endpoint)
        row["endpoint_sha256"] = sha(endpoint)
        row["endpoint_tensor_sha256"] = tensor_sha(current)
        if stage != "third":
            reserve(ledger, "commit")
            torch.cuda.synchronize(); tick = time.perf_counter()
            with current_prefix_feedback(), torch.no_grad():
                _ = interval_cached(model, current, start=start, index=index,
                    cache=cache, full_packed=layout, prompt=prompt, anchor=anchor,
                    audio=audio, sigma=0, commit=True)
            torch.cuda.synchronize()
            row["commit_seconds"] = time.perf_counter() - tick
            row["commit_forwards"] = 1
            row["cache_after"] = cache_info(cache, index + 1)
            cache_out = output_dir / ("cache_through12.pt" if stage == "first" else "cache_through17.pt")
            temp = Path(str(cache_out) + ".partial")
            torch.save(cache, temp); os.replace(temp, cache_out)
            row["cache_after_sha256"] = sha(cache_out)
        if stage != "first":
            assert sha(cache_file) == row["cache_file_sha256"]
            row["source_cache_file_unchanged"] = True
        row["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
        del velocity
        pipe.load_models_to_device(["video_vae"])
        reserve(ledger, "vae")
        torch.cuda.synchronize(); tick = time.perf_counter()
        decoded = decode_video(pipe, current if history is None else torch.cat((history, current), dim=2))
        row["vae_calls"] = 1
        assert len(decoded) == rgb_stop(stop)
        row["decode_seconds"] = time.perf_counter() - tick
        if stage == "first":
            appended = decoded
        else:
            old_hash = rgb_sha(published)
            appended = stitch_immutable(published, decoded, start, stop)
            assert rgb_sha(appended[:len(published)]) == old_hash
            row["old_RGB_unchanged"] = True
            row["past5_redecode_MAD"] = float(np.abs(
                decoded[rgb_stop(start)-5:rgb_stop(start)].astype(np.float32) -
                published[-5:].astype(np.float32)).mean())
        rgb_file = output_dir / f"published_{rgb_stop(stop)}.npy"
        np.save(rgb_file, appended)
        row["published_file_sha256"] = sha(rgb_file)
        row["published_RGB_sha256"] = rgb_sha(appended)
        video_file = output_dir / f"rollout_{rgb_stop(stop)}.mp4"
        write_video([Image.fromarray(f) for f in appended], video_file)
        row["video_sha256"] = sha(video_file)
        new_frames = appended if stage == "first" else appended[rgb_stop(start):]
        contact_sheet(new_frames, output_dir / f"chunk_{start}_{stop}_all_frames.jpg")
        if stage != "first":
            active_file = output_dir / f"chunk_{start}_{stop}.mp4"
            write_video([Image.fromarray(f) for f in new_frames], active_file)
            row["active_video_sha256"] = sha(active_file)
            row["flow"] = flow_metrics(active_file)
            gray = np.stack([np.asarray(Image.fromarray(f).convert("L"), dtype=np.float32)
                             for f in appended[rgb_stop(start)-1:]])
            row["boundary_gray_MAD"] = float(np.abs(gray[1]-gray[0]).mean())
            row["inside_gray_MAD"] = float(np.abs(np.diff(gray[1:], axis=0)).mean())
        else:
            row["flow"] = flow_metrics(video_file)
            gray = np.stack([np.asarray(Image.fromarray(f).convert("L"), dtype=np.float32)
                             for f in appended])
            row["inside_gray_MAD"] = float(np.abs(np.diff(gray, axis=0)).mean())
        row["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
        row["peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 2**30
        assert row["peak_allocated_gib"] <= CFG["max_allocated_gib"]
        row["peak_cpu_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        row["frozen_parameters_unchanged"] = all(p._version == v for p, v in versions)
        assert row["frozen_parameters_unchanged"]
        row["wall_seconds"] = time.perf_counter() - began
        row["forward_count_end"] = ledger["total_forwards"]
        row["status"] = "complete_pending_visual_review"
        atomic_json(row_path, row)
        print(json.dumps({"stage": stage, "path": path, "video": str(video_file),
                          "status": row["status"]}), flush=True)
    except BaseException as exc:
        row["status"] = "failed"
        row["error"] = repr(exc)
        row["wall_seconds"] = time.perf_counter() - began
        row["forward_count_end"] = ledger["total_forwards"]
        atomic_json(row_path, row)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--stage", choices=["first", "second", "third"])
    parser.add_argument("--path", choices=["AA", "AD"])
    parser.add_argument("--gpu", type=int)
    args = parser.parse_args()
    setup_paths()
    verify_manifest()
    if args.preflight:
        preflight()
        return
    if args.stage is None or args.gpu is None:
        parser.error("--stage and --gpu required outside preflight")
    if (args.stage == "first") != (args.path is None):
        parser.error("first stage is shared; later stages require --path")
    free = int(subprocess.check_output(["nvidia-smi", f"--id={args.gpu}",
        "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True).strip())
    if free < 44000:
        raise RuntimeError(f"GPU {args.gpu} is not sufficiently idle: {free} MiB")
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "worker.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        budget_path = OUT / "budget.json"
        ledger = json.loads(budget_path.read_text()) if budget_path.exists() else dict(
            task="EXP-006/v1", first_start=time.time(), gpu_seconds=0.,
            total_forwards=0, sampling=0, commits=0, vae_decodes=0, runs=[])
        start_time = time.time()
        ledger["active_start"] = start_time
        ledger["active_stage"] = args.stage
        ledger["active_path"] = args.path
        atomic_json(budget_path, ledger)
        check_budget(ledger)
        deadline = datetime.fromisoformat(CFG["deadline_hkt"]).timestamp()
        remaining = min(CFG["max_gpu_hours"] * 3600 - ledger["gpu_seconds"],
                        CFG["max_elapsed_minutes"] * 60 - (start_time - ledger["first_start"]),
                        deadline - start_time)
        def timeout_handler(_signum, _frame):
            raise TimeoutError("EXP-006 absolute signal deadline")
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.setitimer(signal.ITIMER_REAL, remaining)
        os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu), ABOT_VRAM_RESERVE_GIB="18",
            HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
            DIFFSYNTH_SKIP_DOWNLOAD="True", TOKENIZERS_PARALLELISM="false",
            PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
        status = "failed"
        try:
            run(args.stage, args.path, args.gpu, ledger, verify_manifest())
            status = "complete_pending_visual_review"
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            ledger["gpu_seconds"] += time.time() - start_time
            ledger.pop("active_start", None)
            ledger.pop("active_stage", None)
            ledger.pop("active_path", None)
            ledger["runs"].append(dict(stage=args.stage, path=args.path, gpu=args.gpu,
                                       start=start_time, end=time.time(), status=status))
            atomic_json(budget_path, ledger)


if __name__ == "__main__":
    main()
