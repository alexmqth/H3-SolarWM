"""EXP-012 frozen AF2 step32: one FM8-C1 clean commit and AF8 AA/AD C2.

Only a Judge marker can enable one scene. No training, no new C1 generation,
no GT reset, no third chunk, and no FM8 baseline rerun.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import resource
import signal
import time

from common import (CFG, FROZEN, HERE, OUT, ROOT, Ledger, atomic_json,
                    check_machine, require_marker, setup_paths, sha, verify_sources)
from checkpoint import load_checkpoint

O11 = ROOT / "H3-World/outputs/EXP-011_v3_scene_transfer"


def tensor_sha(tensor) -> str:
    import torch
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(str((tuple(value.shape), str(value.dtype))).encode() +
                          value.view(torch.uint8).numpy().tobytes()).hexdigest()


def cache_signature(cache):
    return (cache.commits, tuple((layer, tuple(
        (entry.index, id(entry), tuple((id(t), t.data_ptr(), t._version, tuple(t.shape))
                                     for t in (entry.key, entry.value, entry.rope)))
        for entry in entries)) for layer, entries in sorted(cache.layers.items())))


def load_inputs(scene: str):
    import numpy as np
    import torch
    sources = json.loads((HERE / "source_manifest.json").read_text())["sources"]
    p1 = json.loads((O11 / "P1_result.json").read_text())
    fixture_path = ROOT / p1["fixtures"][scene]["path"]
    assert sha(fixture_path) == sources[f"fixture_{scene}"]["sha256"] == p1["fixtures"][scene]["sha256"]
    fixture = torch.load(fixture_path, map_location="cpu", weights_only=True)
    assert fixture["format"] == "exp011_native_single_i0_full37_v1"
    assert fixture["scene"] == scene
    assert fixture["initial_noise"].shape == (1, 24, 37, 30, 52)
    assert fixture["anchor"].shape == (390, 96)
    assert len(fixture["packed"]["action_text_spans_local"]) == 37
    g1 = O11 / "G1" / scene / "FM8"
    row_path = g1 / "result.json"
    assert sha(row_path) == sources[f"g1_{scene}_row"]["sha256"]
    row = json.loads(row_path.read_text())
    assert row["status"] == "complete_pending_visual_review"
    assert row["method"] == "FM8" and row["scene"] == scene
    first_path = g1 / "first12.pt"
    rgb_path = g1 / "published_39.npy"
    assert sha(first_path) == sources[f"g1_{scene}_first12"]["sha256"] == row["endpoint_sha256"]
    assert sha(rgb_path) == sources[f"g1_{scene}_rgb39"]["sha256"] == row["published_sha256"]
    first = torch.load(first_path, map_location="cpu", weights_only=True)
    published = np.load(rgb_path)
    assert first.shape == (1, 24, 12, 30, 52) and torch.isfinite(first).all()
    assert published.shape == (39, 480, 832, 3) and published.dtype == np.uint8
    return fixture, first, published, {"fixture_sha256": sha(fixture_path),
                                      "first_endpoint_sha256": sha(first_path),
                                      "first_published_sha256": sha(rgb_path),
                                      "g1_row_sha256": sha(row_path)}


def move(tree):
    import torch
    if torch.is_tensor(tree):
        return tree.to("cuda:0")
    if isinstance(tree, dict):
        return {key: move(value) for key, value in tree.items()}
    return tree


def visible(fixture, action: str, stop: int):
    import torch
    from causal.local_topology import visible_inputs
    packed = move(fixture["packed"])
    prompt = fixture["prompts"]["A"].clone()
    if action == "AD":
        donor = fixture["prompts"]["D"]
        for lo, hi in fixture["packed"]["action_text_spans_local"][12:stop]:
            prompt[lo:hi] = donor[lo:hi]
    elif action != "A":
        raise ValueError(action)
    layout, cropped = visible_inputs(packed, prompt.to("cuda:0"), stop, 390)
    assert len(layout["action_text_rows"]) == stop
    assert layout["seq_len"] == layout["action_video_start"] + stop * 390
    assert layout["img_pos"].numel() == (stop + 1) * 390
    return layout, cropped


def load_model():
    import torch
    import infer as abot
    from causal.anyflow import install_anyflow
    from causal.pretrained_lora import install_adapters
    from causal.h3_precision import configure_precision
    qkv_state, target_state, trainer_state = load_checkpoint()
    pipe = abot.load_pipeline("cuda:0")
    released = FROZEN / "runtime/checkpoints/H3-World/step-10000.safetensors"
    sources = json.loads((HERE / "source_manifest.json").read_text())["sources"]
    assert sha(released) == sources["released_action_lora"]["sha256"]
    pipe.load_lora(pipe.dit, state_dict=abot.load_checkpoint_lora(released), hotload=True)
    model = pipe.dit.eval().requires_grad_(False)
    precision = configure_precision(model, CFG["precision"],
        native_transformer_dir=FROZEN / "runtime/DiffSynth-Studio-h3-v2/models/MiniMax/MiniMax-H3/FL2VA/transformer")
    pipe.load_models_to_device(["dit"])
    adapters, indices = install_adapters(model, rank=8, block_indices=tuple(range(42, 50)), device="cuda:0")
    assert indices == list(range(42, 50)) or tuple(indices) == tuple(range(42, 50))
    target = install_anyflow(model, device="cuda:0", gate=0.25)
    with torch.no_grad():
        for adapter, a, b in zip(adapters, qkv_state["lora_A"], qkv_state["lora_B"]):
            adapter.lora_A.copy_(a.to("cuda:0"))
            adapter.lora_B.copy_(b.to("cuda:0"))
    target.load_state_dict(target_state["weights"], strict=True)
    model.requires_grad_(False)
    return pipe, model, precision, {"step": trainer_state["step"],
                                    "qkv_sha256": sources["af2_qkv"]["sha256"],
                                    "target_sha256": sources["af2_target"]["sha256"]}


def check_peak():
    import torch
    peak = torch.cuda.max_memory_allocated() / 2**30
    if peak > CFG["gpu_allocated_cap_gib"]:
        raise MemoryError(f"allocated peak {peak:.3f} GiB exceeds cap")
    return peak


def sample(model, fixture, cache, ledger: Ledger, branch: str, row: dict):
    import torch
    from causal.anyflow import finite_map_step
    from causal.anyflow_sampling import configure_video_schedule
    from diffsynth.diffusion.flow_match import FlowMatchScheduler
    from current_prefix import current_prefix_feedback
    from interval_student import interval_student

    layout, prompt = visible(fixture, "A" if branch == "AA" else "AD", 17)
    audio = fixture["audio_noise"].to("cuda:0")
    anchor = fixture["anchor"].to("cuda:0")
    current = fixture["initial_noise"][:, :, 12:17].to("cuda:0", torch.float32).clone()
    initial_sha = tensor_sha(current)
    # configure_video_schedule uses the scheduler's native H3 8-step grid.
    # FlowMatchScheduler is used only for an independent reference assertion.
    from causal.anyflow_sampling import configure_video_schedule as schedule
    expected = schedule(FlowMatchScheduler("MiniMax-H3"), steps=8,
                        grid="native", flow_shift=CFG["flow_shift"])
    row.update(sigmas=expected, prompt_sha256=tensor_sha(prompt),
               position_sha256=tensor_sha(layout["img_position_ids"]),
               initial_noise_sha256=initial_sha, sampling_forwards=0)
    atomic_json(Path(row["row_file"]), row)
    signature = cache_signature(cache)
    torch.cuda.synchronize(); began = time.perf_counter()
    with torch.no_grad(), current_prefix_feedback():
        for step, (sigma, next_sigma) in enumerate(zip(expected[:-1], expected[1:]), 1):
            ledger.reserve("sampling_forwards")
            velocity = interval_student(model, current, start=12, index=1,
                cache=cache, full_packed=layout, prompt=prompt, anchor=anchor,
                audio=audio, sigma=sigma, target_sigma=next_sigma)
            current = finite_map_step(current, velocity, sigma, next_sigma)
            if not torch.isfinite(current).all():
                raise FloatingPointError("nonfinite AF8 latent")
            row["sampling_forwards"] = step
            row["step_completed"] = step
            check_peak()
            atomic_json(Path(row["row_file"]), row)
    torch.cuda.synchronize()
    assert cache_signature(cache) == signature
    assert tensor_sha(fixture["initial_noise"][:, :, 12:17].to(torch.float32)) == initial_sha
    row["sampling_seconds"] = time.perf_counter() - began
    row["endpoint_tensor_sha256"] = tensor_sha(current)
    return current


def decode(pipe, first, current, ledger: Ledger):
    import numpy as np
    import torch
    pipe.load_models_to_device(["video_vae"])
    ledger.reserve("vae_decodes")
    torch.cuda.synchronize(); began = time.perf_counter()
    with torch.no_grad():
        rgb = pipe.video_vae.decode_video(torch.cat((first, current), dim=2),
            dtype=pipe.torch_dtype, process_image=False, tiled=True,
            tile_size=256, tile_overlap=64)
        frames = pipe.vae_output_to_video(rgb, min_value=0, max_value=1)
    torch.cuda.synchronize()
    return np.stack([np.asarray(frame.convert("RGB"), dtype=np.uint8)
                     for frame in frames]), time.perf_counter() - began


def write_video_and_sheet(frames, video: Path, sheet: Path):
    from PIL import Image, ImageDraw
    from benchmark import write_video
    video.parent.mkdir(parents=True, exist_ok=True)
    write_video([Image.fromarray(frame) for frame in frames], video)
    width, height, columns = 208, 120, 5
    output = Image.new("RGB", (columns * width,
                       ((len(frames) + columns - 1) // columns) * (height + 20)), "black")
    draw = ImageDraw.Draw(output)
    for i, frame in enumerate(frames):
        x, y = (i % columns) * width, (i // columns) * (height + 20)
        output.paste(Image.fromarray(frame).resize((width, height)), (x, y))
        draw.text((x + 4, y + height + 2), str(i), fill="white")
    output.save(sheet, quality=90)


def run_scene(scene: str, gpu: int, ledger: Ledger):
    import numpy as np
    import torch
    from causal.h3_cached import H3ChunkCache
    from current_prefix import current_prefix_feedback
    from interval_student import interval_student

    stage_started = time.perf_counter()
    fixture, first_cpu, published, refs = load_inputs(scene)
    input_load_seconds = time.perf_counter() - stage_started
    directory = OUT / "G1" / scene
    if directory.exists():
        raise FileExistsError(f"prior attempt exists: {directory}")
    directory.mkdir(parents=True)
    row_path = directory / "result.json"
    row = {"task": CFG["task"], "stage": "G1", "scene": scene,
           "status": "loading", "gpu": gpu, "row_file": str(row_path),
           "runner_sha256": sha(__file__), **refs,
           "sampling_forwards": 0, "commit_forwards": 0,
           "vae_decodes": 0, "branches": {}}
    atomic_json(row_path, row)
    started = stage_started
    try:
        load_tick = time.perf_counter()
        pipe, model, precision, checkpoint = load_model()
        torch.cuda.synchronize()
        row["load_model_seconds"] = time.perf_counter() - load_tick
        row["input_load_seconds"] = input_load_seconds
        row.update(precision=precision, checkpoint=checkpoint)
        first = first_cpu.to("cuda:0", torch.float32)
        first_tensor_sha = tensor_sha(first)
        row["first_tensor_sha256"] = first_tensor_sha
        cache = H3ChunkCache(max_history=CFG["max_history"], storage_device="cpu")
        layout, prompt = visible(fixture, "A", 12)
        audio = fixture["audio_noise"].to("cuda:0")
        anchor = fixture["anchor"].to("cuda:0")
        ledger.reserve("commit_forwards")
        torch.cuda.synchronize(); tick = time.perf_counter()
        with torch.no_grad(), current_prefix_feedback():
            _ = interval_student(model, first, start=0, index=0, cache=cache,
                full_packed=layout, prompt=prompt, anchor=anchor, audio=audio,
                sigma=0.0, target_sigma=0.0, commit=True)
        torch.cuda.synchronize()
        check_peak()
        assert len(cache.layers) == 50 and all(len(entries) == 1 for entries in cache.layers.values())
        assert all(entries[0].index == 0 for entries in cache.layers.values())
        row["commit_forwards"] = 1
        row["commit_seconds"] = time.perf_counter() - tick
        row["cpu_raw_kv_bytes"] = cache.nbytes
        row["commit_source"] = "shared EXP-011 FM8 C1 endpoint, re-committed under AF2 step32 student weights"
        cache_path = directory / "af_cache_through12.pt"
        save_tick = time.perf_counter()
        torch.save(cache, cache_path)
        row["cache_sha256"] = sha(cache_path)
        row["cache_save_and_hash_seconds"] = time.perf_counter() - save_tick
        frozen_signature = cache_signature(cache)
        atomic_json(row_path, row)
        for branch in ("AA", "AD"):
            branch_dir = directory / branch
            branch_dir.mkdir()
            part = {"row_file": str(branch_dir / "result.json"),
                    "status": "sampling", "branch": branch,
                    "scene": scene, "checkpoint_step": 32,
                    "first_endpoint_sha256": refs["first_endpoint_sha256"],
                    "source_cache_sha256": row["cache_sha256"]}
            atomic_json(branch_dir / "result.json", part)
            current = sample(model, fixture, cache, ledger, branch, part)
            assert cache_signature(cache) == frozen_signature
            endpoint = branch_dir / "chunk_12_17.pt"
            save_tick = time.perf_counter()
            torch.save(current.cpu(), endpoint)
            part["endpoint_sha256"] = sha(endpoint)
            part["endpoint_save_and_hash_seconds"] = time.perf_counter() - save_tick
            assert sha(cache_path) == row["cache_sha256"]
            frames, seconds = decode(pipe, first, current, ledger)
            assert frames.shape == (56, CFG["height"], CFG["width"], 3)
            joined = np.concatenate((published, frames[39:]), axis=0)
            assert np.array_equal(joined[:39], published)
            save_tick = time.perf_counter()
            np.save(branch_dir / "published_56.npy", joined)
            video = branch_dir / "rollout_56.mp4"
            write_video_and_sheet(joined, video, branch_dir / "all_56_frames.jpg")
            write_video_and_sheet(joined[39:], branch_dir / "new_17.mp4",
                                  branch_dir / "new_17_frames.jpg")
            part["rgb_video_save_seconds"] = time.perf_counter() - save_tick
            part.update(status="complete_pending_visual_review",
                        published_sha256=sha(branch_dir / "published_56.npy"),
                        video_sha256=sha(video), decode_seconds=seconds,
                        vae_decodes=1, cpu_raw_kv_bytes=cache.nbytes,
                        peak_allocated_gib=check_peak())
            atomic_json(branch_dir / "result.json", part)
            row["branches"][branch] = {
                "result_file": str((branch_dir / "result.json").relative_to(ROOT)),
                "video_sha256": part["video_sha256"]}
            row["sampling_forwards"] += part["sampling_forwards"]
            row["vae_decodes"] += 1
            atomic_json(row_path, row)
            pipe.load_models_to_device(["dit"])
        assert sha(O11 / "G1" / scene / "FM8/first12.pt") == refs["first_endpoint_sha256"]
        assert sha(O11 / "G1" / scene / "FM8/published_39.npy") == refs["first_published_sha256"]
        assert tensor_sha(first) == first_tensor_sha
        row.update(status="complete_pending_visual_review", peak_allocated_gib=check_peak(),
                   wall_seconds=time.perf_counter() - started,
                   peak_cpu_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
        atomic_json(row_path, row)
    except BaseException as error:
        row.update(status="failed", error=repr(error), wall_seconds=time.perf_counter() - started)
        atomic_json(row_path, row)
        raise


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene", choices=CFG["scenes"], required=True)
    ap.add_argument("--gpu", type=int, required=True)
    args = ap.parse_args()
    setup_paths()
    verify_sources()
    marker = require_marker(f"G1_{args.scene}")
    if marker.get("scene") != args.scene:
        raise PermissionError("Judge marker scene mismatch")
    check_machine(args.gpu)
    import torch
    torch.set_num_threads(4)
    torch.manual_seed(CFG["seed"])
    ledger = Ledger(args.gpu, f"G1_{args.scene}")
    remaining = min(CFG["max_gpu_seconds"] - ledger.data["gpu_seconds"],
                    datetime.fromisoformat(CFG["deadline_hkt"]).timestamp()
                    - datetime.now(timezone.utc).timestamp())
    if remaining <= 0:
        ledger.close("failed_before_model_load")
        raise TimeoutError("EXP-012 budget or 09:00 HKT cutoff")
    def alarm(_signum, _frame):
        raise TimeoutError("EXP-012 absolute GPU alarm")
    previous = signal.signal(signal.SIGALRM, alarm)
    signal.setitimer(signal.ITIMER_REAL, remaining)
    try:
        run_scene(args.scene, args.gpu, ledger)
        ledger.close("complete")
    except BaseException:
        ledger.close("failed")
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


if __name__ == "__main__":
    main()
