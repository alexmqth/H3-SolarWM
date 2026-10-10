"""EXP-014 frozen V3 FM30 teacher C1 + AA/AD C2; requires Judge markers.

C1: A action generates 12 latent / 39 RGB.
C2: commit that same teacher C1 and fork AA/AD to 56 RGB.
No training, GT history, AnyFlow, DMD or old encoded input.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import resource
import time

from common import (CFG, FROZEN, HERE, OUT, ROOT, Ledger, atomic_json,
                    check_machine, require_marker, setup_paths, sha, verify_sources)


def tensor_sha(tensor) -> str:
    import hashlib
    import torch
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(str((tuple(value.shape), str(value.dtype))).encode() +
                          value.view(torch.uint8).numpy().tobytes()).hexdigest()


def cache_signature(cache):
    """Detect in-memory history KV replacement or mutation during sampling."""
    return (cache.commits, tuple((layer, tuple(
        (entry.index, id(entry), tuple((id(t), t.data_ptr(), t._version, tuple(t.shape))
                                     for t in (entry.key, entry.value, entry.rope)))
        for entry in entries)) for layer, entries in sorted(cache.layers.items())))


def load_fixture(scene: str):
    import torch
    result = json.loads((OUT / "P1_result.json").read_text())
    if result["status"] != "complete_pending_cpu_audit":
        raise RuntimeError("P1 not complete")
    entry = result["fixtures"][scene]
    path = ROOT / entry["path"]
    if sha(path) != entry["sha256"]:
        raise RuntimeError("P1 fixture hash changed")
    fixture = torch.load(path, map_location="cpu", weights_only=True)
    if fixture["format"] != "exp014_native_single_i0_full37_v1" or fixture["scene"] != scene:
        raise RuntimeError("wrong fixture protocol or scene")
    if fixture["source_png_sha256"] != json.loads((HERE / "source_manifest.json").read_text())["scenes"][scene]["png_sha256"]:
        raise RuntimeError("wrong initial image")
    packed = fixture["packed"]
    assert fixture["initial_noise"].shape == (1, 24, 37, 30, 52)
    assert fixture["anchor"].shape == (390, 96)
    assert packed["img_pos"].numel() == 38 * 390
    assert len(packed["action_text_spans_local"]) == 37
    assert torch.equal(fixture["prompts"]["A"][:packed["action_text_spans_local"][0][0]],
                       fixture["prompts"]["D"][:packed["action_text_spans_local"][0][0]])
    return fixture, entry


def load_model():
    import torch
    import infer as abot
    from causal.h3_precision import configure_precision
    pipe = abot.load_pipeline("cuda:0")
    released = FROZEN / "runtime/checkpoints/H3-World/step-10000.safetensors"
    baseline = json.loads((ROOT / "submission/experiments/EXP-006_v3_fm8_full/source_manifest.json").read_text())
    assert sha(released) == baseline["sources"]["released_action_lora"]["sha256"]
    pipe.load_lora(pipe.dit, state_dict=abot.load_checkpoint_lora(released), hotload=True)
    model = pipe.dit.eval().requires_grad_(False)
    precision = configure_precision(model, CFG["precision"],
        native_transformer_dir=FROZEN / "runtime/DiffSynth-Studio-h3-v2/models/MiniMax/MiniMax-H3/FL2VA/transformer")
    pipe.load_models_to_device(["dit"])
    return pipe, model, precision


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
    if action == "A":
        prompt = fixture["prompts"]["A"]
    elif action == "AD":
        prompt = fixture["prompts"]["A"].clone()
        donor = fixture["prompts"]["D"]
        for lo, hi in fixture["packed"]["action_text_spans_local"][12:stop]:
            prompt[lo:hi] = donor[lo:hi]
    else:
        raise ValueError(action)
    layout, cropped = visible_inputs(packed, prompt.to("cuda:0"), stop, 390)
    assert len(layout["action_text_rows"]) == stop
    assert layout["seq_len"] == layout["action_video_start"] + stop * 390
    assert layout["img_pos"].numel() == (stop + 1) * 390
    return layout, cropped


def check_peak():
    import torch
    peak = torch.cuda.max_memory_allocated() / 2**30
    if peak > CFG["gpu_allocated_cap_gib"]:
        raise MemoryError(f"allocated peak {peak:.3f} GiB exceeds task cap")
    return peak


def sample(pipe, model, fixture, *, cache, start: int, stop: int, index: int,
           action: str, steps: int, ledger: Ledger, row: dict):
    import torch
    from causal.anyflow_sampling import configure_video_schedule
    from current_prefix import current_prefix_feedback
    from interval_cached import interval_cached
    layout, prompt = visible(fixture, action, stop)
    audio = fixture["audio_noise"].to("cuda:0")
    anchor = fixture["anchor"].to("cuda:0")
    current = fixture["initial_noise"][:, :, start:stop].to("cuda:0", torch.float32).clone()
    initial_sha = tensor_sha(current)
    sigmas = configure_video_schedule(pipe.scheduler, steps=steps,
                                       grid="native", flow_shift=CFG["flow_shift"])
    assert len(sigmas) == steps + 1 and sigmas[0] == 1 and sigmas[-1] == 0
    row.update(sigmas=sigmas, prompt_sha256=tensor_sha(prompt),
               position_sha256=tensor_sha(layout["img_position_ids"]),
               initial_noise_sha256=initial_sha, sampling_forwards=0)
    torch.cuda.synchronize(); began = time.perf_counter()
    with torch.no_grad(), current_prefix_feedback():
        for step, timestep in enumerate(pipe.scheduler.timesteps, 1):
            ledger.reserve("sampling_forwards")
            velocity = interval_cached(model, current, start=start, index=index,
                cache=cache, full_packed=layout, prompt=prompt, anchor=anchor,
                audio=audio, sigma=float(timestep) / 1000)
            current = pipe.scheduler.step(velocity, timestep, current)
            if not torch.isfinite(current).all():
                raise FloatingPointError("nonfinite latent")
            row["sampling_forwards"] = step
            row["step_completed"] = step
            check_peak()
            atomic_json(Path(row["row_file"]), row)
    torch.cuda.synchronize()
    row["sampling_seconds"] = time.perf_counter() - began
    row["endpoint_tensor_sha256"] = tensor_sha(current)
    assert tensor_sha(fixture["initial_noise"][:, :, start:stop].to(torch.float32)) == initial_sha
    return current, layout, prompt, audio, anchor


def decode(pipe, history, current, ledger: Ledger):
    import numpy as np
    import torch
    pipe.load_models_to_device(["video_vae"])
    ledger.reserve("vae_decodes")
    torch.cuda.synchronize(); began = time.perf_counter()
    latents = current if history is None else torch.cat((history, current), dim=2)
    with torch.no_grad():
        rgb = pipe.video_vae.decode_video(latents, dtype=pipe.torch_dtype,
            process_image=False, tiled=True, tile_size=256, tile_overlap=64)
        frames = pipe.vae_output_to_video(rgb, min_value=0, max_value=1)
    torch.cuda.synchronize()
    arr = np.stack([np.asarray(frame.convert("RGB"), dtype=np.uint8) for frame in frames])
    return arr, time.perf_counter() - began


def write_video_and_sheet(frames, video: Path, sheet: Path):
    from PIL import Image, ImageDraw
    from benchmark import write_video
    video.parent.mkdir(parents=True, exist_ok=True)
    write_video([Image.fromarray(frame) for frame in frames], video)
    columns, width, height = 5, 208, 120
    output = Image.new("RGB", (columns * width, ((len(frames)+columns-1)//columns)*(height+20)), "black")
    draw = ImageDraw.Draw(output)
    for i, frame in enumerate(frames):
        x, y = (i % columns)*width, (i//columns)*(height+20)
        output.paste(Image.fromarray(frame).resize((width, height)), (x,y))
        draw.text((x+4,y+height+2), str(i), fill="white")
    output.save(sheet, quality=90)


def run_g1(scene: str, method: str, gpu: int, ledger: Ledger):
    import numpy as np
    import torch
    from causal.h3_cached import H3ChunkCache
    fixture, entry = load_fixture(scene)
    directory = OUT / "G1" / scene / method
    if directory.exists():
        raise FileExistsError(f"prior attempt exists: {directory}")
    directory.mkdir(parents=True)
    row_path = directory / "result.json"
    row = {"task": CFG["task"], "stage": "G1", "scene": scene, "method": method,
           "status": "loading", "row_file": str(row_path), "gpu": gpu,
           "fixture_sha256": entry["sha256"], "runner_sha256": sha(__file__),
           "sampling_forwards": 0, "commit_forwards": 0, "vae_decodes": 0}
    atomic_json(row_path, row)
    started = time.perf_counter()
    try:
        pipe, model, precision = load_model()
        row["precision"] = precision
        cache = H3ChunkCache(max_history=CFG["max_history"], storage_device="cpu")
        assert cache.history(0, 0) == []
        row["status"] = "sampling"; atomic_json(row_path, row)
        current, _, _, _, _ = sample(pipe, model, fixture, cache=cache, start=0, stop=12,
            index=0, action="A", steps=CFG["steps"],
            ledger=ledger, row=row)
        assert not cache.layers, "G1 must not commit hidden KV"
        endpoint = directory / "first12.pt"
        torch.save(current.cpu(), endpoint)
        row["endpoint_sha256"] = sha(endpoint)
        frames, seconds = decode(pipe, None, current, ledger)
        assert frames.shape == (39, CFG["height"], CFG["width"], 3)
        row["vae_decodes"] = 1; row["decode_seconds"] = seconds
        rgb_path = directory / "published_39.npy"
        np.save(rgb_path, frames)
        video = directory / "first39.mp4"
        write_video_and_sheet(frames, video, directory / "all_39_frames.jpg")
        row.update(status="complete_pending_visual_review", video_sha256=sha(video),
                   published_sha256=sha(rgb_path), peak_allocated_gib=check_peak(),
                   wall_seconds=time.perf_counter()-started,
                   peak_cpu_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
        atomic_json(row_path, row)
    except BaseException as error:
        row.update(status="failed", error=repr(error), wall_seconds=time.perf_counter()-started)
        atomic_json(row_path, row)
        raise


def run_g2(scene: str, method: str, gpu: int, ledger: Ledger):
    import numpy as np
    import torch
    from causal.h3_cached import H3ChunkCache
    from current_prefix import current_prefix_feedback
    from interval_cached import interval_cached
    fixture, entry = load_fixture(scene)
    first_dir = OUT / "G1" / scene / method
    first_row = json.loads((first_dir / "result.json").read_text())
    if first_row["status"] != "complete_pending_visual_review":
        raise RuntimeError("G1 not complete")
    first_file = first_dir / "first12.pt"
    if sha(first_file) != first_row["endpoint_sha256"]:
        raise RuntimeError("first chunk changed")
    published_file = first_dir / "published_39.npy"
    if sha(published_file) != first_row["published_sha256"]:
        raise RuntimeError("published first RGB changed")
    published = np.load(published_file)
    assert published.shape == (39, 480, 832, 3)
    first = torch.load(first_file, map_location="cpu", weights_only=True).to("cuda:0", torch.float32)
    directory = OUT / "G2" / scene / method
    if directory.exists():
        raise FileExistsError(f"prior attempt exists: {directory}")
    directory.mkdir(parents=True)
    row_path = directory / "result.json"
    row = {"task": CFG["task"], "stage": "G2", "scene": scene, "method": method,
           "status": "loading", "gpu": gpu, "row_file": str(row_path),
           "fixture_sha256": entry["sha256"], "first_endpoint_sha256": sha(first_file),
           "first_published_sha256": sha(published_file), "runner_sha256": sha(__file__),
           "sampling_forwards": 0, "commit_forwards": 0, "vae_decodes": 0, "branches": {}}
    atomic_json(row_path, row)
    started = time.perf_counter()
    try:
        pipe, model, precision = load_model()
        row["precision"] = precision
        cache = H3ChunkCache(max_history=CFG["max_history"], storage_device="cpu")
        layout, prompt = visible(fixture, "A", 12)
        audio = fixture["audio_noise"].to("cuda:0")
        anchor = fixture["anchor"].to("cuda:0")
        ledger.reserve("commit_forwards")
        torch.cuda.synchronize(); tick = time.perf_counter()
        with torch.no_grad(), current_prefix_feedback():
            _ = interval_cached(model, first, start=0, index=0, cache=cache,
                full_packed=layout, prompt=prompt, anchor=anchor, audio=audio,
                sigma=0.0, commit=True)
        torch.cuda.synchronize()
        assert len(cache.layers) == 50 and all(len(entries) == 1 for entries in cache.layers.values())
        row["commit_forwards"] = 1
        row["commit_seconds"] = time.perf_counter() - tick
        row["cpu_raw_kv_bytes"] = cache.nbytes
        cache_path = directory / "cache_through12.pt"
        torch.save(cache, cache_path)
        row["cache_sha256"] = sha(cache_path)
        frozen_cache_signature = cache_signature(cache)
        atomic_json(row_path, row)
        for branch, action in (("AA", "A"), ("AD", "AD")):
            branch_dir = directory / branch
            branch_dir.mkdir()
            part = {"row_file": str(branch_dir / "result.json"), "status": "sampling",
                    "branch": branch, "scene": scene, "method": method,
                    "first_endpoint_sha256": row["first_endpoint_sha256"],
                    "source_cache_sha256": row["cache_sha256"]}
            atomic_json(branch_dir / "result.json", part)
            current, _, _, _, _ = sample(pipe, model, fixture, cache=cache,
                start=12, stop=17, index=1, action=action,
                steps=CFG["steps"], ledger=ledger, row=part)
            assert cache_signature(cache) == frozen_cache_signature
            endpoint = branch_dir / "chunk_12_17.pt"
            torch.save(current.cpu(), endpoint)
            part["endpoint_sha256"] = sha(endpoint)
            assert sha(cache_path) == row["cache_sha256"]
            frames, seconds = decode(pipe, first, current, ledger)
            assert frames.shape == (56, CFG["height"], CFG["width"], 3)
            # Decode 56 from the available prefix, then append only 17 new RGB.
            joined = np.concatenate((published, frames[39:]), axis=0)
            assert np.array_equal(joined[:39], published)
            np.save(branch_dir / "published_56.npy", joined)
            video = branch_dir / "rollout_56.mp4"
            write_video_and_sheet(joined, video, branch_dir / "all_56_frames.jpg")
            write_video_and_sheet(joined[39:], branch_dir / "new_17.mp4",
                                  branch_dir / "new_17_frames.jpg")
            part.update(status="complete_pending_visual_review", endpoint_sha256=sha(endpoint),
                        published_sha256=sha(branch_dir / "published_56.npy"),
                        video_sha256=sha(video), decode_seconds=seconds, vae_decodes=1,
                        cpu_raw_kv_bytes=cache.nbytes, peak_allocated_gib=check_peak())
            atomic_json(branch_dir / "result.json", part)
            row["branches"][branch] = {"result_file": str((branch_dir / "result.json").relative_to(ROOT)),
                                      "video_sha256": part["video_sha256"]}
            row["sampling_forwards"] += part["sampling_forwards"]
            row["vae_decodes"] += 1
            atomic_json(row_path, row)
            pipe.load_models_to_device(["dit"])
        assert sha(first_file) == row["first_endpoint_sha256"]
        assert sha(published_file) == row["first_published_sha256"]
        row.update(status="complete_pending_visual_review", peak_allocated_gib=check_peak(),
                   wall_seconds=time.perf_counter()-started,
                   peak_cpu_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
        atomic_json(row_path, row)
    except BaseException as error:
        row.update(status="failed", error=repr(error), wall_seconds=time.perf_counter()-started)
        atomic_json(row_path, row)
        raise


def make_training_manifest(scene: str) -> None:
    """Index immutable teacher latents; future student must rebuild its own KV."""
    base = OUT / "G1" / scene / "FM30"
    cont = OUT / "G2" / scene / "FM30"
    first_row = json.loads((base / "result.json").read_text())
    cont_row = json.loads((cont / "result.json").read_text())
    assert first_row["status"] == cont_row["status"] == "complete_pending_visual_review"
    target = {"task": CFG["task"], "scene": scene, "quality_status": "pending_judge",
              "generated_history": True, "gt_reset": False,
              "teacher_C1_latent": {"path": str((base / "first12.pt").relative_to(ROOT)),
                                    "sha256": first_row["endpoint_sha256"]},
              "teacher_C1_rgb": {"path": str((base / "published_39.npy").relative_to(ROOT)),
                                 "sha256": first_row["published_sha256"]},
              "teacher_C2_endpoint": {},
              "student_history_rule": "reuse frozen teacher C1 latent; rebuild sigma0 KV under each current student weight"}
    for branch in ("AA", "AD"):
        row = json.loads((cont / branch / "result.json").read_text())
        assert row["status"] == "complete_pending_visual_review"
        endpoint = cont / branch / "chunk_12_17.pt"
        target["teacher_C2_endpoint"][branch] = {"path": str(endpoint.relative_to(ROOT)),
                                                   "sha256": row["endpoint_sha256"]}
    atomic_json(OUT / "teacher" / scene / "target_manifest.json", target)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=["T1", "T2"], required=True)
    ap.add_argument("--scene", choices=list(CFG["scenes"]), required=True)
    ap.add_argument("--gpu", type=int, required=True)
    args = ap.parse_args()
    setup_paths()
    verify_sources()
    require_marker(args.stage, args.scene)
    import torch
    torch.set_num_threads(4)
    torch.manual_seed(CFG["seed"])
    ledger = Ledger(args.gpu, args.stage, args.scene)
    try:
        check_machine(args.gpu, args.stage)
        run_g1(args.scene, "FM30", args.gpu, ledger)
        # EXP-011 ran C1/C2 in separate processes. Here they share one
        # per-scene process, so explicitly collect the released first pipeline
        # before constructing the second 33B pipeline. This changes no tensor
        # values or history protocol.
        gc.collect()
        torch.cuda.empty_cache()
        remaining_gib = torch.cuda.memory_allocated() / 2**30
        atomic_json(OUT / "teacher" / args.scene / "model_lifecycle.json",
                    {"after_C1_collect_allocated_gib": remaining_gib,
                     "C2_load_allowed_below_gib": 8.0})
        if remaining_gib >= 8.0:
            raise MemoryError(f"C1 model still occupies {remaining_gib:.2f} GiB before C2 load")
        run_g2(args.scene, "FM30", args.gpu, ledger)
        make_training_manifest(args.scene)
        ledger.close("complete")
    except BaseException:
        ledger.close("failed")
        raise


if __name__ == "__main__":
    main()
