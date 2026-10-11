"""EXP-016: CPU P0, then marker-gated source-VAE-only P1 on GPU0."""
from __future__ import annotations

import argparse
import gc
import inspect
import json
import math
import os
from pathlib import Path
import sys
import time

import av
import numpy as np
from PIL import Image

from common import (CFG, FROZEN, HERE, OUT, VAE_FILE, EXPECTED_VAE_SHA, Ledger,
                    check_machine, cpu_affinity, freeze_sources, require_marker,
                    setup_runtime, sha, source_rows, verify_frozen, write)


def read_frames(path: Path) -> list[Image.Image]:
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_count = 1
        assert int(stream.average_rate) == 24
        frames = [Image.fromarray(frame.to_ndarray(format="rgb24"))
                  for frame in container.decode(video=0)]
    assert len(frames) == 56 and all(im.size == (832, 480) for im in frames)
    return frames


def preflight() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("P0 requires CUDA_VISIBLE_DEVICES=''")
    cpus = cpu_affinity()
    setup_runtime()
    import torch
    from diffsynth.models.minimax_h3_video_vae import (MiniMaxH3VideoVAE,
        TemporalIsolatedSpatialGroupNorm)
    from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Pipeline
    torch.set_num_threads(4)
    if sha(VAE_FILE) != EXPECTED_VAE_SHA:
        raise RuntimeError("source VAE weight SHA changed")
    signature = inspect.signature(MiniMaxH3VideoVAE)
    assert signature.parameters["causal_encoder"].default is True
    assert signature.parameters["use_t_isolated_gn"].default is True
    assert signature.parameters["clip_length"].default == 17
    assert signature.parameters["token_drop"].default == 3
    assert TemporalIsolatedSpatialGroupNorm.__name__ == "TemporalIsolatedSpatialGroupNorm"
    assert CFG["counts"] == {"image_encode": 4, "video_encode": 8, "video_decode": 4}
    rows = source_rows()
    probe = MiniMaxH3Pipeline(device="cpu", torch_dtype=torch.float32)
    source_checks = []
    for row in rows:
        frames = read_frames(Path(row["output_files"]["real56.mp4"]["path"]))
        assert np.array_equal(np.asarray(frames[0]),
                              np.asarray(Image.open(row["output_files"]["I0.png"]["path"]).convert("RGB")))
        prefix = frames[:39]
        assert all(np.array_equal(np.asarray(frames[j]), np.asarray(prefix[j])) for j in range(39))
        inp = probe.preprocess_video(frames, torch_dtype=torch.float32,
                                     device="cpu", min_value=0, max_value=1)
        assert inp.dtype == torch.float32 and tuple(inp.shape) == (1, 3, 56, 480, 832)
        assert float(inp.min()) >= 0 and float(inp.max()) <= 1
        assert torch.equal(inp[:, :, :39], probe.preprocess_video(
            prefix, torch_dtype=torch.float32, device="cpu", min_value=0, max_value=1))
        source_checks.append({"scene": row["scene"], "decoded_frames": 56,
                              "input_min": float(inp.min()), "input_max": float(inp.max()),
                              "prefix_exact_same_source": True,
                              "native_latent_lengths": [12, 17]})
        del inp
    frozen = freeze_sources()
    write(HERE / "source_manifest.json", frozen)
    write(HERE / "P0_PREFLIGHT.json", {"task": CFG["task"], "status": "PASS_CPU",
          "cpu_affinity": cpus, "GPU_calls": 0, "model_loads": 0,
          "torch": torch.__version__, "source_checks": source_checks,
          "runtime_vae_source_sha256": frozen["runtime_vae_source_sha256"],
          "runtime_flags_from_constructor": {"causal_encoder": True,
            "use_t_isolated_gn": True, "clip_length": 17, "token_drop": 3}})
    print("P0 CPU PASS", len(source_checks), "scenes", flush=True)


def load_vae():
    import torch
    from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Pipeline, ModelConfig
    dtype = torch.bfloat16
    vram = dict(offload_dtype=dtype, offload_device="cpu", onload_dtype=dtype,
                onload_device="cpu", preparing_dtype=dtype, preparing_device="cuda:0",
                computation_dtype=dtype, computation_device="cuda:0")
    pipe = MiniMaxH3Pipeline.from_pretrained(
        torch_dtype=dtype, device="cuda:0",
        model_configs=[ModelConfig(model_id="MiniMax/MiniMax-H3",
                                   origin_file_pattern="FL2VA/video_vae/source/model.safetensors", **vram)],
        processor_config=None, vram_limit=43.0)
    if pipe.video_vae is None or any(x is not None for x in
                                     (pipe.text_encoder, pipe.dit, pipe.audio_vae, pipe.processor)):
        raise RuntimeError("P1 must load source video VAE only")
    pipe.load_models_to_device(["video_vae"])
    return pipe


def save_tensor(path: Path, tensor) -> dict:
    import torch
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    torch.save(tensor.detach().cpu(), tmp)
    tmp.rename(path)
    return {"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size,
            "shape": list(tensor.shape), "dtype": str(tensor.dtype)}


def write_mp4(path: Path, frames: list[np.ndarray]) -> dict:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".partial.mp4")
    with av.open(str(tmp), "w") as container:
        stream = container.add_stream("libx264", rate=24)
        stream.width = frames[0].shape[1]
        stream.height = frames[0].shape[0]
        stream.pix_fmt = "yuv420p"
        stream.thread_count = 2
        stream.options = {"crf": "14", "preset": "veryfast"}
        for arr in frames:
            for packet in stream.encode(av.VideoFrame.from_ndarray(arr, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    tmp.rename(path)
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        decoded = list(container.decode(video=0))
        assert len(decoded) == 56 and int(stream.average_rate) == 24
    return {"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size,
            "frames": len(decoded), "fps": 24, "resolution": list((stream.width, stream.height))}


def scene_run(pipe, ledger: Ledger, row: dict) -> dict:
    import torch
    from diffsynth.models.minimax_h3_video_vae import BaseConv3d, TemporalIsolatedSpatialGroupNorm
    scene = row["scene"]
    dest = OUT / scene
    if dest.exists():
        raise FileExistsError(f"scene output exists; no overwrite: {dest}")
    frames = read_frames(Path(row["output_files"]["real56.mp4"]["path"]))
    i0 = Image.open(row["output_files"]["I0.png"]["path"]).convert("RGB")
    assert np.array_equal(np.asarray(i0), np.asarray(frames[0]))
    video = pipe.preprocess_video(frames, torch_dtype=torch.float32,
                                  device="cuda:0", min_value=0, max_value=1)
    image = pipe.preprocess_image(i0, torch_dtype=torch.float32,
                                  device="cuda:0", min_value=0, max_value=1)
    assert torch.equal(video[:, :, 0], image)
    prefix = video[:, :, :39].contiguous()
    assert tuple(video.shape) == (1, 3, 56, 480, 832)
    assert tuple(prefix.shape) == (1, 3, 39, 480, 832)
    assert video.dtype == image.dtype == torch.float32
    assert float(video.min()) >= 0 and float(video.max()) <= 1
    vae = pipe.video_vae
    convs = [m for m in vae.encoder.modules() if isinstance(m, BaseConv3d)]
    norms = [m for m in vae.encoder.modules() if isinstance(m, TemporalIsolatedSpatialGroupNorm)]
    assert convs and all(m.causal for m in convs) and norms
    assert (vae.clip_length, vae.token_drop, vae.tile_size, vae.tile_overlap_min)[:2] == (17, 3)
    with torch.no_grad():
        ledger.reserve("image_encode", scene)
        z_i0 = vae.encode_video(image, dtype=torch.bfloat16, process_image=True,
                                tile_size=256, tile_overlap=64)
        ledger.reserve("video_encode", scene)
        z56 = vae.encode_video(video, dtype=torch.bfloat16, process_image=False,
                               tile_size=256, tile_overlap=64)
        ledger.reserve("video_encode", scene)
        z39 = vae.encode_video(prefix, dtype=torch.bfloat16, process_image=False,
                               tile_size=256, tile_overlap=64)
    assert tuple(z_i0.shape) == (1, 24, 1, 30, 52)
    assert tuple(z56.shape) == (1, 24, 17, 30, 52)
    assert tuple(z39.shape) == (1, 24, 12, 30, 52)
    assert all(torch.isfinite(x).all() for x in (z_i0, z56, z39))
    delta = z56[:, :, :12].float() - z39.float()
    max_abs = float(delta.abs().max())
    mean_abs = float(delta.abs().mean())
    relative_rms = float(torch.sqrt(delta.square().mean()) /
                         torch.sqrt(z39.float().square().mean()).clamp_min(1e-12))
    compare = {"exact_equal": bool(torch.equal(z56[:, :, :12], z39)),
               "max_abs": max_abs, "mean_abs": mean_abs, "relative_RMS": relative_rms,
               "dtype_56": str(z56.dtype), "dtype_39": str(z39.dtype),
               "gate_pass": max_abs <= CFG["prefix_max_abs_gate"] and
                            relative_rms <= CFG["prefix_relative_rms_gate"]}
    result = {"task": CFG["task"], "scene": scene, "status": "encoded_pending_roundtrip",
              "source_video_sha256": row["output_files"]["real56.mp4"]["sha256"],
              "source_I0_sha256": row["output_files"]["I0.png"]["sha256"],
              "input_range": [float(video.min()), float(video.max())],
              "input_dtype": str(video.dtype), "first39_same_tensor": True,
              "VAE_runtime": {"encoder_conv_count": len(convs),
                              "encoder_convs_all_causal": True,
                              "temporal_isolated_norm_count": len(norms),
                              "clip_length": vae.clip_length, "token_drop": vae.token_drop,
                              "tile_size": vae.tile_size, "tile_overlap": vae.tile_overlap_min},
              "prefix_comparison": compare, "files": {}}
    dest.mkdir(parents=True)
    result["files"]["I0_latent"] = save_tensor(dest / "I0.pt", z_i0)
    result["files"]["full17"] = save_tensor(dest / "full17.pt", z56)
    result["files"]["prefix12"] = save_tensor(dest / "prefix12.pt", z39)
    write(dest / "metrics.json", result)
    if not compare["gate_pass"]:
        raise RuntimeError(f"{scene}: significant VAE prefix difference; stop subsequent scenes")
    ledger.reserve("video_decode", scene)
    with torch.no_grad():
        recon = vae.decode_video(z56, dtype=torch.bfloat16, process_image=False,
                                 tile_size=256, tile_overlap=64)
    assert tuple(recon.shape) == (1, 3, 56, 480, 832)
    assert torch.isfinite(recon).all()
    rebuilt_pil = pipe.vae_output_to_video(recon, min_value=0, max_value=1)
    assert len(rebuilt_pil) == 56
    orig = [np.asarray(f, dtype=np.uint8) for f in frames]
    rebuilt = [np.asarray(f, dtype=np.uint8) for f in rebuilt_pil]
    assert all(x.shape == (480, 832, 3) for x in rebuilt)
    mse = [float(np.square(a.astype(np.float32) - b.astype(np.float32)).mean())
           for a, b in zip(orig, rebuilt)]
    mad = [float(np.abs(a.astype(np.float32) - b.astype(np.float32)).mean())
           for a, b in zip(orig, rebuilt)]
    psnr = [10 * math.log10(255**2 / x) if x > 0 else float("inf") for x in mse]
    result["files"]["reconstruction"] = write_mp4(dest / "reconstruction.mp4", rebuilt)
    paired = [np.concatenate((a, b), axis=1) for a, b in zip(orig, rebuilt)]
    result["files"]["source_vs_reconstruction"] = write_mp4(
        dest / "source_vs_reconstruction.mp4", paired)
    result.update(status="complete_pending_visual_review",
                  roundtrip={"frames": 56, "MAD_per_frame": mad,
                             "PSNR_per_frame": psnr,
                             "MAD_mean": float(np.mean(mad)),
                             "PSNR_mean": float(np.mean(psnr))})
    write(dest / "metrics.json", result)
    del video, image, prefix, z_i0, z56, z39, recon
    gc.collect(); torch.cuda.empty_cache()
    return result


def run() -> None:
    cpu_affinity()
    setup_runtime()
    require_marker()  # must precede any CUDA or model import
    check_machine(before_start=True)
    if sha(VAE_FILE) != EXPECTED_VAE_SHA:
        raise RuntimeError("source VAE weight SHA changed")
    rows = source_rows()
    import torch
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats()
    free = __import__("shutil").disk_usage(OUT.parent).free
    ledger = Ledger(free)
    summary = {"task": CFG["task"], "status": "loading", "scenes": [],
               "marker_sha256": sha(HERE / "judge/P1_APPROVED.json"),
               "source_manifest_sha256": sha(HERE / "source_manifest.json"),
               "code_manifest_sha256": sha(HERE / "code_manifest.json")}
    write(OUT / "result.json", summary)
    try:
        ledger.guard(); ledger.data["model_loads"] = 1; ledger.save()
        pipe = load_vae()
        summary["status"] = "encoding"; write(OUT / "result.json", summary)
        for row in rows:
            ledger.guard()
            result = scene_run(pipe, ledger, row)
            summary["scenes"].append({"scene": row["scene"], "metrics": result})
            summary["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
            summary["elapsed_seconds"] = time.monotonic() - ledger.start
            write(OUT / "result.json", summary)
            if summary["peak_allocated_gib"] > CFG["max_allocated_gib"]:
                raise MemoryError("allocated VRAM cap exceeded")
        assert all(ledger.data[k] == v for k, v in CFG["counts"].items())
        summary["status"] = "complete_pending_judge"
        write(OUT / "result.json", summary)
        ledger.close("complete")
    except BaseException as exc:
        summary.update(status="failed_or_stopped", error=repr(exc),
                       elapsed_seconds=time.monotonic() - ledger.start)
        write(OUT / "result.json", summary)
        ledger.close("failed_or_stopped", repr(exc))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--encode", action="store_true")
    args = parser.parse_args()
    if args.preflight == args.encode:
        parser.error("choose exactly one mode")
    preflight() if args.preflight else run()
