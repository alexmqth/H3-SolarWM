"""EXP-005/v2 G0: metered old-vs-SW-G C6 regression; no training.

`--preflight` hashes frozen inputs on CPU. `--run` requires that exact manifest,
reserves each of 34 forwards before calling the model, and refuses retries.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = HERE.parents[3]
CONFIG = HERE / "g0_config.json"
AUTH = HERE / "g0_authorization.json"
FROZEN = ROOT / "H3-World/outputs/2026-10-09-22/chunk_partition_cb"
PREV = ROOT / "H3-World/outputs/EXP-003_native_cached_124/AA"
EXP003 = ROOT / "submission/experiments/EXP-003_native_cached_124"
EXP001 = ROOT / "submission/experiments/EXP-001_v2b_124"
ROUTER = ROOT / "submission/reports/stage1_anyflow/02_causal_diagnostics/current_prefix_candidate"
FIRST = ROOT / "submission/experiments/11_causal_12_then5_selfhistory/states/first12_A.pt"
SOURCES = {
    "parking_A": FROZEN / "source_coarse/inputs/parking_A.pt",
    "parking_D": FROZEN / "source_coarse/inputs/parking_D.pt",
    "released_lora": FROZEN / "runtime/checkpoints/H3-World/step-10000.safetensors",
    "cache_through32": PREV / "cache_through32.pt",
    "C6_endpoint": PREV / "chunk_32_37.pt",
    "prior_RGB_107": PREV / "published_107.npy",
    "baseline_RGB_124": PREV / "published_124.npy",
    "C1_endpoint": FIRST,
    "C2_endpoint": ROOT / "H3-World/outputs/EXP-002_native_cached/AA/chunk_12_17.pt",
    "C3_endpoint": ROOT / "H3-World/outputs/EXP-002_native_cached/AA/chunk_17_22.pt",
    "C4_endpoint": PREV / "chunk_22_27.pt",
    "C5_endpoint": PREV / "chunk_27_32.pt",
    "baseline_C6_metrics": EXP003 / "artifacts/metrics/AA_chunk_32_37.json",
    "old_interval": EXP003 / "interval_cached.py",
    "new_interval": EXP / "interval_sw.py",
    "chunk_plan": EXP / "chunk_plan.py",
    "position_sw": EXP / "position_sw.py",
    "router": ROUTER / "current_prefix.py",
    "raw_KV": FROZEN / "runtime/code/causal/h3_cached.py",
    "H3_dit": FROZEN / "runtime/DiffSynth-Studio-h3-v2/diffsynth/models/minimax_h3_dit.py",
    "H3_pipeline": FROZEN / "runtime/DiffSynth-Studio-h3-v2/diffsynth/pipelines/minimax_h3_audio_video.py",
    "infer": FROZEN / "runtime/code/abot/infer.py",
    "h3_precision": FROZEN / "runtime/code/causal/h3_precision.py",
    "base_source_manifest": EXP001 / "source_manifest_v3.json",
    "runner": HERE / "g0_runner.py",
    "config": CONFIG,
}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def preflight() -> dict:
    cfg = json.loads(CONFIG.read_text())
    assert cfg["max_forwards"] == cfg["max_diagnostic"] + cfg["max_sampling"] == 34
    assert cfg["max_vae"] == 1 and cfg["gpu"] == 0
    hashes = {name: {"path": str(path), "sha256": sha(path)}
              for name, path in SOURCES.items()}
    prior = json.loads(SOURCES["baseline_C6_metrics"].read_text())
    assert hashes["cache_through32"]["sha256"] == prior["cache_file_sha256_before"]
    assert hashes["C6_endpoint"]["sha256"] == prior["endpoint_sha256"]
    prior_chunk = json.loads((EXP003 / "artifacts/metrics/AA_chunk_27_32.json").read_text())
    assert hashes["prior_RGB_107"]["sha256"] == prior_chunk["published_file_sha256"]
    assert hashes["baseline_RGB_124"]["sha256"] == prior["published_file_sha256"]
    assert hashes["released_lora"]["sha256"] == prior["source_checkpoint_sha256"]
    assert hashes["old_interval"]["sha256"] == prior["interval_sha256"]
    assert hashes["router"]["sha256"] == prior["accepted_router_sha256"]
    import torch
    pieces = [torch.load(SOURCES[f"C{i}_endpoint"], map_location="cpu", weights_only=True)
              for i in range(1,6)]
    history = torch.cat(pieces, dim=2).contiguous()
    tensor_sha = hashlib.sha256(str((tuple(history.shape),str(history.dtype))).encode()+
                                history.view(torch.uint8).numpy().tobytes()).hexdigest()
    assert history.shape == (1,24,32,30,52)
    assert tensor_sha == prior["history_tensor_sha256"]
    manifest = {"task": "EXP-005/v2", "stage": "G0", "created_utc": time.time(),
                "sources": hashes, "reference": str(SOURCES["baseline_C6_metrics"]),
                "source_history_tensor_sha256": tensor_sha,
                "reference_endpoint_tensor_sha256": prior["endpoint_tensor_sha256"],
                "reference_RGB_sha256": prior["published_rgb_sha256"]}
    atomic_json(HERE / "g0_source_manifest.json", manifest)
    return manifest


def verify_manifest() -> dict:
    path = HERE / "g0_source_manifest.json"
    if not path.exists():
        raise RuntimeError("run G0 CPU preflight first")
    manifest = json.loads(path.read_text())
    if set(manifest["sources"]) != set(SOURCES):
        raise RuntimeError("source set changed after preflight")
    for name, file in SOURCES.items():
        if str(file) != manifest["sources"][name]["path"] or sha(file) != manifest["sources"][name]["sha256"]:
            raise RuntimeError(f"frozen G0 source changed: {name}")
    return manifest


def verify_authorization() -> dict:
    if not AUTH.exists():
        raise RuntimeError("G0 Judge authorization file missing")
    approval = json.loads(AUTH.read_text())
    expected = {"approved": True, "stage": "G0", "task": "EXP-005/v2",
                "runner_sha256": sha(Path(__file__)),
                "config_sha256": sha(CONFIG),
                "manifest_sha256": sha(HERE / "g0_source_manifest.json"),
                "max_forwards": 34, "max_vae": 1, "max_gpu_seconds": 1080}
    for key,value in expected.items():
        if approval.get(key) != value:
            raise RuntimeError(f"G0 Judge authorization mismatch: {key}")
    return approval


class Budget:
    def __init__(self, cfg: dict, root: Path):
        self.cfg, self.root = cfg, root
        self.start = time.time()
        self.ledger = {"task": cfg["task"], "stage": "G0", "first_start": self.start,
                       "gpu": cfg["gpu"], "diagnostic": 0, "sampling": 0,
                       "total_forwards": 0, "vae": 0, "status": "running"}
        self.write()

    def write(self):
        self.ledger["elapsed_seconds"] = time.time() - self.start
        atomic_json(self.root / "budget.json", self.ledger)

    def check_time(self):
        elapsed = time.time() - self.start
        if elapsed >= min(self.cfg["max_gpu_seconds"], self.cfg["max_elapsed_seconds"]):
            raise TimeoutError("G0 wall/GPU time budget exhausted")

    def reserve(self, kind: str):
        self.check_time()
        if kind not in ("diagnostic", "sampling"):
            raise ValueError(kind)
        if self.ledger[kind] >= self.cfg[f"max_{kind}"] or self.ledger["total_forwards"] >= self.cfg["max_forwards"]:
            raise RuntimeError("G0 forward budget exhausted")
        self.ledger[kind] += 1
        self.ledger["total_forwards"] += 1
        self.write()  # before every model attempt; failures consume budget

    def reserve_vae(self):
        self.check_time()
        if self.ledger["vae"] >= self.cfg["max_vae"]:
            raise RuntimeError("G0 VAE budget exhausted")
        self.ledger["vae"] += 1
        self.write()


def run(cfg: dict, manifest: dict, budget: Budget) -> None:
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
    from interval_cached import interval_cached
    from interval_sw import interval_sw
    from chunk_plan import PLAN, cache_identity, validate_cache
    from rollout_contract import prompt_for_path, rgb_sha, stitch_immutable

    torch.set_num_threads(4)
    torch.manual_seed(cfg["seed"])
    device = "cuda:0"
    data = {action: torch.load(SOURCES[f"parking_{action}"], map_location="cpu", weights_only=True)
            for action in "AD"}
    for name in ("initial_noise", "audio_noise", "anchor"):
        assert torch.equal(data["A"][name], data["D"][name]), name
    baseline = json.loads(SOURCES["baseline_C6_metrics"].read_text())
    pipe = abot.load_pipeline(device)
    pipe.load_lora(pipe.dit, state_dict=abot.load_checkpoint_lora(SOURCES["released_lora"]), hotload=True)
    model = pipe.dit.eval().requires_grad_(False)
    precision = configure_precision(model, "h3_fp32", native_transformer_dir=
        FROZEN / "runtime/DiffSynth-Studio-h3-v2/models/MiniMax/MiniMax-H3/FL2VA/transformer")
    pipe.load_models_to_device(["dit"])
    torch.cuda.synchronize()
    versions = [(p, p._version) for p in model.parameters()]

    def move(x):
        if torch.is_tensor(x): return x.to(device)
        if isinstance(x, dict): return {k: move(v) for k,v in x.items()}
        return x

    current_noise = data["A"]["initial_noise"][:, :, 32:37].to(device, torch.float32)
    endpoint_ref = torch.load(SOURCES["C6_endpoint"], map_location="cpu", weights_only=True).to(device, torch.float32)
    audio = data["A"]["audio_noise"].to(device)
    anchor = data["A"]["anchor"].to(device)
    full = move(data["A"]["packed"])
    prompt = prompt_for_path(data, "AA", 37).to(device)
    layout, prompt = visible_inputs(full, prompt, 37, 390)
    cache = torch.load(SOURCES["cache_through32"], map_location="cpu", weights_only=False)
    assert isinstance(cache, H3ChunkCache) and cache.storage_device == "cpu"
    cache_audit = validate_cache(cache, plan=PLAN, index=5, frame_rows=390, expected_layers=50)
    assert cache_audit["ancestors"] == [0,1,2,3,4]
    signature = cache_identity(cache)
    sigmas = configure_video_schedule(pipe.scheduler, steps=30, grid="native", flow_shift=2.22)
    assert sigmas == baseline["sigmas"]
    result = {"task": cfg["task"], "stage": "G0", "status": "running",
              "source_manifest_sha256": sha(HERE / "g0_source_manifest.json"),
              "config_sha256": sha(CONFIG), "runner_sha256": sha(Path(__file__)),
              "authorization_sha256": sha(AUTH),
              "history_cache_before": cache_audit, "precision": precision,
              "environment": {
                  "torch": torch.__version__, "torch_cuda": torch.version.cuda,
                  "gpu_name": torch.cuda.get_device_name(0),
                  "cuda_device_total_bytes": torch.cuda.get_device_properties(0).total_memory,
                  "sdpa_flash_enabled": torch.backends.cuda.flash_sdp_enabled(),
                  "sdpa_math_enabled": torch.backends.cuda.math_sdp_enabled(),
                  "sdpa_mem_efficient_enabled": torch.backends.cuda.mem_efficient_sdp_enabled(),
                  "sdpa_cudnn_enabled": torch.backends.cuda.cudnn_sdp_enabled(),
                  "cudnn_version": torch.backends.cudnn.version(),
                  "model_input_dtype": str(getattr(model,"_h3_input_dtype",None)),
                  "sdpa_function": "torch.nn.functional.scaled_dot_product_attention",
              },
              "sigmas": sigmas, "diagnostics": [], "sampling_steps": 0}
    out = budget.root
    atomic_json(out / "result.json", result)

    def forward(fn, state, sigma, kind):
        budget.reserve(kind)
        with torch.no_grad(), current_prefix_feedback():
            output = fn(model, state, start=32, index=5, cache=cache,
                        full_packed=layout, prompt=prompt, anchor=anchor, audio=audio,
                        sigma=sigma)
        if torch.cuda.max_memory_allocated()/2**30 > cfg["max_allocated_gib"]:
            raise MemoryError("G0 allocated peak exceeded 44 GiB")
        return output

    def compare(a, b):
        if not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise FloatingPointError("nonfinite G0 velocity")
        diff = (a.float() - b.float()).abs()
        rel = float(diff.square().mean().sqrt() /
                    a.float().square().mean().sqrt().clamp_min(1e-12))
        return {"max_abs": float(diff.max()), "relative_RMS": rel,
                "allclose_1e-5": bool(torch.allclose(a,b,rtol=1e-5,atol=1e-5))}

    states = [("initial_noise", current_noise, float(sigmas[0])),
              ("interpolated_half", torch.lerp(endpoint_ref,current_noise,0.5), 0.5)]
    for label, state, sigma in states:
        old = forward(interval_cached, state, sigma, "diagnostic")
        new = forward(interval_sw, state, sigma, "diagnostic")
        check = compare(old, new)
        result["diagnostics"].append({"state": label, "sigma": sigma, **check})
        atomic_json(out / "result.json", result)
        if not check["allclose_1e-5"]:
            raise AssertionError(f"G0 old/SW-G mismatch at {label}: {check}")
        if cache_identity(cache) != signature:
            raise RuntimeError("G0 diagnostic mutated history cache")
        del old,new

    current = current_noise.clone()
    tick = time.perf_counter()
    for j,t in enumerate(pipe.scheduler.timesteps):
        velocity = forward(interval_sw, current, float(t)/1000, "sampling")
        current = pipe.scheduler.step(velocity,t,current)
        if not torch.isfinite(current).all():
            raise FloatingPointError("G0 nonfinite sampled latent")
        peak = torch.cuda.max_memory_allocated()/2**30
        if peak > cfg["max_allocated_gib"]:
            raise MemoryError(f"G0 peak allocation {peak:.3f} GiB > cap")
        result["sampling_steps"] = j+1
        if (j+1)%5 == 0:
            result["peak_allocated_gib"] = peak
            atomic_json(out / "result.json", result)
        del velocity
    torch.cuda.synchronize()
    result["sampling_seconds"] = time.perf_counter()-tick
    result["cache_read_unchanged"] = cache_identity(cache)==signature
    if not result["cache_read_unchanged"]:
        raise RuntimeError("G0 sampling mutated history cache")
    endpoint_check = compare(endpoint_ref, current)
    result["endpoint_vs_baseline"] = endpoint_check
    torch.save(current.detach().cpu(), out / "C6_replay_endpoint.pt")
    result["peak_allocated_gib"] = torch.cuda.max_memory_allocated()/2**30
    atomic_json(out / "result.json", result)

    history_parts = [torch.load(SOURCES[f"C{i}_endpoint"],map_location="cpu",weights_only=True)
                     for i in range(1,6)]
    history = torch.cat(history_parts,dim=2).to(device,torch.float32)
    assert history.shape==(1,24,32,30,52)
    prior_rgb = np.load(SOURCES["prior_RGB_107"])
    baseline_rgb = np.load(SOURCES["baseline_RGB_124"])
    assert len(prior_rgb)==107 and len(baseline_rgb)==124
    assert np.array_equal(prior_rgb,baseline_rgb[:107])
    budget.reserve_vae()
    pipe.load_models_to_device(["video_vae"])
    torch.cuda.synchronize()
    tick = time.perf_counter()
    with torch.no_grad():
        decoded = pipe.video_vae.decode_video(torch.cat((history,current),dim=2),
            dtype=pipe.torch_dtype, process_image=False, tiled=True,
            tile_size=256, tile_overlap=64)
        frames = pipe.vae_output_to_video(decoded,min_value=0,max_value=1)
    rgb = np.stack([np.asarray(frame.convert("RGB"),dtype=np.uint8) for frame in frames])
    result["vae_seconds"] = time.perf_counter()-tick
    assert len(rgb)==124
    stitched = stitch_immutable(prior_rgb,rgb,32,37)
    assert len(stitched)==124 and np.array_equal(stitched[:107],prior_rgb)
    np.save(out / "C6_replay_124.npy",stitched)
    write_video([Image.fromarray(frame) for frame in stitched],out / "C6_replay_124.mp4")
    result["RGB_sha256"] = rgb_sha(stitched)
    result["old107_unchanged"] = True
    result["new17_MAD_vs_baseline"] = float(np.abs(
        stitched[107:].astype(np.float32)-baseline_rgb[107:].astype(np.float32)).mean())
    result["peak_allocated_gib"] = torch.cuda.max_memory_allocated()/2**30
    if result["peak_allocated_gib"] > cfg["max_allocated_gib"]:
        raise MemoryError("G0 peak allocation exceeded 44 GiB")
    result["frozen_parameters_unchanged"] = all(p._version==version for p,version in versions)
    if not result["frozen_parameters_unchanged"]:
        raise RuntimeError("frozen model parameters changed")
    result["status"] = "complete_pending_judge_review"
    atomic_json(out / "result.json",result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.preflight == args.run:
        parser.error("choose exactly one of --preflight or --run")
    if args.preflight:
        manifest = preflight()
        print(f"G0 CPU preflight: {len(manifest['sources'])} sources verified", flush=True)
        return
    cfg = json.loads(CONFIG.read_text())
    if cfg["stage"] != "G0" or cfg["gpu"] != 0:
        raise RuntimeError("G0 frozen config changed")
    out = Path(cfg["output_root"])
    out.mkdir(parents=True,exist_ok=True)
    with (out / "worker.lock").open("a+") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (out / "budget.json").exists():
            raise RuntimeError("G0 already attempted; no automatic retry")
        free = int(subprocess.check_output(["nvidia-smi",f"--id={cfg['gpu']}",
            "--query-gpu=memory.free","--format=csv,noheader,nounits"],text=True).strip())
        if free < 44000:
            raise RuntimeError(f"GPU0 not idle: {free} MiB free")
        manifest = verify_manifest()
        verify_authorization()
        os.environ.update(CUDA_VISIBLE_DEVICES="0",ABOT_VRAM_RESERVE_GIB="18",
            HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",
            DIFFSYNTH_SKIP_DOWNLOAD="True",TOKENIZERS_PARALLELISM="false",
            PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
        sys.path[:0] = [str(FROZEN/"runtime/code"),str(FROZEN/"runtime/code/abot"),
            str(FROZEN/"runtime/code/causal"),str(FROZEN/"runtime/DiffSynth-Studio-h3-v2"),
            str(EXP003),str(EXP001),str(ROUTER),str(EXP)]
        budget = Budget(cfg,out)
        def timeout_handler(_signum,_frame):
            raise TimeoutError("G0 absolute 0.30 GPU-hour deadline")
        signal.signal(signal.SIGALRM,timeout_handler)
        signal.setitimer(signal.ITIMER_REAL,cfg["max_gpu_seconds"])
        try:
            run(cfg,manifest,budget)
        except BaseException as exc:
            budget.ledger["status"]="failed"
            budget.ledger["error"]=repr(exc)
            budget.write()
            result=out/"result.json"
            if result.exists():
                data=json.loads(result.read_text());data.update(status="failed",error=repr(exc))
                atomic_json(result,data)
            raise
        else:
            budget.ledger["status"]="complete_pending_judge_review"
            budget.write()
        finally:
            signal.setitimer(signal.ITIMER_REAL,0)


if __name__ == "__main__":
    main()
