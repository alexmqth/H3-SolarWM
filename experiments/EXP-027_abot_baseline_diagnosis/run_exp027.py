"""One-shot EXP-027/v2 two-clip C2 learnability gate; reuses frozen EXP-025 hooks."""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
EXP23 = ROOT / "submission/experiments/EXP-023_v3_fm_small_data"
EXP24 = ROOT / "submission/experiments/EXP-024_v3_fm_coverage_preparation"
EXP25 = ROOT / "submission/experiments/EXP-025_v3_fm_full_coverage"
sys.path.insert(0, str(EXP25))
import run_exp025 as base

torch = base.torch
old = base.old
cov = base.cov
profile = base.profile
sha = base.sha
write = base.write
rss_gib = base.rss_gib

FIT = (
    "43866101158a538d22c52ac5b2fa606b_src0621_rgb56",
    "7199292ca430b478bfd0d83cf2e519d8_src1670_rgb56",
)
DEV = (
    "118eb5d8b75e1b8ac23a4e9ae77af9a9_src0000_rgb56",
    "dfec8ed3237860eba14d67c089ecd041_src0483_rgb56",
)
ALL = FIT + DEV
SIGMAS = (0.10, 0.25, 0.50, 0.75, 0.90)
PROBE_SEEDS = (113, 127)
ORDER = FIT * 32
CAPS = {"dit": 336, "backward": 64, "update": 64, "vae_decode": 6,
        "vae_encode": 0, "text": 0}
STAGE_CAPS = {"setup": 400, "probe": 1400, "train": 3200, "video": 1000}
AUTH = HERE / "AUTHORIZATION_v2.json"
SOURCE = EXP24 / "SOURCE_MANIFEST.json"
INITIAL = EXP25 / "checkpoint_000.pt"
INITIAL_SHA = "65fc21a9afa45db9efa7de4b89c6323df694cbd601142c9a86669afe9276c659"


def setup_media():
    import imageio_ffmpeg
    executable = Path(imageio_ffmpeg.get_ffmpeg_exe()).resolve()
    if not executable.is_file():
        raise RuntimeError("imageio-ffmpeg binary missing")
    bindir = HERE / "bin"
    bindir.mkdir(exist_ok=True)
    link = bindir / "ffmpeg"
    if link.is_symlink():
        if link.resolve() != executable:
            raise RuntimeError("task-local ffmpeg symlink changed")
    elif link.exists():
        raise RuntimeError("task-local ffmpeg path occupied")
    else:
        link.symlink_to(executable)
    os.environ["PATH"] = str(bindir) + os.pathsep + os.environ.get("PATH", "")
    return str(executable)


def preflight(gpu: bool):
    a = json.loads(AUTH.read_text())
    assert a["status"] == "AUTHORIZED_CPU_GATES_THEN_ONE_GPU_TASK"
    assert (a["gpu_seconds"], a["max_top_level_dit_attempts"], a["max_updates"]) == (6000, 336, 64)
    assert sha(INITIAL) == INITIAL_SHA
    source = json.loads(SOURCE.read_text())
    for relative, digest in source["files"].items():
        if sha(ROOT / relative) != digest:
            raise RuntimeError(f"frozen source changed: {relative}")
    media_binary = setup_media()
    selected = json.loads((EXP23 / "SELECTION.json").read_text())
    assert set(ALL) <= (set(selected["train_clip_ids"]) |
                        {r["clip_id"] for r in selected["evaluation"]})
    inputs = {}
    for clip in ALL:
        folder = EXP23 / "encoded" / clip
        record = json.loads((folder / "record.json").read_text())
        files = {}
        for filename, key in (("I0.pt", "I0"), ("condition.pt", "condition"),
                              ("full17.pt", "full17")):
            digest = sha(folder / filename)
            expected = (record["condition"]["sha256"] if key == "condition"
                        else record["latents"][key]["sha256"])
            if digest != expected:
                raise RuntimeError(f"encoded source changed: {clip}/{filename}")
            files[filename] = digest
        inputs[clip] = files
    if not gpu:
        state = torch.load(INITIAL, map_location="cpu", weights_only=True)
        assert state["step"] == 0 and len(state["adapter"]) == 50
        assert all(a["A"].shape == (8, 5376) and a["B"].shape == (21504, 8)
                   and torch.count_nonzero(a["B"]) == 0 for a in state["adapter"])
        assert sum(v.numel() for a in state["adapter"] for v in a.values()) == 10752000
        del state
        with subprocess.Popen(
            ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=c=black:s=16x16:r=24",
             "-frames:v", "1", "-f", "mp4", "-y", str(HERE / "CPU_MEDIA_TEST.mp4")],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        ) as process:
            _, errors = process.communicate(timeout=10)
            if process.returncode:
                raise RuntimeError("CPU ffmpeg smoke failed: " + errors.decode()[:200])
        (HERE / "CPU_MEDIA_TEST.mp4").unlink()
        result = {"status": "PASS_CPU_NO_CUDA", "cuda_initialized": torch.cuda.is_initialized(),
                  "fit": FIT, "dev_observed": DEV, "train_order": ORDER,
                  "encoded_sha256": inputs, "initial_checkpoint_sha256": INITIAL_SHA,
                  "source_manifest_sha256": sha(SOURCE),
                  "source_code_sha256": {str(p.relative_to(ROOT)): sha(p) for p in
                      (EXP25 / "run_exp025.py", EXP23 / "attempt_00/fm_run.py",
                       EXP24 / "coverage.py", EXP24 / "profile_gpu.py")},
                  "implementation_sha256": sha(__file__), "ffmpeg_binary": media_binary,
                  "taskbook_sha256": sha(HERE / "taskbook_v2.md"),
                  "authorization_sha256": sha(AUTH),
                  "expected_max_counts": {"regression_dit": 6, "train_dit": 128,
                                          "probe_dit": 132, "video_dit": 70, "total_dit": 336}}
        assert not torch.cuda.is_initialized()
        write(HERE / "CPU_PREFLIGHT.json", result)
        return
    cpu = json.loads((HERE / "CPU_PREFLIGHT.json").read_text())
    assert cpu["status"] == "PASS_CPU_NO_CUDA" and cpu["implementation_sha256"] == sha(__file__)
    physical = os.environ.get("CUDA_VISIBLE_DEVICES")
    if physical is None or not physical.isdigit():
        raise RuntimeError("exactly one physical GPU must be exposed")
    return int(physical)


class Ledger:
    def __init__(self, physical_gpu: int):
        self.path = HERE / "GPU_LEDGER.json"
        if self.path.exists():
            raise FileExistsError("EXP-027 already started; no automatic retry")
        self.start = time.monotonic()
        self.phase = "setup"
        self.phase_start = self.start
        self.counts = {name: 0 for name in CAPS}
        self.events = []
        self.status = "RUNNING"
        self.error = None
        self.physical_gpu = physical_gpu
        self.phase_seconds = {}
        signal.signal(signal.SIGALRM, self._alarm)
        signal.setitimer(signal.ITIMER_REAL, 5980)
        self.save()

    @staticmethod
    def _alarm(_signum, _frame):
        raise TimeoutError("EXP-027 total watchdog")

    def save(self):
        write(self.path, {"task": "EXP-027/v2", "status": self.status,
              "physical_gpu": self.physical_gpu, "pid": os.getpid(), "phase": self.phase,
              "elapsed_seconds": time.monotonic()-self.start,
              "phase_elapsed_seconds": time.monotonic()-self.phase_start,
              "phase_seconds": self.phase_seconds, "counts": self.counts.copy(),
              "caps": CAPS, "gpu_seconds_cap": 6000,
              "peak_allocated_gib": torch.cuda.max_memory_allocated()/2**30,
              "peak_reserved_gib": torch.cuda.max_memory_reserved()/2**30,
              "rss_gib": rss_gib(), "events": self.events[-500:], "error": self.error})

    def gate(self):
        elapsed = time.monotonic()-self.start
        if elapsed >= 5980: raise TimeoutError("total GPU time budget")
        if self.phase in ("probe", "train") and elapsed >= 6000-800:
            raise TimeoutError("mandatory video reserve reached")
        if torch.cuda.max_memory_allocated()/2**30 >= 40:
            raise MemoryError("allocated VRAM >=40GiB")
        if rss_gib() >= 96:
            raise MemoryError("RSS >=96GiB")

    def reserve(self, kind, note):
        self.gate()
        if self.counts[kind] >= CAPS[kind]:
            raise RuntimeError(f"{kind} budget exceeded")
        self.counts[kind] += 1
        self.events.append({"kind": kind, "note": note,
                            "seconds": time.monotonic()-self.start})
        self.save()

    def measure(self, name, fn):
        torch.cuda.synchronize()
        begin = time.monotonic()
        value = fn()
        torch.cuda.synchronize()
        self.gate()
        self.events.append({"kind": "timing", "note": name,
                            "duration_seconds": time.monotonic()-begin,
                            "seconds": time.monotonic()-self.start})
        self.save()
        return value

    def next_phase(self, phase):
        self.phase_seconds[self.phase] = (self.phase_seconds.get(self.phase,0.0)
                                          + time.monotonic()-self.phase_start)
        self.phase = phase
        self.phase_start = time.monotonic()
        self.save()
        print("PHASE", phase, self.counts, flush=True)

    def close(self, status, error=None):
        signal.setitimer(signal.ITIMER_REAL, 0)
        self.phase_seconds[self.phase] = (self.phase_seconds.get(self.phase,0.0)
                                          + time.monotonic()-self.phase_start)
        self.status = status
        self.error = repr(error) if error is not None else None
        self.save()


def save_checkpoint(adapters, optimizer, step, noise_gen):
    path = HERE / f"checkpoint_{step:03d}.pt"
    if path.exists(): raise FileExistsError(path)
    state = {"step": step,
             "adapter": [{"A": a.lora_A.detach().cpu().clone(),
                          "B": a.lora_B.detach().cpu().clone()} for a in adapters],
             "optimizer": old.cpu_tree(optimizer.state_dict()),
             "cpu_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(),
             "noise_rng": noise_gen.get_state(), "order": ORDER, "order_pos": step,
             "metadata": {"task": "EXP-027/v2", "objective": "C2_ordinary_FM",
                          "fit_clips": FIT, "init_sha256": INITIAL_SHA,
                          "history": "GT_C1_clean_commit", "precision": "h3_fp32"}}
    partial = path.with_suffix(".partial")
    torch.save(state, partial)
    partial.replace(path)
    readback = torch.load(path, map_location="cpu", weights_only=True)
    if readback["step"] != step or len(readback["adapter"]) != 50:
        raise RuntimeError("checkpoint metadata readback mismatch")
    if not torch.equal(readback["noise_rng"],state["noise_rng"]) or not torch.equal(readback["cpu_rng"],state["cpu_rng"]):
        raise RuntimeError("checkpoint RNG readback mismatch")
    if len(readback["optimizer"]["state"]) != len(state["optimizer"]["state"]):
        raise RuntimeError("checkpoint optimizer readback mismatch")
    if not all(torch.equal(v[key],w[key]) for v,w in zip(state["adapter"],readback["adapter"])
               for key in ("A","B")):
        raise RuntimeError("checkpoint adapter readback mismatch")
    write(path.with_suffix(".json"), {"step": step, "sha256": sha(path),
                                     "bytes": path.stat().st_size,
                                     "metadata": state["metadata"]})
    del state, readback
    gc.collect()


def load_adapter(path, adapters):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if len(state["adapter"]) != 50:
        raise RuntimeError("wrong checkpoint adapter count")
    with torch.no_grad():
        for adapter, saved in zip(adapters, state["adapter"]):
            adapter.lora_A.copy_(saved["A"].to(adapter.lora_A.device))
            adapter.lora_B.copy_(saved["B"].to(adapter.lora_B.device))
    return state["step"]


def run_probe(model, ledger, step):
    rows = []
    for clip in ALL:
        data = old.load_clip(clip)
        clean = data["z"][:, :, 12:17]
        cache = old.make_cache(model, data, data["z"][:, :, :12], ledger,
                               f"probe{step}_{clip}_GT_C1_commit")
        assert cache.nbytes == 6799104000
        for seed in PROBE_SEEDS:
            generator = torch.Generator(device="cpu").manual_seed(seed)
            noise = torch.randn(clean.shape, generator=generator).to("cuda:0")
            for sigma in SIGMAS:
                noisy = (1-sigma)*clean + sigma*noise
                predicted = old.velocity(model, data, cache, 1, noisy, sigma, ledger,
                                         f"probe{step}_{clip}_{seed}_{sigma}")
                target = noise-clean
                mse = float(torch.nn.functional.mse_loss(predicted.float(), target.float()))
                rms = float(predicted.float().square().mean().sqrt())
                if not (torch.isfinite(predicted).all() and 0 <= mse < 100 and rms < 100):
                    raise FloatingPointError("nonfinite or unreasonable fixed probe")
                rows.append({"step": step, "clip_id": clip, "seed": seed,
                             "sigma": sigma, "mse": mse, "velocity_rms": rms})
                del predicted
        del data, clean, cache
        gc.collect()
    write(HERE / f"PROBE_{step:03d}.json", rows)
    print("PROBE", step, {clip: sum(r["mse"] for r in rows if r["clip_id"]==clip)/10
                          for clip in FIT}, flush=True)
    return rows


def train_step(model, adapters, trainable, optimizer, noise_gen, ledger, step, clip):
    import current_prefix as route
    data0 = old.load_clip(clip)
    f = data0["f"]
    data = {"z": data0["z"],
            "full_packed": {k: v.to("cuda:0") if torch.is_tensor(v) else v
                            for k,v in f["packed"].items()},
            "prompt": f["prompt"].to("cuda:0"),
            "anchor": data0["anchor"], "audio": data0["audio"]}
    clean = data["z"][:, :, 12:17]
    noise = torch.randn(clean.shape, generator=noise_gen, dtype=torch.float32).to("cuda:0")
    sigma = float(torch.rand((), generator=noise_gen))
    sigma = max(1e-6, min(1-1e-6, sigma))
    phase = ["idle"]
    trace = []
    source = route.current_prefix_attend
    def audit(control, *args, **kwargs):
        if control.action_prefix_mode != "own" or not control.action_feedback or control.index != 1:
            raise RuntimeError("C2 action route changed")
        trace.append((phase[0], control.index, int(kwargs["layer"])))
        return source(control, *args, **kwargs)
    def stage(name, when):
        kind = {"commit": "dit", "forward": "dit", "backward": "backward",
                "optimizer": "update"}[name]
        if when == "before": ledger.reserve(kind, f"step{step}_{name}")
        else: ledger.gate()
    def gate(_, cache):
        proof = profile.validate_block_gate("C2", trace, model, trainable,
                                            adapters if step == 1 else None)
        if step == 1:
            proof.update({"step": 1, "clip_id": clip, "cpu_kv_bytes": cache.nbytes})
            write(HERE / "FIRST_C2_GRADIENT_GATE.json", proof)
    row = cov.fm_step(model, data, optimizer, block="C2", noise=noise,
                      sigma=sigma, checkpoint=True, offload=True,
                      audit=audit, phase_hook=lambda name: phase.__setitem__(0,name),
                      stage_callback=stage, pre_optimizer_gate=gate)
    row.update(step=step, clip_id=clip, peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
               rss_gib=rss_gib())
    if not (0 <= row["loss"] < 100 and 0 < row["grad_l2"] < 100):
        raise FloatingPointError("nonfinite loss/gradient")
    del data0, data, clean, noise
    gc.collect()
    return row


def probe_gate():
    before = json.loads((HERE / "PROBE_000.json").read_text())
    after = json.loads((HERE / "PROBE_064.json").read_text())
    report = {"fit": {}, "optional_dev_videos": False}
    for clip in FIT:
        a = [r for r in before if r["clip_id"] == clip]
        b = [r for r in after if r["clip_id"] == clip]
        by = {(r["seed"],r["sigma"]):r for r in a}
        assert len(a) == len(b) == 10
        def ratio(predicate):
            bb = [r["mse"] for r in b if predicate(r)]
            aa = [by[(r["seed"],r["sigma"])]["mse"] for r in b if predicate(r)]
            return sum(bb)/sum(aa)-1
        report["fit"][clip] = {"mean_change": ratio(lambda r: True),
                               "low_change": ratio(lambda r: r["sigma"] <= .25),
                               "high_change": ratio(lambda r: r["sigma"] >= .75)}
    report["optional_dev_videos"] = all(
        x["mean_change"] <= -.10 and x["low_change"] < 0 and x["high_change"] < 0
        for x in report["fit"].values())
    write(HERE / "PROBE_GATE.json", report)
    return report["optional_dev_videos"]


def video_one(pipe, model, ledger, clip, step, history):
    from common import raw_source_rgb, inputs
    from causal.h3_cached import H3ChunkCache
    import numpy as np
    _, _, _, train, dev, _ = inputs()
    data = old.load_clip(clip)
    if history == "GT39":
        clean1 = data["z"][:, :, :12]
        cache = old.make_cache(model, data, clean1, ledger,
                               f"video{step}_{clip}_GT_C1_commit")
        c2, sampling_c2 = old.sample_chunk(pipe, model, data, cache, 1, ledger,
                                           f"video{step}_{clip}_GT")
        latents = torch.cat((clean1,c2), dim=2)
        sampling_c1 = None
    else:
        empty = H3ChunkCache(5, "cpu")
        c1, sampling_c1 = old.sample_chunk(pipe, model, data, empty, 0, ledger,
                                           f"video{step}_{clip}_SELF")
        cache = old.make_cache(model, data, c1, ledger,
                               f"video{step}_{clip}_SELF_C1_commit")
        c2, sampling_c2 = old.sample_chunk(pipe, model, data, cache, 1, ledger,
                                           f"video{step}_{clip}_SELF")
        latents = torch.cat((c1,c2), dim=2)
    frames = old.decode(pipe, latents, ledger, f"video{step}_{clip}_{history}")
    if history == "GT39":
        source = raw_source_rgb((train if clip in train else dev)[clip])
        frames[:39] = source[:39]
    folder = HERE / "videos"
    folder.mkdir(exist_ok=True)
    path = folder / f"{clip}_{history}_checkpoint_{step:03d}.mp4"
    if path.exists(): raise FileExistsError(path)
    old.save_video(frames, path)
    import av
    with av.open(str(path)) as media:
        fps = float(media.streams.video[0].average_rate)
        frame_count = sum(1 for _ in media.decode(video=0))
    if frame_count != 56 or fps != 24.0:
        raise RuntimeError(f"video media check failed: {frame_count}f {fps}fps")
    row = {"clip_id": clip, "history": history, "checkpoint_updates": step,
           "nfe_per_chunk": 8, "generated_rgb_range": [39,55] if history=="GT39" else [0,55],
           "video": str(path.relative_to(ROOT)), "sha256": sha(path),
           "bytes": path.stat().st_size, "fps": 24, "frames": 56,
           "boundary_mad": float(np.abs(frames[39].astype(float)-frames[38].astype(float)).mean()),
           "frame_mad": float(np.abs(np.diff(frames.astype(float),axis=0)).mean()),
           "sampling_c1_seconds": sampling_c1, "sampling_c2_seconds": sampling_c2,
           "cpu_kv_bytes": cache.nbytes}
    del data, cache, latents, frames, c2
    gc.collect()
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--gpu", type=int)
    args = parser.parse_args()
    if args.cpu_only:
        preflight(False)
        print("EXP027_CPU_PREFLIGHT_PASS", flush=True)
        return
    physical = preflight(True)
    if args.gpu != physical:
        raise RuntimeError("physical GPU argument differs")
    import run as prior
    prior.setup()
    torch.set_num_threads(4)
    torch.manual_seed(13)
    torch.cuda.manual_seed_all(13)
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats()
    ledger = Ledger(physical)
    try:
        pipe, model, precision = ledger.measure("model_load", prior.model)
        adapters, params, trainable = base.install(model)
        if load_adapter(INITIAL, adapters) != 0:
            raise RuntimeError("initial adapter checkpoint not step0")
        optimizer = torch.optim.AdamW(params, lr=1e-4, betas=(.9,.95), weight_decay=.01)
        noise_gen = torch.Generator(device="cpu").manual_seed(13)
        save_checkpoint(adapters, optimizer, 0, noise_gen)
        base.HERE = HERE
        base.regression(model, adapters, ledger)
        ledger.next_phase("probe")
        run_probe(model, ledger, 0)
        updates = []
        ledger.next_phase("train")
        for step, clip in enumerate(ORDER, 1):
            row = train_step(model, adapters, trainable, optimizer, noise_gen, ledger, step, clip)
            updates.append(row)
            write(HERE / "TRAIN_UPDATES.json", updates)
            print("UPDATE",step,clip,row["sigma"],row["loss"],row["grad_l2"],flush=True)
            if step in (32,64):
                save_checkpoint(adapters, optimizer, step, noise_gen)
                ledger.next_phase("probe")
                run_probe(model, ledger, step)
                if step < 64: ledger.next_phase("train")
        optional = probe_gate()
        ledger.next_phase("video")
        videos = []
        for step in (0,64):
            ckpt = HERE / f"checkpoint_{step:03d}.pt"
            if load_adapter(ckpt, adapters) != step:
                raise RuntimeError("video checkpoint identity mismatch")
            for clip in FIT:
                videos.append(video_one(pipe,model,ledger,clip,step,"GT39"))
                write(HERE / "VIDEO_RESULTS.json", videos)
        if optional:
            for step in (0,64):
                if load_adapter(HERE / f"checkpoint_{step:03d}.pt", adapters) != step:
                    raise RuntimeError("optional video checkpoint identity mismatch")
                videos.append(video_one(pipe,model,ledger,DEV[1],step,"SELF56"))
                write(HERE / "VIDEO_RESULTS.json", videos)
        ledger.close("COMPLETE_PENDING_JUDGE")
        print("EXP027_COMPLETE",ledger.counts,"optional_dev",optional,repr(precision),flush=True)
    except BaseException as error:
        ledger.close("FAILED_STOPPED",error)
        print("EXP027_STOPPED",repr(error),flush=True)
        raise


if __name__ == "__main__": main()
