"""EXP-029/v2: recover only s3 OFF_D from the frozen generated C1."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = HERE.parents[3]
sys.path[:0] = [str(EXP), str(ROOT / "submission/experiments/EXP-014_v3_multiscene_teacher")]

import numpy as np
import torch

import run_exp029 as frozen
from common import setup_paths, sha, atomic_json

OUT = ROOT / "H3-World/outputs/EXP-029_current_prefix_behavior/recovery01"
ORIGINAL = ROOT / "H3-World/outputs/EXP-029_current_prefix_behavior"
SOURCE = ROOT / "H3-World/outputs/EXP-014_v3_multiscene_teacher"
SCENE = "s3_b784d995"
HARD_STOP = datetime.fromisoformat("2026-10-17T05:00:00+00:00")
EXPECTED_KV = "2cd1f8e6d4659e74fa70618c0b0317272c18c575244bd55f1c2df14d2cfd8ca9"


def rss_gib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20


class Ledger:
    def __init__(self):
        self.start = time.monotonic()
        self.counts = dict(dit=0, commit=0, sampling=0, decode=0,
                           backward=0, update=0, text_encode=0, vae_encode=0)
        self.events = []
        self.status = "RUNNING"
        self.save()

    def save(self, **extra):
        disk = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
        atomic_json(OUT / "GPU_LEDGER.json", dict(task="EXP-029/recovery01",
            status=self.status, elapsed_in_process_seconds=time.monotonic()-self.start,
            counts=self.counts, caps=dict(dit=31, commit=1, sampling=30, decode=1),
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
            max_rss_gib=rss_gib(), disk_bytes=disk, events=self.events, **extra))

    def reserve(self, kind, label):
        if kind not in ("commit", "sampling", "decode"):
            raise ValueError("unauthorized call kind")
        self.gate()
        cap = dict(commit=1, sampling=30, decode=1)[kind]
        if self.counts[kind] >= cap or (kind != "decode" and self.counts["dit"] >= 31):
            raise RuntimeError("recovery call cap")
        self.counts[kind] += 1
        if kind != "decode": self.counts["dit"] += 1
        self.events.append(dict(kind=kind, label=label,
                                elapsed=time.monotonic()-self.start))
        self.save()

    def gate(self):
        torch.cuda.synchronize()
        if datetime.now(timezone.utc) >= HARD_STOP or time.monotonic()-self.start >= 1180:
            raise TimeoutError("recovery deadline/wall cap")
        if torch.cuda.max_memory_allocated()/2**30 >= 40 or rss_gib() >= 96:
            raise MemoryError("allocated GPU/RSS cap")
        if sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file()) >= 2**30:
            raise OSError("recovery 1GiB disk cap")


def verify_freeze():
    manifest = json.loads((HERE / "ORIGINAL_FREEZE.json").read_text())
    assert manifest["status"] == "PASS_SEVEN_FROZEN_CPU" and len(manifest["frozen_seven"]) == 7
    assert sha(EXP / "SCHEDULER_STATUS.json") == manifest["original_scheduler_sha256"]
    assert sha(ORIGINAL / "GPU_LEDGER.json") == manifest["original_ledger_sha256"]
    for entry in manifest["frozen_seven"].values():
        for relative, expected in entry["files"].items():
            if sha(ROOT / relative) != expected:
                raise RuntimeError("original completed output changed: " + relative)


def main():
    gate = HERE / "START_GATE.json"
    for _ in range(100):
        if gate.is_file(): break
        time.sleep(.1)
    if not gate.is_file(): raise TimeoutError("registration gate missing")
    registration = json.loads(gate.read_text())
    auth = json.loads((HERE / "AUTHORIZATION.json").read_text())
    assert registration["python_pid"] == os.getpid()
    assert int(os.environ["CUDA_VISIBLE_DEVICES"]) == registration["physical_gpu"]
    assert registration["physical_gpu"] in auth["allowed_gpu"] == [1, 6]
    assert os.environ.get("ABOT_VRAM_RESERVE_GIB") == "18"
    assert auth["max_dit_attempts"] == 31 and auth["max_vae_decodes"] == 1
    cpu = json.loads((HERE / "CPU_PREFLIGHT.json").read_text())
    assert cpu["status"] == "PASS_CPU"
    for relative, expected in cpu["sha256"].items():
        if sha(ROOT / relative) != expected:
            raise RuntimeError("frozen source/input changed: " + relative)
    verify_freeze()
    setup_paths()
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats()
    OUT.mkdir(parents=True, exist_ok=True)
    branch_dir = OUT / "OFF_D"
    branch_dir.mkdir(exist_ok=False)
    ledger = Ledger()
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("wall alarm")))
    signal.alarm(1185)
    try:
        import run_teacher as old
        from causal.h3_cached import H3ChunkCache
        from causal.anyflow_sampling import configure_video_schedule
        from current_prefix import current_prefix_feedback
        from interval_cached import interval_cached

        pipe, model, precision = old.load_model()
        fixture, entry = old.load_fixture(SCENE)
        first_file = SOURCE / "G1" / SCENE / "FM30/first12.pt"
        published_file = SOURCE / "G1" / SCENE / "FM30/published_39.npy"
        source_row = json.loads((SOURCE / "G1" / SCENE / "FM30/result.json").read_text())
        assert sha(first_file) == source_row["endpoint_sha256"]
        assert sha(published_file) == source_row["published_sha256"]
        first = torch.load(first_file, map_location="cpu", weights_only=True).to("cuda:0", torch.float32)
        published = np.load(published_file)
        assert published.shape == (39, 480, 832, 3)
        audio = fixture["audio_noise"].to("cuda:0")
        anchor = fixture["anchor"].to("cuda:0")
        cache = H3ChunkCache(max_history=5, storage_device="cpu")
        layout, prompt = old.visible(fixture, "A", 12)
        ledger.reserve("commit", "s3/C1_clean_ON")
        with torch.no_grad(), current_prefix_feedback():
            interval_cached(model, first, start=0, index=0, cache=cache,
                full_packed=layout, prompt=prompt, anchor=anchor, audio=audio,
                sigma=0., commit=True)
        digest = frozen.cache_digest(cache)
        ledger.save(cache_sha256=digest, cpu_kv_bytes=cache.nbytes)
        if digest != EXPECTED_KV:
            raise RuntimeError("reconstructed C1 cache differs from original s3")
        ledger.gate()

        layout, prompt = old.visible(fixture, "AD", 17)
        noise = fixture["initial_noise"][:, :, 12:17].to("cuda:0", torch.float32)
        current = noise.clone()
        sigmas = configure_video_schedule(pipe.scheduler, steps=30, grid="native", flow_shift=2.22)
        assert len(sigmas) == 31 and sigmas[0] == 1 and sigmas[-1] == 0
        old_partial = json.loads((ORIGINAL / SCENE / "OFF_D/RESULT.json").read_text())
        assert frozen.tensor_sha(prompt) == old_partial["prompt_sha256"]
        assert frozen.tensor_sha(layout["img_position_ids"]) == old_partial["position_sha256"]
        assert frozen.tensor_sha(noise) == old_partial["initial_noise_sha256"]
        assert sigmas == old_partial["sigmas"]
        row = dict(task="EXP-029/recovery01", scene=SCENE, branch="OFF_D",
            status="SAMPLING", current_prefix_feedback=False, action="D",
            fixture_sha256=entry["sha256"], first12_sha256=sha(first_file),
            published39_sha256=sha(published_file), source_cache_sha256=digest,
            prompt_sha256=frozen.tensor_sha(prompt),
            position_sha256=frozen.tensor_sha(layout["img_position_ids"]),
            initial_noise_sha256=frozen.tensor_sha(noise), sigmas=sigmas,
            sampling_forwards=0, precision=precision)
        atomic_json(branch_dir / "RESULT.json", row)
        torch.cuda.synchronize(); start = time.monotonic()
        with torch.no_grad():
            for step, timestep in enumerate(pipe.scheduler.timesteps, 1):
                ledger.reserve("sampling", f"s3/OFF_D/{step}")
                velocity = interval_cached(model, current, start=12, index=1,
                    cache=cache, full_packed=layout, prompt=prompt, anchor=anchor,
                    audio=audio, sigma=float(timestep)/1000)
                current = pipe.scheduler.step(velocity, timestep, current)
                if not bool(torch.isfinite(current).all()):
                    raise FloatingPointError("nonfinite latent")
                row["sampling_forwards"] = step
                atomic_json(branch_dir / "RESULT.json", row)
                ledger.gate()
        torch.cuda.synchronize()
        row["sampling_seconds"] = time.monotonic() - start
        assert frozen.cache_digest(cache) == digest
        endpoint = branch_dir / "endpoint5.pt"
        torch.save(current.detach().cpu(), endpoint)
        row["endpoint_sha256"] = sha(endpoint)
        decoded, row["decode_seconds"] = frozen.decode_native(
            pipe, first, current, ledger, "s3/OFF_D")
        row["decoded_history_mad"] = float(np.abs(
            decoded[:39].astype(np.int16)-published.astype(np.int16)).mean())
        generated = decoded[39:].copy()
        joined = np.concatenate((published, generated), axis=0)
        assert joined.shape == (56, 480, 832, 3) and np.array_equal(joined[:39], published)
        np.save(branch_dir / "generated17.npy", generated)
        np.save(branch_dir / "published56.npy", joined)
        row["boundary_mad"] = float(np.abs(published[-1].astype(np.int16)-generated[0].astype(np.int16)).mean())
        row["adjacent_new_mad"] = float(np.abs(generated[1:].astype(np.int16)-generated[:-1].astype(np.int16)).mean())
        row["rollout_video"] = frozen.write_mp4(joined, branch_dir / "rollout_56.mp4")
        row["new_video"] = frozen.write_mp4(generated, branch_dir / "new_17.mp4")
        row["contact_sheet"] = frozen.contact_sheet(published[-1], generated,
                                                     branch_dir / "contact_18.jpg")
        row["cache_sha256_after"] = frozen.cache_digest(cache)
        assert row["cache_sha256_after"] == digest
        row["status"] = "COMPLETE_PENDING_REVIEW"
        atomic_json(branch_dir / "RESULT.json", row)
        assert ledger.counts["dit"] == 31 and ledger.counts["decode"] == 1
        ledger.status = "COMPLETE_PENDING_JUDGE"
        ledger.save(cache_sha256=digest, cpu_kv_bytes=cache.nbytes)
    except BaseException as exc:
        ledger.status = "FAILED_PENDING_JUDGE"
        ledger.save(error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        signal.alarm(0)


if __name__ == "__main__": main()
