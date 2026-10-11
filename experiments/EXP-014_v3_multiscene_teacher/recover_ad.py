"""EXP-014/v2 CPU preflight and gated recovery of only the missing AD branch."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import gc
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import time

from common import (HERE, OUT, ROOT, atomic_json, check_machine, setup_paths,
                    sha, verify_code_manifest, verify_sources)
from run_teacher import (cache_signature, check_peak, decode, load_fixture,
                         load_model, sample, tensor_sha, write_video_and_sheet)

CONFIG = json.loads((HERE / "recovery_config.json").read_text())
RECOVERY = OUT / CONFIG["output_subdir"]


def snapshot_files() -> dict:
    files = [
        HERE / "config.json", HERE / "source_manifest.json", HERE / "code_manifest.json",
        HERE / "run_teacher.py", HERE / "common.py",
        OUT / "P1_result.json",
    ]
    for scene in CONFIG["scenes"]:
        files.extend([
            OUT / "G1" / scene / "FM30/result.json",
            OUT / "G1" / scene / "FM30/first12.pt",
            OUT / "G1" / scene / "FM30/published_39.npy",
            OUT / "G2" / scene / "FM30/result.json",
            OUT / "G2" / scene / "FM30/cache_through12.pt",
            OUT / "G2" / scene / "FM30/AA/result.json",
            OUT / "G2" / scene / "FM30/AA/chunk_12_17.pt",
            OUT / "G2" / scene / "FM30/AD/result.json",
            OUT / "teacher" / scene / "budget.json",
        ])
    return {str(path.relative_to(ROOT)): sha(path) for path in files}


def verify_snapshot() -> None:
    expected = json.loads((HERE / "recovery_source_manifest.json").read_text())
    if expected["files"] != snapshot_files():
        raise RuntimeError("recovery source snapshot changed")
    if expected["parent_task"] != "EXP-014/v1":
        raise RuntimeError("wrong parent task")


def verify_recovery_code() -> None:
    expected = json.loads((HERE / "recovery_code_manifest.json").read_text())
    for path, digest in expected["files"].items():
        if sha(ROOT / path) != digest:
            raise RuntimeError(f"recovery code changed after freeze: {path}")


def require_recovery_marker(scene: str) -> dict:
    path = HERE / "judge/RECOVERY_APPROVED.json"
    if not path.is_file():
        raise PermissionError(f"Judge recovery marker missing: {path}")
    marker = json.loads(path.read_text())
    if marker.get("task") != CONFIG["task"] or marker.get("approved") is not True:
        raise PermissionError("invalid recovery marker")
    if scene not in marker.get("scenes", []):
        raise PermissionError(f"scene not approved for recovery: {scene}")
    for field, filename in (
        ("config_sha256", "recovery_config.json"),
        ("source_manifest_sha256", "recovery_source_manifest.json"),
        ("code_manifest_sha256", "recovery_code_manifest.json"),
    ):
        if marker.get(field) != sha(HERE / filename):
            raise PermissionError(f"marker {field} does not match frozen recovery")
    verify_recovery_code()
    return marker


def preflight_scene(scene: str) -> dict:
    import numpy as np
    import torch
    from causal.h3_cached import H3ChunkCache
    from causal.local_topology import visible_inputs

    fixture, entry = load_fixture(scene)
    c1 = OUT / "G1" / scene / "FM30"
    g2 = OUT / "G2" / scene / "FM30"
    first = json.loads((c1 / "result.json").read_text())
    parent = json.loads((g2 / "result.json").read_text())
    aa = json.loads((g2 / "AA/result.json").read_text())
    old_ad = json.loads((g2 / "AD/result.json").read_text())
    old_budget = json.loads((OUT / "teacher" / scene / "budget.json").read_text())
    assert first["status"] == aa["status"] == "complete_pending_visual_review"
    assert parent["status"] == "loading" and old_ad["status"] == "sampling"
    assert old_budget["events"][-1]["event"] == "reserve"
    assert old_budget["sampling_forwards"] in (78, 80)
    assert not (g2 / "AD/chunk_12_17.pt").exists()
    assert not (g2 / "AD/rollout_56.mp4").exists()
    assert sha(c1 / "first12.pt") == first["endpoint_sha256"] == parent["first_endpoint_sha256"]
    assert sha(c1 / "published_39.npy") == first["published_sha256"] == parent["first_published_sha256"]
    assert sha(g2 / "AA/chunk_12_17.pt") == aa["endpoint_sha256"]
    assert sha(g2 / "cache_through12.pt") == parent["cache_sha256"] == aa["source_cache_sha256"]
    rgb = np.load(c1 / "published_39.npy", mmap_mode="r")
    assert rgb.shape == (39, 480, 832, 3) and rgb.dtype == np.uint8
    cache = torch.load(g2 / "cache_through12.pt", map_location="cpu", weights_only=False)
    assert isinstance(cache, H3ChunkCache)
    assert len(cache.layers) == 50 and cache.commits == 50
    assert all([item.index for item in cache.history(layer, 1)] == [0]
               for layer in range(50))
    assert cache.nbytes == aa["cpu_raw_kv_bytes"] == 6_799_104_000
    del cache

    prompt = fixture["prompts"]["A"].clone()
    donor = fixture["prompts"]["D"]
    allowed = torch.zeros(prompt.shape[0], dtype=torch.bool)
    for lo, hi in fixture["packed"]["action_text_spans_local"][12:17]:
        prompt[lo:hi] = donor[lo:hi]
        allowed[lo:hi] = True
    assert torch.equal(prompt[~allowed], fixture["prompts"]["A"][~allowed])
    assert not torch.equal(prompt[allowed], fixture["prompts"]["A"][allowed])
    layout, cropped = visible_inputs(fixture["packed"], prompt, 17, 390)
    assert tensor_sha(cropped) == old_ad["prompt_sha256"]
    assert tensor_sha(layout["img_position_ids"]) == old_ad["position_sha256"]
    assert tensor_sha(fixture["initial_noise"][:, :, 12:17].float()) == old_ad["initial_noise_sha256"]
    assert old_ad["sigmas"] == aa["sigmas"] and len(old_ad["sigmas"]) == 31
    assert old_ad["first_endpoint_sha256"] == first["endpoint_sha256"]
    assert old_ad["source_cache_sha256"] == parent["cache_sha256"]
    return {
        "scene": scene, "fixture_sha256": entry["sha256"],
        "c1_endpoint_sha256": first["endpoint_sha256"],
        "c1_rgb_sha256": first["published_sha256"],
        "aa_endpoint_sha256": aa["endpoint_sha256"],
        "cache_sha256": parent["cache_sha256"], "cache_bytes": aa["cpu_raw_kv_bytes"],
        "old_ad_result_sha256": sha(g2 / "AD/result.json"),
        "ad_prompt_sha256": old_ad["prompt_sha256"],
        "ad_position_sha256": old_ad["position_sha256"],
        "ad_noise_sha256": old_ad["initial_noise_sha256"],
        "ad_sigmas": old_ad["sigmas"],
        "old_reserved_sampling": old_budget["sampling_forwards"],
        "old_completed_ad_steps": old_ad["step_completed"],
        "preflight": "PASS_CPU_ONLY",
    }


class RecoveryLedger:
    def __init__(self, scene: str, gpu: int):
        self.scene, self.gpu = scene, gpu
        self.directory = RECOVERY / scene
        if self.directory.exists():
            raise FileExistsError(f"recovery attempt already exists: {self.directory}")
        self.directory.mkdir(parents=True)
        self.lock = (self.directory / "worker.lock").open("a+")
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.path = self.directory / "budget.json"
        self.began = time.monotonic()
        self.data = {"task": CONFIG["task"], "scene": scene, "gpu": gpu,
                     "gpu_seconds": 0.0, "sampling_forwards": 0, "vae_decodes": 0,
                     "commit_forwards": 0, "image_vae_encodes": 0,
                     "training_updates": 0,
                     "events": [{"event": "stage_start", "at": self.now()}]}
        self.save()

    @staticmethod
    def now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def save(self) -> None:
        atomic_json(self.path, self.data)

    def reserve(self, field: str) -> None:
        if datetime.now(timezone.utc) >= datetime.fromisoformat(CONFIG["deadline_hkt"]):
            raise TimeoutError("09:00 HKT cutoff")
        if shutil.disk_usage(OUT.parent).free / 2**30 < CONFIG["minimum_free_disk_gib"]:
            raise OSError("disk free floor")
        if time.monotonic() - self.began >= CONFIG["max_gpu_seconds_per_scene"]:
            raise TimeoutError("recovery per-scene GPU seconds cap")
        cap = CONFIG[f"new_{field}_per_scene"]
        if self.data[field] >= cap:
            raise RuntimeError(f"recovery {field} budget exhausted")
        self.data[field] += 1
        self.data["events"].append({"event": "reserve", "field": field,
                                    "count": self.data[field], "at": self.now()})
        self.save()

    def close(self, status: str, error: str | None = None) -> None:
        self.data["gpu_seconds"] = time.monotonic() - self.began
        self.data["events"].append({"event": "stage_stop", "status": status,
                                    "error": error, "at": self.now()})
        self.save()
        self.lock.close()


def run(scene: str, gpu: int) -> None:
    import numpy as np
    import torch
    from causal.h3_cached import H3ChunkCache

    require_recovery_marker(scene)
    verify_snapshot()
    preflight = preflight_scene(scene)
    check_machine(gpu, "T2")
    if datetime.now(timezone.utc) >= datetime.fromisoformat(CONFIG["no_new_recovery_after_hkt"]):
        raise TimeoutError("08:40 HKT no-new-recovery cutoff")
    if (RECOVERY / scene).exists():
        raise FileExistsError("recovery scene already attempted")
    torch.set_num_threads(4)
    torch.manual_seed(13)
    ledger = RecoveryLedger(scene, gpu)
    row_path = ledger.directory / "AD/result.json"
    row = {"task": CONFIG["task"], "scene": scene, "branch": "AD",
           "status": "loading", "row_file": str(row_path),
           "runner_sha256": sha(__file__), "gpu": gpu,
           "preflight": preflight, "sampling_forwards": 0, "vae_decodes": 0}
    row_path.parent.mkdir(parents=True)
    atomic_json(row_path, row)
    began = time.perf_counter()
    try:
        fixture, _ = load_fixture(scene)
        g1 = OUT / "G1" / scene / "FM30"
        g2 = OUT / "G2" / scene / "FM30"
        first = torch.load(g1 / "first12.pt", map_location="cpu", weights_only=True).to("cuda:0", torch.float32)
        cache = torch.load(g2 / "cache_through12.pt", map_location="cpu", weights_only=False)
        assert isinstance(cache, H3ChunkCache)
        frozen_signature = cache_signature(cache)
        pipe, model, precision = load_model()
        row["precision"] = precision
        row["status"] = "sampling"; atomic_json(row_path, row)
        current, _, _, _, _ = sample(pipe, model, fixture, cache=cache,
            start=12, stop=17, index=1, action="AD", steps=30, ledger=ledger, row=row)
        assert cache_signature(cache) == frozen_signature
        for field, key in (("prompt_sha256", "ad_prompt_sha256"),
                           ("position_sha256", "ad_position_sha256"),
                           ("initial_noise_sha256", "ad_noise_sha256")):
            assert row[field] == preflight[key]
        assert row["sigmas"] == preflight["ad_sigmas"]
        endpoint = row_path.parent / "chunk_12_17.pt"
        torch.save(current.cpu(), endpoint)
        assert sha(g2 / "cache_through12.pt") == preflight["cache_sha256"]
        frames, decode_seconds = decode(pipe, first, current, ledger)
        assert frames.shape == (56, 480, 832, 3)
        published = np.load(g1 / "published_39.npy")
        joined = np.concatenate((published, frames[39:]), axis=0)
        assert np.array_equal(joined[:39], published)
        rgb_file = row_path.parent / "published_56.npy"
        np.save(rgb_file, joined)
        video = row_path.parent / "rollout_56.mp4"
        write_video_and_sheet(joined, video, row_path.parent / "all_56_frames.jpg")
        write_video_and_sheet(joined[39:], row_path.parent / "new_17.mp4",
                              row_path.parent / "new_17_frames.jpg")
        row.update(status="complete_pending_visual_review",
                   endpoint_sha256=sha(endpoint), published_sha256=sha(rgb_file),
                   video_sha256=sha(video), decode_seconds=decode_seconds,
                   vae_decodes=1, cpu_raw_kv_bytes=cache.nbytes,
                   peak_allocated_gib=check_peak(),
                   wall_seconds=time.perf_counter() - began,
                   peak_cpu_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
        atomic_json(row_path, row)
        target = {"task": CONFIG["task"], "scene": scene,
                  "quality_status": "pending_judge", "generated_history": True,
                  "teacher_C1_latent": {"path": str((g1 / "first12.pt").relative_to(ROOT)),
                                        "sha256": preflight["c1_endpoint_sha256"]},
                  "teacher_C2_endpoint": {
                      "AA": {"path": str((g2 / "AA/chunk_12_17.pt").relative_to(ROOT)),
                             "sha256": preflight["aa_endpoint_sha256"]},
                      "AD": {"path": str(endpoint.relative_to(ROOT)),
                             "sha256": row["endpoint_sha256"]}},
                  "recovered_from_interrupted_attempt":
                      str((g2 / "AD/result.json").relative_to(ROOT)),
                  "student_history_rule": "rebuild sigma0 KV under student weights"}
        atomic_json(ledger.directory / "target_manifest.json", target)
        ledger.close("complete")
    except BaseException as error:
        row.update(status="failed", error=repr(error), wall_seconds=time.perf_counter() - began)
        atomic_json(row_path, row)
        ledger.close("failed", repr(error))
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--scene", choices=CONFIG["scenes"])
    parser.add_argument("--gpu", type=int)
    args = parser.parse_args()
    if args.preflight == args.run:
        parser.error("choose exactly one of --preflight or --run")
    setup_paths()
    verify_sources()
    verify_code_manifest()
    verify_snapshot()
    verify_recovery_code()
    if args.preflight:
        scenes = [args.scene] if args.scene else CONFIG["scenes"]
        rows = [preflight_scene(scene) for scene in scenes]
        target = HERE / "artifacts/recovery_v2_cpu_preflight.json"
        result = {"task": CONFIG["task"], "scenes": rows}
        if target.exists():
            if json.loads(target.read_text()) != result:
                raise RuntimeError("existing recovery CPU preflight differs")
        else:
            target.write_text(json.dumps(result, indent=2) + "\n")
        print(target)
        return
    if args.scene is None or args.gpu is None:
        parser.error("--run requires --scene and --gpu")
    run(args.scene, args.gpu)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM,
                  lambda signum, frame: (_ for _ in ()).throw(SystemExit(143)))
    main()
