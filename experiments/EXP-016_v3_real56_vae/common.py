"""Frozen EXP-016 sources, CPU gate, and write-ahead VAE call accounting."""
from __future__ import annotations

from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CFG = json.loads((HERE / "config.json").read_text())
FROZEN = ROOT / "H3-World/outputs/2026-10-09-22/chunk_partition_cb/runtime/DiffSynth-Studio-h3-v2"
MODEL_ROOT = FROZEN / "models"
VAE_FILE = MODEL_ROOT / "MiniMax/MiniMax-H3/FL2VA/video_vae/source/model.safetensors"
OUT = ROOT / "H3-World/outputs/EXP-016_v3_real56_vae"
PARENT = ROOT / "submission/experiments/EXP-015_v3_real56_data/OUTPUT_MANIFEST.json"
EXPECTED_VAE_SHA = "5f0c2e161d895a9fee7645ca32d4a7e3a22b90cacfcbeba62ec999cdbbefe0d3"


def sha(path: Path | str) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")
    os.replace(tmp, path)


def now() -> float:
    return datetime.now().timestamp()


def when(name: str) -> float:
    return datetime.fromisoformat(CFG[name]).timestamp()


def cpu_affinity() -> list[int]:
    cpus = sorted(os.sched_getaffinity(0))[:4]
    if len(cpus) != 4:
        raise RuntimeError("Need four available CPU cores")
    os.sched_setaffinity(0, cpus)
    return cpus


def setup_runtime() -> None:
    os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = str(MODEL_ROOT)
    sys.path.insert(0, str(FROZEN))


def source_rows() -> list[dict]:
    if sha(PARENT) != CFG["parent_manifest_sha256"]:
        raise RuntimeError("EXP-015 parent manifest SHA changed")
    parent = json.loads(PARENT.read_text())
    rows = parent["scenes"]
    if [row["scene"] for row in rows] != CFG["scenes"]:
        raise RuntimeError("scene order changed")
    for row in rows:
        for info in row["output_files"].values():
            p = Path(info["path"])
            if not p.is_file() or sha(p) != info["sha256"]:
                raise RuntimeError(f"input SHA mismatch: {p}")
        assert row["video_probe"]["frames"] == 56
        assert row["video_probe"]["resolution"] == [832, 480]
        assert row["video_probe"]["fps"] == 24
    return rows


def freeze_sources() -> dict:
    rows = source_rows()
    return {"task": CFG["task"], "parent_manifest_path": str(PARENT),
            "parent_manifest_sha256": sha(PARENT),
            "video_vae_path": str(VAE_FILE), "video_vae_sha256": EXPECTED_VAE_SHA,
            "runtime_vae_source": str(FROZEN / "diffsynth/models/minimax_h3_video_vae.py"),
            "runtime_vae_source_sha256": sha(FROZEN / "diffsynth/models/minimax_h3_video_vae.py"),
            "scenes": [{"scene": r["scene"],
                        "video": r["output_files"]["real56.mp4"],
                        "I0": r["output_files"]["I0.png"]} for r in rows]}


def verify_frozen() -> None:
    if json.loads((HERE / "source_manifest.json").read_text()) != freeze_sources():
        raise RuntimeError("source manifest changed")
    code = json.loads((HERE / "code_manifest.json").read_text())
    for rel, expected in code["files"].items():
        if sha(ROOT / rel) != expected:
            raise RuntimeError(f"code SHA changed: {rel}")


def require_marker() -> dict:
    marker_path = HERE / "judge/P1_APPROVED.json"
    if not marker_path.is_file():
        raise PermissionError(f"Judge marker missing: {marker_path}")
    marker = json.loads(marker_path.read_text())
    if marker.get("task") != CFG["task"] or marker.get("stage") != "P1" or marker.get("approved") is not True:
        raise PermissionError("invalid P1 marker")
    for field, path in (("config_sha256", HERE / "config.json"),
                        ("source_manifest_sha256", HERE / "source_manifest.json"),
                        ("code_manifest_sha256", HERE / "code_manifest.json")):
        if marker.get(field) != sha(path):
            raise PermissionError(f"P1 marker {field} mismatch")
    verify_frozen()
    return marker


def check_machine(*, before_start: bool = False) -> dict:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("P1 requires CUDA_VISIBLE_DEVICES=0")
    if before_start and now() >= when("start_before_hkt"):
        raise TimeoutError("08:50 HKT start cutoff")
    if now() >= when("no_new_call_after_hkt"):
        raise TimeoutError("08:57 HKT model-call cutoff")
    free = shutil.disk_usage(OUT.parent).free
    if free / 2**30 < CFG["minimum_free_disk_gib"]:
        raise OSError("disk free below 60 GiB")
    available = int(subprocess.check_output(["nvidia-smi", "--id=0", "--query-gpu=memory.free",
                                              "--format=csv,noheader,nounits"], text=True).strip())
    if available < 45056:
        raise RuntimeError(f"GPU0 not free enough: {available} MiB")
    return {"free_disk_bytes": free, "gpu0_free_mib": available}


class Ledger:
    def __init__(self, initial_free: int):
        self.path = OUT / "budget.json"
        if self.path.exists():
            raise FileExistsError("EXP-016 budget already exists; no automatic rerun")
        self.start = time.monotonic()
        self.initial_free = initial_free
        self.data = {"task": CFG["task"], "stage": "P1", "status": "started", "gpu": 0,
                     "image_encode": 0, "video_encode": 0, "video_decode": 0,
                     "model_loads": 0, "events": [], "gpu_wall_seconds": 0.0}
        self.save()

    def save(self) -> None:
        self.data["gpu_wall_seconds"] = time.monotonic() - self.start
        write(self.path, self.data)

    def guard(self) -> None:
        if now() >= when("no_new_call_after_hkt"):
            raise TimeoutError("08:57 HKT model-call cutoff")
        if time.monotonic() - self.start >= CFG["max_gpu_seconds"]:
            raise TimeoutError("600 GPU-second cap")
        free = shutil.disk_usage(OUT.parent).free
        if free / 2**30 < CFG["minimum_free_disk_gib"] or self.initial_free - free >= CFG["max_new_bytes"]:
            raise OSError("disk budget exceeded")

    def reserve(self, kind: str, scene: str) -> None:
        self.guard()
        if kind not in CFG["counts"] or self.data[kind] >= CFG["counts"][kind]:
            raise RuntimeError(f"call budget exceeded: {kind}")
        self.data[kind] += 1
        self.data["events"].append({"call": kind, "scene": scene,
                                    "at": datetime.now().isoformat()})
        self.save()

    def close(self, status: str, error: str | None = None) -> None:
        self.data["status"] = status
        if error is not None:
            self.data["error"] = error
        self.save()
