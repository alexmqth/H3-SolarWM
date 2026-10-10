"""Frozen EXP-011 paths, approval gates and write-ahead budget accounting."""
from __future__ import annotations

from datetime import datetime, timezone
import fcntl
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
FROZEN = ROOT / "H3-World/outputs/2026-10-09-22/chunk_partition_cb"
OUT = ROOT / "H3-World/outputs/EXP-011_v3_scene_transfer"
APPROVALS = HERE / "judge"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".partial")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    os.replace(temp, path)


def setup_paths() -> None:
    # Same source precedence as accepted EXP-006. Importing infer verifies v2.
    sources = [FROZEN / "runtime/code", FROZEN / "runtime/code/abot",
               FROZEN / "runtime/code/causal",
               FROZEN / "runtime/DiffSynth-Studio-h3-v2",
               ROOT / "submission/experiments/EXP-001_v2b_124",
               ROOT / "submission/reports/stage1_anyflow/02_causal_diagnostics/current_prefix_candidate",
               ROOT / "submission/experiments/EXP-002_native_cached"]
    sys.path[:0] = [str(x) for x in sources]


def verify_sources() -> None:
    from source_audit import inspect_sources
    manifest = json.loads((HERE / "source_manifest.json").read_text())
    if manifest != inspect_sources():
        raise RuntimeError("fixed source manifest changed")
    accepted = json.loads((ROOT / "submission/experiments/EXP-006_v3_fm8_full/source_manifest.json").read_text())["sources"]
    for name in ("released_action_lora", "accepted_interval", "accepted_current_prefix", "raw_cache",
                 "precision", "visible_inputs", "schedule_helper", "flow_scheduler", "h3_dit",
                 "h3_pipeline", "infer_loader", "video_writer", "flow_metric", "rollout_contract"):
        entry = accepted[name]
        if sha(ROOT / entry["path"]) != entry["sha256"]:
            raise RuntimeError(f"frozen runtime source changed: {name}")


def verify_code_manifest() -> None:
    manifest = json.loads((HERE / "code_manifest.json").read_text())
    for rel, expected in manifest["files"].items():
        if sha(ROOT / rel) != expected:
            raise RuntimeError(f"EXP-011 code changed after freeze: {rel}")


def require_marker(stage: str) -> dict:
    path = APPROVALS / f"{stage}_APPROVED.json"
    if not path.is_file():
        raise PermissionError(f"Judge stage marker missing: {path}")
    marker = json.loads(path.read_text())
    if marker.get("task") != CFG["task"] or marker.get("stage") != stage or marker.get("approved") is not True:
        raise PermissionError(f"invalid Judge stage marker: {path}")
    if marker.get("config_sha256") != sha(HERE / "config.json"):
        raise PermissionError("marker refers to a different config")
    if marker.get("source_manifest_sha256") != sha(HERE / "source_manifest.json"):
        raise PermissionError("marker refers to a different source manifest")
    verify_code_manifest()
    if marker.get("code_manifest_sha256") != sha(HERE / "code_manifest.json"):
        raise PermissionError("marker refers to a different code manifest")
    return marker


def check_machine(gpu: int) -> None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != str(gpu):
        raise RuntimeError(f"set CUDA_VISIBLE_DEVICES={gpu} for --gpu {gpu}; got {visible!r}")
    if datetime.now(timezone.utc).timestamp() >= datetime.fromisoformat(CFG["deadline_hkt"]).timestamp():
        raise TimeoutError("09:00 HKT absolute GPU cutoff")
    free_gib = shutil.disk_usage(OUT.parent).free / 2**30
    if free_gib < CFG["minimum_free_disk_gib"]:
        raise OSError(f"free disk {free_gib:.1f} GiB is below task floor")
    free_mib = int(subprocess.check_output(
        ["nvidia-smi", f"--id={gpu}", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        text=True).strip())
    if free_mib < 44000:
        raise RuntimeError(f"GPU {gpu} has only {free_mib} MiB free")


class Ledger:
    """Exclusive, persistent counters. Reserve before each expensive call."""
    def __init__(self, gpu: int, stage: str):
        OUT.mkdir(parents=True, exist_ok=True)
        self.lock = (OUT / "worker.lock").open("a+")
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.path = OUT / "budget.json"
        if self.path.exists():
            self.data = json.loads(self.path.read_text())
        else:
            self.data = {"task": CFG["task"], "p1_gpu_seconds": 0.0,
                         "inference_gpu_seconds": 0.0, "text_encoder_calls": 0,
                         "image_vae_encodes": 0, "video_vae_encodes": 0,
                         "sampling_forwards": 0, "commit_forwards": 0,
                         "vae_decodes": 0, "events": []}
        self.gpu, self.stage, self.began = gpu, stage, time.monotonic()
        self.data["events"].append({"event": "stage_start", "stage": stage, "gpu": gpu,
                                    "at": datetime.now(timezone.utc).isoformat()})
        self.save()

    def save(self):
        atomic_json(self.path, self.data)

    def check_time(self):
        if datetime.now(timezone.utc).timestamp() >= datetime.fromisoformat(CFG["deadline_hkt"]).timestamp():
            raise TimeoutError("09:00 HKT cutoff")
        free_gib = shutil.disk_usage(OUT.parent).free / 2**30
        if free_gib < CFG["minimum_free_disk_gib"]:
            raise OSError(f"free disk {free_gib:.1f} GiB below task floor")
        elapsed = time.monotonic() - self.began
        if self.stage == "P1":
            if self.data["p1_gpu_seconds"] + elapsed >= CFG["p1"]["max_gpu_seconds"]:
                raise TimeoutError("P1 GPU seconds cap")
        elif self.data["inference_gpu_seconds"] + elapsed >= CFG["inference_total"]["max_gpu_seconds"]:
            raise TimeoutError("inference GPU seconds cap")

    def reserve(self, field: str):
        self.check_time()
        stage_key = "P1" if self.stage == "P1" else ("G1" if self.stage == "G1" else "G2")
        cap = CFG["p1"].get(field) if stage_key == "P1" else CFG["inference_total"].get(field)
        stage_cap = CFG[stage_key.lower()].get(field)
        used_here = sum(e.get("field") == field and
                        (e.get("stage") == stage_key if stage_key != "G2" else
                         str(e.get("stage", "")).startswith("G2_"))
                        for e in self.data["events"] if e.get("event") == "reserve")
        if cap is None or self.data[field] >= cap or stage_cap is None or used_here >= stage_cap:
            raise RuntimeError(f"budget exhausted or forbidden: {field}")
        self.data[field] += 1
        self.data["events"].append({"event": "reserve", "stage": self.stage,
                                    "field": field, "count": self.data[field],
                                    "at": datetime.now(timezone.utc).isoformat()})
        self.save()

    def close(self, status: str):
        field = "p1_gpu_seconds" if self.stage == "P1" else "inference_gpu_seconds"
        self.data[field] += time.monotonic() - self.began
        self.data["events"].append({"event": "stage_stop", "stage": self.stage,
                                    "gpu": self.gpu, "status": status,
                                    "at": datetime.now(timezone.utc).isoformat()})
        self.save()
        self.lock.close()
