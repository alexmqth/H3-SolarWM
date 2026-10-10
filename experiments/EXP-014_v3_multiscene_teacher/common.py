"""EXP-014 frozen paths, Judge gates, and per-scene write-ahead GPU accounting."""
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
OUT = ROOT / "H3-World/outputs/EXP-014_v3_multiscene_teacher"
APPROVALS = HERE / "judge"
MODEL_ROOT = FROZEN / "runtime/DiffSynth-Studio-h3-v2/models"


def sha(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".partial")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    os.replace(temp, path)


def setup_paths() -> None:
    # Freeze model resolution explicitly; EXP-011 P1 attempt2 failed without it.
    if not (MODEL_ROOT / "MiniMax/MiniMax-H3/FL2VA/text_encoder").is_dir():
        raise FileNotFoundError(MODEL_ROOT)
    os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = str(MODEL_ROOT)
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
        raise RuntimeError("frozen EXP-014 source manifest changed")
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
            raise RuntimeError(f"EXP-014 code changed after freeze: {rel}")


def require_marker(stage: str, scene: str | None = None) -> dict:
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
    if stage == "T1" and scene != CFG["t1_scene"]:
        raise PermissionError("T1 may only run the first predetermined scene")
    if stage == "T2" and (scene is None or scene == CFG["t1_scene"]):
        raise PermissionError("T2 may only run the remaining predetermined scenes")
    if scene is not None and scene not in marker.get("scenes", []):
        raise PermissionError(f"scene {scene} not in Judge marker")
    return marker


def now_hkt_timestamp() -> float:
    return datetime.now(timezone.utc).timestamp()


def deadline_timestamp(name: str) -> float:
    return datetime.fromisoformat(CFG[name]).timestamp()


def check_machine(gpu: int, stage: str) -> None:
    if gpu in (3, 4):
        raise RuntimeError("GPU3/4 reserved by other VLLM processes for this task")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != str(gpu):
        raise RuntimeError(f"set CUDA_VISIBLE_DEVICES={gpu} for --gpu {gpu}; got {visible!r}")
    if now_hkt_timestamp() >= deadline_timestamp("deadline_hkt"):
        raise TimeoutError("09:00 HKT GPU cutoff")
    if stage in ("T1", "T2") and now_hkt_timestamp() >= deadline_timestamp("no_new_scene_after_hkt"):
        raise TimeoutError("08:40 HKT new-scene cutoff")
    free_gib = shutil.disk_usage(OUT.parent).free / 2**30
    if free_gib < CFG["minimum_free_disk_gib"]:
        raise OSError(f"free disk {free_gib:.1f} GiB below task floor")
    free_mib = int(subprocess.check_output([
        "nvidia-smi", f"--id={gpu}", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        text=True).strip())
    if free_mib < 44000:
        raise RuntimeError(f"GPU {gpu} has only {free_mib} MiB free")


class Ledger:
    """One exclusive journal per P1 or teacher scene; reserve before every model call."""
    def __init__(self, gpu: int, stage: str, scene: str | None = None):
        if stage not in ("P1", "T1", "T2"):
            raise ValueError(stage)
        if stage != "P1" and scene not in CFG["scenes"]:
            raise ValueError(scene)
        self.stage, self.scene, self.gpu = stage, scene, gpu
        self.directory = OUT / ("P1" if stage == "P1" else f"teacher/{scene}")
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = (self.directory / "worker.lock").open("a+")
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.path = self.directory / "budget.json"
        baseline_file = OUT / "task_baseline.json"
        if not baseline_file.exists():
            # P1 is sequentially first. Exclusive creation avoids T2 races if
            # a stage is launched after an interrupted attempt.
            baseline = {"task": CFG["task"],
                        "initial_free_bytes": shutil.disk_usage(OUT.parent).free,
                        "max_new_disk_gib": CFG["max_new_disk_gib"]}
            try:
                fd = os.open(baseline_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o664)
            except FileExistsError:
                pass
            else:
                with os.fdopen(fd, "w") as stream:
                    json.dump(baseline, stream)
                    stream.write("\n")
        self.baseline_free_bytes = json.loads(baseline_file.read_text())["initial_free_bytes"]
        if self.path.exists():
            self.data = json.loads(self.path.read_text())
        else:
            self.data = {"task": CFG["task"], "stage": stage, "scene": scene,
                         "gpu_seconds": 0.0, "text_encoder_calls": 0,
                         "image_vae_encodes": 0, "video_vae_encodes": 0,
                         "sampling_forwards": 0, "commit_forwards": 0,
                         "vae_decodes": 0, "events": []}
        if self.data["stage"] != stage or self.data["scene"] != scene:
            raise RuntimeError("cannot reuse ledger for different stage/scene")
        self.began = time.monotonic()
        self.data["events"].append({"event": "stage_start", "gpu": gpu,
                                    "at": datetime.now(timezone.utc).isoformat()})
        self.save()

    def save(self) -> None:
        atomic_json(self.path, self.data)

    def check_time(self) -> None:
        if now_hkt_timestamp() >= deadline_timestamp("deadline_hkt"):
            raise TimeoutError("09:00 HKT cutoff")
        free_gib = shutil.disk_usage(OUT.parent).free / 2**30
        if free_gib < CFG["minimum_free_disk_gib"]:
            raise OSError(f"free disk {free_gib:.1f} GiB below task floor")
        consumed_gib = (self.baseline_free_bytes - shutil.disk_usage(OUT.parent).free) / 2**30
        if consumed_gib >= CFG["max_new_disk_gib"]:
            raise OSError(f"disk growth {consumed_gib:.1f} GiB exceeds task cap")
        elapsed = time.monotonic() - self.began
        cap = CFG["p1" if self.stage == "P1" else "teacher_per_scene"]["max_gpu_seconds"]
        if self.data["gpu_seconds"] + elapsed >= cap:
            raise TimeoutError(f"{self.stage} GPU seconds cap")

    def reserve(self, field: str) -> None:
        self.check_time()
        cap = CFG["p1" if self.stage == "P1" else "teacher_per_scene"].get(field)
        if cap is None or self.data[field] >= cap:
            raise RuntimeError(f"budget exhausted or forbidden: {field}")
        self.data[field] += 1
        self.data["events"].append({"event": "reserve", "field": field,
                                    "count": self.data[field],
                                    "at": datetime.now(timezone.utc).isoformat()})
        self.save()

    def close(self, status: str) -> None:
        self.data["gpu_seconds"] += time.monotonic() - self.began
        self.data["events"].append({"event": "stage_stop", "gpu": self.gpu,
                                    "status": status,
                                    "at": datetime.now(timezone.utc).isoformat()})
        self.save()
        self.lock.close()
