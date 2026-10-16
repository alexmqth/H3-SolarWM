"""Expiry-bounded, single-attempt GPU capacity admission for EXP-028."""
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
EXPIRY = datetime.fromisoformat("2026-10-17T05:15:00+00:00")
CANDIDATES = (0, 1, 6, 7)
STATUS = HERE / "SCHEDULER_STATUS.json"
LOCK = ROOT / "submission/.gwm_exp_gpu.lock"


def save(**kwargs):
    previous = json.loads(STATUS.read_text()) if STATUS.exists() else {}
    previous.update(task="EXP-028/v1", pid=os.getpid(),
                    expiry_utc=EXPIRY.isoformat(), updated_utc=datetime.now(timezone.utc).isoformat(), **kwargs)
    partial = STATUS.with_suffix(".partial")
    partial.write_text(json.dumps(previous, indent=2) + "\n")
    partial.replace(STATUS)


def snapshot():
    info = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,memory.used,memory.free",
        "--format=csv,noheader,nounits"], text=True)
    apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
        "--format=csv,noheader"], text=True)
    gpu = {}
    for line in info.splitlines():
        idx, uuid, used, free = (s.strip() for s in line.split(","))
        gpu[int(idx)] = dict(uuid=uuid, used_mib=int(used), free_mib=int(free), processes=[])
    for line in apps.splitlines():
        parts = [s.strip() for s in line.split(",")]
        if len(parts) >= 2:
            for item in gpu.values():
                if item["uuid"] == parts[0]:
                    item["processes"].append(parts[1])
    avail_kib = 0
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            avail_kib = int(line.split()[1]); break
    return gpu, avail_kib / 2**20


def eligible(gpu, host_gib):
    return host_gib >= 100 and [i for i in CANDIDATES
        if i in gpu and gpu[i]["free_mib"] >= 40960]


def main():
    cpu = json.loads((HERE / "CPU_PREFLIGHT.json").read_text())
    auth = json.loads((HERE / "AUTHORIZATION.json").read_text())
    if (cpu["status"] != "PASS_CPU" or (HERE / "GPU_LEDGER.json").exists()
        or auth.get("admission_revision") != 2 or auth.get("min_free_gpu_mib") != 40960):
        raise RuntimeError("CPU gate failed or GPU attempt already exists")
    save(status="WAITING_FOR_GPU_CAPACITY", attempt_count=0, admission_revision=2)
    while datetime.now(timezone.utc) < EXPIRY:
        first, host = snapshot()
        save(status="WAITING_FOR_GPU_CAPACITY", last_gpu_snapshot=first, host_mem_available_gib=host)
        possible = eligible(first, host)
        if possible:
            time.sleep(10.2)
            second, host2 = snapshot()
            possible = [i for i in possible if i in eligible(second, host2)
                        and second[i]["uuid"] == first[i]["uuid"]]
            if possible and datetime.now(timezone.utc) < EXPIRY:
                with LOCK.open("a+") as lock:
                    try:
                        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        possible = []
                    if possible:
                        third, host3 = snapshot()
                        possible = [i for i in possible if i in eligible(third, host3)
                                    and third[i]["uuid"] == first[i]["uuid"]]
                        if possible and datetime.now(timezone.utc) < EXPIRY:
                            gpu_id = possible[0]
                            env = os.environ.copy()
                            env.update(CUDA_VISIBLE_DEVICES=str(gpu_id), OMP_NUM_THREADS="4",
                                       OPENBLAS_NUM_THREADS="1")
                            command = ["timeout", "--signal=TERM", "--kill-after=10s", "2400s",
                                       str(ROOT / ".venvs/h3world/bin/python"), "-u", str(HERE / "gpu_run.py")]
                            save(status="RUNNING", physical_gpu=gpu_id, gpu_uuid=first[gpu_id]["uuid"],
                                 command=command, attempt_count=1, launched_utc=datetime.now(timezone.utc).isoformat())
                            with (HERE / "GPU_RUN.log").open("w") as output:
                                outcome = subprocess.run(command, cwd=ROOT, env=env,
                                                         stdout=output, stderr=subprocess.STDOUT)
                            save(status="FINISHED_PENDING_JUDGE" if outcome.returncode == 0 else "FAILED_PENDING_JUDGE",
                                 returncode=outcome.returncode, finished_utc=datetime.now(timezone.utc).isoformat())
                            return
        time.sleep(60)
    save(status="EXPIRED_NO_GPU_CAPACITY", attempt_count=0)


if __name__ == "__main__":
    main()
