"""Exactly one EXP-028 startup recovery, GPU1 preferred, 2396 seconds max."""
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
from datetime import datetime, timezone

from admit import ROOT, HERE, EXPIRY, LOCK, snapshot, eligible

OUT = HERE / "recovery01"
OUT.mkdir(exist_ok=True)
STATUS = OUT / "SCHEDULER_STATUS.json"


def save(**kwargs):
    previous = json.loads(STATUS.read_text()) if STATUS.exists() else {}
    previous.update(task="EXP-028/v1-recovery01", pid=os.getpid(),
        expiry_utc=EXPIRY.isoformat(), updated_utc=datetime.now(timezone.utc).isoformat(), **kwargs)
    temp = STATUS.with_suffix(".partial")
    temp.write_text(json.dumps(previous, indent=2) + "\n")
    temp.replace(STATUS)


def main():
    auth = json.loads((HERE / "AUTHORIZATION_RECOVERY_01.json").read_text())
    cpu = json.loads((HERE / "CPU_PREFLIGHT.json").read_text())
    if (auth.get("recovery_revision") != 1 or auth.get("gpu_process_seconds") != 2396
        or cpu.get("status") != "PASS_CPU" or (OUT / "GPU_LEDGER.json").exists()
        or (OUT / "GPU_RUN.log").exists()):
        raise RuntimeError("recovery authorization, CPU gate or prior attempt invalid")
    save(status="WAITING_FOR_GPU1_CAPACITY", attempt_count=0,
         prior_startup_seconds_charged=4)
    while datetime.now(timezone.utc) < EXPIRY:
        first, host = snapshot()
        save(last_gpu_snapshot=first, host_mem_available_gib=host)
        if 1 in eligible(first, host):
            time.sleep(10.2)
            second, host2 = snapshot()
            if (1 in eligible(second, host2)
                and second[1]["uuid"] == first[1]["uuid"]
                and datetime.now(timezone.utc) < EXPIRY):
                with LOCK.open("a+") as lock:
                    try:
                        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        pass
                    else:
                        third, host3 = snapshot()
                        if (1 in eligible(third, host3)
                            and third[1]["uuid"] == first[1]["uuid"]
                            and datetime.now(timezone.utc) < EXPIRY):
                            env = os.environ.copy()
                            env.update(CUDA_VISIBLE_DEVICES="1", OMP_NUM_THREADS="4",
                                       OPENBLAS_NUM_THREADS="1")
                            command = ["timeout", "--signal=TERM", "--kill-after=10s", "2396s",
                                       str(ROOT / ".venvs/h3world/bin/python"), "-u", str(HERE / "gpu_run.py")]
                            save(status="RUNNING", physical_gpu=1, gpu_uuid=first[1]["uuid"],
                                 command=command, attempt_count=1,
                                 launched_utc=datetime.now(timezone.utc).isoformat(),
                                 other_compute_pids=third[1]["processes"])
                            with (OUT / "GPU_RUN.log").open("x") as output:
                                child = subprocess.Popen(command, cwd=ROOT, env=env,
                                                         stdout=output, stderr=subprocess.STDOUT)
                                save(child_pid=child.pid)
                                code = child.wait()
                            save(status="FINISHED_PENDING_JUDGE" if code == 0 else "FAILED_PENDING_JUDGE",
                                 returncode=code, finished_utc=datetime.now(timezone.utc).isoformat())
                            return
        time.sleep(60)
    save(status="EXPIRED_NO_GPU1_CAPACITY", attempt_count=0)


if __name__ == "__main__":
    main()
