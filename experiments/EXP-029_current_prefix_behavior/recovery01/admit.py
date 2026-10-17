"""Single-attempt daytime admission for EXP-029 s3 OFF_D recovery."""
from __future__ import annotations

from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
STATUS = HERE / "SCHEDULER_STATUS.json"
REGISTRY = HERE / "PROCESS_REGISTRY.json"
LATEST = datetime.fromisoformat("2026-10-17T04:35:00+00:00")
HARD = datetime.fromisoformat("2026-10-17T05:00:00+00:00")
LOCK = ROOT / "submission/.gwm_exp_gpu.lock"


def save(**updates):
    row = json.loads(STATUS.read_text()) if STATUS.exists() else {}
    row.update(task="EXP-029/recovery01", controller_pid=os.getpid(),
               updated_utc=datetime.now(timezone.utc).isoformat(), **updates)
    temp = STATUS.with_suffix(".partial")
    temp.write_text(json.dumps(row, indent=2) + "\n")
    temp.replace(STATUS)


def identity(pid):
    fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    return dict(pid=int(pid), start_ticks=int(fields[19]), pgid=int(fields[2]))


def snapshot():
    raw = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,memory.free,memory.used",
                                   "--format=csv,noheader,nounits"], text=True)
    rows = {}
    for line in raw.splitlines():
        index, uuid, free, used = (x.strip() for x in line.split(","))
        rows[int(index)] = dict(uuid=uuid, free_mib=int(free), used_mib=int(used))
    return rows


def eligible(rows, order, threshold):
    return next((i for i in order if i in rows and rows[i]["free_mib"] >= threshold), None)


def child_pid(timeout_pid):
    path = Path(f"/proc/{timeout_pid}/task/{timeout_pid}/children")
    for _ in range(100):
        if path.exists():
            children = path.read_text().split()
            if len(children) == 1: return int(children[0])
        time.sleep(.1)
    raise TimeoutError("actual Python child PID not found before gate")


def main():
    auth = json.loads((HERE / "AUTHORIZATION.json").read_text())
    cpu = json.loads((HERE / "CPU_PREFLIGHT.json").read_text())
    assert auth["task"] == "EXP-029/recovery01" and cpu["status"] == "PASS_CPU"
    assert auth["allowed_gpu"] == [1, 6] and auth["min_free_gpu_mib"] == 32768
    assert not STATUS.exists() and not REGISTRY.exists() and not (HERE / "START_GATE.json").exists()
    REGISTRY.write_text(json.dumps(dict(controller=identity(os.getpid()), workers=[]), indent=2) + "\n")
    save(status="WAITING_FOR_CAPACITY", attempt_count=0, allowed_gpu=[1, 6],
         min_free_gpu_mib=32768, reserve_gib=18)
    while datetime.now(timezone.utc) < LATEST:
        first = snapshot()
        save(last_gpu_snapshot=first)
        index = eligible(first, [1, 6], 32768)
        if index is not None:
            time.sleep(10.2)
            second = snapshot()
            if eligible(second, [index], 32768) == index and second[index]["uuid"] == first[index]["uuid"]:
                with LOCK.open("a+") as lock:
                    try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError: pass
                    else:
                        third = snapshot()
                        if (eligible(third, [index], 32768) == index and
                            third[index]["uuid"] == first[index]["uuid"] and
                            datetime.now(timezone.utc) < LATEST):
                            remain = (HARD - datetime.now(timezone.utc)).total_seconds()
                            timeout_seconds = min(1200, int(remain) - 10)
                            if timeout_seconds < 60: raise TimeoutError("insufficient time")
                            cmd = ["timeout", "--signal=TERM", "--kill-after=10s",
                                   f"{timeout_seconds}s", str(ROOT / ".venvs/h3world/bin/python"),
                                   "-u", str(HERE / "run_recovery.py")]
                            env = os.environ.copy()
                            env.update(CUDA_VISIBLE_DEVICES=str(index), ABOT_VRAM_RESERVE_GIB="18",
                                       OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1")
                            log = (HERE / "GPU_RUN.log").open("x")
                            child = subprocess.Popen(cmd, cwd=ROOT, env=env,
                                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                            try:
                                actual = child_pid(child.pid)
                                records = [identity(child.pid), identity(actual)]
                                REGISTRY.write_text(json.dumps(dict(controller=identity(os.getpid()),
                                    workers=records), indent=2) + "\n")
                                gate = dict(controller_pid=os.getpid(), timeout_pid=child.pid,
                                    python_pid=actual, physical_gpu=index, gpu_uuid=first[index]["uuid"],
                                    workers=records, created_utc=datetime.now(timezone.utc).isoformat())
                                (HERE / "START_GATE.json").write_text(json.dumps(gate, indent=2) + "\n")
                                save(status="RUNNING", attempt_count=1, physical_gpu=index,
                                     gpu_uuid=first[index]["uuid"], timeout_pid=child.pid,
                                     python_pid=actual, command=cmd, initial_gpu_free_mib=third[index]["free_mib"],
                                     launched_utc=datetime.now(timezone.utc).isoformat())
                                code = child.wait()
                            except BaseException:
                                child.terminate()
                                child.wait(timeout=20)
                                raise
                            finally:
                                log.close()
                            save(status="FINISHED_PENDING_JUDGE" if code == 0 else "FAILED_PENDING_JUDGE",
                                 returncode=code, finished_utc=datetime.now(timezone.utc).isoformat())
                            return
        time.sleep(20)
    save(status="NO_LAUNCH_BEFORE_LATEST_START", attempt_count=0)


if __name__ == "__main__": main()
