"""Single-attempt GPU1 admission and identity registration for EXP-029."""
from __future__ import annotations

from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
REGISTRY=ROOT/"submission/research/overnight_20261017/PROCESS_REGISTRY.json"
REG_LOCK=REGISTRY.with_suffix(".lock")
STATUS=HERE/"SCHEDULER_STATUS.json"
SOFT=datetime.fromisoformat("2026-10-17T00:59:45+00:00")
PROJECT_LOCK=ROOT/"submission/.gwm_exp_gpu.lock"


def save(**kwargs):
    value=json.loads(STATUS.read_text()) if STATUS.exists() else {}
    value.update(task="EXP-029/v1",controller_pid=os.getpid(),
                 updated_utc=datetime.now(timezone.utc).isoformat(),**kwargs)
    temp=STATUS.with_suffix(".partial")
    temp.write_text(json.dumps(value,indent=2)+"\n")
    temp.replace(STATUS)


def identity(pid):
    fields=Path(f"/proc/{pid}/stat").read_text().rsplit(")",1)[1].split()
    return dict(task="EXP-029/v1",pid=int(pid),start_ticks=int(fields[19]),pgid=int(fields[2]))


def register(entries):
    with REG_LOCK.open("a+") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        data=json.loads(REGISTRY.read_text())
        assert isinstance(data["processes"],list)
        old={x["pid"] for x in data["processes"]}
        if any(x["pid"] in old for x in entries):
            raise RuntimeError("PID already registered")
        data["processes"].extend(entries)
        temp=REGISTRY.with_suffix(".partial")
        temp.write_text(json.dumps(data,indent=2)+"\n")
        temp.replace(REGISTRY)


def snapshot():
    info=subprocess.check_output(["nvidia-smi","--query-gpu=index,uuid,memory.free,memory.used",
                                  "--format=csv,noheader,nounits"],text=True)
    rows={}
    for line in info.splitlines():
        index,uuid,free,used=(x.strip() for x in line.split(","))
        rows[int(index)]=dict(uuid=uuid,free_mib=int(free),used_mib=int(used))
    apps=subprocess.check_output(["nvidia-smi","--query-compute-apps=gpu_uuid,pid",
                                  "--format=csv,noheader"],text=True)
    for row in rows.values():
        row["other_compute_pids"]=[x.split(",")[1].strip() for x in apps.splitlines()
                                   if x.split(",")[0].strip()==row["uuid"]]
    host=0
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            host=int(line.split()[1])/2**20;break
    return rows,host


def eligible(rows,host,index):
    return host>=100 and index in rows and rows[index]["free_mib"]>=40960


def first_eligible(rows,host,allowed):
    return next((index for index in allowed if eligible(rows,host,index)),None)


def child_pid(timeout_pid):
    path=Path(f"/proc/{timeout_pid}/task/{timeout_pid}/children")
    for _ in range(100):
        if path.exists():
            children=path.read_text().split()
            if len(children)==1:return int(children[0])
        time.sleep(.1)
    raise TimeoutError("actual Python child PID not found before registration gate")


def main():
    auth=json.loads((HERE/"AUTHORIZATION.json").read_text())
    cpu=json.loads((HERE/"CPU_PREFLIGHT.json").read_text())
    if (auth.get("task_id")!="EXP-029" or cpu.get("status")!="PASS_CPU"
        or STATUS.exists() or (HERE/"START_GATE.json").exists()):
        raise RuntimeError("authorization/gate/duplicate scheduler invalid")
    register([identity(os.getpid())])
    allowed=auth["allowed_gpu"]
    assert allowed == [1,0,6,7] and auth["admission_revision"] == 2
    save(status="WAITING_FOR_GPU_CAPACITY",attempt_count=0,allowed_gpu=allowed)
    while datetime.now(timezone.utc).timestamp()+3600 < SOFT.timestamp():
        first,host=snapshot()
        save(last_gpu_snapshot=first,host_mem_available_gib=host)
        selected=first_eligible(first,host,allowed)
        if selected is not None:
            time.sleep(10.2)
            second,host2=snapshot()
            if eligible(second,host2,selected) and second[selected]["uuid"]==first[selected]["uuid"]:
                with PROJECT_LOCK.open("a+") as lock:
                    try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    except BlockingIOError: pass
                    else:
                        third,host3=snapshot()
                        if (eligible(third,host3,selected) and third[selected]["uuid"]==first[selected]["uuid"]
                            and datetime.now(timezone.utc).timestamp()+3600 < SOFT.timestamp()):
                            command=["timeout","--signal=TERM","--kill-after=10s","3600s",
                                     str(ROOT/".venvs/h3world/bin/python"),"-u",str(HERE/"run_exp029.py")]
                            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(selected),OMP_NUM_THREADS="4",
                                                              OPENBLAS_NUM_THREADS="1")
                            output=(HERE/"GPU_RUN.log").open("x")
                            child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=output,
                                                   stderr=subprocess.STDOUT,start_new_session=True)
                            try:
                                python_pid=child_pid(child.pid)
                                records=[identity(child.pid),identity(python_pid)]
                                register(records)
                                gate=dict(python_pid=python_pid,timeout_pid=child.pid,
                                          controller_pid=os.getpid(),physical_gpu=selected,
                                          gpu_uuid=first[selected]["uuid"],
                                          registered=records,created_utc=datetime.now(timezone.utc).isoformat())
                                (HERE/"START_GATE.json").write_text(json.dumps(gate,indent=2)+"\n")
                                save(status="RUNNING",attempt_count=1,command=command,
                                     physical_gpu=selected,gpu_uuid=first[selected]["uuid"],
                                     other_compute_pids=third[selected]["other_compute_pids"],
                                     launched_utc=datetime.now(timezone.utc).isoformat(),
                                     timeout_pid=child.pid,python_pid=python_pid,process_registry=records)
                                code=child.wait()
                            except BaseException:
                                child.terminate();child.wait(timeout=20)
                                raise
                            finally: output.close()
                            save(status="FINISHED_PENDING_JUDGE" if code==0 else "FAILED_PENDING_JUDGE",
                                 returncode=code,finished_utc=datetime.now(timezone.utc).isoformat())
                            return
        time.sleep(60)
    save(status="NO_LAUNCH_BEFORE_DEADLINE",attempt_count=0)


if __name__=="__main__":main()
