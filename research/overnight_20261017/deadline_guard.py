"""Stop only identity-registered GWM process groups at the user's cutoff."""
import json
import os
from pathlib import Path
import signal
import sys
import time
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
REGISTRY = HERE / "PROCESS_REGISTRY.json"
STATUS = HERE / "DEADLINE_STATUS.json"
SOFT = datetime.fromisoformat("2026-10-17T00:59:45+00:00").timestamp()
HARD = datetime.fromisoformat("2026-10-17T01:00:00+00:00").timestamp()

def identity(pid):
    try:
        # stat fields after comm: state=3, ppid=4, pgrp=5, session=6, starttime=22.
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return {"pid": int(pid), "start_ticks": int(fields[19]), "pgid": int(fields[2])}
    except (FileNotFoundError, ProcessLookupError):
        return None

def matches(entry):
    live = identity(entry["pid"])
    return live is not None and all(live[k] == entry[k] for k in ("pid", "start_ticks", "pgid"))

def registered():
    return json.loads(REGISTRY.read_text())["processes"]

def save(state, events):
    temp = STATUS.with_suffix(".tmp")
    temp.write_text(json.dumps({"status": state, "guard_pid": os.getpid(),
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "soft_stop_utc": "2026-10-17T00:59:45Z", "hard_stop_utc": "2026-10-17T01:00:00Z",
        "events": events}, indent=2) + "\n")
    temp.replace(STATUS)

def stop(signum, events):
    seen = set()
    for entry in registered():
        if not matches(entry):
            continue
        pgid = entry["pgid"]
        if pgid <= 1 or pgid == os.getpgrp() or pgid in seen:
            continue
        seen.add(pgid)
        try:
            os.killpg(pgid, signum)
            events.append({"task": entry["task"], "pid": entry["pid"],
                "pgid": pgid, "signal": signum, "time": time.time()})
        except ProcessLookupError:
            pass

if __name__ == "__main__":
    if "--check" in sys.argv:
        print(json.dumps([{**e, "identity_matches_live": matches(e)} for e in registered()], indent=2))
    else:
        events = []
        while time.time() < SOFT:
            save("ARMED", events)
            time.sleep(min(15, max(0.05, SOFT-time.time())))
        stop(signal.SIGTERM, events)
        save("SOFT_STOP_SENT", events)
        while time.time() < HARD:
            time.sleep(min(1, max(0.05, HARD-time.time())))
        stop(signal.SIGKILL, events)
        save("CUTOFF_ENFORCED_PENDING_JUDGE_PROCESS_AUDIT", events)
