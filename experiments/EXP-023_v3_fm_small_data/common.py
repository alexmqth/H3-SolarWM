"""Shared immutable selection, native raw-source RGB extraction and accounting."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import imageio_ffmpeg
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
FROZEN = ROOT / "H3-World/outputs/2026-10-09-22/chunk_partition_cb/runtime"
EXP021 = ROOT / "submission/experiments/EXP-021_v3_fm_baseline_preparation"
EXP022 = ROOT / "submission/experiments/EXP-022_v3_fm_training_profile"
VAE_OLD = ROOT / "H3-World/outputs/EXP-016_v3_real56_vae"
COND_OLD = ROOT / "H3-World/outputs/EXP-019_v3_mixed_action_validation/E"
ENCODED = HERE / "encoded"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def inputs():
    m = json.loads((EXP021 / "DATA_MANIFEST.json").read_text())
    s = json.loads((HERE / "SELECTION.json").read_text())
    a = json.loads((HERE / "AUTHORIZATION.json").read_text())
    assert a["approved"] and a["task"] == "EXP-023/v1"
    assert sha(HERE / "taskbook_v1.md") == a["taskbook_sha256"]
    assert sha(HERE / "SELECTION.json") == a["selection_sha256"]
    addendum = json.loads((HERE / "AUTHORIZATION_ADDENDUM_01.json").read_text())
    assert addendum["approved"] and addendum["task"] == a["task"]
    assert sha(HERE / "AUTHORIZATION.json") == addendum["base_authorization_sha256"]
    assert sha(HERE / "ADDENDUM_01_SOURCE_RGB.md") == addendum["addendum_sha256"]
    assert addendum["effective_limits"]["vae_encode"] == 70
    train = {r["clip_id"]:r for r in m["train_candidates"]}
    dev = {r["clip_id"]:r for r in m["dev_candidates"]}
    assert len(train) == 32 and len(dev) == 8
    assert set(train) == set(s["train_clip_ids"])
    roles = {r["role"]:r for r in s["evaluation"]}
    assert set(roles) == {"train_constant","train_switch","dev_constant","dev_switch"}
    for role, r in roles.items():
        row = (train if role.startswith("train") else dev)[r["clip_id"]]
        assert row["episode_id"] == r["episode_id"] and row["category"] == r["category"]
    assert not {r["episode_id"] for r in train.values()} & {r["episode_id"] for r in dev.values()}
    return m,s,a,train,dev,roles


def new_rows():
    _,_,_,train,dev,roles = inputs()
    new_train = [r for r in train.values() if not r["materialized_real56"]]
    assert len(new_train) == 28
    assert len(train) == 32
    selected_dev = [dev[roles[x]["clip_id"]] for x in ("dev_constant","dev_switch")]
    assert len(selected_dev) == 2
    pilot_constant = train[roles["train_constant"]["clip_id"]]
    if pilot_constant["materialized_real56"]:
        pilot_constant = min((r for r in new_train if r["category"] == "constant"),key=lambda r:r["clip_id"])
    pilot = [pilot_constant,dev[roles["dev_switch"]["clip_id"]]]
    assert all(not r["materialized_real56"] for r in pilot)
    # Addendum 01 supersedes reuse of the older four H.264-derived latents.
    # Every selected clip is re-encoded from the same original-source RGB path.
    rest = [r for r in train.values()] + selected_dev
    rest = [r for r in rest if r["clip_id"] not in {x["clip_id"] for x in pilot}]
    assert len(rest) == 32
    return pilot,rest


def raw_source_rgb(row):
    """Decode selected original source frames; no intermediate H.264 encode."""
    source = ROOT / row["source_video"]
    start = int(row["source_start"])
    assert row["source_indices"][-1] == start + 68
    vf = (f"select='between(n\\,{start}\\,{start+68})"
          f"*not(eq(mod(n-{start}\\,5)\\,4))',"
          "scale=-2:480,crop=832:480")
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-threads", "1",
           "-filter_threads", "1", "-i", str(source), "-vf", vf, "-vsync", "0",
           "-frames:v", "56", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    raw = subprocess.check_output(cmd)
    assert len(raw) == 56 * 480 * 832 * 3
    return np.frombuffer(raw, dtype=np.uint8).reshape(56, 480, 832, 3).copy()


def rss_gib():
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) / 2**20
    return -1


class PhaseLedger:
    def __init__(self, phase, gpu, max_seconds):
        import torch
        self.phase,self.gpu,self.limit=phase,gpu,max_seconds
        self.started=time.monotonic()
        self.path=HERE/f"{phase.upper()}_BUDGET.json"
        if self.path.exists():
            raise FileExistsError(f"{phase} already ran; manual audit required")
        self.prior=[]
        for old in ("encode_pilot","encode_rest","train","eval"):
            path=HERE/f"{old.upper()}_BUDGET.json"
            if old!=phase and path.exists():
                item=json.loads(path.read_text())
                if item["status"]!="COMPLETE_PENDING_JUDGE":
                    raise RuntimeError(f"prior phase {old} not complete")
                self.prior.append(item)
        self.counts={"dit":0,"backward":0,"update":0,"vae_encode":0,"vae_decode":0,"text":0}
        self.events=[]
        self.status="RUNNING"
        self.save()

    def save(self, **extra):
        import torch
        own_wall=time.monotonic()-self.started
        summed={k:self.counts[k]+sum(x["counts"][k] for x in self.prior) for k in self.counts}
        total_wall=own_wall+sum(x["wall_seconds"] for x in self.prior)
        write(self.path,{"task":"EXP-023/v1","phase":self.phase,"status":self.status,
            "gpu":self.gpu,"wall_seconds":own_wall,
            "limits":{"wall_seconds":self.limit,"allocated_gib":44},
            "counts":dict(self.counts),"peak_allocated_gib":torch.cuda.max_memory_allocated()/2**30,
            "peak_reserved_gib":torch.cuda.max_memory_reserved()/2**30,
            "rss_gib":rss_gib(),"events":self.events,
            "cumulative_counts":summed,"cumulative_gpu_seconds":total_wall,**extra})
        write(HERE/"CUMULATIVE_BUDGET.json",{"task":"EXP-023/v1",
            "authorization_base_sha256":sha(HERE/"AUTHORIZATION.json"),
            "authorization_addendum_sha256":sha(HERE/"AUTHORIZATION_ADDENDUM_01.json"),
            "gpu_process_seconds_sum":total_wall,"counts":summed,
            "completed_phases":[x["phase"] for x in self.prior],
            "active_phase":self.phase,"active_status":self.status,
            "limits":json.loads((HERE/"AUTHORIZATION_ADDENDUM_01.json").read_text())["effective_limits"]})

    def reserve(self,kind,note):
        if time.monotonic()-self.started>=self.limit-30:
            raise TimeoutError(f"{self.phase} hard stop")
        total=time.monotonic()-self.started+sum(x["wall_seconds"] for x in self.prior)
        if total>=13800-30:
            raise TimeoutError("cumulative GPU seconds exhausted")
        max_counts={"dit":440,"backward":128,"update":128,"vae_encode":70,"vae_decode":12,"text":600}
        prior=sum(x["counts"][kind] for x in self.prior)
        if self.counts[kind]+prior>=max_counts[kind]:
            raise RuntimeError(f"{kind} budget exhausted")
        self.counts[kind]+=1
        self.events.append({"kind":kind,"note":note,"elapsed":time.monotonic()-self.started})
        self.save()

    def measure(self,name,func):
        import torch
        torch.cuda.synchronize()
        start=time.monotonic()
        value=func()
        torch.cuda.synchronize()
        elapsed=time.monotonic()-start
        self.events.append({"kind":"timing","name":name,"seconds":elapsed,
                            "rss_gib":rss_gib(),"elapsed":time.monotonic()-self.started})
        if torch.cuda.max_memory_allocated()/2**30>44:
            raise MemoryError("allocated >44 GiB")
        self.save()
        print("TIMING",name,round(elapsed,3),flush=True)
        return value

    def close(self,status,error=None):
        self.status=status
        self.save(error=error)
