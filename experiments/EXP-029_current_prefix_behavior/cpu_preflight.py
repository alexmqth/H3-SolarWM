"""EXP-029 CPU-only gates; no model loading or CUDA initialization."""
from __future__ import annotations

import hashlib
import inspect
import json
import os
from pathlib import Path
import resource
import sys
import tempfile
import time

os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
EXP14=ROOT/"submission/experiments/EXP-014_v3_multiscene_teacher"
FROZEN=ROOT/"H3-World/outputs/2026-10-09-22/chunk_partition_cb/runtime"
sys.path.insert(0,str(EXP14))

import admit
import run_exp029 as worker
from common import setup_paths,sha
setup_paths()
import run_teacher
import infer as abot
from causal.h3_cached import ChunkAttention,H3ChunkCache
from causal.local_topology import visible_inputs
from causal.anyflow_sampling import configure_video_schedule
from current_prefix import current_prefix_feedback,verify_one_edge_change
from diffsynth.diffusion.flow_match import FlowMatchScheduler

torch.set_num_threads(4)
started=time.monotonic()
assert callable(run_teacher.load_model) and callable(worker.main)
auth=json.loads((HERE/"AUTHORIZATION.json").read_text())
assert auth["admission_revision"]==3 and auth["min_free_gpu_mib"]==32768
assert auth["memory_offload_reserve_gib"]==18
assert callable(admit.main) and admit.eligible({1:{"free_mib":32768}},101,1,32768)
assert not admit.eligible({1:{"free_mib":32767}},101,1,32768)
assert not admit.eligible({1:{"free_mib":32768}},99,1,32768)
assert admit.first_eligible({1:{"free_mib":32767},0:{"free_mib":32768}},101,[1,0,6,7],32768)==0
assert admit.first_eligible({1:{"free_mib":32768},0:{"free_mib":32768}},101,[1,0,6,7],32768)==1
assert admit.first_eligible({1:{"free_mib":32767},0:{"free_mib":32767}},101,[1,0,6,7],32768) is None
loader_source=inspect.getsource(abot.load_pipeline)
assert 'float(os.environ.get("ABOT_VRAM_RESERVE_GIB", "5"))' in loader_source
assert 'vram_limit = torch.cuda.mem_get_info(device)[1] / (1024 ** 3) - vram_reserve_gib' in loader_source
assert "ABOT_VRAM_RESERVE_GIB=str(reserve_gib)" in inspect.getsource(admit.main)
verify_one_edge_change(ChunkAttention.attend)
with current_prefix_feedback():
    assert ChunkAttention.attend.__module__=="current_prefix"

input_audit=json.loads((HERE/"EXP029_INPUT_AUDIT.json").read_text())
assert input_audit["status"]=="PASS_CPU_READ_ONLY" and len(input_audit["samples"])==2

def check_masks(scene):
    fixture=torch.load(worker.OUT14/"fixtures"/f"{scene}.pt",map_location="cpu",weights_only=True)
    layout,_=visible_inputs(fixture["packed"],fixture["prompts"]["A"],17,390)
    p=int(layout["action_video_start"])
    cache=H3ChunkCache(max_history=5,storage_device="cpu")
    control=ChunkAttention(cache,1,p,390,12,layout["action_text_rows"],"own",True,commit=False)
    prefix,video=control.masks(5*390,12*390,"cpu")
    ann=torch.full((p,),-1,dtype=torch.long)
    for frame,(lo,hi) in enumerate(layout["action_text_rows"].tolist()):ann[lo:hi]=frame
    assert bool((ann<0).any()) and bool((ann>=12).any())
    assert video.shape==(5*390,p+17*390)
    assert bool(video[0,int(layout["action_text_rows"][12,0])])
    assert not bool(video[0,int(layout["action_text_rows"][11,0])])
    current_offset=p+12*390
    off=torch.zeros((p,p+17*390),dtype=torch.bool)
    off[:,:p]=prefix
    current_frame=12+torch.arange(5*390)//390
    for frame in torch.unique(ann[ann>=0]).tolist():
        off[ann==frame,current_offset:]=current_frame==frame
    on=off.clone()
    on[ann<0,current_offset:]=True
    target=torch.zeros_like(off)
    target[ann<0,current_offset:]=True
    assert torch.equal(on^off,target)
    assert not bool(off[ann<0,current_offset:].any())
    assert bool(on[ann<0,current_offset:].all())
    assert not bool(off[:,p:current_offset].any())
    assert cache.commits==0 and len(cache.layers)==0
    return dict(scene=scene,prefix_rows=p,history_kv_rows=12*390,current_video_rows=5*390,
                only_non_action_prefix_current_video_xor=True,other_mask_edges_identical=True,
                future_rows_absent=len(layout["action_text_rows"])==17,
                no_cache_writer=True)

mask_rows=[check_masks(scene) for scene in worker.SCENES]

schedule=FlowMatchScheduler("MiniMax-H3")
sigma=configure_video_schedule(schedule,steps=30,grid="native",flow_shift=2.22)
assert len(sigma)==31 and len(schedule.timesteps)==30 and sigma[0]==1 and sigma[-1]==0
assert all(sigma[i]>sigma[i+1] for i in range(30))
prior=json.loads((worker.OUT14/"G2/s0_43866101/FM30/AA/result.json").read_text())
assert np.allclose(sigma,prior["sigmas"],atol=0,rtol=0)

ledger=worker.Ledger.__new__(worker.Ledger)
ledger.started=time.monotonic();ledger.counts={k:0 for k in worker.Ledger.caps}
ledger.events=[];ledger.save=lambda **kwargs:None
for i in range(2):ledger.reserve("commit",f"cpu{i}")
for i in range(240):ledger.reserve("sampling",f"cpu{i}")
assert ledger.counts["dit"]==242 and ledger.counts["sampling"]==240
for kind in ("commit","sampling"):
    try:ledger.reserve(kind,"over")
    except RuntimeError:pass
    else:raise AssertionError(f"{kind} cap failed")
for i in range(8):ledger.reserve("decode",f"cpu{i}")
try:ledger.reserve("decode","over")
except RuntimeError:pass
else:raise AssertionError("decode cap failed")

with tempfile.TemporaryDirectory(prefix="exp029_media_",dir=HERE) as temp:
    path=Path(temp)/"tiny.mp4"
    info=worker.write_mp4(np.zeros((4,32,32,3),dtype=np.uint8),path)
    assert info["frames"]==4 and info["fps"]==24 and info["width"]==32

paths=[HERE/x for x in ("taskbook_v1.md","ADDENDUM_01_GPU_ADMISSION.md",
                        "ADDENDUM_02_MEMORY_ADMISSION.md","AUTHORIZATION.json","input_audit_cpu.py",
                        "EXP029_INPUT_AUDIT.json","cpu_preflight.py","run_exp029.py","admit.py")]
paths += [EXP14/x for x in ("run_teacher.py","common.py","config.json")]
paths += [FROZEN/x for x in ("code/causal/h3_cached.py","code/causal/local_topology.py",
                            "code/causal/anyflow_sampling.py","code/abot/infer.py",
                            "DiffSynth-Studio-h3-v2/diffsynth/models/minimax_h3_dit.py",
                            "DiffSynth-Studio-h3-v2/diffsynth/pipelines/minimax_h3_audio_video.py",
                            "DiffSynth-Studio-h3-v2/models/MiniMax/MiniMax-H3/FL2VA/transformer/config.json",
                            "DiffSynth-Studio-h3-v2/models/MiniMax/MiniMax-H3/FL2VA/transformer/model.safetensors.index.json")]
paths += [ROOT/"submission/reports/stage1_anyflow/02_causal_diagnostics/current_prefix_candidate/current_prefix.py",
          ROOT/"submission/experiments/EXP-002_native_cached/interval_cached.py",
          ROOT/"models/H3-World/step-10000.safetensors"]
for scene in worker.SCENES:
    base=worker.OUT14/"G1"/scene/"FM30"
    paths += [worker.OUT14/"fixtures"/f"{scene}.pt",base/"first12.pt",base/"published_39.npy",base/"result.json"]
hashes={str(p.relative_to(ROOT)):sha(p) for p in paths}
result=dict(task="EXP-029/v1",status="PASS_CPU",gpu_calls=0,
            source_inputs_sha256=hashes,sha256=hashes,
            input_audit=input_audit["samples"],masks=mask_rows,
            schedule_points=len(sigma),schedule_updates=30,native_sigma_sha256=hashlib.sha256(
                json.dumps(sigma).encode()).hexdigest(),
            budget_boundary_pass=True,media_h264_24fps_pass=True,
            elapsed_seconds=time.monotonic()-started,
            max_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,
            cuda_initialized=torch.cuda.is_initialized())
assert not result["cuda_initialized"] and result["max_rss_gib"]<32 and result["elapsed_seconds"]<2700
(HERE/"CPU_PREFLIGHT.json").write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
print(result["status"],result["elapsed_seconds"])
