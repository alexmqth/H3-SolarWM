"""CPU-only source, input and budget gate for one-branch EXP-029 recovery."""
from __future__ import annotations

import hashlib
import inspect
import json
import os
from pathlib import Path
import resource
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = HERE.parents[3]
sys.path[:0] = [str(HERE), str(EXP), str(ROOT / "submission/experiments/EXP-014_v3_multiscene_teacher")]

import torch
import numpy as np
import admit
import run_recovery as worker
from common import setup_paths, sha

started = time.monotonic()
setup_paths()
import run_teacher as old
import infer as abot
from causal.anyflow_sampling import configure_video_schedule
from diffsynth.diffusion.flow_match import FlowMatchScheduler
from causal.local_topology import visible_inputs

auth = json.loads((HERE / "AUTHORIZATION.json").read_text())
assert auth["status"] == "AUTHORIZED_BY_ADDENDUM_03"
assert auth["allowed_gpu"] == [1, 6] and auth["min_free_gpu_mib"] == 32768
assert auth["memory_offload_reserve_gib"] == 18
assert auth["max_dit_attempts"] == 31 and auth["max_vae_decodes"] == 1
assert admit.eligible({1:{"free_mib":32768}},[1,6],32768)==1
assert admit.eligible({1:{"free_mib":32767},6:{"free_mib":32768}},[1,6],32768)==6
assert admit.eligible({1:{"free_mib":32767},6:{"free_mib":32767}},[1,6],32768) is None
assert 'ABOT_VRAM_RESERVE_GIB="18"' in inspect.getsource(admit.main)
assert 'float(os.environ.get("ABOT_VRAM_RESERVE_GIB", "5"))' in inspect.getsource(abot.load_pipeline)
assert worker.HARD_STOP.isoformat() == "2026-10-17T05:00:00+00:00"
assert admit.LATEST.isoformat() == "2026-10-17T04:35:00+00:00"
assert '1185' in inspect.getsource(worker.main)

freeze = json.loads((HERE / "ORIGINAL_FREEZE.json").read_text())
assert freeze["status"] == "PASS_SEVEN_FROZEN_CPU" and len(freeze["frozen_seven"]) == 7
assert freeze["s3_cache_sha256"] == worker.EXPECTED_KV
assert sha(EXP / "SCHEDULER_STATUS.json") == freeze["original_scheduler_sha256"]
assert sha(worker.ORIGINAL / "GPU_LEDGER.json") == freeze["original_ledger_sha256"]
for entry in freeze["frozen_seven"].values():
    for relative, expected in entry["files"].items():
        assert sha(ROOT / relative) == expected

fixture, entry = old.load_fixture(worker.SCENE)
first_file = worker.SOURCE / "G1" / worker.SCENE / "FM30/first12.pt"
published_file = worker.SOURCE / "G1" / worker.SCENE / "FM30/published_39.npy"
source_row = json.loads((first_file.parent / "result.json").read_text())
assert sha(first_file) == source_row["endpoint_sha256"]
assert sha(published_file) == source_row["published_sha256"]
partial = json.loads((worker.ORIGINAL / worker.SCENE / "OFF_D/RESULT.json").read_text())
assert partial["status"] == "SAMPLING" and partial["branch"] == "OFF_D"
assert not (worker.ORIGINAL / worker.SCENE / "OFF_D/endpoint5.pt").exists()
assert not (worker.ORIGINAL / worker.SCENE / "OFF_D/rollout_56.mp4").exists()
layout,prompt = visible_inputs(fixture["packed"],fixture["prompts"]["A"],17,390)
assert len(layout["action_text_rows"]) == 17 and layout["seq_len"] == layout["action_video_start"]+17*390
donor = fixture["prompts"]["D"]
counterfactual = fixture["prompts"]["A"].clone()
for lo,hi in fixture["packed"]["action_text_spans_local"][12:17]:
    counterfactual[lo:hi] = donor[lo:hi]
visible, cropped = visible_inputs(fixture["packed"],counterfactual,17,390)
assert worker.frozen.tensor_sha(cropped) == partial["prompt_sha256"]
assert worker.frozen.tensor_sha(visible["img_position_ids"]) == partial["position_sha256"]
assert worker.frozen.tensor_sha(fixture["initial_noise"][:,:,12:17].to(torch.float32)) == partial["initial_noise_sha256"]
assert len(visible["action_text_rows"]) == 17
assert torch.equal(cropped[:layout["action_text_rows"][12,0]],prompt[:layout["action_text_rows"][12,0]])

scheduler = FlowMatchScheduler("MiniMax-H3")
sigmas = configure_video_schedule(scheduler,steps=30,grid="native",flow_shift=2.22)
assert len(sigmas)==31 and len(scheduler.timesteps)==30 and sigmas==partial["sigmas"]

ledger = worker.Ledger.__new__(worker.Ledger)
ledger.start=time.monotonic(); ledger.counts=dict(dit=0,commit=0,sampling=0,decode=0,
    backward=0,update=0,text_encode=0,vae_encode=0);ledger.events=[]
ledger.gate=lambda:None;ledger.save=lambda **kwargs:None
ledger.reserve("commit","cpu")
for i in range(30): ledger.reserve("sampling",f"cpu/{i}")
ledger.reserve("decode","cpu")
assert ledger.counts["dit"]==31 and ledger.counts["decode"]==1
for kind in ("commit","sampling","decode"):
    try:ledger.reserve(kind,"over")
    except RuntimeError:pass
    else:raise AssertionError(f"{kind} cap failed")

frozen_runtime = ROOT / "H3-World/outputs/2026-10-09-22/chunk_partition_cb/runtime"
paths = [HERE/x for x in ("AUTHORIZATION.json","ORIGINAL_FREEZE.json","cpu_preflight.py",
                          "run_recovery.py","admit.py")]
paths += [EXP/x for x in ("ADDENDUM_03_FINITE_RECOVERY.md","run_exp029.py")]
paths += [ROOT/x for x in ("submission/experiments/EXP-014_v3_multiscene_teacher/run_teacher.py",
                           "submission/experiments/EXP-014_v3_multiscene_teacher/common.py",
                           "submission/experiments/EXP-014_v3_multiscene_teacher/config.json",
                           "submission/experiments/EXP-002_native_cached/interval_cached.py",
                           "submission/reports/stage1_anyflow/02_causal_diagnostics/current_prefix_candidate/current_prefix.py")]
paths += [frozen_runtime/x for x in ("code/abot/infer.py","code/causal/h3_cached.py",
    "code/causal/local_topology.py","code/causal/anyflow_sampling.py",
    "code/causal/h3_precision.py","DiffSynth-Studio-h3-v2/diffsynth/models/minimax_h3_dit.py",
    "DiffSynth-Studio-h3-v2/diffsynth/pipelines/minimax_h3_audio_video.py")]
paths += [worker.ORIGINAL/"GPU_LEDGER.json",EXP/"SCHEDULER_STATUS.json",
          worker.ORIGINAL/worker.SCENE/"OFF_D/RESULT.json",
          worker.SOURCE/"fixtures"/f"{worker.SCENE}.pt",first_file,published_file,
          first_file.parent/"result.json",ROOT/"models/H3-World/step-10000.safetensors"]
hashes={str(p.relative_to(ROOT)):sha(p) for p in paths}
result=dict(task="EXP-029/recovery01",status="PASS_CPU",gpu_calls=0,
    candidate_gpu=[1,6],min_free_gpu_mib=32768,reserve_gib=18,
    only_branch="s3_b784d995/OFF_D",same_source_prompt_noise_schedule=True,
    frozen_original_outputs=7,original_partial_sampling_attempts=23,
    cache_digest_expected=worker.EXPECTED_KV,schedule_points=31,schedule_updates=30,
    budget_boundary_pass=True,sha256=hashes,
    cuda_initialized=torch.cuda.is_initialized(),
    max_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,
    elapsed_seconds=time.monotonic()-started)
assert not result["cuda_initialized"] and result["max_rss_gib"]<32
(HERE/"CPU_PREFLIGHT.json").write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
print("PASS_CPU",result["elapsed_seconds"])
