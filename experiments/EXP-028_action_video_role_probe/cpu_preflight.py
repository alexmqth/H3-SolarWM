"""CPU-only gates for EXP-028; no model or CUDA initialization."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time
import types
from unittest.mock import patch

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "submission/code"))
sys.path.insert(0, str(ROOT / "H3-World/code"))
sys.path.insert(0, str(ROOT / "submission/experiments/EXP-023_v3_fm_small_data"))
sys.path.insert(0, str(ROOT / "submission/reports/stage1_anyflow/02_causal_diagnostics/current_prefix_candidate"))
from causal.pretrained_lora import CausalQKVLoRA
from causal.local_transition import DifferentiableWindowAttention, transition_forward
from causal.local_topology import OriginalWindowAttention, visible_inputs
from interval_train import interval_train
from fm_run import verified_attention_scope
import admit
import admit_recovery
import gpu_run
import run as original_loader
from original_clean import original_clean_forward
from role_probe import RoleAccumulator, ROLE_NAMES, diagnostic_forward, original_layout, v3_layout

torch.set_num_threads(4)
torch.manual_seed(28)
started = time.monotonic()
rows = 3
spans = torch.tensor([[1+i*2, 3+i*2] for i in range(17)], dtype=torch.long)
packed = {"action_video_start": 36, "action_text_rows": spans}
orig = original_layout(packed, rows)
v3 = v3_layout(packed, rows)
assert len(orig) == 36+17*rows and len(v3) == 36+5*rows
assert (orig == 0).sum() == 24 and (orig == 1).sum() == 10
assert (orig == 2).sum() == 12*rows and (orig == 3).sum() == 5*rows
assert (v3 == 2).sum() == 0 and (v3 == 3).sum() == 5*rows
mask = OriginalWindowAttention(packed, 17, rows).build_mask("cpu")
p = int(packed["action_video_start"])
# Original Single-Egress is directed in both action/video directions.
assert bool(mask[p + 12*rows, int(spans[12,0])])
assert not bool(mask[p + 12*rows, int(spans[11,0])])
assert bool(mask[int(spans[12,0]), p + 12*rows])
assert not bool(mask[int(spans[12,0]), p + 11*rows])
assert DifferentiableWindowAttention.allow_grad_read
assert callable(transition_forward) and callable(interval_train)
assert callable(original_loader.model) and callable(gpu_run.main)
ledger = gpu_run.Ledger.__new__(gpu_run.Ledger)
ledger.start = time.monotonic()
ledger.counts = dict(dit=14, backward=7, commit=2, identity=4, update=0, video=0)
ledger.events = []
ledger.save = lambda **kwargs: None
ledger.reserve("backward", "CPU cap boundary")
assert ledger.counts["backward"] == 8 and ledger.counts["dit"] == 14
try:
    ledger.reserve("main", "over cap")
    raise AssertionError("DiT cap was not enforced")
except RuntimeError as exc:
    assert "DiT call cap" in str(exc)
try:
    ledger.reserve("backward", "over cap")
    raise AssertionError("backward cap was not enforced")
except RuntimeError as exc:
    assert "call cap" in str(exc)
synthetic_gpu = {1: {"free_mib": 40960, "processes": ["other-process"]},
                 6: {"free_mib": 40959, "processes": []}}
assert admit.eligible(synthetic_gpu, 101) == [1]
assert admit.eligible(synthetic_gpu, 99) is False
assert admit_recovery.OUT.name == "recovery01"
with verified_attention_scope():
    from causal.h3_cached import ChunkAttention
    assert ChunkAttention.attend.__module__ == "current_prefix"

base = nn.Linear(16, 3*4*8, bias=False)
adapter = CausalQKVLoRA(base, rank=8, device="cpu")
adapter.forward = diagnostic_forward.__get__(adapter, CausalQKVLoRA)
adapter.role_accumulator = None
with torch.no_grad():
    adapter.lora_B.zero_()
x = torch.randn(len(orig), 16)
assert torch.equal(adapter(x), base(x))
cases = []
for name, labels, use_checkpoint in (("original", orig, False), ("v3_checkpoint", v3, True)):
    data = torch.randn(len(labels), 16)
    adapter.lora_A.grad = adapter.lora_B.grad = None
    acc = RoleAccumulator(labels, 96, 8, adapter.scale, tile=7)
    adapter.role_accumulator = acc
    y = checkpoint(adapter, data, use_reentrant=False) if use_checkpoint else adapter(data)
    (y.float().square().mean()).backward()
    check = acc.check(adapter.lora_B.grad)
    assert check["pass_role_sum"], check
    assert adapter.lora_A.grad is not None and torch.count_nonzero(adapter.lora_A.grad) == 0
    assert acc.counts["other"] == 0
    assert acc.cotangent_invocations == 1, check
    qkv = adapter.lora_B.grad.view(4, 3, 8, 8)
    assert qkv.shape == (4, 3, 8, 8)
    cases.append(dict(name=name, **check,
                      qkv_norms={n: float(qkv[:, i].norm()) for i,n in enumerate("qkv")}))
    adapter.role_accumulator = None

# Production-like BF16 residual cast: hook sees FP32 cotangent after the cast.
adapter.lora_B.grad = None
with torch.no_grad():
    adapter.base.weight.data = adapter.base.weight.data.to(torch.bfloat16)
data = torch.randn(len(v3), 16, dtype=torch.bfloat16)
acc = RoleAccumulator(v3, 96, 8, adapter.scale)
adapter.role_accumulator = acc
assert torch.equal(adapter(data), adapter.base(data))
(adapter(data).float().square().mean()).backward()
cast_check = acc.check(adapter.lora_B.grad)
assert cast_check["pass_role_sum"], cast_check
# A separate detached history writer must have no gradient in the reader loss.
writer = nn.Linear(4, 4, bias=False)
reader = nn.Linear(4, 4, bias=False)
h = writer(torch.randn(3, 4)).detach()
(reader(h).square().mean()).backward()
assert writer.weight.grad is None and reader.weight.grad is not None

paths = [
    HERE / "taskbook_v1.md", HERE / "AUTHORIZATION.json", HERE / "ADDENDUM_01_SHARED_GPU.md",
    HERE / "role_probe.py", HERE / "cpu_preflight.py",
    HERE / "admit.py",
    HERE / "admit_recovery.py",
    HERE / "gpu_run.py",
    HERE / "original_clean.py",
    HERE / "ADDENDUM_02_STARTUP_RECOVERY.md",
    HERE / "AUTHORIZATION_RECOVERY_01.json",
    ROOT / "submission/code/causal/pretrained_lora.py",
    ROOT / "H3-World/code/causal/local_transition.py",
    ROOT / "H3-World/outputs/2026-10-09-22/chunk_partition_cb/runtime/code/causal/local_topology.py",
    ROOT / "submission/reports/stage1_anyflow/02_causal_diagnostics/current_prefix_candidate/current_prefix.py",
    ROOT / "H3-World/DiffSynth-Studio-h3-v2/diffsynth/models/minimax_h3_dit.py",
    ROOT / "H3-World/DiffSynth-Studio-h3-v2/diffsynth/pipelines/minimax_h3_audio_video.py",
    ROOT / "models/H3-World/step-10000.safetensors",
    ROOT / "H3-World/outputs/2026-10-09-22/chunk_partition_cb/runtime/DiffSynth-Studio-h3-v2/models/MiniMax/MiniMax-H3/FL2VA/transformer/config.json",
    ROOT / "H3-World/outputs/2026-10-09-22/chunk_partition_cb/runtime/DiffSynth-Studio-h3-v2/models/MiniMax/MiniMax-H3/FL2VA/transformer/model.safetensors.index.json",
    ROOT / "submission/experiments/EXP-023_v3_fm_small_data/interval_train.py",
    ROOT / "submission/experiments/EXP-023_v3_fm_small_data/fm_run.py",
    ROOT / "submission/experiments/EXP-019_v3_mixed_action_validation/run.py",
    ROOT / "submission/experiments/EXP-025_v3_fm_full_coverage/checkpoint_000.pt",
]
for clip in ("43866101158a538d22c52ac5b2fa606b_src1245_rgb56",
             "b784d995827f53977a8067e5b9dc4b5e_src1635_rgb56"):
    for leaf in ("record.json", "condition.pt", "full17.pt", "I0.pt"):
        paths.append(ROOT / "submission/experiments/EXP-023_v3_fm_small_data/encoded" / clip / leaf)
def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()

hashes = {str(p.relative_to(ROOT)): sha(p) for p in paths}
real_layouts = []
for clip in ("43866101158a538d22c52ac5b2fa606b_src1245_rgb56",
             "b784d995827f53977a8067e5b9dc4b5e_src1635_rgb56"):
    folder = ROOT / "submission/experiments/EXP-023_v3_fm_small_data/encoded" / clip
    record = json.loads((folder / "record.json").read_text())
    assert record["clip_id"] == clip and record["latents"]["full17"]["sha256"] == sha(folder / "full17.pt")
    assert record["condition"]["sha256"] == sha(folder / "condition.pt")
    condition = torch.load(folder / "condition.pt", map_location="cpu", weights_only=True)
    assert condition["clip_id"] == clip and condition["episode_id"] == record["episode_id"]
    assert condition["action_script_sha256"] == record["action_script_sha256"]
    actual, _ = visible_inputs(condition["packed"], condition["prompt"], 17, 390)
    actual_original = original_layout(actual, 390)
    actual_v3 = v3_layout(actual, 390)
    assert len(actual_original) == actual["seq_len"]
    assert len(actual_v3) == actual["action_video_start"] + 5*390
    assert (actual_original == 2).sum() == 12*390 and (actual_original == 3).sum() == 5*390
    assert (actual_v3 == 2).sum() == 0 and (actual_v3 == 3).sum() == 5*390
    real_layouts.append(dict(clip=clip, episode=record["episode_id"],
                             original_rows=len(actual_original), v3_rows=len(actual_v3),
                             action_history_rows=int((actual_original == 0).sum()),
                             action_current_rows=int((actual_original == 1).sum())))
    if len(real_layouts) == 1:
        observed = {}
        history = torch.randn(1,24,12,30,52)
        current = torch.randn(1,24,5,30,52)
        def fake_model_fn(dit, video, audio, packed, prompt, **kwargs):
            fixed = kwargs["denoise_mask_video"]
            assert torch.count_nonzero(fixed[:,:,:12]) == 0
            assert torch.all(fixed[:,:,12:] == 1)
            assert torch.equal(kwargs["input_latents_video"], video)
            assert kwargs["fixed_prefix_timesteps"] is False
            assert kwargs["use_gradient_checkpointing"] and kwargs["use_gradient_checkpointing_offload"]
            mask = kwargs["causal_control"].build_mask("cpu")
            assert mask.shape == (actual["seq_len"], actual["seq_len"])
            observed["sigma_native"] = float(kwargs["timestep_video"])
            observed["history_clean"] = torch.equal(video[:,:,:12], history.to(video.dtype))
            return (video, None)
        module = types.ModuleType("diffsynth.pipelines.minimax_h3_audio_video")
        module.model_fn_minimax_h3 = fake_model_fn
        fake_modules = {"diffsynth":types.ModuleType("diffsynth"),
                        "diffsynth.pipelines":types.ModuleType("diffsynth.pipelines"),
                        "diffsynth.pipelines.minimax_h3_audio_video":module}
        with patch.dict(sys.modules, fake_modules):
            out = original_clean_forward(object(),current,history=history,
                full_packed=actual,prompt=condition["prompt"],
                anchor=condition["anchor"],audio=condition["audio_noise"],
                sigma=.25,checkpoint=True,offload=True)
        assert torch.equal(out, current.to(out.dtype)) and observed == {"sigma_native":250.0,"history_clean":True}, observed
result = dict(task="EXP-028/v1", status="PASS_CPU", cases=cases, cast=cast_check,
              source_sha256=hashes, real_layouts=real_layouts, elapsed_seconds=time.monotonic()-started,
              rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
              cuda_initialized=torch.cuda.is_initialized(), gpu_calls=0,
              role_sum_threshold=dict(relative_rms=1e-3, max_abs="1e-3 * max(1,max_abs(reference))"),
              identity_threshold=dict(relative_rms=1e-6, max_abs="1e-6 * max(1,max_abs(reference))"))
assert not result["cuda_initialized"] and result["rss_gib"] < 32 and result["elapsed_seconds"] < 3600
(HERE / "CPU_PREFLIGHT.json").write_text(json.dumps(result, indent=2) + "\n")
print("PASS_CPU", result["elapsed_seconds"], flush=True)
