"""Guarded two-update EXP-024 full50-rank8 GPU profile; DO NOT RUN in CPU phase.

Requires a *new* Judge authorization JSON with gpu_authorized=true. This file
is intentionally inert without that document. It never starts a training loop.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import torch

import coverage as cov
import fm_run as old

HERE = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_source_manifest():
    manifest = json.loads((HERE / "SOURCE_MANIFEST.json").read_text())
    for relative, expected in manifest["files"].items():
        if sha(cov.ROOT / relative) != expected:
            raise RuntimeError(f"EXP-024 source SHA changed: {relative}")
    return sha(HERE / "SOURCE_MANIFEST.json")


def atomic_json(path, obj):
    temp = path.with_suffix(path.suffix + ".partial")
    temp.write_text(json.dumps(obj, indent=2) + "\n")
    temp.replace(path)


def require_new_authorization(path):
    value = json.loads(Path(path).read_text())
    if (value.get("task_id") != "EXP-024" or value.get("gpu_authorized") is not True
            or value.get("candidate") != "full50_rank8_C1_C2"
            or value.get("profile_updates_max") != 2
            or value.get("top_level_dit_attempts_max") != 3
            or value.get("backward_max") != 2
            or value.get("allocated_gib_stop") != 44
            or value.get("rss_gib_stop") != 128
            or value.get("outer_timeout_seconds") != 1770
            or value.get("retry_authorized") is not False
            or value.get("taskbook_sha256") != sha(HERE / "taskbook_v2_gpu_profile.md")
            or not 0 < int(value.get("gpu_seconds_max", 0)) <= 1800):
        raise PermissionError("EXP-024 GPU profile needs new Judge authorization")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if len(visible.split(",")) != 1 or not visible.isdigit():
        raise RuntimeError("set exactly one CUDA_VISIBLE_DEVICES physical GPU")
    return value, int(visible)


class Recorder:
    def __init__(self, authorized, physical_gpu):
        self.path = HERE / "GPU_PROFILE_RESULT.json"
        if self.path.exists():
            raise FileExistsError(self.path)
        self.started = time.monotonic()
        self.cap = int(authorized["gpu_seconds_max"])
        self.record = {"task": "EXP-024/v1", "status": "RUNNING",
                       "candidate": "full50_rank8_C1_C2", "physical_gpu": physical_gpu,
                       "pid": os.getpid(), "started_unix_seconds": time.time(),
                       "gpu_seconds_cap": self.cap, "updates": [], "stages": [],
                       "attempted_blocks": [], "completed_updates": 0,
                       "stage_attempted": {}, "stage_completed": {},
                       "route_trace": [], "pre_optimizer_gates": {},
                       "error": None}
        self.active = {}
        self.save()

    def save(self):
        self.record["elapsed_seconds"] = time.monotonic() - self.started
        attempted, completed = self.record["stage_attempted"], self.record["stage_completed"]
        self.record["counts"] = {
            "top_level_dit_attempted": sum(attempted.get(x, 0) for x in
                                           ("C1_forward", "C2_commit", "C2_forward")),
            "top_level_dit_completed": sum(completed.get(x, 0) for x in
                                           ("C1_forward", "C2_commit", "C2_forward")),
            "backward_attempted": attempted.get("C1_backward", 0)+attempted.get("C2_backward", 0),
            "backward_completed": completed.get("C1_backward", 0)+completed.get("C2_backward", 0),
            "optimizer_attempted": attempted.get("C1_optimizer", 0)+attempted.get("C2_optimizer", 0),
            "optimizer_completed": completed.get("C1_optimizer", 0)+completed.get("C2_optimizer", 0),
        }
        atomic_json(self.path, self.record)

    def check(self):
        torch.cuda.synchronize()
        elapsed = time.monotonic() - self.started
        peak = torch.cuda.max_memory_allocated() / 2**30
        rss = old.rss_gib()
        if elapsed >= self.cap - 30:
            raise TimeoutError("EXP-024 GPU profile wall cap")
        if peak >= 44:
            raise MemoryError("EXP-024 profile exceeded 44 GiB allocated")
        if rss >= 128:
            raise MemoryError("EXP-024 profile exceeded 128 GiB process RSS")
        return {"elapsed_seconds": elapsed, "allocated_gib": torch.cuda.memory_allocated()/2**30,
                "reserved_gib": torch.cuda.memory_reserved()/2**30,
                "peak_allocated_gib": peak,
                "peak_reserved_gib": torch.cuda.max_memory_reserved()/2**30,
                "rss_gib": rss}

    def stage(self, name, when):
        if when == "before":
            counts = self.record["counts"]
            if name in ("C1_forward", "C2_commit", "C2_forward") and counts["top_level_dit_attempted"] >= 3:
                raise RuntimeError("authorized top-level DiT attempt cap")
            if name in ("C1_backward", "C2_backward") and counts["backward_attempted"] >= 2:
                raise RuntimeError("authorized backward attempt cap")
            if name in ("C1_optimizer", "C2_optimizer") and counts["optimizer_attempted"] >= 2:
                raise RuntimeError("authorized optimizer attempt cap")
            attempted = self.record["stage_attempted"]
            attempted[name] = attempted.get(name, 0) + 1
            self.active[name] = time.monotonic()
            self.save()
            self.check()
        elif when == "after":
            stats = self.check()
            stats.update({"name": name, "seconds": time.monotonic()-self.active.pop(name)})
            self.record["stages"].append(stats)
            completed = self.record["stage_completed"]
            completed[name] = completed.get(name, 0) + 1
            self.save()
        else:
            raise ValueError(when)


def validate_block_gate(block, trace, model, trainable_ids, adapters=None):
    """Run before optimizer.step; C1 failure prevents entering C2."""
    expected = 0 if block == "C1" else 1
    phases = {}
    for tag, index, layer in trace:
        phases.setdefault(f"{tag}_C{index+1}", set()).add(layer)
    required = [f"forward_C{expected+1}", f"backward_C{expected+1}"]
    if block == "C2":
        required.append("commit_C1")
    for name in required:
        if phases.get(name) != set(range(50)):
            raise RuntimeError(f"pre-optimizer route gate: missing layer in {name}")
    if any(p.grad is not None for p in model.parameters() if id(p) not in trainable_ids):
        raise RuntimeError("pre-optimizer frozen base/action parameter has gradient")
    grad_l2_by_layer = None
    if adapters is not None:
        grad_l2_by_layer = []
        for index, adapter in enumerate(adapters):
            grads = (adapter.lora_A.grad, adapter.lora_B.grad)
            if any(g is None or not torch.isfinite(g).all() for g in grads):
                raise RuntimeError(f"pre-optimizer adapter layer{index} gradient invalid")
            grad_l2_by_layer.append(float(torch.sqrt(sum(g.float().square().sum() for g in grads))))
    return {"route_layers": {name: sorted(phases[name]) for name in required},
            "grad_l2_by_layer": grad_l2_by_layer,
            "frozen_grad_absent": True}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--authorization", required=True)
    args = p.parse_args()
    authorized, physical_gpu = require_new_authorization(args.authorization)
    source_manifest_sha = verify_source_manifest()
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats()
    rec = Recorder(authorized, physical_gpu)
    route_trace = []
    try:
        rec.record["source_sha256"] = {
            "source_manifest": source_manifest_sha,
            "gpu_authorization": sha(args.authorization),
            "profile_gpu.py": sha(__file__),
            "coverage.py": sha(HERE / "coverage.py")}
        rec.record["command"] = ["profile_gpu.py", "--authorization", str(args.authorization)]
        rec.save()
        import run as prior
        from causal.pretrained_lora import install_adapters
        prior.setup()
        torch.manual_seed(13)
        torch.cuda.manual_seed_all(13)
        rec.stage("model_load", "before")
        pipe, model, precision = prior.model()
        rec.record["precision"] = {"mode": "h3_fp32", "runtime": repr(precision)}
        rec.save()
        rec.stage("model_load", "after")
        rec.stage("adapter_optimizer_install", "before")
        adapters, indices = install_adapters(model, rank=8,
                                             block_indices=tuple(range(50)),
                                             device="cuda:0")
        params = [p for a in adapters for p in (a.lora_A, a.lora_B)]
        trainable_ids = {id(p) for p in params}
        assert indices == list(range(50))
        assert sum(p.numel() for p in params) == 10752000
        assert all(not p.requires_grad for p in model.parameters()
                   if id(p) not in trainable_ids)
        optimizer = torch.optim.AdamW(params, lr=1e-4, betas=(.9, .95), weight_decay=.01)
        rec.stage("adapter_optimizer_install", "after")

        # One frozen training clip, C1 then C2, with independent CPU RNG noise.
        selection = json.loads((cov.EXP23 / "SELECTION.json").read_text())
        clip = selection["train_clip_ids"][0]
        rec.record["fixed_clip_id"] = clip
        rec.record["source_sha256"].update({
            name: sha(cov.EXP23 / "encoded" / clip / name)
            for name in ("condition.pt", "full17.pt", "I0.pt")})
        rec.save()
        rec.stage("input_load", "before")
        loaded = old.load_clip(clip, device="cuda:0")
        f = loaded["f"]
        data = {"z": loaded["z"],
                "full_packed": {k: v.to("cuda:0") if torch.is_tensor(v) else v
                                for k, v in f["packed"].items()},
                "prompt": f["prompt"].to("cuda:0"),
                "anchor": loaded["anchor"], "audio": loaded["audio"]}
        cov.prepare_visible(data, "C1")
        cov.prepare_visible(data, "C2")
        rec.stage("input_load", "after")
        noise_gen = torch.Generator(device="cpu").manual_seed(130024)
        phase = ["idle"]
        import current_prefix as route
        original = route.current_prefix_attend

        def audit(control, *a, **kw):
            assert control.action_prefix_mode == "own" and control.action_feedback
            assert control.index in (0, 1)
            route_trace.append((phase[0], control.index, int(kw["layer"])))
            return original(control, *a, **kw)

        def block_gate(block, cache, start):
            evidence = validate_block_gate(block, route_trace[start:], model,
                                           trainable_ids, adapters)
            rec.record["pre_optimizer_gates"][block] = evidence
            rec.save()

        for block in ("C1", "C2"):
            rec.check()
            rec.record["attempted_blocks"].append(block)
            rec.save()
            trace_start = len(route_trace)
            _, start, stop = cov.BLOCKS[block]
            clean = data["z"][:, :, start:stop]
            noise = torch.randn(clean.shape, generator=noise_gen,
                                dtype=torch.float32).to("cuda:0")
            row = cov.fm_step(model, data, optimizer, block=block, noise=noise,
                              sigma=.5, checkpoint=True, offload=True, audit=audit,
                              phase_hook=lambda tag: phase.__setitem__(0, tag),
                              stage_callback=lambda tag, when, b=block:
                                  rec.stage(b+"_"+tag, when),
                              pre_optimizer_gate=lambda b, cache:
                                  block_gate(b, cache, trace_start))
            rec.record["updates"].append(row)
            rec.record["completed_updates"] += 1
            rec.record["route_trace"] = route_trace
            rec.save()
        phases = {}
        for tag, index, layer in route_trace:
            phases.setdefault(f"{tag}_C{index+1}", set()).add(layer)
        for name in ("forward_C1", "backward_C1", "commit_C1",
                     "forward_C2", "backward_C2"):
            if phases.get(name) != set(range(50)):
                raise RuntimeError(f"missing routed layer in {name}")
        rec.record["route_layers_by_phase"] = {k: sorted(v) for k, v in phases.items()}
        rec.record["status"] = "COMPLETE_PENDING_JUDGE"
        rec.record["python_exit_code"] = 0
        rec.save()
    except BaseException as exc:
        rec.record["status"] = "FAILED_STOPPED"
        rec.record["error"] = repr(exc)
        rec.record["python_exit_code"] = 1
        rec.record["route_trace"] = route_trace
        rec.save()
        raise


if __name__ == "__main__":
    main()
