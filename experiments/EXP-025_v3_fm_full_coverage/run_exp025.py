"""EXP-025/v1 guarded, single-GPU full50 C1/C2 ordinary-FM experiment.

The frozen EXP-023/024 implementation is imported, never edited. One invocation
runs regression, pre-evaluation, 256 updates, then post-evaluation; any failed
gate stops the invocation and preserves the ledger. No automatic retry.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gc
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
EXP23 = ROOT / "submission/experiments/EXP-023_v3_fm_small_data"
EXP24 = ROOT / "submission/experiments/EXP-024_v3_fm_coverage_preparation"
for path in (EXP24, EXP23):
    sys.path.insert(0, str(path))

import torch
import coverage as cov
import fm_run as old
import profile_gpu as profile
from common import inputs, sha, write, rss_gib

ORIGINAL_C2_PROBE = old.probe

PHASE_CAPS = {"regression": 400, "pre": 1400, "train": 11000, "post": 1400}
CAPS = {"dit": 576, "backward": 256, "update": 256,
        "vae_decode": 12, "vae_encode": 0, "text": 0}


def atomic_save(path: Path, state: dict):
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_suffix(".partial")
    if temporary.exists():
        raise FileExistsError(temporary)
    torch.save(state, temporary)
    temporary.replace(path)
    write(path.with_suffix(".json"), {
        "task": "EXP-025/v1", "step": state["step"], "sha256": sha(path),
        "bytes": path.stat().st_size, "adapter_layers": len(state["adapter"]),
        "metadata": state["metadata"]})


def checkpoint(adapters, optimizer, step, order, position, noise_gen):
    state = {
        "step": step,
        "adapter": [{"A": a.lora_A.detach().cpu().clone(),
                     "B": a.lora_B.detach().cpu().clone()} for a in adapters],
        "optimizer": old.cpu_tree(optimizer.state_dict()),
        "cpu_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state(),
        "noise_rng": noise_gen.get_state(),
        "order": order, "order_pos": position,
        "metadata": {"task": "EXP-025/v1", "objective": "ordinary_fm",
                     "adapter": "all50_qkv_rank8", "partition": [12, 5],
                     "train_clips": 32, "seed": 13,
                     "precision": "h3_fp32", "history": "GT_C1_clean_commit"}}
    path = HERE / f"checkpoint_{step:03d}.pt"
    atomic_save(path, state)
    restored = torch.load(path, map_location="cpu", weights_only=True)
    if (restored["step"] != step or len(restored["adapter"]) != 50
            or restored["order_pos"] != position
            or not torch.equal(restored["noise_rng"], state["noise_rng"])):
        raise RuntimeError("checkpoint CPU reload mismatch")
    for got, expected in zip(restored["adapter"], state["adapter"]):
        if not all(torch.equal(got[key], expected[key]) for key in ("A", "B")):
            raise RuntimeError("checkpoint adapter reload mismatch")
    def identical(x, y):
        if torch.is_tensor(x):
            return torch.is_tensor(y) and torch.equal(x, y)
        if isinstance(x, dict):
            return isinstance(y, dict) and x.keys() == y.keys() and all(identical(x[k], y[k]) for k in x)
        if isinstance(x, (list, tuple)):
            return isinstance(y, (list, tuple)) and len(x) == len(y) and all(identical(a,b) for a,b in zip(x,y))
        return x == y
    for key in ("optimizer", "cpu_rng", "cuda_rng", "order"):
        if not identical(restored[key], state[key]):
            raise RuntimeError(f"checkpoint {key} reload mismatch")
    del restored, state
    gc.collect()


def preflight(*, require_gpu: bool):
    auth = json.loads((HERE / "AUTHORIZATION.json").read_text())
    if (auth.get("task_id"), auth.get("plan_version"), auth.get("gpu_count_max")) != ("EXP-025", 1, 1):
        raise PermissionError("EXP-025/v1 single-GPU authorization required")
    if not auth.get("gpu_authorized") or auth.get("retry_authorized") is not False:
        raise PermissionError("authorization mismatch")
    for field, expected in (("optimizer_updates_max", 256), ("backward_max", 256),
                            ("top_level_dit_attempts_max", 576),
                            ("gpu_seconds_max", 13800), ("vae_decode_max", 12),
                            ("vae_encode_max", 0), ("text_encode_max", 0)):
        if auth.get(field) != expected:
            raise PermissionError(f"authorization mismatch: {field}")
    if require_gpu and (os.environ.get("CUDA_VISIBLE_DEVICES", "").count(",")
                        or not os.environ.get("CUDA_VISIBLE_DEVICES", "").isdigit()):
        raise RuntimeError("exactly one physical GPU must be exposed")
    source = json.loads((EXP24 / "SOURCE_MANIFEST.json").read_text())
    for relative, expected in source["files"].items():
        if sha(ROOT / relative) != expected:
            raise RuntimeError(f"frozen source changed: {relative}")
    _, selection, _, _, _, roles = inputs()
    ids = selection["train_clip_ids"]
    order = cov.balanced_block_order(ids, seed=13, epochs=4)
    assert len(ids) == 32 and len(order) == 256
    counts = Counter(order)
    assert all(counts[(clip, block)] == 4 for clip in ids for block in ("C1", "C2"))
    all_ids = sorted(set(ids) | {row["clip_id"] for row in roles.values()})
    assert len(all_ids) == 34
    records = {}
    for clip in all_ids:
        folder = EXP23 / "encoded" / clip
        record = json.loads((folder / "record.json").read_text())
        records[clip] = {name: sha(folder / name) for name in ("condition.pt", "full17.pt", "I0.pt")}
        for name, key in (("condition.pt", "condition"), ("full17.pt", "full17"), ("I0.pt", "I0")):
            expected = record[key]["sha256"] if key == "condition" else record["latents"][key]["sha256"]
            if records[clip][name] != expected:
                raise RuntimeError(f"encoded source mismatch: {clip}/{name}")
    expected_dit = 6 + 2 * (4 * 9 + 2 * 17 + 4 * 5) + 384
    # Per evaluation: 4 GT*(commit+8) + 2 self*(8+commit+8) + 4 probes*(C1 2 + C2 commit+2).
    assert expected_dit == 570
    result = {"task": "EXP-025/v1", "status": "PASS", "train_clips": 32,
              "dev_clips": 2, "order_length": len(order), "per_clip_block_quota": 4,
              "block_counts": dict(Counter(block for _, block in order)),
              "expected_dit": expected_dit, "budget_dit": CAPS["dit"],
              "frozen_source_manifest_sha256": sha(EXP24 / "SOURCE_MANIFEST.json"),
              "authorization_sha256": sha(HERE / "AUTHORIZATION.json"),
              "taskbook_sha256": sha(HERE / "taskbook_v1.md"),
              "input_sha256": records,
              "implementation_sha256": sha(__file__),
              "source_sha256": {str(path.relative_to(ROOT)): sha(path) for path in
                  (EXP23 / "fm_run.py", EXP23 / "interval_train.py",
                   EXP24 / "coverage.py", EXP24 / "profile_gpu.py")}}
    write(HERE / "CPU_PREFLIGHT.json", result)
    return order


def cpu_checkpoint_lifecycle():
    """Exercise full50 adapter and populated AdamW/RNG serialization on CPU."""
    from types import SimpleNamespace
    torch.set_num_threads(4)
    gen = torch.Generator(device="cpu").manual_seed(13)
    adapters = []
    for _ in range(50):
        adapters.append(SimpleNamespace(
            lora_A=torch.nn.Parameter(torch.randn((8, 6720), generator=gen)*.02),
            lora_B=torch.nn.Parameter(torch.zeros((20160, 8)))))
    params = [p for a in adapters for p in (a.lora_A, a.lora_B)]
    assert sum(p.numel() for p in params) == 10752000
    optimizer = torch.optim.AdamW(params, lr=1e-4, betas=(.9,.95), weight_decay=.01)
    # Populate optimizer tensors without launching a GPU forward.
    for p in params:
        p.grad = torch.ones_like(p)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    state = {"adapter": [{"A": a.lora_A.detach().clone(),
                           "B": a.lora_B.detach().clone()} for a in adapters],
             "optimizer": optimizer.state_dict(), "noise_rng": gen.get_state(),
             "cpu_rng": torch.get_rng_state()}
    with tempfile.TemporaryDirectory(prefix="exp025_cpu_ckpt_") as folder:
        path = Path(folder) / "state.pt"
        torch.save(state, path)
        loaded = torch.load(path, map_location="cpu", weights_only=True)
        assert len(loaded["adapter"]) == 50
        assert torch.equal(loaded["noise_rng"], state["noise_rng"])
        assert torch.equal(loaded["cpu_rng"], state["cpu_rng"])
        assert all(torch.equal(x[k], y[k]) for x,y in zip(state["adapter"],loaded["adapter"])
                   for k in ("A","B"))
        assert len(loaded["optimizer"]["state"]) == 100
        for index, record in state["optimizer"]["state"].items():
            reread = loaded["optimizer"]["state"][index]
            assert all(torch.equal(value, reread[key]) for key,value in record.items())
        result = {"status": "PASS", "adapter_layers": 50,
                  "adapter_parameters": 10752000,
                  "optimizer_state_parameters": len(loaded["optimizer"]["state"]),
                  "serialized_bytes": path.stat().st_size,
                  "serialization_sha256": sha(path)}
    write(HERE / "CPU_CHECKPOINT_LIFECYCLE.json", result)


class Ledger:
    def __init__(self, physical_gpu):
        self.path = HERE / "GPU_LEDGER.json"
        if self.path.exists():
            raise FileExistsError("EXP-025 has already started; no automatic retry")
        self.start = time.monotonic()
        self.phase = "regression"
        self.phase_start = self.start
        self.counts = {k: 0 for k in CAPS}
        self.phase_counts = {}
        self.phase_seconds = {}
        self.events = []
        self.status = "RUNNING"
        self.gpu = physical_gpu
        self.error = None
        signal.signal(signal.SIGALRM, self._alarm)
        signal.setitimer(signal.ITIMER_REAL, PHASE_CAPS[self.phase]-10)
        self.save()

    @staticmethod
    def _alarm(signum, frame):
        raise TimeoutError("EXP-025 phase watchdog")

    def totals(self):
        return dict(self.counts)

    def save(self):
        write(self.path, {"task": "EXP-025/v1", "status": self.status,
              "physical_gpu": self.gpu, "pid": os.getpid(), "phase": self.phase,
              "elapsed_seconds": time.monotonic()-self.start,
              "phase_elapsed_seconds": time.monotonic()-self.phase_start,
              "counts": self.totals(), "caps": CAPS,
              "phase_seconds": self.phase_seconds, "phase_counts": self.phase_counts,
              "phase_caps": PHASE_CAPS, "gpu_seconds_cap": 13800,
              "peak_allocated_gib": torch.cuda.max_memory_allocated()/2**30,
              "peak_reserved_gib": torch.cuda.max_memory_reserved()/2**30,
              "rss_gib": rss_gib(), "events": self.events[-300:],
              "error": self.error})

    def gate(self):
        if time.monotonic()-self.start >= 13800-30:
            raise TimeoutError("total GPU-second cap")
        if time.monotonic()-self.phase_start >= PHASE_CAPS[self.phase]-30:
            raise TimeoutError(f"{self.phase} GPU-second cap")
        if self.phase == "train" and time.monotonic()-self.start >= 13800-1400-30:
            raise TimeoutError("post-evaluation reserve")
        if torch.cuda.max_memory_allocated()/2**30 >= 44:
            raise MemoryError("allocated VRAM >=44 GiB")
        if rss_gib() >= 128:
            raise MemoryError("process RSS >=128 GiB")

    def reserve(self, kind, note):
        self.gate()
        if self.counts[kind] >= CAPS[kind]:
            raise RuntimeError(f"{kind} cap reached")
        self.counts[kind] += 1
        self.events.append({"event": "attempt", "kind": kind, "note": note,
                            "seconds": time.monotonic()-self.start})
        self.save()

    def measure(self, name, fn):
        torch.cuda.synchronize()
        started = time.monotonic()
        value = fn()
        torch.cuda.synchronize()
        self.gate()
        self.events.append({"event": "complete", "note": name,
                            "duration_seconds": time.monotonic()-started,
                            "seconds": time.monotonic()-self.start})
        self.save()
        print("TIMING", name, round(time.monotonic()-started, 3), flush=True)
        return value

    def next_phase(self, phase):
        self.phase_seconds[self.phase] = time.monotonic()-self.phase_start
        self.phase_counts[self.phase] = dict(self.counts)
        self.phase = phase
        self.phase_start = time.monotonic()
        signal.setitimer(signal.ITIMER_REAL, PHASE_CAPS[phase]-10)
        self.save()
        print("PHASE", phase, self.totals(), flush=True)

    def close(self, status, error=None):
        signal.setitimer(signal.ITIMER_REAL, 0)
        self.status = status
        self.error = repr(error) if error is not None else None
        self.phase_seconds[self.phase] = time.monotonic()-self.phase_start
        self.phase_counts[self.phase] = dict(self.counts)
        self.save()


def install(pipe_model):
    from causal.pretrained_lora import install_adapters
    adapters, indices = install_adapters(pipe_model, rank=8,
                                         block_indices=tuple(range(50)), device="cuda:0")
    params = [p for a in adapters for p in (a.lora_A, a.lora_B)]
    if indices != list(range(50)) or sum(p.numel() for p in params) != 10752000:
        raise RuntimeError("wrong full50 adapter")
    trainable = {id(p) for p in params}
    if any(p.requires_grad for p in pipe_model.parameters() if id(p) not in trainable):
        raise RuntimeError("frozen H3/action LoRA is trainable")
    if any(torch.count_nonzero(a.lora_B) for a in adapters):
        raise RuntimeError("adapter not zero-initialized")
    return adapters, params, trainable


def regression(model, adapters, ledger):
    from causal.h3_cached import H3ChunkCache
    from interval_train import interval_train
    data = old.load_clip(json.loads((EXP23 / "SELECTION.json").read_text())["train_clip_ids"][0])
    z = data["z"]
    noise = data["f"]["initial_noise"].float().to("cuda:0")
    outputs = []
    digests = []
    for enabled in (False, True):
        for a in adapters:
            a.enabled = enabled
        cache = H3ChunkCache(5, "cpu")
        row = []
        for index, sigma, current, commit in ((0, .5, .5*z[:,:,:12]+.5*noise[:,:,:12], False),
                                              (0, 0., z[:,:,:12], True),
                                              (1, .5, .5*z[:,:,12:17]+.5*noise[:,:,12:17], False)):
            stop, start = (12, 0) if index == 0 else (17, 12)
            packed, prompt = data["vis"][stop]
            ledger.reserve("dit", f"regression_{enabled}_C{index+1}_{sigma}")
            def call():
                with torch.no_grad(), old.verified_attention_scope():
                    return interval_train(model, current, start=start, index=index,
                        cache=cache, full_packed=packed, prompt=prompt,
                        anchor=data["anchor"], audio=data["audio"],
                        sigma=sigma, commit=commit)
            row.append(ledger.measure("regression_forward", call).detach().cpu())
        outputs.append(row)
        digests.append(cov.cache_integrity(cache)[1])
        del cache
        gc.collect()
    for a in adapters:
        a.enabled = True
    errors = [float((x.float()-y.float()).abs().max())
              for x, y in zip(outputs[0], outputs[1])]
    passed = all(x == 0.0 for x in errors) and digests[0] == digests[1]
    write(HERE / "ZERO_ADAPTER_REGRESSION.json", {
        "status": "PASS" if passed else "FAIL", "max_abs_by_call": errors,
        "full_KV_sha256_disabled_enabled": digests, "dit_attempts": 6,
        "precision": "h3_fp32"})
    if not passed:
        raise RuntimeError("zero adapter regression failed")
    del outputs, data
    gc.collect()


def eval_probe(model, data, budget, label):
    clean1 = data["z"][:,:,:12]
    clean2 = data["z"][:,:,12:17]
    noise1 = data["f"]["initial_noise"][:,:,:12].float().to("cuda:0")
    noise2 = data["f"]["initial_noise"][:,:,12:17].float().to("cuda:0")
    output = {"C1": [], "C2": []}
    from causal.h3_cached import H3ChunkCache
    empty = H3ChunkCache(5, "cpu")
    for sigma in (.25, .75):
        state = (1-sigma)*clean1 + sigma*noise1
        pred = old.velocity(model, data, empty, 0, state, sigma, budget,
                            f"{label}_C1_{sigma}")
        output["C1"].append({"sigma": sigma,
            "fm_mse": float(torch.nn.functional.mse_loss(pred.float(), noise1-clean1)),
            "velocity_rms": float(pred.float().square().mean().sqrt())})
    output["C2"] = ORIGINAL_C2_PROBE(model, data, budget, label+"_C2")
    return output


def run_evaluation(pipe, model, ledger, phase):
    # Frozen EXP-023 sampling/decoding/media implementation, with EXP-025
    # output root and explicit checkpoint identity. It consumes 90 DiT/6 decode.
    old.HERE = HERE
    old.probe = eval_probe
    def identity(data, role, _, history):
        return {"task": "EXP-025/v1", "clip_id": data["f"]["clip_id"],
                "role": role, "history": history, "checkpoint_updates":
                0 if phase == "pre" else 256, "nfe_per_chunk": 8,
                "seed": 13,
                "condition_sha256": data["record"]["condition"]["sha256"],
                "source_video_sha256": data["record"]["source_video_sha256"],
                "initial_noise_sha256": old.tensor_sha(data["f"]["initial_noise"])}
    old.video_identity = identity
    rows, probes = old.evaluate(pipe, model, ledger,
                                "eval_pre" if phase == "pre" else "eval_post")
    write(HERE / f"EVAL_{phase.upper()}_SUMMARY.json", {
        "phase": phase, "checkpoint_updates": 0 if phase == "pre" else 256,
        "videos": rows, "probes": probes,
        "budget": ledger.totals()})


def train(model, adapters, params, trainable, optimizer, order, ledger, noise_gen):
    import current_prefix as route
    source = route.current_prefix_attend
    updates = []
    for position, (clip, block) in enumerate(order):
        step = position + 1
        data0 = old.load_clip(clip)
        f = data0["f"]
        data = {"z": data0["z"],
                "full_packed": {k: v.to("cuda:0") if torch.is_tensor(v) else v
                                for k, v in f["packed"].items()},
                "prompt": f["prompt"].to("cuda:0"),
                "anchor": data0["anchor"], "audio": data0["audio"]}
        _, start, stop = cov.BLOCKS[block]
        clean = data["z"][:,:,start:stop]
        noise = torch.randn(clean.shape, generator=noise_gen,
                            dtype=torch.float32).to("cuda:0")
        sigma = float(torch.rand((), generator=noise_gen))
        sigma = max(1e-6, min(1-1e-6, sigma))
        phase = ["idle"]
        trace = []
        def audit(control, *args, **kwargs):
            if (control.action_prefix_mode != "own" or not control.action_feedback
                    or control.index not in (0, 1)):
                raise RuntimeError("action route changed")
            trace.append((phase[0], control.index, int(kwargs["layer"])))
            return source(control, *args, **kwargs)
        def stage(name, when):
            kind = {"commit": "dit", "forward": "dit",
                    "backward": "backward", "optimizer": "update"}[name]
            if when == "before":
                ledger.reserve(kind, f"step{step}_{block}_{name}")
            else:
                ledger.gate()
        def gate(_, cache):
            proof = profile.validate_block_gate(block, trace, model, trainable,
                                                adapters if step <= 2 else None)
            if step <= 2:
                proof["step"] = step
                proof["block"] = block
                proof["cpu_kv_bytes"] = cache.nbytes
                write(HERE / f"FIRST_{block}_GRADIENT_GATE.json", proof)
        row = cov.fm_step(model, data, optimizer, block=block, noise=noise,
                          sigma=sigma, checkpoint=True, offload=True,
                          audit=audit, phase_hook=lambda name: phase.__setitem__(0, name),
                          stage_callback=stage, pre_optimizer_gate=gate)
        row.update(step=step, clip_id=clip,
                   peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                   rss_gib=rss_gib())
        updates.append(row)
        write(HERE / "TRAIN_UPDATES.json", updates)
        print("UPDATE", step, block, clip, row["loss"], row["grad_l2"], flush=True)
        if step in (64, 128, 256):
            checkpoint(adapters, optimizer, step, order, step, noise_gen)
        del data0, data, clean, noise
        gc.collect()
    return updates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--gpu", type=int)
    args = parser.parse_args()
    order = preflight(require_gpu=not args.cpu_only)
    if args.cpu_only:
        cpu_checkpoint_lifecycle()
        print("CPU_PREFLIGHT_PASS", len(order), flush=True)
        return
    if not (HERE / "CPU_CHECKPOINT_LIFECYCLE.json").exists():
        raise RuntimeError("CPU checkpoint lifecycle preflight required")
    if args.gpu is None or os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.gpu):
        raise RuntimeError("physical GPU selector mismatch")
    import run as prior
    prior.setup()
    torch.set_num_threads(4)
    torch.manual_seed(13)
    torch.cuda.manual_seed_all(13)
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats()
    ledger = Ledger(args.gpu)
    try:
        pipe, model, precision = ledger.measure("model_load", prior.model)
        adapters, params, trainable = install(model)
        optimizer = torch.optim.AdamW(params, lr=1e-4, betas=(.9,.95),
                                      weight_decay=.01)
        noise_gen = torch.Generator(device="cpu").manual_seed(130025)
        checkpoint(adapters, optimizer, 0, order, 0, noise_gen)
        regression(model, adapters, ledger)
        ledger.next_phase("pre")
        run_evaluation(pipe, model, ledger, "pre")
        ledger.next_phase("train")
        train(model, adapters, params, trainable, optimizer, order, ledger, noise_gen)
        ledger.next_phase("post")
        run_evaluation(pipe, model, ledger, "post")
        ledger.close("COMPLETE_PENDING_JUDGE")
        print("EXP025_COMPLETE", ledger.totals(), repr(precision), flush=True)
    except BaseException as error:
        ledger.close("FAILED_STOPPED", error)
        print("EXP025_STOPPED", repr(error), flush=True)
        raise


if __name__ == "__main__":
    main()
