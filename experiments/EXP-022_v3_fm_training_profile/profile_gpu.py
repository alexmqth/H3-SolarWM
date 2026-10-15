"""Authorized single-GPU V3 ordinary-FM equivalence and two-update profile."""
from __future__ import annotations

import gc
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
EXP019 = ROOT / "submission/experiments/EXP-019_v3_mixed_action_validation"
sys.path.insert(0, str(EXP019))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".partial")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def rss_bytes():
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024
    return -1


def tree_cpu(value):
    import torch
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: tree_cpu(v) for k, v in value.items()}
    if isinstance(value, list):
        return [tree_cpu(v) for v in value]
    return value


def diff(a, b):
    import torch
    delta = a.float() - b.float()
    return {"exact": bool(torch.equal(a, b)), "max_abs": float(delta.abs().max()),
            "relative_rms": float(delta.square().mean().sqrt() /
                                  a.float().square().mean().sqrt().clamp_min(1e-12))}


class Ledger:
    def __init__(self, gpu):
        self.started = time.monotonic()
        self.gpu = gpu
        self.path = HERE / "GPU_BUDGET.json"
        if self.path.exists():
            raise FileExistsError("GPU run already exists; no automatic retry")
        self.counts = {"top_level_forward": 0, "backward": 0, "update": 0,
                       "vae": 0, "text": 0, "sampling": 0}
        self.events = []
        self.status = "RUNNING"
        self.save()

    def save(self, **other):
        import torch
        self.d = {"task": "EXP-022/v1", "status": self.status,
                  "gpu_physical": self.gpu, "elapsed_gpu_process_seconds": time.monotonic() - self.started,
                  "counts": dict(self.counts), "limits": {"top_level_forward": 20,
                      "backward": 2, "update": 2, "vae": 0, "text": 0, "sampling": 0,
                      "wall_seconds": 3600, "peak_allocated_gib": 44},
                  "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                  "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
                  "rss_gib": rss_bytes() / 2**30, "events": self.events, **other}
        write(self.path, self.d)

    def reserve(self, kind, note):
        if time.monotonic() - self.started >= 3500:
            raise TimeoutError("GPU wall hard stop before 3600s")
        cap = {"top_level_forward": 20, "backward": 2, "update": 2}[kind]
        if self.counts[kind] >= cap:
            raise RuntimeError(f"{kind} cap")
        self.counts[kind] += 1
        self.events.append({"kind": kind, "note": note, "reserved_at_seconds": time.monotonic() - self.started})
        self.save()

    def check(self, name):
        import torch
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() / 2**30
        if peak > 44:
            raise MemoryError(f"allocated peak {peak:.3f}GiB exceeds 44")
        if time.monotonic() - self.started >= 3500:
            raise TimeoutError("GPU wall hard stop")
        self.save(last_stage=name)

    def close(self, status, error=None):
        self.status = status
        self.save(error=error)


def timed(ledger, name, fn):
    import torch
    torch.cuda.synchronize()
    start = time.monotonic()
    result = fn()
    torch.cuda.synchronize()
    seconds = time.monotonic() - start
    ledger.events.append({"kind": "timing", "name": name, "seconds": seconds,
                          "elapsed_at_end": time.monotonic() - ledger.started,
                          "allocated_gib": torch.cuda.memory_allocated() / 2**30,
                          "rss_gib": rss_bytes() / 2**30})
    ledger.check(name)
    print(f"TIMING {name} {seconds:.3f}s", flush=True)
    return result


def precheck(gpu):
    import torch
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(gpu):
        raise RuntimeError("physical GPU selector does not match CUDA_VISIBLE_DEVICES")
    if not (HERE / "CPU_PREFLIGHT.json").is_file():
        raise RuntimeError("CPU preflight required")
    auth = json.loads((HERE / "AUTHORIZATION.json").read_text())
    assert auth["approved"] and auth["limits"]["gpus"] == 1
    assert sha(HERE / "taskbook_v1.md") == auth["taskbook_sha256"]
    info = subprocess.check_output(["nvidia-smi", f"--id={gpu}",
        "--query-gpu=name,uuid,memory.free", "--format=csv,noheader,nounits"], text=True).strip()
    if int(info.split(",")[-1]) < 44000:
        raise RuntimeError(f"selected GPU not free: {info}")
    torch.cuda.set_device(0)
    torch.manual_seed(13)
    torch.cuda.manual_seed_all(13)
    torch.cuda.reset_peak_memory_stats()
    return info


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    args = parser.parse_args()
    import run as prior
    prior.setup()
    import torch
    from causal.h3_cached import H3ChunkCache
    from causal.local_topology import visible_inputs
    from causal.pretrained_lora import install_adapters
    from current_prefix import current_prefix_feedback
    from interval_cached import interval_cached
    from interval_train import interval_train

    gpu_info = precheck(args.gpu)
    ledger = Ledger(args.gpu)
    signal.alarm(3550)
    try:
        print("GPU START", gpu_info, flush=True)
        source = json.loads((HERE / "CPU_PREFLIGHT.json").read_text())
        assert sha(ROOT / source["fixture"]["path"]) == source["fixture"]["sha256"]
        f = torch.load(ROOT / source["fixture"]["path"], map_location="cpu", weights_only=True)
        clean = torch.load(ROOT / "H3-World/outputs/EXP-016_v3_real56_vae/b784d995/full17.pt",
                           map_location="cpu", weights_only=True).float()
        past = clean[:, :, :12].to("cuda:0")
        clean_c2 = clean[:, :, 12:17].to("cuda:0")
        noise = f["initial_noise"][:, :, 12:17].float().to("cuda:0")
        state = .5 * clean_c2 + .5 * noise
        target = noise - clean_c2
        assert torch.isfinite(state).all() and torch.isfinite(target).all()
        packed = prior.move(f["packed"])
        prompt = f["prompt"].to("cuda:0")
        vis = {s: visible_inputs(packed, prompt, s, 390) for s in (12, 17)}
        anchor = f["anchor"].to("cuda:0")
        audio = f["audio_noise"].to("cuda:0")
        pipe, model, precision = timed(ledger, "load_model", prior.model)
        ledger.save(gpu_info=gpu_info, precision=precision,
                    fixture_sha256=source["fixture"]["sha256"])

        def reference(x, cache, index, sigma, commit=False):
            stop, start = (12, 0) if index == 0 else (17, 12)
            p, pr = vis[stop]
            ledger.reserve("top_level_forward", f"reference index={index} sigma={sigma} commit={commit}")
            with torch.no_grad(), current_prefix_feedback():
                return interval_cached(model, x, start=start, index=index, cache=cache,
                    full_packed=p, prompt=pr, anchor=anchor, audio=audio,
                    sigma=sigma, commit=commit)

        def candidate(x, cache, index, sigma, commit=False, grad=False):
            stop, start = (12, 0) if index == 0 else (17, 12)
            p, pr = vis[stop]
            ledger.reserve("top_level_forward", f"new index={index} sigma={sigma} commit={commit} grad={grad}")
            with torch.enable_grad() if grad else torch.no_grad(), current_prefix_feedback():
                return interval_train(model, x, start=start, index=index, cache=cache,
                    full_packed=p, prompt=pr, anchor=anchor, audio=audio,
                    sigma=sigma, commit=commit, checkpoint=grad, offload=grad)

        ref_cache = H3ChunkCache(5, "cpu")
        timed(ledger, "reference_clean_commit", lambda: reference(past, ref_cache, 0, 0., True))
        ref_velocity = timed(ledger, "reference_c2_velocity", lambda: reference(state, ref_cache, 1, .5)).detach().cpu()
        adapters, indices = install_adapters(model, rank=8,
                                              block_indices=tuple(range(len(model.blocks)-8, len(model.blocks))),
                                              device="cuda:0")
        trainable = [p for a in adapters for p in (a.lora_A, a.lora_B)]
        nparam = sum(p.numel() for p in trainable)
        assert indices == list(range(42, 50)) and nparam == 1720320
        assert all(p.requires_grad for p in trainable)
        assert all(not p.requires_grad for name,p in model.named_parameters()
                   if not (name.endswith("lora_A") or name.endswith("lora_B")))
        assert all(bool(torch.count_nonzero(a.lora_B) == 0) for a in adapters)
        ledger.save(adapter={"indices": indices, "rank": 8, "parameters": nparam,
                             "initial_lora_B_zero": True})
        metadata = {"task": "EXP-022/v1", "objective": "ordinary_fm", "seed": 13,
                    "precision": "h3_fp32", "partition": [12, 5], "adapter": "tail8_qkv_rank8",
                    "source_fixture_sha256": source["fixture"]["sha256"]}
        checkpoint0 = HERE / "checkpoint0.pt"
        if checkpoint0.exists():
            raise FileExistsError(checkpoint0)
        timed(ledger, "save_checkpoint0", lambda: torch.save({
            "adapter": [tree_cpu({"A": a.lora_A, "B": a.lora_B}) for a in adapters],
            "metadata": metadata, "step": 0, "cpu_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state()}, checkpoint0))
        new_cache = H3ChunkCache(5, "cpu")
        timed(ledger, "new_clean_commit", lambda: candidate(past, new_cache, 0, 0., True))
        cache_diffs = []
        for layer in range(50):
            r = ref_cache.history(layer, 1)[0]
            n = new_cache.history(layer, 1)[0]
            for field in ("key", "value", "rope"):
                cache_diffs.append({"layer": layer, "tensor": field,
                                    **diff(getattr(r, field), getattr(n, field))})
        write(HERE / "CACHE_EQUIVALENCE.json", cache_diffs)
        if not all(x["exact"] for x in cache_diffs):
            raise RuntimeError("checkpoint0 KV mismatch; stop before training")
        new_velocity = timed(ledger, "new_c2_velocity", lambda: candidate(state, new_cache, 1, .5)).detach().cpu()
        velocity_diff = diff(ref_velocity, new_velocity)
        write(HERE / "FORWARD_EQUIVALENCE.json", {"sigma": .5, "state": "same GT C1 and same C2 noisy latent",
              "velocity": velocity_diff, "cache_tensors": len(cache_diffs), "cache_exact": True,
              "dtype": str(new_velocity.dtype), "backend": "torch SDPA/current_prefix"})
        print("EQUIVALENCE", velocity_diff, flush=True)
        if not velocity_diff["exact"]:
            raise RuntimeError("checkpoint0 velocity mismatch; stop before training")
        del ref_velocity, new_velocity, ref_cache, new_cache
        gc.collect()

        optimizer = torch.optim.AdamW(trainable, lr=1e-4, betas=(.9, .95), weight_decay=.01)
        updates = []
        for step, sigma in enumerate((.5, .75), 1):
            clean_cache = H3ChunkCache(5, "cpu")
            timed(ledger, f"step{step}_clean_commit", lambda: candidate(past, clean_cache, 0, 0., True))
            z = (1-sigma) * clean_c2 + sigma * noise
            optimizer.zero_grad(set_to_none=True)
            before = [p.detach().clone() for p in trainable]
            velocity = timed(ledger, f"step{step}_forward", lambda: candidate(z, clean_cache, 1, sigma, grad=True))
            loss = torch.nn.functional.mse_loss(velocity.float(), target.float())
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite FM loss")
            ledger.reserve("backward", f"step{step}")
            timed(ledger, f"step{step}_backward", loss.backward)
            grads = [p.grad for p in trainable]
            if any(g is None or not torch.isfinite(g).all() for g in grads):
                raise FloatingPointError("missing/nonfinite adapter gradient")
            grad_norm = float(torch.sqrt(sum(g.float().square().sum() for g in grads)))
            if grad_norm <= 0:
                raise FloatingPointError("zero adapter gradient")
            if any(p.grad is not None for name,p in model.named_parameters()
                   if not (name.endswith("lora_A") or name.endswith("lora_B"))):
                raise RuntimeError("frozen base/action LoRA received gradient")
            ledger.reserve("update", f"step{step}")
            timed(ledger, f"step{step}_optimizer", optimizer.step)
            change = float(torch.sqrt(sum((p.detach().float()-b.float()).square().sum()
                                          for p,b in zip(trainable, before))))
            if not change > 0:
                raise FloatingPointError("adapter did not change")
            row = {"step": step, "sigma": sigma, "loss": float(loss.detach()),
                   "grad_l2": grad_norm, "parameter_delta_l2": change,
                   "cpu_kv_bytes": clean_cache.nbytes,
                   "peak_allocated_gib": torch.cuda.max_memory_allocated()/2**30,
                   "peak_reserved_gib": torch.cuda.max_memory_reserved()/2**30,
                   "rss_gib": rss_bytes()/2**30}
            updates.append(row)
            write(HERE / "UPDATES.json", updates)
            print("UPDATE", row, flush=True)
            del velocity, loss, grads, before, clean_cache
            gc.collect()

        # A fresh post-update clean commit is required: the prior cache was
        # produced before the second optimizer step and is now stale.
        final_cache = H3ChunkCache(5, "cpu")
        timed(ledger, "final_clean_commit", lambda: candidate(past, final_cache, 0, 0., True))
        final_state = .5 * clean_c2 + .5 * noise
        final_before = timed(ledger, "pre_reload_velocity", lambda: candidate(final_state, final_cache, 1, .5)).detach().cpu()
        checkpoint = HERE / "step02.pt"
        if checkpoint.exists():
            raise FileExistsError(checkpoint)
        state = {"adapter": [tree_cpu({"A": a.lora_A, "B": a.lora_B}) for a in adapters],
                 "optimizer": tree_cpu(optimizer.state_dict()), "metadata": metadata,
                 "step": 2, "cpu_rng": torch.get_rng_state(),
                 "cuda_rng": torch.cuda.get_rng_state()}
        timed(ledger, "save_step02", lambda: torch.save(state, checkpoint))
        expected_random = torch.rand(4, device="cuda:0")
        with torch.no_grad():
            adapters[0].lora_B[0, 0].add_(.25)
        read = timed(ledger, "reload_step02", lambda: torch.load(checkpoint, map_location="cpu", weights_only=True))
        assert read["metadata"] == metadata and read["step"] == 2
        with torch.no_grad():
            for a, saved in zip(adapters, read["adapter"]):
                a.lora_A.copy_(saved["A"].to(a.lora_A.device))
                a.lora_B.copy_(saved["B"].to(a.lora_B.device))
        optimizer.load_state_dict(read["optimizer"])
        torch.set_rng_state(read["cpu_rng"])
        torch.cuda.set_rng_state(read["cuda_rng"])
        random_exact = bool(torch.equal(torch.rand(4, device="cuda:0"), expected_random))
        final_after = timed(ledger, "post_reload_velocity", lambda: candidate(final_state, final_cache, 1, .5)).detach().cpu()
        reload_diff = diff(final_before, final_after)
        write(HERE / "RELOAD_CHECK.json", {"velocity": reload_diff, "cuda_rng_exact": random_exact,
             "optimizer_state_items": len(optimizer.state_dict()["state"]),
             "checkpoint_sha256": sha(checkpoint), "checkpoint_bytes": checkpoint.stat().st_size,
             "checkpoint0_sha256": sha(checkpoint0), "checkpoint0_bytes": checkpoint0.stat().st_size})
        if not reload_diff["exact"] or not random_exact:
            raise RuntimeError("checkpoint reload mismatch")
        result = {"status": "COMPLETE_PENDING_JUDGE", "equivalence": velocity_diff,
                  "cache_exact": True, "updates": updates, "reload": reload_diff,
                  "cuda_rng_exact": random_exact, "precision": precision,
                  "checkpoint": str(checkpoint.relative_to(ROOT)),
                  "counts": dict(ledger.counts), "gpu_wall_seconds": time.monotonic()-ledger.started,
                  "peak_allocated_gib": torch.cuda.max_memory_allocated()/2**30,
                  "peak_reserved_gib": torch.cuda.max_memory_reserved()/2**30,
                  "rss_gib": rss_bytes()/2**30,
                  "scope": "engineering profile; no quality or action evaluation"}
        write(HERE / "GPU_RESULT.json", result)
        ledger.close("COMPLETE_PENDING_JUDGE")
        print("GPU PROFILE COMPLETE", json.dumps({k:result[k] for k in ("counts","gpu_wall_seconds","peak_allocated_gib")}), flush=True)
    except BaseException as exc:
        ledger.close("FAILED_STOPPED", repr(exc))
        print("GPU PROFILE STOPPED", repr(exc), flush=True)
        raise
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    main()
