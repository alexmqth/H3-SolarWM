"""EXP-007/v2: one real H3 target-time initialization and finite-map update.

Never loads old AnyFlow weights. Every model call/backward/update is reserved
in an atomic ledger before execution; a failed call still consumes budget.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import fcntl
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
CFG = json.loads((HERE / "config_v2.json").read_text())
OUT = ROOT / CFG["output_root"]
FROZEN = ROOT / "H3-World/outputs/2026-10-09-22/chunk_partition_cb"
EXP005 = HERE.parent / "EXP-005_v3_sliding_window"
ROUTER = ROOT / "submission/reports/stage1_anyflow/02_causal_diagnostics/current_prefix_candidate"
CONTRACT = HERE.parent / "EXP-001_v2b_124"
EXP002 = HERE.parent / "EXP-002_native_cached"
SOURCES = {
    "first12": ROOT / "submission/experiments/11_causal_12_then5_selfhistory/states/first12_A.pt",
    "second5": ROOT / "H3-World/outputs/EXP-002_native_cached/AA/chunk_12_17.pt",
    "input_A": FROZEN / "source_coarse/inputs/parking_A.pt",
    "input_D": FROZEN / "source_coarse/inputs/parking_D.pt",
    "released_action_lora": FROZEN / "runtime/checkpoints/H3-World/step-10000.safetensors",
    "interval_student": HERE / "interval_student.py",
    "taskbook": HERE / "taskbook_v2.md",
    "config": HERE / "config_v2.json",
    "runner": HERE / "run_af1.py",
    "anyflow": FROZEN / "runtime/code/causal/anyflow.py",
    "pretrained_lora": FROZEN / "runtime/code/causal/pretrained_lora.py",
    "precision": FROZEN / "runtime/code/causal/h3_precision.py",
    "interval_sw": EXP005 / "interval_sw.py",
}
EXPECTED_SOURCE = {
    "first12": "242a1db06bc3423fe21207af5d2ccf59c9f27f07c9d88f22ce9f77921f6b0eea",
    "released_action_lora": "ddd9187b920b1e52c2d090f4e264fd83d8d433efc2a5b159e58883aeaf96e526",
}


def sha(path: Path | str) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(str(path) + ".partial")
    partial.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(partial, path)


def setup_paths() -> None:
    sys.path[:0] = [str(FROZEN / "runtime/code"),
                    str(FROZEN / "runtime/code/abot"),
                    str(FROZEN / "runtime/DiffSynth-Studio-h3-v2"),
                    str(EXP005), str(EXP005 / "stage2"), str(ROUTER),
                    str(CONTRACT), str(EXP002), str(HERE)]


def source_manifest() -> dict:
    entries = {name: {"path": str(path.relative_to(ROOT)), "sha256": sha(path)}
               for name, path in SOURCES.items()}
    for name, expected in EXPECTED_SOURCE.items():
        assert entries[name]["sha256"] == expected, name
    frozen = HERE / "source_manifest_v2.json"
    if frozen.exists():
        if json.loads(frozen.read_text()) != {"task": CFG["task"], "sources": entries}:
            raise RuntimeError("source/config changed after manifest freeze")
    else:
        atomic_json(frozen, {"task": CFG["task"], "sources": entries})
    return entries


def preflight() -> None:
    assert CFG["max_forwards"] == 22 and CFG["max_backward"] == 4 and CFG["max_updates"] == 1
    assert CFG["rank"] == CFG["tail_blocks"] == 8
    assert CFG["max_vae"] == 0 and CFG["source_action"] == "AA"
    af0 = json.loads((HERE / "cpu_preflight.json").read_text())
    assert af0["status"] == "CPU_PASS"
    entries = source_manifest()
    import torch
    first = torch.load(SOURCES["first12"], map_location="cpu", weights_only=True)
    second = torch.load(SOURCES["second5"], map_location="cpu", weights_only=True)
    assert first.shape == (1, 24, 12, 30, 52) and second.shape == (1, 24, 5, 30, 52)
    assert torch.isfinite(first).all() and torch.isfinite(second).all()
    print(json.dumps({"status": "CPU_PASS", "task": CFG["task"],
                      "source_count": len(entries), "first_shape": list(first.shape),
                      "second_shape": list(second.shape)}), flush=True)


def run(gpu: int, ledger: dict, manifest: dict) -> None:
    import torch
    import infer as abot
    from causal.anyflow import (install_anyflow, save_anyflow, logical_time_pairs,
                                anyflow_sample_loss, adaptive_scale)
    from causal.pretrained_lora import install_adapters, save_adapters
    from causal.h3_cached import H3ChunkCache
    from causal.h3_precision import configure_precision
    from causal.local_topology import visible_inputs
    from current_prefix import current_prefix_feedback
    from interval_student import interval_student
    from interval_sw import interval_sw
    from rollout_contract import prompt_for_path

    torch.set_num_threads(4)
    device = "cuda:0"
    torch.manual_seed(CFG["seed"])
    result = dict(task=CFG["task"], status="loading", gpu=gpu,
                  started=time.time(), runner_sha256=sha(__file__),
                  manifest_sha256=sha(HERE / "source_manifest_v2.json"),
                  source=manifest, torch=torch.__version__,
                  cuda_backend=str(torch.backends.cuda.sdp_kernel))
    atomic_json(OUT / "result.json", result)
    deadline = datetime.fromisoformat(CFG["deadline_hkt"]).timestamp()
    def reserve(kind: str) -> None:
        if time.time() >= deadline or time.time() - ledger["start"] >= CFG["max_gpu_hours"] * 3600:
            raise TimeoutError("AF1 absolute GPU-time deadline")
        key, maximum = {"forward": ("forwards", 22), "backward": ("backward", 4),
                        "update": ("updates", 1)}[kind]
        if ledger[key] >= maximum:
            raise RuntimeError(f"AF1 {kind} budget exhausted")
        ledger[key] += 1
        atomic_json(OUT / "budget.json", ledger)

    def move(value):
        if torch.is_tensor(value): return value.to(device)
        if isinstance(value, dict): return {k: move(v) for k, v in value.items()}
        return value

    try:
        inputs = {a: torch.load(SOURCES[f"input_{a}"], map_location="cpu", weights_only=True)
                  for a in "AD"}
        for name in ("initial_noise", "audio_noise", "anchor"):
            assert torch.equal(inputs["A"][name], inputs["D"][name]), name
        first = torch.load(SOURCES["first12"], map_location="cpu", weights_only=True).to(device, torch.float32)
        second = torch.load(SOURCES["second5"], map_location="cpu", weights_only=True).to(device, torch.float32)
        cond = inputs["A"]
        full = move(cond["packed"])
        prompt12 = prompt_for_path(inputs, "AA", 12).to(device)
        prompt17 = prompt_for_path(inputs, "AA", 17).to(device)
        layout12, prompt12 = visible_inputs(full, prompt12, 12, 390)
        layout17, prompt17 = visible_inputs(full, prompt17, 17, 390)
        assert len(layout17["action_text_rows"]) == 17
        anchor, audio = cond["anchor"].to(device), cond["audio_noise"].to(device)
        pipe = abot.load_pipeline(device)
        pipe.load_lora(pipe.dit, state_dict=abot.load_checkpoint_lora(SOURCES["released_action_lora"]), hotload=True)
        model = pipe.dit.eval().requires_grad_(False)
        result["precision"] = configure_precision(model, "h3_fp32",
            native_transformer_dir=FROZEN / "runtime/DiffSynth-Studio-h3-v2/models/MiniMax/MiniMax-H3/FL2VA/transformer")
        pipe.load_models_to_device(["dit"])
        base_versions = [(name, p, p._version) for name, p in model.named_parameters()]
        result["base_parameter_count"] = len(base_versions)
        atomic_json(OUT / "result.json", result)

        def call(function, cache, index, sample, sigma, target_sigma=None,
                 commit=False, gradient=False):
            reserve("forward")
            kwargs = dict(start=(0 if index == 0 else 12), index=index,
                cache=cache, full_packed=(layout12 if index == 0 else layout17),
                prompt=(prompt12 if index == 0 else prompt17), anchor=anchor, audio=audio,
                sigma=sigma, commit=commit)
            if function is interval_student:
                kwargs.update(target_sigma=target_sigma,
                              use_gradient_checkpointing=gradient,
                              use_gradient_checkpointing_offload=gradient)
            output = function(model, sample, **kwargs)
            if not torch.isfinite(output).all():
                raise FloatingPointError("AF1 model output is nonfinite")
            if torch.cuda.max_memory_allocated()/2**30 > CFG["max_allocated_gib"]:
                raise MemoryError("AF1 allocated VRAM cap exceeded")
            return output

        with current_prefix_feedback():
            baseline_cache = H3ChunkCache(5, "cpu")
            with torch.no_grad():
                call(interval_sw, baseline_cache, 0, first, 0, commit=True)
                fixed_generator = torch.Generator().manual_seed(CFG["seed"])
                fixed_noise = torch.randn(second.shape, generator=fixed_generator, dtype=torch.float32).to(device)
                fixed_noisy = (0.4 * second + 0.6 * fixed_noise).float()
                baseline = call(interval_sw, baseline_cache, 1, fixed_noisy, 0.6).float().detach()
            result["baseline_velocity_rms"] = float(baseline.square().mean().sqrt())
            baseline_cache.clear()

            adapters, indices = install_adapters(model, rank=8,
                block_indices=tuple(range(len(model.blocks)-8, len(model.blocks))), device=device)
            target = install_anyflow(model, device=device, gate=CFG["target_gate"])
            params = [p for a in adapters for p in (a.lora_A, a.lora_B)] + list(target.parameters())
            assert {id(p) for p in model.parameters() if p.requires_grad} == {id(p) for p in params}
            result["trainable_names"] = [name for name, p in model.named_parameters() if p.requires_grad]
            result["trainable_count"] = sum(p.numel() for p in params)
            result["qkv_indices"] = indices
            result["target_gate"] = target.gate
            student_cache = H3ChunkCache(5, "cpu")
            with torch.no_grad():
                call(interval_student, student_cache, 0, first, 0, target_sigma=0, commit=True)
                diagonal = call(interval_student, student_cache, 1, fixed_noisy, .6, target_sigma=.6).float()
            delta = diagonal - baseline
            result["diagonal_max_abs"] = float(delta.abs().max())
            result["diagonal_relative_rms"] = float(delta.square().mean().sqrt() /
                                                     baseline.square().mean().sqrt().clamp_min(1e-8))
            result["diagonal_allclose_1e5"] = bool(torch.allclose(diagonal, baseline, rtol=1e-5, atol=1e-5))
            atomic_json(OUT / "result.json", result)
            if not result["diagonal_allclose_1e5"]:
                raise AssertionError("AF1 initialized diagonal regression failed")
            with torch.no_grad():
                finite = call(interval_student, student_cache, 1, fixed_noisy, .6, target_sigma=.2)
            result["off_diagonal_finite"] = bool(torch.isfinite(finite).all())
            result["off_diagonal_relative_rms"] = float((finite.float()-diagonal).square().mean().sqrt() /
                                                        diagonal.square().mean().sqrt().clamp_min(1e-8))
            del finite, diagonal, baseline, delta
            if not result["off_diagonal_finite"]:
                raise FloatingPointError("nonfinite initialized off-diagonal output")

            def save_checkpoint(directory: Path, optimizer, step: int, generator):
                directory.mkdir(parents=True, exist_ok=False)
                metadata = dict(task=CFG["task"], step=step,
                                config_sha256=sha(HERE / "config_v2.json"),
                                manifest_sha256=sha(HERE / "source_manifest_v2.json"),
                                protocol="V3 strict causal/global/current-prefix/Single I0/12+5/sigma0 student KV",
                                precision="h3_fp32", source_kind="generated V3 FM30 C1/C2")
                save_adapters(directory / "qkv.pt", adapters, indices, metadata)
                save_anyflow(directory / "target_time.pt", target, metadata)
                state = dict(task=CFG["task"], step=step,
                             optimizer=move_optimizer_to_cpu(optimizer.state_dict()),
                             logical_rng_state=generator.get_state(),
                             cpu_rng_state=torch.get_rng_state(),
                             cuda_rng_state=torch.cuda.get_rng_state(device),
                             qkv_sha256=sha(directory / "qkv.pt"),
                             target_sha256=sha(directory / "target_time.pt"),
                             metadata=metadata)
                torch.save(state, directory / "trainer_state.pt")
                return {p.name: sha(p) for p in directory.iterdir() if p.is_file()}

            def move_optimizer_to_cpu(value):
                if torch.is_tensor(value): return value.detach().cpu()
                if isinstance(value, dict): return {k: move_optimizer_to_cpu(v) for k, v in value.items()}
                if isinstance(value, list): return [move_optimizer_to_cpu(v) for v in value]
                return value

            optimizer = torch.optim.AdamW(params, lr=CFG["lr"], betas=(.9,.95), weight_decay=.01)
            generator = torch.Generator().manual_seed(CFG["seed"])
            result["initial_checkpoint"] = save_checkpoint(OUT / "step_00", optimizer, 0, generator)
            before = [p.detach().clone() for p in params]
            # Fixed student history for the entire logical batch; no graph through KV.
            from chunk_plan import cache_identity
            cache_before = cache_identity(student_cache)
            pairs = logical_time_pairs(generator, shift=CFG["sample_shift"], batch_size=4)
            diffusion = []
            samples = []
            optimizer.zero_grad(set_to_none=True)
            for i in range(4):
                noise = torch.randn(second.shape, generator=generator, dtype=torch.float32).to(device)
                t, r = float(pairs.t[i]), float(pairs.r[i])
                def velocity(x, current_sigma, target_sigma):
                    grad = torch.is_grad_enabled()
                    return call(interval_student, student_cache, 1, x, current_sigma,
                                target_sigma=target_sigma, gradient=grad)
                raw, weight, item = anyflow_sample_loss(velocity, second, noise, t, r,
                    shift=CFG["weight_shift"], epsilon=CFG["epsilon"], preserve_fp32_inputs=True)
                is_diffusion = bool(pairs.is_diffusion[i])
                scale = adaptive_scale(raw, is_diffusion=is_diffusion, diffusion_losses=diffusion)
                if is_diffusion: diffusion.append(float(raw.detach()))
                value = raw * weight * scale / 4
                if not torch.isfinite(value): raise FloatingPointError("nonfinite AF1 sample loss")
                reserve("backward")
                value.backward()
                item.update(sample_type=("diffusion" if is_diffusion else
                           "endpoint" if int(pairs.sample_type[i]) == 1 else "flow_map"),
                           adaptive_scale=float(scale), weighted_loss=float(value.detach())*4)
                samples.append(item)
                result["samples"] = samples
                atomic_json(OUT / "result.json", result)
                print(f"AF1 sample {i+1}/4: t={t:.6f} r={r:.6f} raw={item['raw_loss']:.6f}", flush=True)
                del raw, value, noise
            assert cache_identity(student_cache) == cache_before
            assert all(not e.key.requires_grad and not e.value.requires_grad
                       for entries in student_cache.layers.values() for e in entries)
            grads = [p.grad for p in params]
            assert all(g is not None and torch.isfinite(g).all() for g in grads)
            result["grad_norm_target"] = float(torch.linalg.vector_norm(torch.stack(
                [g.float().norm() for g in grads[-len(list(target.parameters())):]])))
            result["grad_norm_qkv"] = float(torch.linalg.vector_norm(torch.stack(
                [g.float().norm() for g in grads[:-len(list(target.parameters()))]])))
            if not result["grad_norm_target"] > 0 or not result["grad_norm_qkv"] > 0:
                raise AssertionError("missing target/QKV gradient")
            result["grad_clip_pre_norm"] = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
            reserve("update")
            optimizer.step()
            result["qkv_parameters_changed"] = any(not torch.equal(p, b) for p, b in zip(params[:-len(list(target.parameters()))], before))
            result["target_parameters_changed"] = any(not torch.equal(p, b) for p, b in zip(params[-len(list(target.parameters())):], before[-len(list(target.parameters())):]))
            assert result["qkv_parameters_changed"] and result["target_parameters_changed"]
            result["step1_checkpoint"] = save_checkpoint(OUT / "step_01", optimizer, 1, generator)
            fresh = H3ChunkCache(5, "cpu")
            with torch.no_grad():
                call(interval_student, fresh, 0, first, 0, target_sigma=0, commit=True)
            result["recomputed_KV_differs"] = any(
                not torch.equal(old.key, new.key)
                for layer in fresh.layers for old, new in zip(student_cache.layers[layer], fresh.layers[layer]))
            assert result["recomputed_KV_differs"]
            student_cache.clear(); fresh.clear()
            result["frozen_parameter_versions_unchanged"] = all(p._version == version for _, p, version in base_versions)
            assert result["frozen_parameter_versions_unchanged"]
        result["peak_allocated_gib"] = torch.cuda.max_memory_allocated()/2**30
        result["peak_reserved_gib"] = torch.cuda.max_memory_reserved()/2**30
        assert result["peak_allocated_gib"] <= CFG["max_allocated_gib"]
        result["status"] = "complete_pending_judge"
        result["seconds"] = time.time() - result["started"]
        atomic_json(OUT / "result.json", result)
    except BaseException as exc:
        result["status"] = "failed"
        result["error"] = repr(exc)
        result["seconds"] = time.time() - result["started"]
        atomic_json(OUT / "result.json", result)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--gpu", type=int, default=CFG["gpu"])
    args = parser.parse_args()
    setup_paths()
    manifest = source_manifest()
    if args.preflight:
        preflight(); return
    free = int(subprocess.check_output(["nvidia-smi", f"--id={args.gpu}",
        "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True).strip())
    if free < 44000:
        raise RuntimeError(f"GPU {args.gpu} not sufficiently idle: {free} MiB")
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "worker.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (OUT / "budget.json").exists():
            raise RuntimeError("AF1 prior GPU attempt exists; no automatic retry")
        ledger = dict(task=CFG["task"], start=time.time(), gpu=args.gpu,
                      forwards=0, backward=0, updates=0, vae=0)
        atomic_json(OUT / "budget.json", ledger)
        deadline = min(datetime.fromisoformat(CFG["deadline_hkt"]).timestamp(),
                       ledger["start"] + CFG["max_gpu_hours"]*3600)
        def timeout(_signum, _frame):
            raise TimeoutError("AF1 absolute alarm deadline")
        signal.signal(signal.SIGALRM, timeout)
        signal.setitimer(signal.ITIMER_REAL, deadline - time.time())
        os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu), ABOT_VRAM_RESERVE_GIB="18",
            HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", DIFFSYNTH_SKIP_DOWNLOAD="True",
            TOKENIZERS_PARALLELISM="false", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
        try:
            run(args.gpu, ledger, manifest)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            ledger["end"] = time.time(); ledger["gpu_seconds"] = ledger["end"] - ledger["start"]
            atomic_json(OUT / "budget.json", ledger)


if __name__ == "__main__":
    main()
