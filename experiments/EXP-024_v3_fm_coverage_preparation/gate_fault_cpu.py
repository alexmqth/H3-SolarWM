"""CPU fault injection: same-byte-size KV content and route gates reject before update."""
from __future__ import annotations

import json
from pathlib import Path

import torch

import coverage as cov
import cpu_test as basic
import full_layer_lifecycle_cpu as full
import profile_gpu as profile
import current_prefix as route
from diffsynth.pipelines import minimax_h3_audio_video as pipe_module

HERE = Path(__file__).resolve().parent


def spy_optimizer(optimizer):
    called = [0]
    original = optimizer.step

    def counted(*args, **kwargs):
        called[0] += 1
        return original(*args, **kwargs)

    optimizer.step = counted
    return called


def same_size_kv_mutation():
    data = basic.toy_data()
    model, optimizer = basic.make_model()
    calls = spy_optimizer(optimizer)
    before = [p.detach().clone() for p in model.parameters()]
    phase = ["idle"]
    changed = [False]
    legacy_same = [False]
    old = route.current_prefix_attend
    noise = torch.randn(data["z"][:, :, 12:].shape,
                        generator=torch.Generator().manual_seed(31))

    def audit(control, *args, **kwargs):
        if (phase[0] == "backward" and control.index == 1
                and kwargs["layer"] == 0 and not changed[0]):
            cache = control.cache
            legacy_before = cov.cache_signature(cache)
            byte_count = cache.nbytes
            with torch.no_grad():
                cache.layers[0][0].key.flatten()[20] += .125
            legacy_same[0] = legacy_before == cov.cache_signature(cache)
            assert byte_count == cache.nbytes
            changed[0] = True
        return old(control, *args, **kwargs)

    old_model_fn = pipe_module.model_fn_minimax_h3
    pipe_module.model_fn_minimax_h3 = basic.fake_model_fn
    error = None
    try:
        try:
            cov.fm_step(model, data, optimizer, block="C2", noise=noise,
                        sigma=.5, audit=audit,
                        phase_hook=lambda x: phase.__setitem__(0, x))
        except RuntimeError as exc:
            error = str(exc)
    finally:
        pipe_module.model_fn_minimax_h3 = old_model_fn
    assert changed[0] and legacy_same[0]
    assert error == "training forward/backward mutated history KV content or structure"
    assert calls[0] == 0
    assert all(torch.equal(p, q) for p, q in zip(model.parameters(), before))
    assert not optimizer.state
    return {"injected": "KV key element 20 changed in-place during backward replay before attention",
            "legacy_exp023_cache_signature_unchanged": legacy_same[0],
            "raw_kv_byte_size_unchanged": True,
            "full_payload_digest_rejected": True,
            "optimizer_step_calls": calls[0], "error": error}


def missing_route_layer():
    data = basic.toy_data()
    torch.manual_seed(15)
    model = full.Tiny50()
    params = [p for a in model.blocks for p in (a.lora_A, a.lora_B)]
    optimizer = torch.optim.AdamW(params, lr=1e-3)
    calls = spy_optimizer(optimizer)
    before = [p.detach().clone() for p in model.parameters()]
    trainable_ids = {id(p) for p in params}
    phase = ["idle"]
    trace = []
    attempted = []
    old = route.current_prefix_attend
    noise = torch.randn(data["z"][:, :, :12].shape,
                        generator=torch.Generator().manual_seed(32))

    def audit(control, *args, **kwargs):
        trace.append((phase[0], control.index, int(kwargs["layer"])))
        return old(control, *args, **kwargs)

    old_model_fn = pipe_module.model_fn_minimax_h3
    pipe_module.model_fn_minimax_h3 = full.fake50
    error = None
    try:
        for block in ("C1", "C2"):
            attempted.append(block)
            try:
                cov.fm_step(model, data, optimizer, block=block, noise=noise,
                            sigma=.5, audit=audit,
                            phase_hook=lambda x: phase.__setitem__(0, x),
                            pre_optimizer_gate=lambda b, cache:
                                profile.validate_block_gate(
                                    b, [x for x in trace if x[2] != 49],
                                    model, trainable_ids))
            except RuntimeError as exc:
                error = str(exc)
                break
    finally:
        pipe_module.model_fn_minimax_h3 = old_model_fn
    assert attempted == ["C1"]
    assert "pre-optimizer route gate: missing layer in" in error
    assert calls[0] == 0
    assert all(torch.equal(p, q) for p, q in zip(model.parameters(), before))
    assert not optimizer.state
    return {"injected": "remove layer49 from route gate evidence",
            "attempted_blocks": attempted, "completed_updates": 0,
            "optimizer_step_calls": calls[0], "error": error}


def frozen_gradient_injection():
    data = basic.toy_data()
    torch.manual_seed(15)
    model = full.Tiny50()
    params = [p for a in model.blocks for p in (a.lora_A, a.lora_B)]
    optimizer = torch.optim.AdamW(params, lr=1e-3)
    calls = spy_optimizer(optimizer)
    before = [p.detach().clone() for p in model.parameters()]
    trainable_ids = {id(p) for p in params}
    phase = ["idle"]
    trace = []
    old = route.current_prefix_attend
    noise = torch.randn(data["z"][:, :, :12].shape,
                        generator=torch.Generator().manual_seed(33))

    def audit(control, *args, **kwargs):
        trace.append((phase[0], control.index, int(kwargs["layer"])))
        return old(control, *args, **kwargs)

    def poison_then_gate(block, cache):
        frozen = model.blocks[0].base.weight
        frozen.grad = torch.ones_like(frozen)
        profile.validate_block_gate(block, trace, model, trainable_ids)

    old_model_fn = pipe_module.model_fn_minimax_h3
    pipe_module.model_fn_minimax_h3 = full.fake50
    error = None
    try:
        try:
            cov.fm_step(model, data, optimizer, block="C1", noise=noise,
                        sigma=.5, audit=audit,
                        phase_hook=lambda x: phase.__setitem__(0, x),
                        pre_optimizer_gate=poison_then_gate)
        except RuntimeError as exc:
            error = str(exc)
    finally:
        pipe_module.model_fn_minimax_h3 = old_model_fn
    assert error == "pre-optimizer frozen base/action parameter has gradient"
    assert calls[0] == 0
    assert all(torch.equal(p, q) for p, q in zip(model.parameters(), before))
    assert not optimizer.state
    return {"injected": "frozen base weight.grad set immediately before gate",
            "optimizer_step_calls": calls[0], "error": error}


def main():
    torch.set_num_threads(4)
    result = {"status": "PASS_CPU_PRE_OPTIMIZER_FAULT_GATES", "gpu_calls": 0,
              "same_size_kv_mutation": same_size_kv_mutation(),
              "missing_route_layer": missing_route_layer(),
              "frozen_gradient_injection": frozen_gradient_injection(),
              "actual_33b_backward_tested": False}
    path = HERE / "GATE_FAULT_CPU.json"
    if path.exists():
        previous = json.loads(path.read_text())
        assert previous["status"] == result["status"]
        assert previous["same_size_kv_mutation"]["full_payload_digest_rejected"]
        assert previous["missing_route_layer"]["optimizer_step_calls"] == 0
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
