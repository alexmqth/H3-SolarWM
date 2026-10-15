"""Tiny 50-layer checkpoint recompute lifecycle test; no 33B/GPU weights."""
from __future__ import annotations

import json
from torch import nn
import torch
from torch.utils.checkpoint import checkpoint

import coverage as cov
import cpu_test as basic
import profile_gpu as profile
import current_prefix as route
from causal.h3_cached import ChunkAttention
from causal.pretrained_lora import CausalQKVLoRA
from diffsynth.pipelines import minimax_h3_audio_video as pipe_module


class Tiny50(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([CausalQKVLoRA(nn.Linear(4, 12, bias=False), rank=2)
                                     for _ in range(50)])
        self._h3_input_dtype = torch.float32


def fake50(dit, current, audio, packed, prompt, **kwargs):
    control = kwargs["causal_control"]
    t = current.shape[2]
    p = int(packed["action_video_start"])
    prefix = torch.zeros((p, 4), dtype=current.dtype)
    prefix[0] = kwargs["keyframe_cond_anchor"][0]
    prefix[packed["text_pos"]] = prompt
    x = current.mean(dim=(0, 1, 3, 4)).unsqueeze(-1) * torch.tensor([1., .5, -.25, .75])
    tokens = torch.cat((prefix, x), dim=0)
    total = torch.zeros_like(tokens)
    for layer, adapter in enumerate(dit.blocks):
        def one(features, layer=layer, adapter=adapter):
            q, k, v = [z.unsqueeze(1) for z in adapter(features).chunk(3, dim=-1)]
            rope = torch.zeros((len(features), 1, 1), dtype=features.dtype)
            return control.attend(q, k, v, rope_freqs=rope, layer=layer,
                                  apply_rope=lambda z, _: z, scale=.5).squeeze(1)
        if kwargs.get("use_gradient_checkpointing"):
            out = checkpoint(one, tokens, use_reentrant=False)
        else:
            out = one(tokens)
        total = total + out * .02
        tokens = tokens + out * .01
    prediction = total[-t:].mean(dim=-1).view(1, 1, t, 1, 1).expand_as(current)
    return (prediction,)


def main():
    torch.set_num_threads(4)
    torch.manual_seed(15)
    data = basic.toy_data()
    model = Tiny50()
    params = [p for a in model.blocks for p in (a.lora_A, a.lora_B)]
    optimizer = torch.optim.AdamW(params, lr=1e-3)
    trainable_ids = {id(p) for p in params}
    original_fn = pipe_module.model_fn_minimax_h3
    original_class = ChunkAttention.attend
    original_attend = route.current_prefix_attend
    pipe_module.model_fn_minimax_h3 = fake50
    phase = ["idle"]
    trace = []
    gate_passed = []

    def audit(control, *args, **kwargs):
        trace.append((phase[0], control.index, int(kwargs["layer"])))
        return original_attend(control, *args, **kwargs)

    def gate(block, cache):
        evidence = profile.validate_block_gate(block, trace, model,
                                               trainable_ids, model.blocks)
        assert evidence["frozen_grad_absent"]
        assert len(evidence["grad_l2_by_layer"]) == 50
        gate_passed.append(block)

    try:
        z = data["z"]
        n1 = torch.randn(z[:, :, :12].shape, generator=torch.Generator().manual_seed(23))
        n2 = torch.randn(z[:, :, 12:].shape, generator=torch.Generator().manual_seed(24))
        one = cov.fm_step(model, data, optimizer, block="C1", noise=n1,
                          sigma=.35, audit=audit, phase_hook=lambda x: phase.__setitem__(0, x),
                          pre_optimizer_gate=gate)
        two = cov.fm_step(model, data, optimizer, block="C2", noise=n2,
                          sigma=.65, audit=audit, phase_hook=lambda x: phase.__setitem__(0, x),
                          pre_optimizer_gate=gate)
        assert two["cpu_kv_bytes"] > 0
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
    finally:
        pipe_module.model_fn_minimax_h3 = original_fn
    assert ChunkAttention.attend is original_class
    assert route.current_prefix_attend is original_attend
    phases = {}
    for tag, index, layer in trace:
        phases.setdefault(f"{tag}_C{index+1}", set()).add(layer)
    assert all(phases[k] == set(range(50)) for k in
               ("forward_C1", "backward_C1", "commit_C1", "forward_C2", "backward_C2"))
    report = {"status": "PASS_CPU_TINY_50_LAYER_RECOMPUTE", "gpu_calls": 0,
              "actual_h3_offload_tested": False, "actual_h3_activations_tested": False,
              "C1": one, "C2": two,
              "route_layers_by_phase": {k: sorted(v) for k, v in phases.items()},
              "all_50_backward_layers_keep_route": True,
              "pre_optimizer_gates_passed": gate_passed,
              "context_restored": True,
              "note": "Tiny fake DiT uses actual 50-layer ChunkAttention, interval_train, route scope and torch checkpoint; real model offload remains GPU-profile gate."}
    path = cov.ROOT / "submission/experiments/EXP-024_v3_fm_coverage_preparation/FULL_LAYER_CPU_RESULTS.json"
    if path.exists():
        previous = json.loads(path.read_text())
        assert all(report[k] == value for k, value in previous.items())
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "C1", "C2")}, indent=2))


if __name__ == "__main__":
    main()
