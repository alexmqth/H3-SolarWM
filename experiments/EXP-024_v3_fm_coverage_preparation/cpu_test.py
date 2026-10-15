"""Small-tensor CPU test of the EXP-024 production C1/C2 training boundary."""
from __future__ import annotations

import io
import json

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

import coverage as cov
import current_prefix as route
from causal.h3_cached import ChunkAttention
from causal.pretrained_lora import CausalQKVLoRA
from diffsynth.pipelines import minimax_h3_audio_video as pipe_module


def toy_data():
    torch.manual_seed(13)
    prefix = 18  # one common row + 17 one-token action spans
    length = prefix + 17
    text_pos = torch.arange(1, prefix)
    full = {
        "action_video_start": prefix, "seq_len": length,
        "action_text_rows": torch.tensor([(i+1, i+2) for i in range(17)]),
        "action_text_spans_local": [(i, i+1) for i in range(17)],
        "text_pos": text_pos, "audio_pos": torch.empty(0, dtype=torch.long),
        "img_pos": torch.tensor([0] + list(range(prefix, length))),
        "img_position_ids": torch.arange(length).view(1, length, 1).repeat(1, 1, 3),
        "token_tags": torch.zeros(length, dtype=torch.long),
        "cu_seqlens": torch.tensor([0, length], dtype=torch.int32),
    }
    return {"z": torch.randn(1, 1, 17, 2, 2), "full_packed": full,
            "prompt": torch.randn(17, 4), "anchor": torch.ones(1, 4),
            "audio": torch.zeros(1), "cpu_toy": True}


class ToyDiT(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = CausalQKVLoRA(nn.Linear(4, 12, bias=False), rank=2)
        self._h3_input_dtype = torch.float32


def fake_model_fn(dit, current, audio, packed, prompt, **kwargs):
    control = kwargs["causal_control"]
    t = current.shape[2]
    p = int(packed["action_video_start"])
    assert int(packed["seq_len"]) == p+t
    prefix = torch.zeros((p, 4), dtype=current.dtype)
    prefix[0] = kwargs["keyframe_cond_anchor"][0]
    prefix[packed["text_pos"]] = prompt
    x = current.mean(dim=(0, 1, 3, 4)).unsqueeze(-1) * torch.tensor([1., .5, -.25, .75])
    tokens = torch.cat((prefix, x), dim=0)

    def attend_layer(features):
        q, k, v = [part.unsqueeze(1) for part in dit.qkv(features).chunk(3, dim=-1)]
        rope = torch.zeros((len(features), 1, 1), dtype=features.dtype)
        result = control.attend(q, k, v, rope_freqs=rope, layer=0,
                                apply_rope=lambda value, _: value, scale=.5)
        return result[-t:].mean(dim=(1, 2)).view(1, 1, t, 1, 1).expand_as(current)

    if kwargs.get("use_gradient_checkpointing"):
        return (checkpoint(attend_layer, tokens, use_reentrant=False),)
    return (attend_layer(tokens),)


def make_model():
    torch.manual_seed(91)
    model = ToyDiT()
    optimizer = torch.optim.AdamW((model.qkv.lora_A, model.qkv.lora_B),
                                  lr=1e-3, betas=(.9, .95), weight_decay=.01)
    return model, optimizer


def main():
    torch.set_num_threads(4)
    data = toy_data()
    order = cov.balanced_block_order(["clip_a", "clip_b"], seed=13, epochs=2)
    assert len(order) == 8
    assert all(order.count((clip, block)) == 2 for clip in ("clip_a", "clip_b")
               for block in ("C1", "C2"))
    clean_c1, clean_c2 = data["z"][:, :, :12], data["z"][:, :, 12:17]
    noise_c1 = torch.randn(clean_c1.shape, generator=torch.Generator().manual_seed(21))
    noise_c2 = torch.randn(clean_c2.shape, generator=torch.Generator().manual_seed(22))
    c1_packed, c1_prompt = cov.prepare_visible(data, "C1")
    c2_packed, _ = cov.prepare_visible(data, "C2")
    assert len(c1_packed["action_text_rows"]) == 12
    assert len(c2_packed["action_text_rows"]) == 17
    assert c1_packed["seq_len"] == c1_packed["action_video_start"]+12
    assert c2_packed["seq_len"] == c2_packed["action_video_start"]+17
    altered = dict(data)
    altered["prompt"] = data["prompt"].clone()
    altered["prompt"][12:] += 1000
    altered["z"] = data["z"].clone()
    altered["z"][:, :, 12:] += 1000
    c1_altered, p_altered = cov.prepare_visible(altered, "C1")
    assert torch.equal(c1_prompt, p_altered)
    assert torch.equal(c1_packed["img_position_ids"], c1_altered["img_position_ids"])
    assert torch.equal(data["z"][:, :, :12], altered["z"][:, :, :12])

    original_fn = pipe_module.model_fn_minimax_h3
    original_route = ChunkAttention.attend
    original_attend = route.current_prefix_attend
    pipe_module.model_fn_minimax_h3 = fake_model_fn
    model, optimizer = make_model()
    phase = ["idle"]
    trace = []

    def audit(control, *args, **kwargs):
        assert ChunkAttention.attend.__module__ == "current_prefix"
        assert control.action_prefix_mode == "own" and control.action_feedback
        index = control.index
        assert len(control.action_rows) == (12 if index == 0 else 17)
        assert control.frame_start == (0 if index == 0 else 12)
        history = sum(e.key.shape[0] for e in control.cache.history(kwargs["layer"], index))
        _, vm = control.masks(args[0].shape[0]-control.prefix, history, args[0].device)
        assert bool(vm[0, 0])
        assert bool(vm[0, int(control.action_rows[control.frame_start, 0])])
        if index == 1:
            assert not bool(vm[0, int(control.action_rows[11, 0])])
        trace.append({"phase": phase[0], "index": index,
                      "history_rows": history, "route": "current_prefix"})
        return original_attend(control, *args, **kwargs)

    try:
        c1 = cov.fm_step(model, data, optimizer, block="C1", noise=noise_c1,
                         sigma=.25, audit=audit, phase_hook=lambda x: phase.__setitem__(0, x))
        assert c1["cpu_kv_bytes"] == 0
        c2 = cov.fm_step(model, data, optimizer, block="C2", noise=noise_c2,
                         sigma=.75, audit=audit, phase_hook=lambda x: phase.__setitem__(0, x))
        assert c2["cpu_kv_bytes"] > 0
        assert any(t["phase"] == "backward" and t["index"] == 0 for t in trace)
        assert any(t["phase"] == "backward" and t["index"] == 1 for t in trace)
        assert any(t["phase"] == "commit" and t["index"] == 0 for t in trace)
        assert all(t["history_rows"] == 0 for t in trace if t["index"] == 0)
        assert all(t["history_rows"] == 12 for t in trace if t["index"] == 1)
        assert all(p.grad is None for p in model.qkv.base.parameters())

        # Save/reload adapter, optimizer and RNG state, then repeat a fixed step.
        saved = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "rng": torch.get_rng_state(), "step": 2}
        stream = io.BytesIO()
        torch.save(saved, stream)
        stream.seek(0)
        restored = torch.load(stream, map_location="cpu", weights_only=True)
        model2, optimizer2 = make_model()
        model2.load_state_dict(restored["model"])
        optimizer2.load_state_dict(restored["optimizer"])
        assert restored["step"] == 2
        assert torch.equal(restored["rng"], saved["rng"])
        for p1, p2 in zip(model.parameters(), model2.parameters()):
            assert torch.equal(p1, p2)
        repeat1 = cov.fm_step(model, data, optimizer, block="C1", noise=noise_c1,
                              sigma=.5, audit=audit, phase_hook=lambda x: phase.__setitem__(0, x))
        repeat2 = cov.fm_step(model2, data, optimizer2, block="C1", noise=noise_c1,
                              sigma=.5, audit=audit, phase_hook=lambda x: phase.__setitem__(0, x))
        assert repeat1["loss"] == repeat2["loss"]
        for p1, p2 in zip(model.parameters(), model2.parameters()):
            assert torch.equal(p1, p2)
    finally:
        pipe_module.model_fn_minimax_h3 = original_fn
    assert ChunkAttention.attend is original_route
    assert route.current_prefix_attend is original_attend
    report = {"status": "PASS_CPU_TOY_PRODUCTION_BOUNDARY", "gpu_calls": 0,
              "real_h3_backward_tested": False, "dtype": "float32", "backend": "CPU_SDPA",
              "C1": c1, "C2": c2, "trace": trace,
              "future_action_video_perturbation_invisible_to_C1": True,
              "balanced_sampler_two_epochs_exact": True,
              "frozen_base_grad_absent": True, "save_restore_optimizer_rng_exact": True,
              "repeat_update_exact": True,
              "attention_scope_restored": True,
              "note": "Mock DiT model_fn; actual interval_train, visible_inputs, ChunkAttention and verified_attention_scope used."}
    path = cov.ROOT / "submission/experiments/EXP-024_v3_fm_coverage_preparation/CPU_TEST_RESULTS.json"
    if path.exists():
        previous = json.loads(path.read_text())
        assert all(report[k] == value for k, value in previous.items())
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "C1", "C2",
                                              "repeat_update_exact")}, indent=2))


if __name__ == "__main__":
    main()
