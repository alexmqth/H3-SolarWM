"""Actual small-tensor H3 attention/checkpoint topology regression on CPU."""
from __future__ import annotations

import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
FROZEN = ROOT / "H3-World/outputs/2026-10-09-22/chunk_partition_cb/runtime"
sys.path[:0] = [str(FROZEN / "code"), str(FROZEN / "code/causal"),
                str(FROZEN / "DiffSynth-Studio-h3-v2"),
                str(ROOT / "submission/reports/stage1_anyflow/02_causal_diagnostics/current_prefix_candidate")]

import torch
from torch.utils.checkpoint import checkpoint
from causal.h3_cached import ChunkAttention, H3ChunkCache
from current_prefix import current_prefix_feedback


def trial(*, checkpointed: bool, old_scope: bool):
    torch.manual_seed(13)
    param = torch.nn.Parameter(torch.tensor(.71, dtype=torch.float64))
    q = torch.randn((4, 1, 4), dtype=torch.float64)
    k = torch.randn((4, 1, 4), dtype=torch.float64)
    base_v = torch.randn((4, 1, 4), dtype=torch.float64)
    video_only = torch.zeros_like(base_v)
    video_only[2:] = torch.tensor([[[.4, -.2, .8, .1]], [[-.3, .9, .2, -.1]]])
    rope = torch.zeros((4, 1, 1), dtype=torch.float64)
    control = ChunkAttention(H3ChunkCache(5, "cpu"), 0, 2, 2, 0,
                             torch.tensor([[1, 2]]), "own", True,
                             allow_grad_read=True)
    trace = []
    phase = "forward"

    def layer(p):
        route = ChunkAttention.attend.__module__
        trace.append({"phase": phase, "route": route})
        result = control.attend(q, k, base_v + p * video_only,
                                rope_freqs=rope, layer=0,
                                apply_rope=lambda x, _: x, scale=.5)
        return result[0].sum()

    def run_forward():
        if checkpointed:
            with torch.autograd.graph.save_on_cpu():
                return checkpoint(layer, param, use_reentrant=False)
        return layer(param)

    if old_scope:
        with current_prefix_feedback():
            loss = run_forward()
        phase = "backward"
        try:
            loss.backward()
            return {"output": float(loss.detach()), "grad": float(param.grad),
                    "trace": trace, "error": None}
        except Exception as error:
            return {"output": float(loss.detach()), "grad": None,
                    "trace": trace, "error": repr(error)}
    with current_prefix_feedback():
        loss = run_forward()
        phase = "backward"
        loss.backward()
    return {"output": float(loss.detach()), "grad": float(param.grad),
            "trace": trace, "error": None}


def main():
    direct = trial(checkpointed=False, old_scope=False)
    correct = trial(checkpointed=True, old_scope=False)
    wrong = trial(checkpointed=True, old_scope=True)
    assert len(correct["trace"]) >= 2 and any(x["phase"] == "backward" for x in correct["trace"])
    assert all(x["route"] == "current_prefix" for x in correct["trace"])
    assert abs(direct["grad"]) > .01
    assert abs(direct["output"] - correct["output"]) < 1e-10
    assert abs(direct["grad"] - correct["grad"]) < 1e-10
    assert any(x["phase"] == "backward" and x["route"] != "current_prefix"
               for x in wrong["trace"])
    assert wrong["error"] or wrong["grad"] is None or abs(wrong["grad"]-correct["grad"]) > 1e-6
    result = {"status": "PASS_CPU_ACTUAL_ATTENTION_GRADIENT", "dtype": "float64",
              "tolerance_abs": 1e-10, "direct": direct, "correct_checkpoint_offload": correct,
              "old_forward_only_negative_control": wrong,
              "meaning": "common-prefix output depends on current-video trainable value; checkpointed backward must retain route"}
    path = HERE / "GRADIENT_PROTOCOL_CPU.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
