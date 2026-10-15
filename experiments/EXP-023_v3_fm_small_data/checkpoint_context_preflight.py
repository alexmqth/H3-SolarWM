"""CPU check that non-reentrant checkpoint recomputation sees the live route.

This is a structural test. The first real H3 training backward separately
instruments the actual attention function and is the GPU protocol gate.
"""
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
from causal.h3_cached import ChunkAttention
from current_prefix import current_prefix_feedback


def main():
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        raise RuntimeError("CPU-only test")
    initial = ChunkAttention.attend
    trace = []
    x = torch.tensor(2.0, requires_grad=True)

    def layer(t):
        active = ChunkAttention.attend.__module__ == "current_prefix"
        trace.append({"active": active, "grad_enabled": torch.is_grad_enabled()})
        return t.square() if active else 3 * t.square()

    # Mirrors the intended EXP-023 training scope: forward, loss and backward
    # all live inside the same route context.
    with current_prefix_feedback():
        y = checkpoint(layer, x, use_reentrant=False)
        y.backward()
    assert ChunkAttention.attend is initial
    assert len(trace) >= 2 and all(t["active"] for t in trace), trace
    assert float(x.grad) == 4.0, x.grad
    result = {"status": "PASS_CPU_CHECKPOINT_CONTEXT", "layer_calls": trace,
              "gradient": float(x.grad), "attention_restored": True,
              "scope": "dynamic route scope only; real H3 backward hook remains mandatory"}
    path = HERE / "CHECKPOINT_CONTEXT_CPU.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
