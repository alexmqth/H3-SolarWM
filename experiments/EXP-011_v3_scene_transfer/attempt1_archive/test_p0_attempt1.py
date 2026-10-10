"""EXP-011 CPU-only P0 checks; no model weights or GPU access."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile

import common
from source_audit import inspect_sources


def main():
    import torch
    from run_exp011 import cache_signature, tensor_sha
    assert inspect_sources() == json.loads((common.HERE / "source_manifest.json").read_text())
    common.verify_sources()
    common.verify_code_manifest()
    # Native noise is bf16; denoising input is cast to fp32. Audit must use
    # the same dtype on both sides of a tensor hash comparison.
    native = torch.arange(16, dtype=torch.bfloat16).reshape(1, 1, 1, 4, 4)
    current = native.to(torch.float32).clone()
    assert tensor_sha(current) == tensor_sha(native.to(torch.float32))
    assert tensor_sha(current) != tensor_sha(native)
    class Entry:
        def __init__(self):
            self.index = 0
            self.key = torch.zeros(2, 1)
            self.value = torch.zeros(2, 1)
            self.rope = torch.zeros(2, 1)
    class Cache:
        commits = 1
        layers = {i: [Entry()] for i in range(50)}
    cache = Cache()
    signature = cache_signature(cache)
    cache.layers[0][0].key.add_(1)
    assert cache_signature(cache) != signature
    assert all(row[0].index == 0 for row in cache.layers.values())
    original_out = common.OUT
    with tempfile.TemporaryDirectory(prefix="exp011_p0_") as temp:
        common.OUT = Path(temp)
        ledger = common.Ledger(0, "P1")
        for _ in range(6):
            ledger.reserve("text_encoder_calls")
        for _ in range(2):
            ledger.reserve("image_vae_encodes")
        for kind in ("text_encoder_calls", "image_vae_encodes", "video_vae_encodes",
                     "sampling_forwards", "vae_decodes"):
            try:
                ledger.reserve(kind)
            except RuntimeError:
                pass
            else:
                raise AssertionError(f"P1 ledger accepted forbidden/over-budget {kind}")
        ledger.close("cpu_test")
    common.OUT = original_out
    print(json.dumps({"task": "EXP-011/v1", "cpu_p0": "PASS",
                      "provenance": "PASS", "code_manifest": "PASS",
                      "bf16_to_fp32_noise_hash": "PASS", "cache_mutation_detection": "PASS",
                      "p1_budget_rejection": "PASS", "gpu_calls": 0}))


if __name__ == "__main__":
    main()
