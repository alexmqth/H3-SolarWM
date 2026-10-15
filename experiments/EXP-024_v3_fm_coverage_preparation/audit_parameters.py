"""CPU-only EXP-023 adapter delta and true QKV-shape capacity audit."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OLD = ROOT / "submission/experiments/EXP-023_v3_fm_small_data"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def low_rank_fro(a, b, scale):
    # ||B A||_F^2 = tr((B^T B)(A A^T)); never materialize 21,504x5,376.
    aa = a.double() @ a.double().T
    bb = b.double().T @ b.double()
    return float(torch.clamp(torch.sum(aa * bb.T), min=0).sqrt() * scale)


def main():
    files = [OLD / "checkpoint_000.pt", OLD / "checkpoint_128.pt"]
    meta = [json.loads(p.with_suffix(".json").read_text()) for p in files]
    for p, m in zip(files, meta):
        assert digest(p) == m["sha256"]
    zero, final = [torch.load(p, map_location="cpu", weights_only=True) for p in files]
    assert zero["step"] == 0 and final["step"] == 128
    assert len(zero["adapter"]) == len(final["adapter"]) == 8
    rows = []
    for i, (old, new) in enumerate(zip(zero["adapter"], final["adapter"])):
        a0, b0 = old["A"], old["B"]
        a1, b1 = new["A"], new["B"]
        assert a0.shape == a1.shape == (8, 5376)
        assert b0.shape == b1.shape == (21504, 8)
        assert not torch.count_nonzero(b0)
        assert all(torch.isfinite(x).all() for x in (a0, b0, a1, b1))
        scale = 1.0 / a1.shape[0]
        effect = low_rank_fro(a1, b1, scale)
        rows.append({"layer": 42 + i, "rank": 8,
                     "A_delta_fro": float(torch.linalg.vector_norm((a1-a0).double())),
                     "B_delta_fro": float(torch.linalg.vector_norm((b1-b0).double())),
                     "A_final_fro": float(torch.linalg.vector_norm(a1.double())),
                     "B_final_fro": float(torch.linalg.vector_norm(b1.double())),
                     "delta_weight_fro": effect,
                     "unit_isotropic_input_output_rms": effect / 5376**0.5,
                     "nonzero_B": int(torch.count_nonzero(b1)),
                     "note": "weight-space proxy only; no H3 hidden-state/output effect measured"})
    in_dim, out_dim, layers = 5376, 21504, 50
    variants = {}
    for name, count, rank in (("tail8_rank8", 8, 8),
                              ("full50_rank8", layers, 8),
                              ("full50_rank32", layers, 32)):
        params = count * rank * (in_dim + out_dim)
        variants[name] = {"layers": count, "rank": rank,
                          "trainable_parameters": params,
                          "parameter_bytes_fp32": params*4,
                          "gradient_bytes_fp32": params*4,
                          "adamw_m_v_bytes_fp32": params*8,
                          "static_train_bytes_sum": params*16,
                          "checkpoint_adapter_plus_m_v_bytes": params*12,
                          "excludes": ["base/released-action weights", "activations",
                                       "checkpoint recomputation", "CPU raw KV",
                                       "allocator fragmentation", "optimizer temporaries"]}
    assert variants["tail8_rank8"]["trainable_parameters"] == 1720320
    out = {"status": "CPU_STATIC_AUDIT", "task": "EXP-024/v1",
           "source_checkpoint_sha256": {p.name: digest(p) for p in files},
           "source_module": "causal.pretrained_lora.CausalQKVLoRA",
           "module_shape": {"in_features": in_dim, "out_features": out_dim,
                            "blocks": layers, "scale": "1/rank"},
           "per_layer": rows, "variants": variants,
           "observed_exp023_peak_allocated_gib": 24.539616107940674,
           "observed_exp023_cpu_raw_kv_bytes": 6799104000,
           "caution": "Static parameter cost cannot predict full-layer backward VRAM/time or video quality."}
    path = HERE / "PARAMETER_AUDIT.json"
    if path.exists():
        assert json.loads(path.read_text()) == out
    else:
        path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"layers": len(rows), "variants": variants,
                      "max_effect_fro": max(x["delta_weight_fro"] for x in rows)},
                     indent=2))


if __name__ == "__main__":
    main()
