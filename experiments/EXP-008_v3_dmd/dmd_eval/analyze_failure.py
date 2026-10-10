"""CPU-only latent endpoint diagnostic for the stopped AA C2 evaluation."""
from __future__ import annotations

import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
INPUT = ROOT / "H3-World/outputs/2026-10-09-22/chunk_partition_cb/source_coarse/inputs/parking_A.pt"
FILES = {
    "FM8": ROOT / "H3-World/outputs/EXP-006_v3_fm8_full/AA/chunk_12_17.pt",
    "AF8": ROOT / "H3-World/outputs/EXP-007_v3_anyflow_af3/AA/chunk_12_17.pt",
    "DMD8_cycle8": ROOT / "H3-World/outputs/EXP-008_v3_dmd_eval/AA/chunk_12_17.pt",
}


def cosine(x: torch.Tensor, y: torch.Tensor) -> float:
    return float(torch.nn.functional.cosine_similarity(x.flatten(), y.flatten(), dim=0))


def main() -> None:
    source = torch.load(INPUT, map_location="cpu", weights_only=True)
    noise = source["initial_noise"][:, :, 12:17].float()
    values = {name: torch.load(path, map_location="cpu", weights_only=True).float()
              for name, path in FILES.items()}
    rows = {}
    for name, x in values.items():
        assert x.shape == noise.shape and bool(torch.isfinite(x).all())
        rows[name] = {
            "shape": list(x.shape), "mean": float(x.mean()), "std": float(x.std()),
            "cosine_to_initial_noise": cosine(x, noise),
            "rms_change_from_initial_noise": float((x - noise).square().mean().sqrt()),
            "horizontal_latent_total_variation": float((x[..., 1:] - x[..., :-1]).abs().mean()),
            "vertical_latent_total_variation": float((x[:, :, :, 1:, :] - x[:, :, :, :-1, :]).abs().mean()),
        }
    result = {
        "task": "EXP-008/v2-DMD8-EVAL-AA-failure",
        "scope": "CPU comparison of the same C2 initial noise and saved finite endpoints; no new model inference",
        "initial_noise_total_variation": {
            "horizontal": float((noise[..., 1:] - noise[..., :-1]).abs().mean()),
            "vertical": float((noise[:, :, :, 1:, :] - noise[:, :, :, :-1, :]).abs().mean()),
        },
        "endpoints": rows,
        "pair_cosine": {
            "FM8_AF8": cosine(values["FM8"], values["AF8"]),
            "FM8_DMD8": cosine(values["FM8"], values["DMD8_cycle8"]),
            "AF8_DMD8": cosine(values["AF8"], values["DMD8_cycle8"]),
        },
        "interpretation_limit": "DMD8 endpoint remains close to input noise; this supports denoising failure, not a unique cause for it.",
    }
    (HERE / "failure_latent_diagnostic.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
