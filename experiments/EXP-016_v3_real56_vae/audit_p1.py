"""Independent CPU-only tensor/video verification and boundary contact sheets."""
from __future__ import annotations

import json
import os
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw
import torch

from common import HERE, OUT, CFG, cpu_affinity, sha, write


def video(path: Path, width: int) -> list[np.ndarray]:
    with av.open(str(path)) as container:
        container.streams.video[0].thread_count = 1
        assert int(container.streams.video[0].average_rate) == 24
        frames = list(container.decode(video=0))
    assert len(frames) == 56
    assert all(f.width == width and f.height == 480 for f in frames)
    pts = [f.pts for f in frames]
    assert len(set(pts)) == 56 and all(x < y for x, y in zip(pts, pts[1:]))
    return [f.to_ndarray(format="rgb24") for f in frames]


def make_sheet(path: Path, paired: list[np.ndarray], name: str) -> None:
    positions = [0, 38, 39, 55]
    sheet = Image.new("RGB", (1664, 4 * 504), "black")
    draw = ImageDraw.Draw(sheet)
    for row, index in enumerate(positions):
        y = row * 504
        draw.text((8, y + 4), f"{name} frame {index}    LEFT: source    RIGHT: VAE roundtrip", fill="white")
        sheet.paste(Image.fromarray(paired[index]), (0, y + 24))
    sheet.save(path, quality=88)


def main() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("CPU audit requires CUDA_VISIBLE_DEVICES=''")
    cpus = cpu_affinity()
    torch.set_num_threads(4)
    result = json.loads((OUT / "result.json").read_text())
    ledger = json.loads((OUT / "budget.json").read_text())
    assert result["status"] == "complete_pending_judge"
    assert ledger["status"] == "complete"
    assert {k: ledger[k] for k in CFG["counts"]} == CFG["counts"]
    rows = []
    for scene in CFG["scenes"]:
        directory = OUT / scene
        metrics = json.loads((directory / "metrics.json").read_text())
        tensors = {key: torch.load(directory / filename, map_location="cpu", weights_only=True)
                   for key, filename in (("I0", "I0.pt"), ("full", "full17.pt"),
                                         ("prefix", "prefix12.pt"))}
        assert [list(tensors[k].shape) for k in ("I0", "full", "prefix")] == [
            [1, 24, 1, 30, 52], [1, 24, 17, 30, 52], [1, 24, 12, 30, 52]]
        assert all(torch.isfinite(t).all() for t in tensors.values())
        exact = bool(torch.equal(tensors["full"][:, :, :12], tensors["prefix"]))
        assert exact and metrics["prefix_comparison"]["exact_equal"]
        files = metrics["files"]
        for item in files.values():
            assert sha(item["path"]) == item["sha256"]
        recon = video(directory / "reconstruction.mp4", 832)
        paired = video(directory / "source_vs_reconstruction.mp4", 1664)
        assert all(x.shape == (480, 832, 3) for x in recon)
        # Encoded side-by-side pixels differ from pre-encode arrays, but must
        # closely match its independently encoded right video.
        paired_right_mad = float(np.mean([np.abs(a[:, 832:].astype(np.float32) -
                                                b.astype(np.float32)).mean()
                                           for a, b in zip(paired, recon)]))
        sheet_path = HERE / f"{scene}_boundary_contact_sheet.jpg"
        make_sheet(sheet_path, paired, scene)
        rows.append({"scene": scene, "prefix_exact_from_saved_tensors": exact,
                     "tensor_dtypes": {k: str(t.dtype) for k, t in tensors.items()},
                     "reconstruction_frames": len(recon), "comparison_frames": len(paired),
                     "paired_right_vs_reconstruction_encoded_MAD": paired_right_mad,
                     "MAD_mean_pre_encoding": metrics["roundtrip"]["MAD_mean"],
                     "PSNR_mean_pre_encoding": metrics["roundtrip"]["PSNR_mean"],
                     "contact_sheet": str(sheet_path), "contact_sheet_sha256": sha(sheet_path)})
        print(scene, "CPU audit PASS", flush=True)
    write(HERE / "P1_CPU_AUDIT.json", {"task": CFG["task"], "status": "PASS_CPU_ARTIFACTS",
           "cpu_affinity": cpus, "GPU_calls": 0, "scenes": rows,
           "ledger_counts": {k: ledger[k] for k in CFG["counts"]},
           "gpu_wall_seconds": ledger["gpu_wall_seconds"],
           "peak_allocated_gib": result["peak_allocated_gib"]})


if __name__ == "__main__":
    main()
