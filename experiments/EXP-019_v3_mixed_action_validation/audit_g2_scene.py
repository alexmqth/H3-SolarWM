"""CPU-only, per-scene integrity and motion audit for approved EXP-019 G2 rows."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import av
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASE = ROOT / "H3-World/outputs/EXP-019_v3_mixed_action_validation"
REAL = ROOT / "submission/experiments/EXP-015_v3_real56_data/artifacts"
ALLOWED = ("7199292c", "9dc2e588", "b784d995")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(1 << 20), b""):
            h.update(part)
    return h.hexdigest()


def decode(path: Path, width: int, height: int) -> tuple[np.ndarray, dict]:
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        frames = list(container.decode(stream))
        pts = [frame.pts for frame in frames]
        assert len(frames) == 56 and stream.average_rate == 24
        assert (stream.width, stream.height) == (width, height)
        assert len(set(pts)) == 56 and all(a < b for a, b in zip(pts, pts[1:]))
        rgb = np.stack([frame.to_ndarray(format="rgb24") for frame in frames])
    return rgb, {"path": str(path.relative_to(ROOT)), "sha256": sha(path),
                 "frames": 56, "fps": "24", "resolution": [width, height],
                 "pts_unique_increasing": True}


def motion(rgb: np.ndarray) -> dict:
    gray = [cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in rgb]
    mad = [float(np.abs(a.astype(np.float32) - b.astype(np.float32)).mean())
           for a, b in zip(rgb, rgb[1:])]
    fx = []
    for a, b in zip(gray[39:55], gray[40:56]):
        flow = cv2.calcOpticalFlowFarneback(a, b, None, .5, 3, 15, 3, 5, 1.2, 0)
        fx.append(float(np.median(flow[80:400, 160:672, 0])))
    return {"boundary_mad_38_39": mad[38],
            "within_c2_mad_mean": float(np.mean(mad[39:55])),
            "within_c2_mad_max": float(np.max(mad[39:55])),
            "central_median_horizontal_flow_sum_c2": float(sum(fx)),
            "central_median_horizontal_flow_per_transition": fx}


def main(scene: str) -> None:
    out = BASE / scene
    budget = json.loads((out / "budget.json").read_text())
    result = json.loads((out / "result.json").read_text())
    launch = json.loads((HERE / f"G2_{scene}_LAUNCH_ENV.json").read_text())
    marker = json.loads((HERE / "judge" / f"{scene}_APPROVED.json").read_text())
    assert marker["approved"] and marker["stage"] == scene
    assert marker["source_manifest_sha256"] == sha(HERE / "SOURCE_MANIFEST.json")
    assert budget["status"] == "complete" and result["status"] == "COMPLETE_PENDING_JUDGE"
    assert budget["counts"] == {"forward": 39, "commit": 1, "decode": 2, "text": 0}
    assert budget["elapsed"] <= 900 and budget["peak_allocated_gib"] <= 44
    assert [sum(e["kind"] == kind for e in budget["events"]) for kind in ("forward", "commit", "decode")] == [39, 1, 2]
    assert [r["nfe_per_chunk"] for r in result["runs"]] == [30, 8]
    assert result["runs"][0]["initial_noise_sha256"] == result["runs"][1]["initial_noise_sha256"]
    assert all(r["history_tensor_unchanged"] and r["history_kv_unchanged"] for r in result["runs"])
    assert result["cpu_kv_bytes"] == 6_799_104_000
    assert launch["selected_env"]["PATH"].startswith(str(BASE / "bin") + ":")
    assert launch["selected_env"]["CUDA_VISIBLE_DEVICES"] == str(budget["gpu"])
    assert launch["ffmpeg_sha256"] == sha(Path(launch["ffmpeg_realpath"]))
    e = json.loads((BASE / "E/result.json").read_text())
    entry = next(x for x in e["fixtures"] if x["scene"] == scene)
    assert sha(ROOT / entry["path"]) == entry["sha256"]
    fixture = torch.load(ROOT / entry["path"], map_location="cpu", weights_only=True)
    assert fixture["script"] == json.loads((REAL / scene / "action_script17.json").read_text())
    cache = json.loads((out / "cache_before.json").read_text())
    assert len(cache) == 50 and all([x["index"] for x in v] == [0] for v in cache.values())

    source, source_meta = decode(REAL / scene / "real56.mp4", 832, 480)
    fm30, fm30_meta = decode(out / "fm30_gt39_plus_generated17.mp4", 832, 480)
    fm8, fm8_meta = decode(out / "fm8_gt39_plus_generated17.mp4", 832, 480)
    _, raw30_meta = decode(out / "fm30_raw_decode56.mp4", 832, 480)
    _, raw8_meta = decode(out / "fm8_raw_decode56.mp4", 832, 480)
    _, combined_meta = decode(out / "source_vs_fm30_vs_fm8.mp4", 2496, 512)

    endpoints = {}
    for steps in (30, 8):
        path = out / f"fm{steps}_C2.pt"
        tensor = torch.load(path, map_location="cpu", weights_only=True)
        assert tuple(tensor.shape) == (1, 24, 5, 30, 52) and torch.isfinite(tensor).all()
        recorded = next(r for r in result["runs"] if r["nfe_per_chunk"] == steps)
        assert sha(path) == recorded["endpoint_sha256"]
        endpoints[str(steps)] = {"sha256": sha(path), "shape": list(tensor.shape), "finite": True}

    contact = Image.new("RGB", (3 * 416, 17 * 264), "black")
    draw = ImageDraw.Draw(contact)
    for row, index in enumerate(range(39, 56)):
        for col, (label, frames) in enumerate((("GT", source), ("FM-30", fm30), ("FM-8", fm8))):
            x, y = col * 416, row * 264
            contact.paste(Image.fromarray(frames[index]).resize((416, 240)), (x, y))
            draw.text((x + 8, y + 243), f"{label} RGB {index}", fill="white")
    contact_path = HERE / f"G2_{scene}_ALL_NEW17_CONTACT.jpg"
    contact.save(contact_path, quality=92)

    audit = {
        "status": "PASS_IMPLEMENTATION_PENDING_QUALITY_JUDGE",
        "scene": scene,
        "scope": "GT-history 17-frame continuation, no free-running",
        "budget": {key: budget[key] for key in ("counts", "elapsed", "peak_allocated_gib", "peak_reserved_gib")},
        "source_fixture_sha256": entry["sha256"],
        "real_script17": fixture["script"],
        "cache_layers": len(cache),
        "cpu_kv_bytes": result["cpu_kv_bytes"],
        "videos": {"source": source_meta, "fm30_display": fm30_meta, "fm8_display": fm8_meta,
                   "fm30_raw": raw30_meta, "fm8_raw": raw8_meta, "combined": combined_meta},
        "endpoints": endpoints,
        "display_prefix_h264_mad_to_source": {
            "fm30": float(np.abs(fm30[:39].astype(np.float32) - source[:39].astype(np.float32)).mean()),
            "fm8": float(np.abs(fm8[:39].astype(np.float32) - source[:39].astype(np.float32)).mean()),
        },
        "motion_auxiliary": {"source": motion(source), "fm30": motion(fm30), "fm8": motion(fm8)},
        "all_17_new_frames_contact": str(contact_path.relative_to(ROOT)),
        "limitations": ["Only RGB39..55 newly generated", "No D counterfactual GT", "Farneback is auxiliary, not action correctness", "No free-running"],
    }
    (HERE / f"G2_{scene}_CPU_AUDIT.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(scene, "CPU audit PASS: six 56-frame videos; 39 forward; 17 new frames per branch")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("scene", choices=ALLOWED)
    args = parser.parse_args()
    main(args.scene)
