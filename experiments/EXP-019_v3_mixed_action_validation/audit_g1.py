"""CPU-only validation of the fixed first GT-history continuation scene."""
from __future__ import annotations

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
OUT = ROOT / "H3-World/outputs/EXP-019_v3_mixed_action_validation/43866101"
SOURCE = ROOT / "submission/experiments/EXP-015_v3_real56_data/artifacts/43866101/real56.mp4"
E_OUT = ROOT / "H3-World/outputs/EXP-019_v3_mixed_action_validation/E"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def video(path: Path, width: int, height: int) -> tuple[np.ndarray, dict]:
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        frames = list(container.decode(stream))
        pts = [f.pts for f in frames]
        assert len(frames) == 56 and stream.average_rate == 24
        assert (stream.width, stream.height) == (width, height)
        assert len(set(pts)) == 56 and all(a < b for a, b in zip(pts, pts[1:]))
        rgb = np.stack([f.to_ndarray(format="rgb24") for f in frames])
    return rgb, {"path": str(path.relative_to(ROOT)), "sha256": sha(path),
                 "frames": len(frames), "fps": str(stream.average_rate),
                 "resolution": [width, height], "pts_unique_increasing": True}


def motion(rgb: np.ndarray) -> dict:
    # Exact EXP-014 CPU auxiliary proxy: central median horizontal Farneback.
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


budget = json.loads((OUT / "budget.json").read_text())
result = json.loads((OUT / "result.json").read_text())
launch = json.loads((HERE / "G1_LAUNCH_ENV.json").read_text())
assert budget["status"] == "complete" and result["status"] == "COMPLETE_PENDING_JUDGE"
assert budget["counts"] == {"forward": 39, "commit": 1, "decode": 2, "text": 0}
assert budget["elapsed"] <= 900 and budget["peak_allocated_gib"] <= 44
assert len([e for e in budget["events"] if e["kind"] == "forward"]) == 39
assert len([e for e in budget["events"] if e["kind"] == "commit"]) == 1
assert len([e for e in budget["events"] if e["kind"] == "decode"]) == 2
assert len(result["runs"]) == 2 and [r["nfe_per_chunk"] for r in result["runs"]] == [30, 8]
assert result["runs"][0]["initial_noise_sha256"] == result["runs"][1]["initial_noise_sha256"]
assert all(r["history_tensor_unchanged"] and r["history_kv_unchanged"] for r in result["runs"])
assert result["cpu_kv_bytes"] == 6_799_104_000
assert launch["selected_env"]["PATH"].startswith(str((OUT.parent / "bin")) + ":")
assert launch["selected_env"]["CUDA_VISIBLE_DEVICES"] == "0"
assert launch["ffmpeg_sha256"] == sha(Path(launch["ffmpeg_realpath"]))
e = json.loads((E_OUT / "result.json").read_text())
entry = next(x for x in e["fixtures"] if x["scene"] == "43866101")
assert sha(ROOT / entry["path"]) == entry["sha256"]
cache = json.loads((OUT / "cache_before.json").read_text())
assert len(cache) == 50 and all([x["index"] for x in v] == [0] for v in cache.values())

source, source_meta = video(SOURCE, 832, 480)
fm30, fm30_meta = video(OUT / "fm30_gt39_plus_generated17.mp4", 832, 480)
fm8, fm8_meta = video(OUT / "fm8_gt39_plus_generated17.mp4", 832, 480)
raw30, raw30_meta = video(OUT / "fm30_raw_decode56.mp4", 832, 480)
raw8, raw8_meta = video(OUT / "fm8_raw_decode56.mp4", 832, 480)
combined, combined_meta = video(OUT / "source_vs_fm30_vs_fm8.mp4", 2496, 512)

prefix_diff = {
    "fm30_source_mean_abs": float(np.abs(fm30[:39].astype(np.float32) - source[:39].astype(np.float32)).mean()),
    "fm8_source_mean_abs": float(np.abs(fm8[:39].astype(np.float32) - source[:39].astype(np.float32)).mean()),
    "fm30_fm8_exact_after_decode": bool(np.array_equal(fm30[:39], fm8[:39])),
    "note": "H.264 re-encoding makes displayed MP4 prefix differ slightly from original decoded source; runner assigns identical source RGB arrays before encoding.",
}

endpoints = {}
for steps in (30, 8):
    p = OUT / f"fm{steps}_C2.pt"
    t = torch.load(p, map_location="cpu", weights_only=True)
    assert tuple(t.shape) == (1, 24, 5, 30, 52) and torch.isfinite(t).all()
    recorded = next(r for r in result["runs"] if r["nfe_per_chunk"] == steps)
    assert sha(p) == recorded["endpoint_sha256"]
    endpoints[str(steps)] = {"sha256": sha(p), "shape": list(t.shape), "finite": True}

# Each generated frame is shown next to the real future; no frame is omitted.
contact = Image.new("RGB", (3 * 416, 17 * 264), "black")
draw = ImageDraw.Draw(contact)
for row, index in enumerate(range(39, 56)):
    for col, (label, frames) in enumerate((("GT", source), ("FM-30", fm30), ("FM-8", fm8))):
        x, y = col * 416, row * 264
        contact.paste(Image.fromarray(frames[index]).resize((416, 240)), (x, y))
        draw.text((x + 8, y + 243), f"{label} RGB {index}", fill="white")
contact_path = HERE / "G1_ALL_NEW17_CONTACT.jpg"
contact.save(contact_path, quality=92)

audit = {
    "status": "PASS_IMPLEMENTATION_PENDING_QUALITY_JUDGE",
    "scope": "CPU validation of first real GT-history C2 continuation; not free-running",
    "budget": {key: budget[key] for key in ("counts", "elapsed", "peak_allocated_gib", "peak_reserved_gib")},
    "source_fixture_sha256": entry["sha256"],
    "cache_layers": len(cache),
    "cpu_kv_bytes": result["cpu_kv_bytes"],
    "videos": {"source": source_meta, "fm30_display": fm30_meta, "fm8_display": fm8_meta,
               "fm30_raw": raw30_meta, "fm8_raw": raw8_meta, "combined": combined_meta},
    "endpoints": endpoints,
    "prefix_after_h264": prefix_diff,
    "motion_auxiliary": {"source": motion(source), "fm30": motion(fm30), "fm8": motion(fm8)},
    "all_17_new_frames_contact": str(contact_path.relative_to(ROOT)),
    "limitations": ["Only RGB39..55 are newly generated", "Source future is loaded for side-by-side display but is not passed into the DiT forward", "One mixed action schedule with no counterfactual", "Farneback sign is not an action correctness verdict"],
}
(HERE / "G1_CPU_AUDIT.json").write_text(json.dumps(audit, indent=2) + "\n")
print("G1 CPU audit PASS: 6 videos x 56 frames, 39 forwards, 17 new frames per generated branch")
