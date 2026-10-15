"""CPU-only media verification and GT/Original comparison for EXP-020."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import av
import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUT = ROOT / "H3-World/outputs/EXP-020_original_mixed_action_reference/b784d995"
GT = ROOT / "submission/experiments/EXP-015_v3_real56_data/artifacts/b784d995/real56.mp4"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def decode(path: Path, wh: tuple[int, int]) -> tuple[np.ndarray, dict]:
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        frames = list(c.decode(s))
        assert (s.width, s.height) == wh and s.average_rate == 24
        assert len(frames) == 56
        pts = [f.pts for f in frames]
        assert all(a < b for a, b in zip(pts, pts[1:]))
        rgb = np.stack([f.to_ndarray(format="rgb24") for f in frames])
        audio_streams = len(c.streams.audio)
    return rgb, {"path": str(path.relative_to(ROOT)), "sha256": sha(path),
                 "width": wh[0], "height": wh[1], "fps": 24, "frames": 56,
                 "pts_strict_increasing": True, "audio_streams": audio_streams}


def metrics(rgb: np.ndarray) -> dict:
    gray = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in rgb]
    sums = {}
    for name, lo, hi in (("early_left", 3, 39), ("late_left", 39, 43),
                         ("still", 43, 47), ("forward", 47, 56)):
        flows = []
        for i in range(lo, hi - 1):
            f = cv2.calcOpticalFlowFarneback(gray[i], gray[i + 1], None,
                                             .5, 3, 15, 3, 5, 1.2, 0)
            flows.append(float(np.median(f[80:400, 160:672, 0])))
        sums[name] = sum(flows)
    return {"central_median_horizontal_flow_sum_by_approximate_action_interval": sums,
            "frame_mad_mean": float(np.mean([np.abs(a.astype(np.float32) - b.astype(np.float32)).mean()
                                             for a, b in zip(rgb[:-1], rgb[1:])]))}


def main() -> None:
    result = json.loads((OUT / "result.json").read_text())
    assert result["status"] == "COMPLETE_PENDING_JUDGE"
    assert result["counts"] == {"dit": 30, "text": 4, "image_encode": 1,
                                 "video_decode": 1, "audio_decode": 1}
    assert result["wall_seconds"] <= 1200 and result["peak_allocated_gib"] <= 44
    original_path = OUT / "original56.mp4"
    assert sha(original_path) == result["output_sha256"]
    gt, gt_meta = decode(GT, (832, 480))
    original, original_meta = decode(original_path, (832, 480))

    ffmpeg = Path(json.loads((HERE / "LAUNCH_ENV.json").read_text())["ffmpeg_realpath"])
    assert sha(ffmpeg) == json.loads((HERE / "LAUNCH_ENV.json").read_text())["ffmpeg_sha256"]
    gallery = OUT / "gt_vs_original56.mp4"
    assert not gallery.exists(), "do not overwrite comparison"
    with av.open(str(gallery), mode="w") as output:
        stream = output.add_stream("libx264", rate=24, options={"crf": "18", "preset": "medium", "threads": "4"})
        stream.width, stream.height, stream.pix_fmt = 1664, 512, "yuv420p"
        for i in range(56):
            canvas = Image.new("RGB", (1664, 512), "black")
            canvas.paste(Image.fromarray(gt[i]), (0, 32))
            canvas.paste(Image.fromarray(original[i]), (832, 32))
            draw = ImageDraw.Draw(canvas)
            draw.text((10, 8), f"GT recorded | RGB{i:02d}", fill="white")
            draw.text((842, 8), f"Original H3 | 30 NFE | RGB{i:02d}", fill="white")
            for packet in stream.encode(av.VideoFrame.from_image(canvas)):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    _, gallery_meta = decode(gallery, (1664, 512))

    contact = Image.new("RGB", (832, 17 * 240), "black")
    draw = ImageDraw.Draw(contact)
    for row, i in enumerate(range(39, 56)):
        for col, (name, frames) in enumerate((("GT", gt), ("Original", original))):
            x, y = col * 416, row * 240
            contact.paste(Image.fromarray(frames[i]).resize((416, 216)), (x, y))
            draw.text((x + 8, y + 220), f"{name} RGB{i}", fill="white")
    contact_path = OUT / "all_late17_contact.jpg"
    assert not contact_path.exists()
    contact.save(contact_path, quality=92)

    audit = {"status": "PASS_IMPLEMENTATION_PENDING_QUALITY_JUDGE",
             "scope": "one Original H3 full56 from I0; not matched-history causal comparison",
             "result": {k: result[k] for k in ("counts", "wall_seconds", "peak_allocated_gib", "peak_reserved_gib")},
             "videos": {"gt": gt_meta, "original": original_meta, "comparison": gallery_meta},
             "contact": {"path": str(contact_path.relative_to(ROOT)), "sha256": sha(contact_path)},
             "motion_auxiliary": {"gt": metrics(gt), "original": metrics(original)},
             "approximate_switch_rgb": {"stand_still": 43, "walk_forward": 47},
             "limitations": ["single train episode and seed", "no same-state action counterfactual",
                             "GT vs Original is not a pixelwise reconstruction target",
                             "Original full56 and EXP-019 GT-history17 are different protocols"]}
    (HERE / "CPU_AUDIT.json").write_text(json.dumps(audit, indent=2) + "\n")
    print("EXP-020 CPU audit PASS: 56-frame GT, Original, comparison", flush=True)


if __name__ == "__main__":
    main()
