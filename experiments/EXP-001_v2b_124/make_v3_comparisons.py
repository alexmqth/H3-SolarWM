"""CPU-only Original-vs-V2b A/D comparison videos from saved real MP4s."""

import json
import hashlib
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = ROOT / "H3-World/outputs/EXP-001_v2b_124"
BASE = ROOT / "H3-World/outputs/2026-10-01-21"
TARGET = HERE / "artifacts/videos"
FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 19)
FONT_SMALL = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def timing(action):
    baseline = json.loads((BASE / f"action_{action}_teacher_latents/baseline.json").read_text())
    incremental = sum(json.loads((OUT / f"{action}{action}/chunk_{start}_{start+5}.json").read_text())["wall_seconds"]
                      for start in (22, 27, 32))
    return baseline["wall_including_shared_setup_seconds"], incremental


def frames(path):
    with av.open(str(path)) as container:
        yield from (frame.to_ndarray(format="rgb24") for frame in container.decode(video=0))


def pair_frames(action, causal_length=124):
    original = BASE / f"action_{action}_teacher_latents/baseline.mp4"
    causal = OUT / f"{action}{action}/rollout_{causal_length}.mp4"
    assert original.is_file() and causal.is_file(), (original, causal)
    original_e2e, v2b_incremental = timing(action) if causal_length == 124 else (float("nan"), float("nan"))
    for number, (left, right) in enumerate(zip(frames(original), frames(causal))):
        assert left.shape == right.shape == (480, 832, 3)
        canvas = Image.new("RGB", (1664, 570), "#17202a")
        canvas.paste(Image.fromarray(left), (0, 57))
        canvas.paste(Image.fromarray(right), (832, 57))
        draw = ImageDraw.Draw(canvas)
        draw.text((16, 13), f"Original H3  |  {action}  |  30 full-video steps", font=FONT, fill="white")
        draw.text((848, 13), f"V2b same-sigma history  |  {action}  |  30 steps/chunk", font=FONT, fill="white")
        if causal_length == 124:
            draw.text((16, 37), f"Measured full E2E: {original_e2e:.1f}s", font=FONT_SMALL, fill="#cdd9e5")
            draw.text((848, 37), f"Measured 73→124f incremental: {v2b_incremental:.1f}s; prior 73f reused", font=FONT_SMALL, fill="#cdd9e5")
        draw.text((16, 544), f"RGB {number:03d} / {causal_length-1}  |  24 fps  |  same I0, seed 13 and noise", font=FONT, fill="white")
        draw.text((848, 544), "Single I0  |  own history  |  no hidden KV  |  cross-protocol comparison", font=FONT, fill="white")
        if number in (39, 56, 73, 90, 107):
            draw.line((832, 57, 832, 537), fill="#fac858", width=4)
        yield np.asarray(canvas)


def encode(path, frame_iter):
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=24, options={"crf": "21", "preset": "fast"})
        stream.width = 1664
        stream.height = 570
        stream.pix_fmt = "yuv420p"
        for rgb in frame_iter:
            frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
            count += 1
        for packet in stream.encode():
            container.mux(packet)
    return count


def main():
    manifest = {"comparison_type": "cross-protocol capability comparison, not a single-variable ablation",
                "fps": 24, "resolution": [1664, 570], "videos": {}}
    for action in "AD":
        output = TARGET / f"Original_vs_V2b_{action}_124.mp4"
        assert not output.exists(), f"Refuse to overwrite {output}"
        count = encode(output, pair_frames(action))
        assert count == 124
        original = BASE / f"action_{action}_teacher_latents/baseline.mp4"
        causal = OUT / f"{action}{action}/rollout_124.mp4"
        original_e2e, v2b_incremental = timing(action)
        manifest["videos"][action] = {"path": str(output.relative_to(ROOT)), "sha256": sha(output),
                                      "frames": count, "original_source_sha256": sha(original),
                                      "v2b_source_sha256": sha(causal),
                                      "original_full_e2e_seconds": original_e2e,
                                      "v2b_73_to_124_incremental_wall_seconds": v2b_incremental,
                                      "timing_note": "Different measurement scopes; not a speedup ratio"}
    overview = TARGET / "Original_vs_V2b_AD_overview_248.mp4"
    assert not overview.exists(), f"Refuse to overwrite {overview}"
    count = encode(overview, (frame for action in "AD" for frame in pair_frames(action)))
    assert count == 248
    manifest["videos"]["overview"] = {"path": str(overview.relative_to(ROOT)), "sha256": sha(overview),
                                      "frames": count}
    (HERE / "artifacts/comparison_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
