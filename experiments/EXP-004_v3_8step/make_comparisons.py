"""CPU-only 30-step V3 vs eight-step FM clips from saved real videos."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import av
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OLD = ROOT / "submission/experiments/EXP-002_native_cached/artifacts/videos"
NEW = ROOT / "H3-World/outputs/EXP-004_v3_8step"
DEST = HERE / "artifacts/videos"


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read(path, expected):
    with av.open(str(path)) as container:
        video = container.streams.video[0]
        assert (video.width, video.height, str(video.average_rate)) == (832, 480, "24")
        frames = [f.to_image().convert("RGB") for f in container.decode(video=0)]
    assert len(frames) == expected
    return frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", choices=["AA", "AD"], required=True)
    parser.add_argument("--frames", type=int, choices=[56, 73], required=True)
    args = parser.parse_args()
    DEST.mkdir(parents=True, exist_ok=True)
    old = OLD / f"{args.path}_rollout_{args.frames}.mp4"
    new = NEW / args.path / f"rollout_{args.frames}.mp4"
    target = DEST / f"V3_30_vs_8_{args.path}_{args.frames}.mp4"
    assert old.is_file() and new.is_file() and not target.exists()
    left, right = read(old, args.frames), read(new, args.frames)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 17)
    container = av.open(str(target), "w")
    stream = container.add_stream("libx264", rate=24)
    stream.width, stream.height, stream.pix_fmt = 1664, 560, "yuv420p"
    stream.options = {"crf": "18", "preset": "medium"}
    try:
        for i, (a, b) in enumerate(zip(left, right)):
            canvas = Image.new("RGB", (1664, 560), "#10151d")
            draw = ImageDraw.Draw(canvas)
            canvas.paste(a, (0, 40)); canvas.paste(b, (832, 40))
            draw.text((10, 10), "V3 30-step/chunk | persistent KV", font=font, fill="white")
            draw.text((842, 10), "V3 8-step new chunks | same weights/KV", font=font, fill="white")
            stage = "shared first12: 30-step" if i < 39 else f"current action {args.path[1]}"
            draw.text((10, 530), f"RGB {i:02d} | {stage}", font=font, fill="white")
            if i >= 56:
                draw.text((842, 530), "each side uses its own generated 2nd chunk", font=font,
                          fill="#ffcf80")
            elif i >= 39:
                draw.text((842, 530), "same first12 history/noise | 30 vs 8 native FM", font=font,
                          fill="#ffcf80")
            for packet in stream.encode(av.VideoFrame.from_image(canvas)):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()
    with av.open(str(target)) as check:
        pts = [frame.pts for frame in check.decode(video=0)]
        assert len(pts) == args.frames and all(a < b for a, b in zip(pts, pts[1:]))
    manifest = dict(left=str(old), left_sha256=sha(old), right=str(new),
        right_sha256=sha(new), output=str(target), output_sha256=sha(target),
        frames=args.frames, fps=24,
        comparison="Same V3 weights/protocol; new chunks use native FM 30 vs 8 steps. "
                   "The first12 30-step latent and its 39 RGB frames are shared. "
                   "From RGB56, each side uses its own generated history.")
    (DEST / f"V3_30_vs_8_{args.path}_{args.frames}.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(target)


if __name__ == "__main__":
    main()
