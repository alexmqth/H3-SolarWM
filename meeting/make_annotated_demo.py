#!/usr/bin/env python3
"""Add recorded end-to-end timings to the existing H3/casual side-by-side MP4s.

The source videos already contain the step-count labels. This helper adds a small
second line to the black title strip and re-encodes as H.264/YUV420P.
"""
from pathlib import Path
import argparse
import av
from PIL import Image, ImageDraw, ImageFont

TIMES = {
    "W": (441.5, 450.2),
    "S": (444.8, 451.9),
    "A": (454.2, 438.8),
    "D": (450.4, 383.4),
}

def font(size: int):
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()

def annotate(src: Path, dst: Path, action: str):
    original_s, causal_s = TIMES[action]
    with av.open(str(src)) as inp:
        stream = inp.streams.video[0]
        dst.parent.mkdir(parents=True, exist_ok=True)
        with av.open(str(dst), "w", options={"movflags": "+faststart"}) as out:
            out_stream = out.add_stream("libx264", rate=stream.average_rate or 24)
            out_stream.width = stream.width
            out_stream.height = stream.height
            out_stream.pix_fmt = "yuv420p"
            out_stream.options = {"crf": "18", "preset": "medium"}
            fnt = font(14)
            for frame in inp.decode(video=0):
                image = frame.to_image().convert("RGB")
                draw = ImageDraw.Draw(image)
                midpoint = image.width // 2
                left = f"recorded end-to-end: {original_s:.1f} s | 124 frames / 5.17 s"
                right = f"recorded end-to-end: {causal_s:.1f} s | 64 noisy forwards + 8 commits"
                draw.text((18, 30), left, fill=(220, 220, 220), font=fnt)
                draw.text((midpoint + 18, 30), right, fill=(220, 220, 220), font=fnt)
                out_frame = av.VideoFrame.from_ndarray(__import__("numpy").asarray(image), format="rgb24")
                for packet in out_stream.encode(out_frame):
                    out.mux(packet)
            for packet in out_stream.encode():
                out.mux(packet)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=Path, default=Path("final"))
    ap.add_argument("--output-dir", type=Path, default=Path("annotated"))
    args = ap.parse_args()
    for action in "WSAD":
        src = args.input_dir / f"h3world_final_{action}_original_vs_causal.mp4"
        dst = args.output_dir / f"h3world_final_{action}_original_vs_causal_timed.mp4"
        annotate(src, dst, action)
        print(dst)

if __name__ == "__main__":
    main()

def annotate_grid(src: Path, dst: Path):
    with av.open(str(src)) as inp:
        stream = inp.streams.video[0]
        with av.open(str(dst), "w", options={"movflags": "+faststart"}) as out:
            out_stream = out.add_stream("libx264", rate=stream.average_rate or 24)
            out_stream.width = stream.width
            out_stream.height = stream.height
            out_stream.pix_fmt = "yuv420p"
            out_stream.options = {"crf": "18", "preset": "medium"}
            fnt = font(13)
            for frame in inp.decode(video=0):
                image = frame.to_image().convert("RGB")
                draw = ImageDraw.Draw(image)
                row_height = image.height // 4
                for row, action in enumerate("WSAD"):
                    original_s, causal_s = TIMES[action]
                    y = row * row_height + 30
                    draw.text((18, y), f"recorded e2e: {original_s:.1f}s", fill=(220, 220, 220), font=fnt)
                    draw.text((image.width // 2 + 18, y), f"recorded e2e: {causal_s:.1f}s", fill=(220, 220, 220), font=fnt)
                out_frame = av.VideoFrame.from_ndarray(__import__("numpy").asarray(image), format="rgb24")
                for packet in out_stream.encode(out_frame):
                    out.mux(packet)
            for packet in out_stream.encode():
                out.mux(packet)
