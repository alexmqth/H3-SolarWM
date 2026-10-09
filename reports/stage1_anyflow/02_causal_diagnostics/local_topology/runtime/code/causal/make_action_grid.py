#!/usr/bin/env python3
"""Make a four-row Original-vs-Causal action intervention grid."""
from __future__ import annotations

import argparse
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def make_grid(pairs, output: Path, tile_width=832, tile_height=480):
    opened = [(action, av.open(str(left)), av.open(str(right)))
              for action, left, right in pairs]
    decoders = [(action, iter(left.decode(video=0)), iter(right.decode(video=0)))
                for action, left, right in opened]
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.mp4")
    try:
        first = opened[0][1].streams.video[0]
        rate = first.average_rate
        canvas_w, canvas_h = tile_width * 2, (tile_height + 32) * len(opened)
        with av.open(str(temporary), "w", options={"movflags": "+faststart"}) as out:
            stream = out.add_stream("libx264", rate=rate)
            stream.width, stream.height = canvas_w, canvas_h
            stream.pix_fmt = "yuv420p"
            stream.options = {"crf": "18", "preset": "medium"}
            font = ImageFont.load_default(size=18)
            count = 0
            while True:
                decoded = []
                for action, left_decoder, right_decoder in decoders:
                    lf = next(left_decoder, None)
                    rf = next(right_decoder, None)
                    if lf is None and rf is None:
                        decoded.append(None)
                        continue
                    if lf is None or rf is None:
                        raise ValueError(f"{action}: original/causal frame counts differ")
                    decoded.append((action, lf.to_ndarray(format="rgb24"),
                                    rf.to_ndarray(format="rgb24")))
                if all(item is None for item in decoded):
                    break
                image = Image.new("RGB", (canvas_w, canvas_h), "black")
                draw = ImageDraw.Draw(image)
                for row, item in enumerate(decoded):
                    if item is None:
                        raise ValueError("action videos have different frame counts")
                    action, left_rgb, right_rgb = item
                    y = row * (tile_height + 32)
                    image.paste(Image.fromarray(left_rgb), (0, y + 32))
                    image.paste(Image.fromarray(right_rgb), (tile_width, y + 32))
                    draw.text((12, y + 8), f"{action} | H3-World original | 30 steps",
                              fill="white", font=font)
                    draw.text((tile_width + 12, y + 8), f"{action} | causal Stage1 | 8 steps/chunk",
                              fill="white", font=font)
                for packet in stream.encode(
                    av.VideoFrame.from_ndarray(np.asarray(image), format="rgb24")
                ):
                    out.mux(packet)
                count += 1
            for packet in stream.encode():
                out.mux(packet)
    finally:
        for _, left, right in opened:
            left.close()
            right.close()
    temporary.replace(output)
    print(f"wrote {output} ({count} frames)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("pairs", nargs=4, metavar="ACTION=ORIGINAL,CAUSAL",
                        help="four pairs in W/S/A/D order")
    args = parser.parse_args()
    pairs = []
    for value in args.pairs:
        action, paths = value.split("=", 1)
        left, right = paths.split(",", 1)
        pairs.append((action, Path(left), Path(right)))
    make_grid(pairs, args.output)


if __name__ == "__main__":
    main()
