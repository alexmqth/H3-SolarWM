#!/usr/bin/env python3
"""Create the interview-demo side-by-side video from two H3 outputs."""

from __future__ import annotations

import argparse
from itertools import zip_longest
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def make_side_by_side(left_path: Path, right_path: Path, output_path: Path,
                     left_label='H3-World original | 30 steps',
                     right_label='Causal mask prototype | 30 steps',
                     action_segments=None) -> int:
    if output_path.resolve() in {left_path.resolve(), right_path.resolve()}:
        raise ValueError('output must differ from both inputs')
    left = av.open(str(left_path))
    right = av.open(str(right_path))
    try:
        left_stream = left.streams.video[0]
        right_stream = right.streams.video[0]
        if (left_stream.width, left_stream.height, left_stream.average_rate) != (
            right_stream.width, right_stream.height, right_stream.average_rate
        ):
            raise ValueError("the two videos must have the same size and frame rate")
        width, height = left_stream.width, left_stream.height
        rate = left_stream.average_rate
        bar_height = 80 if action_segments else 48
        if action_segments:
            stop = 0
            for label, start, end in action_segments:
                if start != stop or end <= start:
                    raise ValueError('action segments must form a contiguous timeline from frame 0')
                stop = end
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_suffix('.tmp.mp4')
        container = av.open(str(temporary), "w", options={'movflags': '+faststart'})
        try:
            stream = container.add_stream("libx264", rate=rate)
            stream.width, stream.height = width * 2, height + bar_height
            stream.pix_fmt = "yuv420p"
            stream.options = {"crf": "18", "preset": "medium"}
            count = 0
            font = ImageFont.load_default(size=20)
            for left_frame, right_frame in zip_longest(left.decode(video=0), right.decode(video=0)):
                if left_frame is None or right_frame is None:
                    raise ValueError('videos have different decoded frame counts')
                if left_frame.time != right_frame.time:
                    raise ValueError('input frame timestamps differ')
                left_rgb = left_frame.to_ndarray(format="rgb24")
                right_rgb = right_frame.to_ndarray(format="rgb24")
                canvas = np.concatenate((left_rgb, right_rgb), axis=1)
                image = Image.new('RGB', (width*2, height+bar_height), 'black')
                image.paste(Image.fromarray(canvas), (0, bar_height))
                draw = ImageDraw.Draw(image)
                draw.text((12, 12), left_label, fill='white', font=font)
                draw.text((width+12, 12), right_label, fill='white', font=font)
                if action_segments:
                    active = next((label for label, start, end in action_segments
                                   if start <= count < end), None)
                    if active is None:
                        raise ValueError(f'no action label for frame {count}')
                    overlay = f'INPUT: {active}   |   frame {count:03d}   |   {count/float(rate):.2f} s'
                    for x in (12, width + 12):
                        draw.text((x, 46), overlay, fill='#ffdc66', font=font)
                for packet in stream.encode(
                    av.VideoFrame.from_ndarray(np.asarray(image), format="rgb24")
                ):
                    container.mux(packet)
                count += 1
            if action_segments and count != action_segments[-1][2]:
                raise ValueError('action timeline length differs from decoded video length')
            for packet in stream.encode():
                container.mux(packet)
        finally:
            container.close()
        temporary.replace(output_path)
        return count
    finally:
        left.close()
        right.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument('--left-label', default='H3-World original | 30 steps')
    parser.add_argument('--right-label', default='Causal mask prototype | 30 steps')
    parser.add_argument('--action-segments',
                        help='nominal input-frame intervals, e.g. W:0:51;A:51:85;D:85:124')
    args = parser.parse_args()
    segments = None
    if args.action_segments:
        segments = []
        for item in args.action_segments.split(';'):
            label, start, end = item.split(':')
            segments.append((label, int(start), int(end)))
    count = make_side_by_side(args.left, args.right, args.output,
                             args.left_label, args.right_label, segments)
    print(f"wrote {args.output} ({count} frames)")


if __name__ == "__main__":
    main()
