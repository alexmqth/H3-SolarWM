#!/usr/bin/env python3
"""Render matched videos from a JSON panel spec, without looping or retiming.

Each row is a list of {video, title, run_json, note} objects. Video/run_json
paths are relative to --source-root. Run JSON supplies the actual recorded
runtime and forward counts. Inputs must have equal sizes, fps, and lengths.
"""
import argparse
from contextlib import ExitStack
from itertools import zip_longest
import json
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def font(size):
    for name in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                 "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"):
        if Path(name).exists():
            return ImageFont.truetype(name, size)
    return ImageFont.load_default()


def render(spec, source_root, output):
    rows = spec["rows"]
    if not rows or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError("Expected a rectangular nonempty panel grid")
    panels = [p for row in rows for p in row]
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.mp4")
    with ExitStack() as stack:
        containers = [stack.enter_context(av.open(str(source_root / p["video"]))) for p in panels]
        streams = [c.streams.video[0] for c in containers]
        w, h, fps = streams[0].width, streams[0].height, streams[0].average_rate
        if any((s.width, s.height, s.average_rate) != (w, h, fps) for s in streams):
            raise ValueError("All panels must have identical resolution and fps")
        metadata = [json.loads((source_root / p["run_json"]).read_text()) for p in panels]
        if any(m["status"] != "complete" for m in metadata):
            raise ValueError("Incomplete run cannot be rendered as a completed result")
        expected = metadata[0]["num_frames"]
        if any(m["num_frames"] != expected for m in metadata):
            raise ValueError("Different run lengths")
        header = 80
        with av.open(str(temporary), "w", options={"movflags": "+faststart"}) as out:
            s = out.add_stream("libx264", rate=fps)
            s.width, s.height = w * len(rows[0]), (h + header) * len(rows)
            s.pix_fmt = "yuv420p"
            s.options = {"crf": "18", "preset": "medium", "threads": "4"}
            title_font, info_font = font(19), font(15)
            count = 0
            for i, frames in enumerate(zip_longest(*(c.decode(video=0) for c in containers))):
                if any(f is None for f in frames):
                    raise ValueError("Unequal decoded lengths; refusing to truncate a comparison")
                canvas = Image.new("RGB", (s.width, s.height), (15, 18, 23))
                draw = ImageDraw.Draw(canvas)
                for j, (frame, panel, meta) in enumerate(zip(frames, panels, metadata)):
                    x, y = (j % len(rows[0])) * w, (j // len(rows[0])) * (h + header)
                    canvas.paste(frame.to_image(), (x, y + header))
                    draw.text((x + 12, y + 4), panel["title"], font=title_font, fill="white")
                    calls = meta["denoiser_forwards"]
                    commits = meta.get("commit_forwards", 0)
                    step = f'{meta["steps"]} steps/chunk' if meta["mode"] == "cached" else f'{meta["steps"]} full-horizon steps'
                    e2e = meta["wall_including_shared_setup_seconds"]
                    draw.text((x + 12, y + 29), f'{step} | {calls} noisy + {commits} commits | recorded e2e {e2e:.1f}s', font=info_font, fill="#e0e3e8")
                    draw.text((x + 12, y + 52), f'{i / float(fps):.2f}s / {expected / float(fps):.3f}s | frame {i+1}/{expected} | {panel.get("note", "")}', font=info_font, fill="#bdc9df")
                for packet in s.encode(av.VideoFrame.from_ndarray(np.asarray(canvas), format="rgb24")):
                    out.mux(packet)
                count += 1
            if count != expected:
                raise ValueError(f"Decoded {count} frames, expected {expected}")
            for packet in s.encode():
                out.mux(packet)
    temporary.replace(output)
    return {"path": str(output), "frames": count, "fps": float(fps), "width": s.width, "height": s.height}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spec", type=Path, required=True)
    ap.add_argument("--source-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    print(json.dumps(render(json.loads(args.spec.read_text()), args.source_root, args.output)))
