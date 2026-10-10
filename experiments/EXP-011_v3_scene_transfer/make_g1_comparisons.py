"""CPU-only 39-frame side-by-side preview. Reads saved MP4, no H3 inference."""
from __future__ import annotations

from pathlib import Path

import av
from PIL import Image, ImageDraw, ImageFont

from common import HERE, OUT


def frames(path: Path):
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        assert stream.width == 832 and stream.height == 480 and stream.average_rate == 24
        images = [Image.fromarray(frame.to_ndarray(format="rgb24"))
                  for frame in container.decode(video=0)]
    assert len(images) == 39
    return images


def make(scene: str):
    left = frames(OUT / "G1" / scene / "FM30/first39.mp4")
    right = frames(OUT / "G1" / scene / "FM8/first39.mp4")
    target = HERE / "artifacts/G1/comparisons" / f"{scene}_FM30_vs_FM8_39.mp4"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(target)
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    title = ImageFont.truetype(font_path, 22)
    body = ImageFont.truetype(font_path, 17)
    temp = target.with_name(target.stem + ".partial.mp4")
    with av.open(str(temp), "w", options={"movflags": "+faststart"}) as output:
        stream = output.add_stream("libx264", rate=24)
        stream.width, stream.height, stream.pix_fmt = 1664, 552, "yuv420p"
        stream.options = {"crf": "19", "preset": "medium"}
        for i, (a, b) in enumerate(zip(left, right)):
            canvas = Image.new("RGB", (1664, 552), "#111111")
            canvas.paste(a, (0, 44)); canvas.paste(b, (832, 44))
            draw = ImageDraw.Draw(canvas)
            draw.text((16, 9), "V3 causal FM30 | 30 NFE", font=title, fill="white")
            draw.text((848, 9), "V3 causal FM8 | 8 NFE", font=title, fill="white")
            draw.text((16, 526), f"{scene} | same I0, seed 13 | independent C1 rollouts | frame {i:02d}/38",
                      font=body, fill="white")
            for packet in stream.encode(av.VideoFrame.from_image(canvas)):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    temp.replace(target)
    with av.open(str(target)) as check:
        decoded = list(check.decode(video=0))
        assert len(decoded) == 39 and len({frame.pts for frame in decoded}) == 39
    print(target)


if __name__ == "__main__":
    for name in ("industrial", "village"):
        make(name)
