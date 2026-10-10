"""CPU-only 56-frame FM8-vs-AF8 matched-history comparison videos."""
from __future__ import annotations

import argparse
from pathlib import Path

import av
from PIL import Image, ImageDraw, ImageFont

from common import HERE, OUT, ROOT

FM8 = ROOT / "H3-World/outputs/EXP-011_v3_scene_transfer/G2"


def frames(path: Path) -> list[Image.Image]:
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        assert (stream.width, stream.height, stream.average_rate) == (832, 480, 24)
        result = [Image.fromarray(frame.to_ndarray(format="rgb24"))
                  for frame in container.decode(video=0)]
    assert len(result) == 56
    return result


def make(scene: str, branch: str) -> Path:
    left = frames(FM8 / scene / "FM8" / branch / "rollout_56.mp4")
    right = frames(OUT / "G1" / scene / branch / "rollout_56.mp4")
    target = HERE / "artifacts/comparisons" / f"{scene}_{branch}_FM8_vs_AF8_56.mp4"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(target)
    font = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    title = ImageFont.truetype(font, 22)
    body = ImageFont.truetype(font, 17)
    temp = target.with_name(target.stem + ".partial.mp4")
    with av.open(str(temp), "w", options={"movflags": "+faststart"}) as output:
        stream = output.add_stream("libx264", rate=24)
        stream.width, stream.height, stream.pix_fmt = 1664, 552, "yuv420p"
        stream.options = {"crf": "19", "preset": "medium"}
        for i, (a, b) in enumerate(zip(left, right)):
            canvas = Image.new("RGB", (1664, 552), "#111111")
            canvas.paste(a, (0, 44))
            canvas.paste(b, (832, 44))
            draw = ImageDraw.Draw(canvas)
            draw.text((16, 9), f"V3 ordinary FM8 | {branch} | 8 NFE", font=title, fill="white")
            draw.text((848, 9), f"V3 AF2 step32 | {branch} | 8 finite maps", font=title, fill="white")
            phase = "shared FM8 C1" if i < 39 else "C2 continuation"
            draw.text((16, 526),
                      f"{scene} | same C1 latent/RGB/noise, own model KV | {phase} | frame {i:02d}/55",
                      font=body, fill="white")
            for packet in stream.encode(av.VideoFrame.from_image(canvas)):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    temp.replace(target)
    with av.open(str(target)) as check:
        decoded = list(check.decode(video=0))
        assert len(decoded) == 56 and len({frame.pts for frame in decoded}) == 56
        assert check.streams.video[0].average_rate == 24
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", choices=["industrial", "village"], required=True)
    args = parser.parse_args()
    for branch in ("AA", "AD"):
        print(make(args.scene, branch))
