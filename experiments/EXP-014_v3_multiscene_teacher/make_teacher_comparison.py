"""CPU-only same-C1 AA/AD teacher comparison; reads saved MP4 only."""
from __future__ import annotations

import argparse
from pathlib import Path

import av
from PIL import Image, ImageDraw, ImageFont

from common import CFG, HERE, OUT


def frames(path: Path) -> list[Image.Image]:
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        assert (stream.width, stream.height, stream.average_rate) == (832, 480, 24)
        result = [Image.fromarray(frame.to_ndarray(format="rgb24"))
                  for frame in container.decode(video=0)]
    assert len(result) == 56
    return result


def make(left_path: Path, right_path: Path, target: Path,
         left_title: str, right_title: str, footer: str) -> None:
    left, right = frames(left_path), frames(right_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(target)
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    title_font = ImageFont.truetype(font_path, 22)
    body_font = ImageFont.truetype(font_path, 17)
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
            draw.text((16, 9), left_title, font=title_font, fill="white")
            draw.text((848, 9), right_title, font=title_font, fill="white")
            phase = "C1 A" if i < 39 else "C2 continuation"
            draw.text((16, 526), f"{footer} | {phase} | frame {i:02d}/55 | boundary f39",
                      font=body_font, fill="white")
            if i >= 39:
                draw.line((832, 44, 832, 524), fill="#ffd34d", width=3)
            for packet in stream.encode(av.VideoFrame.from_image(canvas)):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    temp.replace(target)
    with av.open(str(target)) as check:
        decoded = list(check.decode(video=0))
        assert len(decoded) == 56 and len({frame.pts for frame in decoded}) == 56
        assert check.streams.video[0].average_rate == 24
    print(target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", choices=list(CFG["scenes"]), required=True)
    args = parser.parse_args()
    scene = args.scene
    target_dir = HERE / "artifacts/teacher_comparisons"
    a = OUT / "G2" / scene / "FM30" / "AA/rollout_56.mp4"
    d = OUT / "G2" / scene / "FM30" / "AD/rollout_56.mp4"
    make(a, d, target_dir / f"{scene}_AA_vs_AD_56.mp4",
         "V3 FM30 | AA | 30 NFE/chunk", "V3 FM30 | AD | 30 NFE/chunk",
         f"{scene} | same generated C1/cache/noise; C2 action A vs D")
