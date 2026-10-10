"""CPU-only matched-frame G2 comparison videos; reads saved MP4 only."""
from __future__ import annotations

from pathlib import Path

import av
from PIL import Image, ImageDraw, ImageFont

from common import HERE, OUT


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
    target_dir = HERE / "artifacts/G2/comparisons"
    for scene in ("industrial", "village"):
        for branch in ("AA", "AD"):
            a = OUT / "G2" / scene / "FM30" / branch / "rollout_56.mp4"
            b = OUT / "G2" / scene / "FM8" / branch / "rollout_56.mp4"
            make(a, b, target_dir / f"{scene}_{branch}_FM30_vs_FM8_56.mp4",
                 f"V3 causal FM30 | {branch} | 30 NFE/chunk",
                 f"V3 causal FM8 | {branch} | 8 NFE/chunk",
                 f"{scene} | same I0/noise; own C1 per NFE")
        for method in ("FM30", "FM8"):
            a = OUT / "G2" / scene / method / "AA/rollout_56.mp4"
            b = OUT / "G2" / scene / method / "AD/rollout_56.mp4"
            make(a, b, target_dir / f"{scene}_{method}_AA_vs_AD_56.mp4",
                 f"V3 causal {method} | AA", f"V3 causal {method} | AD",
                 f"{scene} | same C1 cache/noise, C2 action A vs D")
