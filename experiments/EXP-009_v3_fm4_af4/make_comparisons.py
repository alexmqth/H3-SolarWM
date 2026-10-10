"""CPU-only 2x2 gallery after the authorized EXP-009 videos exist."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import av
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
FM4 = ROOT / "H3-World/outputs/EXP-009_v3_fm4_af4/fm4"
AF4 = ROOT / "H3-World/outputs/EXP-009_v3_fm4_af4/af4"
FM8 = ROOT / "H3-World/outputs/EXP-006_v3_fm8_full"
AF8 = ROOT / "H3-World/outputs/EXP-007_v3_anyflow_af3"
DEST = HERE / "artifacts/comparisons"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def frames(path: Path, count: int):
    if not path.is_file():
        raise FileNotFoundError(f"real source video missing: {path}")
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        decoded = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
        if (len(decoded) != count or stream.average_rate != 24 or
                any(f.shape != (480, 832, 3) for f in decoded)):
            raise ValueError(f"unexpected video dimensions/FPS/frames: {path}")
    return decoded


def render(action: str, count: int) -> dict:
    sources = [
        ("Ordinary FM4 | Original H3", FM4 / action),
        ("AnyFlow AF4 | step32 QKV/time", AF4 / action),
        ("Ordinary FM8 | reference", FM8 / action),
        ("AnyFlow AF8 | reference", AF8 / action),
    ]
    paths = [(label, folder / f"rollout_{count}.mp4") for label, folder in sources]
    decoded = [frames(path, count) for _, path in paths]
    published = np.load(FM8 / "published_39.npy", mmap_mode="r")
    for _, folder in sources:
        raw = np.load(folder / f"published_{count}.npy", mmap_mode="r")
        if not np.array_equal(raw[:39], published):
            raise ValueError(f"first39 RGB differ: {folder}")

    DEST.mkdir(parents=True, exist_ok=True)
    output = DEST / f"{action}_FM4_AF4_with_8NFE_{count}.mp4"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    process = subprocess.Popen(
        [ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "rawvideo",
         "-pixel_format", "rgb24", "-video_size", "1248x848", "-framerate", "24",
         "-i", "-", "-an", "-c:v", "libx264", "-crf", "19", "-threads", "4",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output)],
        stdin=subprocess.PIPE,
    )
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    try:
        for index in range(count):
            canvas = Image.new("RGB", (1248, 848), "black")
            draw = ImageDraw.Draw(canvas)
            for panel, (label, _) in enumerate(paths):
                x = (panel % 2) * 624
                y = (panel // 2) * 400
                canvas.paste(Image.fromarray(decoded[panel][index]).resize((624, 360)), (x, y + 28))
                draw.text((x + 8, y + 5), label, font=font, fill="white")
            draw.text((8, 805),
                      f"{action} | shared FM8 RGB 0-38 | C2 39-55 same history | C3 56-72 own history | frame {index}",
                      font=small, fill="yellow")
            draw.text((8, 824), "4 vs 8 NFE; AF uses trained step32 QKV/time. Not a single-variable ablation.",
                      font=small, fill="yellow")
            process.stdin.write(canvas.tobytes())
        process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError("ffmpeg encode failed")
    finally:
        if process.poll() is None:
            process.kill()
    with av.open(str(output)) as container:
        stream = container.streams.video[0]
        result = list(container.decode(video=0))
        if (len(result) != count or stream.average_rate != 24 or
                any(frame.width != 1248 or frame.height != 848 for frame in result)):
            raise ValueError(f"encoded video failed decode verification: {output}")
    return {"action": action, "frames": count, "fps": 24,
            "output": str(output.relative_to(ROOT)), "sha256": sha(output),
            "sources": {label: {"path": str(path.relative_to(ROOT)), "sha256": sha(path)}
                        for label, path in paths}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, choices=(56, 73), required=True)
    args = parser.parse_args()
    manifest = [render(action, args.frames) for action in ("AA", "AD")]
    (DEST / f"comparison_{args.frames}_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
