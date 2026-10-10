"""CPU-only 30-step SW-G vs EXP-010 FM8 SW-G gallery after real 158f outputs exist."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import av
import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASE = ROOT / "H3-World/outputs/EXP-005_v3_sliding_window/G1"
NEW = ROOT / "H3-World/outputs/EXP-010_v3_fm8_sw158"
DEST = HERE / "artifacts/comparisons"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"real video missing: {path}")
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        frames = list(container.decode(video=0))
        pts = [frame.pts for frame in frames]
        if (len(frames) != 158 or stream.average_rate != 24 or
                any(frame.width != 832 or frame.height != 480 for frame in frames) or
                len(set(pts)) != 158 or pts != sorted(pts)):
            raise ValueError(f"invalid source video: {path}")
    return frames


def render(action: str) -> dict:
    left = BASE / action / "rollout_158.mp4"
    right = NEW / action / "rollout_158.mp4"
    old, new = read(left), read(right)
    DEST.mkdir(parents=True, exist_ok=True)
    target = DEST / f"{action}_SWG30_vs_FM8_SWG8_158.mp4"
    if target.exists():
        raise FileExistsError(f"refusing to overwrite {target}")
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    process = subprocess.Popen(
        [ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "rawvideo",
         "-pixel_format", "rgb24", "-video_size", "1248x424", "-framerate", "24",
         "-i", "-", "-an", "-c:v", "libx264", "-crf", "19", "-threads", "4",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(target)],
        stdin=subprocess.PIPE)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    try:
        for index, (old_frame, new_frame) in enumerate(zip(old, new)):
            canvas = Image.new("RGB", (1248, 424), "black")
            canvas.paste(Image.fromarray(old_frame.to_ndarray(format="rgb24")).resize((624,360)), (0,40))
            canvas.paste(Image.fromarray(new_frame.to_ndarray(format="rgb24")).resize((624,360)), (624,40))
            draw = ImageDraw.Draw(canvas)
            draw.text((8,10), "SW-G 30-step | existing reference", font=font, fill="white")
            draw.text((632,10), "FM8 + SW-G | EXP-010", font=font, fill="white")
            draw.text((8,404),
                      f"{action} | different generated histories, same parking seed/input protocol | frame {index}",
                      font=small, fill="yellow")
            process.stdin.write(canvas.tobytes())
        process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError("ffmpeg encoding failed")
    finally:
        if process.poll() is None:
            process.kill()
    with av.open(str(target)) as container:
        stream = container.streams.video[0]
        frames = list(container.decode(video=0))
        pts = [frame.pts for frame in frames]
        if (len(frames) != 158 or stream.average_rate != 24 or
                any(frame.width != 1248 or frame.height != 424 for frame in frames) or
                len(set(pts)) != 158 or pts != sorted(pts)):
            raise ValueError(f"invalid encoded comparison: {target}")
    return {"action": action, "frames": 158, "fps": 24,
            "output": str(target.relative_to(ROOT)), "sha256": sha(target),
            "left": {"path": str(left.relative_to(ROOT)), "sha256": sha(left)},
            "right": {"path": str(right.relative_to(ROOT)), "sha256": sha(right)},
            "comparison_limit": "different generated histories; not a matched-KV or isolated-step ablation"}


def main() -> None:
    result = [render(action) for action in ("A", "D")]
    (DEST / "comparison_158_manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
