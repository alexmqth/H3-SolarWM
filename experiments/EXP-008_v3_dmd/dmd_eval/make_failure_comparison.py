"""CPU-only 56-frame AA comparison for the stopped cycle8 DMD evaluation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import av
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[4]
FM8 = ROOT / "H3-World/outputs/EXP-006_v3_fm8_full"
AF8 = ROOT / "H3-World/outputs/EXP-007_v3_anyflow_af3"
DMD8 = ROOT / "H3-World/outputs/EXP-008_v3_dmd_eval"
HERE = Path(__file__).resolve().parent
DEST = HERE / "artifacts/failure_comparison"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def decode(path: Path, count: int):
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        frames = list(container.decode(video=0))
        pts = [f.pts for f in frames]
        assert len(frames) >= count and len(set(pts)) == len(frames) and pts == sorted(pts)
        assert stream.average_rate == 24
        assert all(f.width == 832 and f.height == 480 for f in frames)
    return frames[:count]


def main() -> None:
    sources = [FM8 / "AA/rollout_73.mp4", AF8 / "AA/rollout_73.mp4",
               DMD8 / "AA/rollout_56.mp4"]
    frames = [decode(path, 56) for path in sources]
    first = np.load(FM8 / "published_39.npy")
    assert np.array_equal(first, np.load(DMD8 / "AA/published_56.npy")[:39])
    DEST.mkdir(parents=True, exist_ok=True)
    target = DEST / "AA_FM8_AF8_DMD8_cycle8_collapse_56.mp4"
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    proc = subprocess.Popen([ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", "1872x424",
        "-framerate", "24", "-i", "-", "-an", "-c:v", "libx264", "-crf", "19",
        "-threads", "4", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(target)],
        stdin=subprocess.PIPE)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    labels = ("V3 ordinary FM8", "V3 AnyFlow 8NFE | AF2 step32",
              "DMD8 | cycle8 | FAILED: full-frame noise")
    try:
        for i in range(56):
            image = Image.new("RGB", (1872, 424), "black")
            for j in range(3):
                rgb = Image.fromarray(frames[j][i].to_ndarray(format="rgb24"))
                image.paste(rgb.resize((624, 360)), (624 * j, 40))
            draw = ImageDraw.Draw(image)
            for j, label in enumerate(labels):
                draw.text((624 * j + 8, 10), label, font=font, fill="white")
            draw.text((8, 404), f"Same FM8 C1 (0–38); AA C2 starts at frame 39 | frame {i}",
                      font=small, fill="yellow")
            proc.stdin.write(image.tobytes())
        proc.stdin.close()
        assert proc.wait() == 0
    finally:
        if proc.poll() is None:
            proc.kill()
    with av.open(str(target)) as container:
        output = list(container.decode(video=0))
        assert len(output) == 56 and all(f.width == 1872 and f.height == 424 for f in output)
    record = {"status": "DMD8_AA_C2_FAILED_STOPPED", "video": str(target.relative_to(ROOT)),
              "sha256": sha(target), "frames": 56, "fps": 24,
              "source_sha256": {label: sha(path) for label, path in zip(("FM8", "AF8", "DMD8"), sources)},
              "note": "Identical first 39 RGB frames; only AA C2 completed. AD was interrupted, C3 not run."}
    (HERE / "failure_comparison_manifest.json").write_text(json.dumps(record,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(record,ensure_ascii=False))


if __name__ == "__main__":
    main()
