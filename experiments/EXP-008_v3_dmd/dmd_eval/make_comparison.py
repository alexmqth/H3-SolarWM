"""CPU-only matched FM8 / AF8 / DMD8 73-frame gallery after video release."""
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
DEST = HERE / "artifacts/comparison"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def decode(path: Path):
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        frames = list(container.decode(video=0))
        pts = [f.pts for f in frames]
        assert len(frames) == len(set(pts)) == 73 and pts == sorted(pts)
        assert stream.average_rate == 24
        assert all(f.width == 832 and f.height == 480 for f in frames)
    return frames


def render(path: str) -> dict:
    sources = [FM8 / path / "rollout_73.mp4",
               AF8 / path / "rollout_73.mp4",
               DMD8 / path / "rollout_73.mp4"]
    frames = [decode(source) for source in sources]
    first = np.load(FM8 / "published_39.npy")
    for root in (FM8, AF8, DMD8):
        assert np.array_equal(first, np.load(root / path / "published_73.npy")[:39])
    final_cycle = json.loads((DMD8 / "prefill_0_12.json").read_text())["checkpoint_cycle"]
    DEST.mkdir(parents=True, exist_ok=True)
    target = DEST / f"{path}_FM8_AF8_DMD8_73.mp4"
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    proc = subprocess.Popen([ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", "1872x424",
        "-framerate", "24", "-i", "-", "-an", "-c:v", "libx264", "-crf", "19",
        "-threads", "4", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(target)],
        stdin=subprocess.PIPE)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    labels = ("V3 FM8 | original weights", "V3 AF8 | AF2 step32", f"V3 DMD8 | cycle{final_cycle}")
    try:
        for i in range(73):
            image = Image.new("RGB", (1872, 424), "black")
            draw = ImageDraw.Draw(image)
            for j in range(3):
                rgb = Image.fromarray(frames[j][i].to_ndarray(format="rgb24"))
                image.paste(rgb.resize((624, 360)), (624 * j, 40))
                draw.text((624 * j + 8, 10), labels[j], font=font, fill="white")
            draw.text((8, 404),
                f"Shared FM8 C1 | C2 matched history | C3 own history | {path} | frame {i}",
                font=small, fill="yellow")
            proc.stdin.write(image.tobytes())
        proc.stdin.close()
        assert proc.wait() == 0
    finally:
        if proc.poll() is None:
            proc.kill()
    with av.open(str(target)) as container:
        output = list(container.decode(video=0))
        assert len(output) == 73 and all(f.width == 1872 and f.height == 424 for f in output)
    return {"path": str(target.relative_to(ROOT)), "sha256": sha(target),
            "dmd_cycle": final_cycle,
            "frames": 73, "fps": 24,
            "source_sha256": {label: sha(file) for label, file in zip(("FM8", "AF8", "DMD8"), sources)},
            "protocol_note": "Identical FM8 C1 RGB; C2 same clean history; C3 own generated history."
                             " Distinct weights/training; not a single-variable ablation."}


def main() -> None:
    if not all((DMD8 / path / "rollout_73.mp4").exists() for path in ("AA", "AD")):
        print(json.dumps({"status": "WAITING_FOR_DMD8_VIDEO", "gpu_calls": 0}))
        return
    items = [render(path) for path in ("AA", "AD")]
    (HERE / "comparison_manifest.json").write_text(json.dumps(items, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(items, ensure_ascii=False))


if __name__ == "__main__":
    main()
