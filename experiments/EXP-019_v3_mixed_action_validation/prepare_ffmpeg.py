"""CPU-only codec preflight for G stages; does not touch the frozen model runner."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import av
import imageio_ffmpeg


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = ROOT / "H3-World/outputs/EXP-019_v3_mixed_action_validation"
BIN = OUT / "bin"
LINK = BIN / "ffmpeg"
SOURCE = ROOT / "submission/experiments/EXP-015_v3_real56_data/artifacts/43866101/real56.mp4"
SHORT = OUT / "codec_preflight_short8.mp4"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def media(path: Path) -> dict:
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        pts = [frame.pts for frame in container.decode(stream)]
        return {
            "decoded_frames": len(pts),
            "fps": str(stream.average_rate),
            "dimensions": [stream.width, stream.height],
            "pts_unique_increasing": all(a < b for a, b in zip(pts, pts[1:])),
        }


actual = Path(imageio_ffmpeg.get_ffmpeg_exe()).resolve(strict=True)
BIN.mkdir(parents=True, exist_ok=True)
if LINK.is_symlink():
    assert LINK.resolve(strict=True) == actual, "Existing task ffmpeg link points elsewhere"
elif LINK.exists():
    raise RuntimeError("Refusing to replace existing non-symlink ffmpeg")
else:
    LINK.symlink_to(actual)

version = subprocess.check_output([str(LINK), "-version"], text=True).splitlines()[0]
subprocess.run([str(LINK), "-v", "error", "-threads", "2", "-i", str(SOURCE),
                "-f", "null", "-"], check=True)
source_media = media(SOURCE)
assert source_media["decoded_frames"] == 56 and source_media["pts_unique_increasing"]
subprocess.run([str(LINK), "-v", "error", "-threads", "2", "-i", str(SOURCE),
                "-frames:v", "8", "-an", "-c:v", "libx264", "-threads", "2",
                "-crf", "18", "-pix_fmt", "yuv420p", "-y", str(SHORT)], check=True)
short_media = media(SHORT)
assert short_media["decoded_frames"] == 8 and short_media["pts_unique_increasing"]

result = {
    "status": "PASS",
    "scope": "CPU codec only; no model inference",
    "task_bin": str(BIN),
    "link": str(LINK),
    "realpath": str(actual),
    "binary_sha256": sha(actual),
    "version": version,
    "launch_path_prefix": str(BIN) + ":" + os.environ.get("PATH", ""),
    "source": str(SOURCE.relative_to(ROOT)),
    "source_sha256": sha(SOURCE),
    "source_media": source_media,
    "encoded_short": str(SHORT.relative_to(ROOT)),
    "encoded_short_sha256": sha(SHORT),
    "encoded_short_media": short_media,
}
(HERE / "FFMPEG_CPU_PREFLIGHT.json").write_text(json.dumps(result, indent=2) + "\n")
print("CPU ffmpeg preflight PASS:", version, "56 decoded; 8 encoded and decoded")
