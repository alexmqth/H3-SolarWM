"""Materialize the 36 EXP-021-selected real RGB windows, CPU only."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import av
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageOps, ImageDraw

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = ROOT / "submission/experiments/EXP-021_v3_fm_baseline_preparation/DATA_MANIFEST.json"
OUT = HERE / "artifacts/data"


def sha(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def inspect(path: Path, sheet_path: Path) -> dict:
    selected = {0, 8, 16, 24, 32, 39, 47, 55}
    tiles = []
    brightness = []
    pts = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_count = 1
        assert stream.average_rate == 24
        assert (stream.width, stream.height) == (832, 480)
        for i, frame in enumerate(container.decode(video=0)):
            x = frame.to_ndarray(format="rgb24")
            assert x.shape == (480, 832, 3)
            pts.append(frame.pts)
            brightness.append(float(x.mean()))
            if i in selected:
                tile = Image.fromarray(x).resize((208, 120))
                tiles.append((i, tile))
    assert len(pts) == 56 and len(set(pts)) == 56
    assert all(a < b for a, b in zip(pts, pts[1:]))
    sheet = Image.new("RGB", (832, 264), "white")
    draw = ImageDraw.Draw(sheet)
    for j, (i, tile) in enumerate(tiles):
        x, y = (j % 4) * 208, (j // 4) * 132
        sheet.paste(tile, (x, y))
        draw.text((x + 4, y + 119), f"RGB {i}", fill="black")
    sheet.save(sheet_path, quality=85)
    return {"frames": 56, "fps": 24, "resolution": [832, 480],
            "unique_pts": 56, "brightness_min": min(brightness),
            "brightness_max": max(brightness),
            "near_black_frames_mean_lt_3": sum(v < 3 for v in brightness),
            "contact_sheet": str(sheet_path.relative_to(ROOT))}


def main():
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in ("", "-1"):
        raise RuntimeError("CPU task only")
    started = time.monotonic()
    baseline_free = shutil.disk_usage(HERE).free
    manifest = json.loads(SOURCE.read_text())
    rows = [x for x in manifest["train_candidates"] + manifest["dev_candidates"]
            if not x["materialized_real56"]]
    assert len(rows) == 36 and len({x["clip_id"] for x in rows}) == 36
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    source_by_episode = {}
    result = []
    for row in rows:
        if time.monotonic() - started > 7200:
            raise TimeoutError("CPU task 2h limit")
        if baseline_free - shutil.disk_usage(HERE).free > 12 * 2**30:
            raise RuntimeError("12GiB new disk limit")
        episode = row["episode_id"]
        source = ROOT / row["source_video"]
        if episode not in source_by_episode:
            assert source.is_file() and sha(source) == row["source_video_sha256"]
            source_by_episode[episode] = True
        start = row["source_start"]
        assert row["source_indices"][-1] == start + 68
        dest = OUT / row["clip_id"]
        dest.mkdir(parents=True, exist_ok=True)
        output = dest / "real56.mp4"
        temp = dest / "real56.partial.mp4"
        sheet = dest / "contact.jpg"
        if not output.exists():
            if temp.exists():
                raise FileExistsError(f"manual audit required: {temp}")
            vf = (f"select='between(n\\,{start}\\,{start+68})"
                  f"*not(eq(mod(n-{start}\\,5)\\,4))',"
                  "setpts=N/24/TB,scale=-2:480,crop=832:480")
            cmd = [ffmpeg, "-y", "-v", "error", "-threads", "1", "-filter_threads", "1",
                   "-i", str(source), "-vf", vf, "-r", "24", "-frames:v", "56",
                   "-c:v", "libx264", "-threads:v", "1", "-crf", "14", "-preset", "veryfast",
                   "-pix_fmt", "yuv420p", "-an", "-movflags", "+faststart", str(temp)]
            p = subprocess.run(cmd, capture_output=True, text=True)
            if p.returncode:
                raise RuntimeError(f"{row['clip_id']}: FFmpeg exit {p.returncode}: {p.stderr[-2000:]}")
            temp.rename(output)
        quality = inspect(output, sheet)
        result.append({"clip_id": row["clip_id"], "split": row["split"],
                       "episode_id": episode, "source_start": start,
                       "source_video_sha256": row["source_video_sha256"],
                       "action_script_sha256": row["action_script_sha256"],
                       "video": str(output.relative_to(ROOT)), "video_sha256": sha(output),
                       "bytes": output.stat().st_size, "quality": quality,
                       "exclusion_reason": None if quality["near_black_frames_mean_lt_3"] == 0
                       else "near-black frame(s); inspect before training"})
        print(f"{len(result):02d}/36 {row['clip_id']} RGB PASS", flush=True)
    record = {"task": "EXP-022/v1", "source_manifest": str(SOURCE.relative_to(ROOT)),
              "source_manifest_sha256": sha(SOURCE), "status": "CPU_RGB_MATERIALIZED",
              "gpu_calls": 0, "vae_calls": 0, "text_calls": 0,
              "elapsed_seconds": time.monotonic() - started,
              "new_disk_bytes": baseline_free - shutil.disk_usage(HERE).free,
              "rows": result}
    path = HERE / "RGB_MATERIALIZATION.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    print("MATERIALIZATION COMPLETE", len(result), flush=True)


if __name__ == "__main__":
    main()
