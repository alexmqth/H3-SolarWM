"""Verify the real56 FFmpeg selection against actual decoded source PTS."""
from __future__ import annotations

import os
import re
import subprocess

import imageio_ffmpeg

from build_real56 import HERE, rows, write_json


def main() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("CPU-only audit requires CUDA_VISIBLE_DEVICES=''")
    # Keep at most four physical CPUs runnable even if FFmpeg starts helpers.
    cpus = sorted(os.sched_getaffinity(0))[:4]
    os.sched_setaffinity(0, cpus)
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    scenes = []
    for item in rows():
        start = item["candidate"]["src_start"]
        last = item["indices"][-1]
        vf = (f"select='between(n\\,{start}\\,{last})"
              f"*not(eq(mod(n-{start}\\,5)\\,4))',showinfo")
        cmd = [ffmpeg, "-v", "info", "-threads", "1", "-filter_threads", "1",
               "-i", str(item["video"]), "-vf", vf, "-frames:v", "56",
               "-fps_mode", "passthrough", "-an", "-f", "null", "-"]
        run = subprocess.run(cmd, capture_output=True, text=True)
        if run.returncode:
            raise RuntimeError(f"FFmpeg selection failed for {start}: {run.stderr[-1200:]}")
        found = [(int(a), int(b)) for a, b in re.findall(
            r"\[Parsed_showinfo_\d+[^\n]*?\] n:\s*(\d+)\s+pts:\s*(\d+)", run.stderr)]
        assert len(found) == 56, (start, len(found))
        assert [n for n, _ in found] == list(range(56))
        actual = [pts // 512 for _, pts in found]
        assert all(pts % 512 == 0 for _, pts in found)
        assert actual == item["indices"], (start, actual, item["indices"])
        scenes.append({"scene": item["candidate"]["episode_id"][:8],
                       "actual_source_indices": actual,
                       "selected_source_pts": [pts for _, pts in found],
                       "matched_expected_56": True, "command": cmd})
        print(scenes[-1]["scene"], "56 source PTS/index PASS", flush=True)
    write_json(HERE / "SOURCE_PTS_AUDIT.json",
               {"method": "FFmpeg select,showinfo before setpts; source 30fps PTS increments 512",
                "cpu_affinity": cpus, "GPU_calls": 0, "scenes": scenes})


if __name__ == "__main__":
    main()
