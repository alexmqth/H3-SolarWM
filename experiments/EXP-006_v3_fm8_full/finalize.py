"""Package small EXP-006 evidence and make explicitly cross-history comparisons.

CPU only. Original measurements and videos remain untouched in H3-World/outputs.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import av
import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUT = ROOT / "H3-World/outputs/EXP-006_v3_fm8_full"
OLD = ROOT / "H3-World/outputs/EXP-002_native_cached"
ART = HERE / "artifacts"
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def inspect_video(path: Path) -> dict:
    with av.open(str(path)) as container:
        video = container.streams.video[0]
        frames = list(container.decode(video=0))
        pts = [int(f.pts) for f in frames]
        assert len(pts) == len(set(pts)) and pts == sorted(pts)
        assert all(f.width == 832 and f.height == 480 for f in frames)
        assert len(frames) in (39, 56, 73)
        return dict(frames=len(frames), width=832, height=480,
                    average_rate=str(video.average_rate), decoded=True)


def comparison(path: str) -> dict:
    old = OLD / path / "rollout_73.mp4"
    new = OUT / path / "rollout_73.mp4"
    target = ART / "videos" / f"{path}_FM30_saved_first_vs_FM8_new_first_73.mp4"
    label_left = "V3 FM30 | saved 30-step first | 30 steps/chunk"
    label_right = "V3 FM8 | new 8-step first | 8 steps/chunk"
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    note_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    process = subprocess.Popen([FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
                                "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", "1248x400",
                                "-framerate", "24", "-i", "-", "-an", "-c:v", "libx264",
                                "-crf", "19", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                                str(target)], stdin=subprocess.PIPE)
    try:
        with av.open(str(old)) as a, av.open(str(new)) as b:
            for i, (left, right) in enumerate(zip(a.decode(video=0), b.decode(video=0))):
                if i == 73:
                    break
                canvas = Image.new("RGB", (1248, 400), "black")
                canvas.paste(Image.fromarray(left.to_ndarray(format="rgb24")).resize((624, 360)), (0, 40))
                canvas.paste(Image.fromarray(right.to_ndarray(format="rgb24")).resize((624, 360)), (624, 40))
                draw = ImageDraw.Draw(canvas)
                draw.text((8, 10), label_left, font=font, fill="white")
                draw.text((632, 10), label_right, font=font, fill="white")
                draw.rectangle((0, 376, 1248, 400), fill="black")
                draw.text((8, 380), f"Different first/history states; cross-protocol | action {path} | frame {i}",
                          font=note_font, fill="yellow")
                process.stdin.write(canvas.tobytes())
        process.stdin.close()
        assert process.wait() == 0
    finally:
        if process.poll() is None:
            process.kill()
    with av.open(str(target)) as container:
        frames = list(container.decode(video=0))
        assert len(frames) == 73 and all(f.width == 1248 and f.height == 400 for f in frames)
    return dict(path=str(target.relative_to(ROOT)), sha256=digest(target),
                frames=73, fps=24, dimensions=[1248, 400], decoded=True,
                note="Different first-window generation and subsequent history; not a single-variable ablation")


def main() -> None:
    ART.joinpath("videos").mkdir(parents=True, exist_ok=True)
    ART.joinpath("json").mkdir(parents=True, exist_ok=True)
    budget = json.loads((OUT / "budget.json").read_text())
    assert (budget["sampling"], budget["commits"], budget["total_forwards"], budget["vae_decodes"]) == (40, 3, 43, 5)
    rows = []
    for stage, action, start, stop in [("first", None, 0, 12),
                                       ("second", "AA", 12, 17),
                                       ("second", "AD", 12, 17),
                                       ("third", "AA", 17, 22),
                                       ("third", "AD", 17, 22)]:
        folder = OUT if action is None else OUT / action
        source = folder / f"chunk_{start}_{stop}.json"
        row = json.loads(source.read_text())
        assert row["status"] == "complete_pending_visual_review"
        target = ART / "json" / (f"{action or 'shared'}_chunk_{start}_{stop}.json")
        shutil.copy2(source, target)
        rgb = {12: 39, 17: 56, 22: 73}[stop]
        video = folder / f"rollout_{rgb}.mp4"
        check = inspect_video(video)
        assert check["frames"] == rgb
        if stop == 22 or stage == "first":
            shutil.copy2(video, ART / "videos" / (f"{action or 'shared'}_FM8_{rgb}.mp4"))
        rows.append(dict(stage=stage, path=action, latent=[start, stop], video=str(video.relative_to(ROOT)),
                         video_sha256=digest(video), verification=check,
                         sampling_seconds=row["sampling_seconds"],
                         commit_seconds=row.get("commit_seconds", 0),
                         decode_seconds=row["decode_seconds"],
                         peak_allocated_gib=row["peak_allocated_gib"],
                         cpu_kv_bytes=row.get("cache_after", row.get("cache_before", {})).get("nbytes", 0),
                         flow_horizontal=row["flow"]["horizontal_flow_px"]["mean"],
                         boundary_gray_MAD=row.get("boundary_gray_MAD"),
                         inside_gray_MAD=row["inside_gray_MAD"]))
    pairs = [comparison(action) for action in ("AA", "AD")]
    result = dict(task="EXP-006/v1", status="worker_complete_pending_judge",
                  gpu_seconds=budget["gpu_seconds"], forwards=budget["total_forwards"],
                  sampling=budget["sampling"], commits=budget["commits"], vae=budget["vae_decodes"],
                  rows=rows, comparisons=pairs,
                  reference="EXP-002 saved 30-step first; generated histories differ after first window")
    (HERE / "worker_metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(dict(status=result["status"], comparisons=pairs), ensure_ascii=False))


if __name__ == "__main__":
    main()
