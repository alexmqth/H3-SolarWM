#!/usr/bin/env python3
"""Audit matched long runs and describe all frames, including late failures."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import av
import cv2
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code" / "causal"))
from evaluate_action_control import evaluate as evaluate_flow


def analyze(video, out):
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate)
        count_hint = stream.frames
        sample_indices = sorted({0, *range(120, count_hint, 120), count_hint - 1})
        samples, diffs, edges = {}, [], []
        previous = None
        for i, frame in enumerate(container.decode(video=0)):
            rgb = frame.to_ndarray(format="rgb24").astype(np.int16)
            gray = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.int16)
            edges.append(float((np.abs(np.diff(gray, axis=0)).mean() + np.abs(np.diff(gray, axis=1)).mean()) / 2))
            if previous is not None:
                diffs.append(float(np.abs(rgb - previous).mean()))
            if i in sample_indices:
                samples[i] = frame.to_image().resize((624, 360))
            previous = rgb
        count = len(edges)
        boundaries = list(range(17, count, 17))
        boundary = {str(i): diffs[i - 1] for i in boundaries}
        windows = []
        # Absorb the 1/3-frame H3 alignment remainder into the last window,
        # rather than presenting it as an independent five-second interval.
        starts = [i for i in range(0, count, 120) if i == 0 or count-i >= 12]
        for start, stop in zip(starts, starts[1:] + [count]):
            # Count transitions by destination frame in each five-second interval.
            transitions = diffs[max(start, 1)-1:stop-1]
            windows.append(dict(start_frame=start, stop_frame_exclusive=stop,
                                mean_rgb_mad=float(np.mean(transitions)) if transitions else None,
                                mean_gray_edge=float(np.mean(edges[start:stop]))))
        result = dict(frames=count, fps=fps, duration_s=count/fps, width=stream.width,
                      height=stream.height, codec=stream.codec_context.name, pix_fmt=stream.pix_fmt,
                      mean_rgb_mad=float(np.mean(diffs)), p95_frame_rgb_mad=float(np.percentile(diffs, 95)),
                      max_frame_rgb_mad=float(np.max(diffs)), boundary_frames=boundaries,
                      boundary_rgb_mad=boundary, boundary_score_mean=float(np.mean(list(boundary.values()))),
                      adjacent_frame_rgb_mad=diffs, spatial_gray_edge=edges, five_second_windows=windows,
                      caveat="MAD/edge/flow are descriptive proxies, not perceptual quality or action accuracy.")
    cols = min(3, len(samples))
    sheet = Image.new("RGB", (624*cols, 390*((len(samples)+cols-1)//cols)), (18, 18, 18))
    draw = ImageDraw.Draw(sheet)
    for j, (i, frame) in enumerate(samples.items()):
        x, y = (j % cols)*624, (j // cols)*390
        sheet.paste(frame, (x, y+30))
        draw.text((x+10, y+8), f"frame {i} / {count-1} | {i/fps:.2f}s", fill="white")
    sheet.save(out / "contact_sheet.jpg", quality=92)
    result["horizontal_flow"] = evaluate_flow(video)
    result["video_sha256"] = hashlib.sha256(video.read_bytes()).hexdigest()
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--frames", type=int, nargs="+", default=[243, 481])
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    pairs = []
    for frames in args.frames:
        configs, records = [], []
        for name, mode in (("original", "baseline"), ("causal", "cached")):
            directory = args.runs_root / f"{name}_{frames}"
            run = json.loads((directory / f"{mode}.json").read_text())
            setup = json.loads((directory / "setup.json").read_text())
            assert run["status"] == "complete" and run["num_frames"] == frames
            configs.append(setup)
            out = args.out / f"{name}_{frames}"
            out.mkdir(parents=True, exist_ok=True)
            for file in (f"{mode}.json", "setup.json", "launch.json"):
                (out / file).write_bytes((directory / file).read_bytes())
            video = directory / f"{mode}.mp4"
            existing = out / "video_metrics.json"
            stats = json.loads(existing.read_text()) if existing.exists() else {}
            if stats.get("video_sha256") != hashlib.sha256(video.read_bytes()).hexdigest():
                stats = analyze(video, out)
            assert stats["frames"] == frames and stats["fps"] == 24
            (out / "video_metrics.json").write_text(json.dumps(stats, indent=2)+"\n")
            records.append(dict(method=name, frames=frames, duration_s=stats["duration_s"],
                                e2e_s=run["wall_including_shared_setup_seconds"], sampling_s=run["sampling_seconds"],
                                peak_gpu_MiB=run["memory_entire_run_peak_MiB"], cpu_kv_MiB=run.get("kv_cache_peak_MiB", 0),
                                noisy_forwards=run["denoiser_forwards"], commits=run.get("commit_forwards", 0),
                                chunks=run.get("chunk_seconds", []), rgb_anchor_total_s=sum(run.get("rgb_anchor_seconds", [])),
                                mean_rgb_mad=stats["mean_rgb_mad"], boundary_score_mean=stats["boundary_score_mean"],
                                horizontal_flow_mean=stats["horizontal_flow"]["horizontal_flow_px"]["mean"],
                                five_second_windows=stats["five_second_windows"]))
        assert configs[0]["input_fingerprints"] == configs[1]["input_fingerprints"], "Pair inputs mismatch"
        for key in ("seed", "num_frames", "flow_shift", "action_preset"):
            assert configs[0]["config"][key] == configs[1]["config"][key], key
        pairs.append(dict(frames=frames, input_fingerprints_equal=True, runs=records))
    payload = dict(pairs=pairs, timing_caveat="One run per setting; concurrent workloads and different CPU-offload reserves. No isolated speedup claim.")
    (args.out / "summary.json").write_text(json.dumps(payload, indent=2)+"\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
