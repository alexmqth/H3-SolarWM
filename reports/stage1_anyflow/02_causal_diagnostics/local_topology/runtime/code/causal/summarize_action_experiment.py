#!/usr/bin/env python3
"""Collect the four fixed-action interventions without treating motion as quality."""
from __future__ import annotations

import argparse
from itertools import zip_longest
import json
from pathlib import Path

import av
import numpy as np

from evaluate_action_control import evaluate as evaluate_flow
from evaluate_videos import evaluate as evaluate_video


def rgb_frames(path):
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            yield frame.to_ndarray(format="rgb24").astype(np.int16)


def rgb_boundary(path, boundaries):
    previous = None
    values = []
    for index, frame in enumerate(rgb_frames(path)):
        if index in boundaries:
            values.append(float(np.abs(frame - previous).mean()))
        previous = frame
    if len(values) != len(boundaries):
        raise ValueError(f"missing boundary frames in {path}")
    return {"frames": boundaries, "rgb_mad": values, "mean": float(np.mean(values))}


def pair_difference(left, right):
    values = []
    for a, b in zip_longest(rgb_frames(left), rgb_frames(right)):
        if a is None or b is None or a.shape != b.shape:
            raise ValueError("paired videos must have identical frame counts and size")
        values.append(float(np.abs(a - b).mean()))
    return {"mean_rgb_abs_diff": float(np.mean(values)),
            "p95_frame_mean_rgb_abs_diff": float(np.percentile(values, 95)),
            "max_frame_mean_rgb_abs_diff": float(np.max(values))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--causal-root", type=Path, required=True)
    parser.add_argument("--causal-suffix", required=True)
    parser.add_argument("--teacher-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows, paths, configs, setups = [], {}, [], []
    for action in "WSAD":
        pair = {"action": action}
        for method, directory, mode in (
            ("original", args.teacher_root / f"action_{action}_teacher_latents", "baseline"),
            ("causal", args.causal_root / f"action_{action}_{args.causal_suffix}", "cached"),
        ):
            config = json.loads((directory / f"{mode}.json").read_text())
            setup = json.loads((directory / "setup.json").read_text())
            assert config["status"] == "complete", directory
            assert config["num_frames"] == 124 and config["seed"] == 13, directory
            assert config["action_preset"] == action, directory
            if method == "causal":
                assert config["denoiser_forwards"] == 64 and config["commit_forwards"] == 8
                assert config["chunk_frames"] == 5 and config["history_chunks"] == 5
                assert config["anchor_mode"] == "dynamic_last_frame_dual"
                assert config["cache_device"] == "cpu"
                configs.append(config)
                setups.append(setup)
            path = directory / f"{mode}.mp4"
            stats = evaluate_video(path)
            assert stats["frames"] == 124 and stats["fps"] == 24.0, path
            flow = evaluate_flow(path)
            boundary = rgb_boundary(path, list(range(17, 124, 17)))
            pair[method] = dict(
                path=str(path), sampling_seconds=config["sampling_seconds"],
                wall_after_conditioning_seconds=config["wall_after_conditioning_seconds"],
                wall_including_setup_seconds=config["wall_including_shared_setup_seconds"],
                gpu_allocated_sampling_peak_GiB=config["sampling_memory"]["allocated_peak_MiB"] / 1024,
                gpu_allocated_entire_run_peak_GiB=config["memory_entire_run_peak_MiB"] / 1024,
                gpu_reserved_sampling_peak_GiB=config["sampling_memory"]["reserved_peak_MiB"] / 1024,
                cpu_kv_peak_GiB=config.get("kv_cache_peak_MiB", 0) / 1024,
                denoiser_forwards=config["denoiser_forwards"], clean_commits=config.get("commit_forwards", 0),
                frame_mad_mean=stats["gray_pixel_difference_mean"],
                frame_pixel_diff_p95=stats["gray_pixel_difference_p95"],
                spatial_edge_difference=stats["mean_spatial_edge_difference"],
                boundary_rgb_mad=boundary,
                horizontal_flow_mean=flow["horizontal_flow_px"]["mean"],
                vertical_flow_mean=flow["vertical_flow_px"]["mean"],
            )
            paths[method, action] = path
        rows.append(pair)
    assert len({c["causal_adapter"] for c in configs}) == 1
    metadata = setups[0]["causal_adapter"]["metadata"]
    payload = dict(
        frames=124, fps=24, seed=13, causal_adapter=configs[0]["causal_adapter"],
        training_replay_max_error=metadata["replay_max_error"],
        training_replay_note="Zero-initialized tail feature replay, not an action or quality score.",
        actions=rows,
        paired_action_difference={
            method: {f"{a}_{b}": pair_difference(paths[method, a], paths[method, b])
                     for a, b in (("W", "S"), ("A", "D"))}
            for method in ("original", "causal")},
        caveats=["Fixed seed, one scene; four teacher clips are training data, not held-out scenes.",
                 "MAD is motion activity; RGB differences only show action sensitivity, not correctness.",
                 "Horizontal flow at 416x240 central crop is a proxy, not character action accuracy.",
                 "Timing includes CPU-offloaded weights/KV and concurrent host contention; no isolated speedup claim.",
                 "Boundary score is mean RGB adjacent-frame MAD at nominal chunk starts 17,34,...,119."])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    lines = ["| Action | Method | Sampling s | Total s | GPU peak GiB | CPU KV GiB | Noisy + commit | Gray MAD | RGB boundary | Horizontal flow |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for pair in rows:
        for method in ("original", "causal"):
            r = pair[method]
            lines.append(f"| {pair['action']} | {method} | {r['sampling_seconds']:.1f} | {r['wall_including_setup_seconds']:.1f} | "
                         f"{r['gpu_allocated_entire_run_peak_GiB']:.2f} | {r['cpu_kv_peak_GiB']:.2f} | "
                         f"{r['denoiser_forwards']} + {r['clean_commits']} | {r['frame_mad_mean']:.3f} | "
                         f"{r['boundary_rgb_mad']['mean']:.3f} | {r['horizontal_flow_mean']:+.3f} |")
    lines += ["", "GPU peak is PyTorch allocated over the entire run; total includes model load, conditioning, sampling, VAE and MP4 writing.",
              "Timings were collected under concurrent workloads and do not establish an isolated speedup.", ""]
    args.out.with_suffix(".md").write_text("\n".join(lines))
    print("\n".join(lines))
    print(json.dumps(payload["paired_action_difference"], indent=2))


if __name__ == "__main__":
    main()
