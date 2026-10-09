#!/usr/bin/env python3
"""Report a small optical-flow action-response proxy for fixed-action videos.

This is deliberately a diagnostic, not an action-accuracy or video-quality
metric.  It reports signed frame-to-frame Farneback flow in the central 80% of
the image so that the result is easy to compare across the W/S/A/D intervention
videos.  The raw sign is retained; the direction convention is stated in the
output rather than silently flipped.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import av
import cv2
import numpy as np


def _frames(path: Path):
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            # Optical flow does not need the original 832x480 sampling.  A
            # fixed small image also keeps this diagnostic quick and stable.
            rgb = frame.to_ndarray(format="rgb24")
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            gray = cv2.resize(gray, (416, 240), interpolation=cv2.INTER_AREA)
            yield gray


def evaluate(path: Path) -> dict:
    previous = None
    horizontal = []
    vertical = []
    magnitude = []
    for gray in _frames(path):
        if previous is not None:
            flow = cv2.calcOpticalFlowFarneback(
                previous, gray, None, pyr_scale=0.5, levels=3,
                winsize=21, iterations=3, poly_n=5, poly_sigma=1.2,
                flags=0,
            )
            h, w = gray.shape
            # Ignore a narrow border where VAE padding and codec edges can
            # dominate the scene motion estimate.
            crop = flow[12:h - 12, 21:w - 21]
            fx, fy = crop[..., 0], crop[..., 1]
            horizontal.append(float(np.mean(fx)))
            vertical.append(float(np.mean(fy)))
            magnitude.append(float(np.mean(np.sqrt(fx * fx + fy * fy))))
        previous = gray
    if not horizontal:
        raise ValueError(f"{path}: fewer than two frames")

    def stats(values):
        a = np.asarray(values, dtype=np.float64)
        return {
            "mean": float(np.mean(a)),
            "median": float(np.median(a)),
            "p05": float(np.percentile(a, 5)),
            "p95": float(np.percentile(a, 95)),
            "mean_abs": float(np.mean(np.abs(a))),
        }

    return {
        "path": str(path),
        "frames": len(horizontal) + 1,
        "horizontal_flow_px": stats(horizontal),
        "vertical_flow_px": stats(vertical),
        "flow_magnitude_px": stats(magnitude),
        "direction_note": (
            "positive horizontal flow means image content moves toward increasing "
            "pixel x; negative means decreasing pixel x. This is a motion proxy, "
            "not a strict action-accuracy score."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("videos", nargs="+", type=Path,
                        help="LABEL=VIDEO, e.g. W=.../cached.mp4")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    rows = []
    for item in args.videos:
        if "=" not in str(item):
            raise SystemExit(f"expected LABEL=VIDEO, got {item}")
        label, raw_path = str(item).split("=", 1)
        row = evaluate(Path(raw_path))
        row["action"] = label
        rows.append(row)

    summary = {"videos": rows}
    by_action = {row["action"]: row for row in rows}
    for left, right in (("W", "S"), ("A", "D")):
        if left in by_action and right in by_action:
            summary[f"{left}_minus_{right}_horizontal_mean"] = (
                by_action[left]["horizontal_flow_px"]["mean"]
                - by_action[right]["horizontal_flow_px"]["mean"]
            )
            summary[f"{left}_minus_{right}_vertical_mean"] = (
                by_action[left]["vertical_flow_px"]["mean"]
                - by_action[right]["vertical_flow_px"]["mean"]
            )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
