"""Decode every EXP-018 source and display MP4 and record its exact media properties."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import av


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
ROWS = json.loads((HERE / "results_index.json").read_text())["rows"]
ORDER = ("v0", "v1", "v2a", "v2b", "v3_30", "v3_8", "v4")


def inspect(relative: str, expected_frames: int) -> dict:
    path = ROOT / relative
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        pts = []
        dimensions = set()
        for frame in container.decode(stream):
            pts.append(frame.pts)
            dimensions.add((frame.width, frame.height))
        fps = str(stream.average_rate)
        time_base = str(stream.time_base)
    valid = (
        len(pts) == expected_frames
        and all(p is not None for p in pts)
        and len(set(pts)) == len(pts)
        and pts == sorted(pts)
        and len(dimensions) == 1
        and fps in ("24", "24/1")
    )
    return {
        "path": relative,
        "sha256": digest.hexdigest(),
        "bytes": path.stat().st_size,
        "decoded_frames": len(pts),
        "fps": fps,
        "time_base": time_base,
        "first_pts": pts[0] if pts else None,
        "last_pts": pts[-1] if pts else None,
        "dimensions": sorted([list(x) for x in dimensions]),
        "expected_frames": expected_frames,
        "pass": valid,
    }


result = {"rows": {}, "all_pass": True}
for row in ORDER:
    info = ROWS[row]
    pair = {
        "full_124": inspect(info["full_video"], 124),
        "display_120": inspect(info["display_video"], 120),
    }
    result["rows"][row] = pair
    result["all_pass"] &= all(x["pass"] for x in pair.values())

(HERE / "media_validation.json").write_text(json.dumps(result, indent=2) + "\n")
for row, pair in result["rows"].items():
    print(row, [(name, value["decoded_frames"], value["fps"], value["dimensions"]) for name, value in pair.items()])
print("all_pass", result["all_pass"])
raise SystemExit(0 if result["all_pass"] else 1)
