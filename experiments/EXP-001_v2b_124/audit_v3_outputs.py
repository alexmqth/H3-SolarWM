"""CPU-only audit of every published EXP-001 V2b continuation chunk."""

import hashlib
import json
from fractions import Fraction
from pathlib import Path

import av
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUTPUT = ROOT / "H3-World/outputs/EXP-001_v2b_124"
RGB_STOPS = {17: 56, 22: 73, 27: 90, 32: 107, 37: 124}


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def check_video(path, expected):
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        pts = [frame.pts for frame in container.decode(video=0)]
        assert stream.codec_context.name == "h264"
        assert stream.average_rate == Fraction(24, 1)
        assert (stream.width, stream.height) == (832, 480)
        assert len(pts) == expected and all(p is not None for p in pts)
        assert all(b > a for a, b in zip(pts, pts[1:]))
        return {"path": str(path.relative_to(ROOT)), "sha256": sha(path),
                "frames": len(pts), "fps": 24, "codec": "h264", "first_pts": pts[0],
                "last_pts": pts[-1], "time_base": str(stream.time_base)}


def main():
    report = {"task_id": "EXP-001", "plan_version": 3, "paths": {}}
    for name in ("AA", "DD", "AD", "DA"):
        directory = OUTPUT / name
        state = json.loads((directory / "state.json").read_text())
        latest = {"AA": 124, "DD": 124, "AD": 73, "DA": 73}[name]
        assert state["next_start"] == {124: 37, 73: 22}[latest]
        published = np.load(directory / "published.npy", mmap_mode="r")
        assert published.shape == (latest, 480, 832, 3)
        assert published.dtype == np.uint8
        assert hashlib.sha256(published.tobytes()).hexdigest() == state["published_rgb_sha256"]
        assert sha(directory / "published.npy") == state["published_file_sha256"]
        intervals = [(17, 22)]
        if name in ("AA", "DD"):
            intervals += [(22, 27), (27, 32), (32, 37)]
        assert [tuple(x["interval"]) for x in state["results"]] == intervals
        clips = {}
        for start, stop in intervals:
            stem = f"chunk_{start}_{stop}"
            row_path = directory / (stem + ".json")
            row = json.loads(row_path.read_text())
            result = next(x for x in state["results"] if x["interval"] == [start, stop])
            assert sha(row_path) == result["result_sha256"]
            assert sha(directory / (stem + ".pt")) == row["endpoint_file_sha256"]
            assert hashlib.sha256(published[:RGB_STOPS[start]].tobytes()).hexdigest() == row["prior_published_rgb_sha256"]
            assert hashlib.sha256(published[:RGB_STOPS[stop]].tobytes()).hexdigest() == row["published_rgb_sha256"]
            full = check_video(directory / f"rollout_{RGB_STOPS[stop]}.mp4", RGB_STOPS[stop])
            chunk = check_video(directory / (stem + ".mp4"), 17)
            boundary = check_video(directory / f"boundary_{start}_{stop}.mp4", 2)
            assert full["sha256"] == row["rollout_mp4_sha256"]
            assert chunk["sha256"] == row["current_mp4_sha256"]
            clips[stem] = {"full": full, "chunk": chunk, "boundary": boundary,
                           "published_prefix_sha256": row["published_rgb_sha256"]}
        report["paths"][name] = {"frames": latest, "published_file_sha256": state["published_file_sha256"],
                                  "clips": clips, "integrity": "PASS"}
    target = HERE / "artifacts/video_integrity_audit_v3.json"
    target.write_text(json.dumps(report, indent=2) + "\n")
    print(target)
    print("PASS: every published video and immutable RGB prefix")


if __name__ == "__main__":
    main()
