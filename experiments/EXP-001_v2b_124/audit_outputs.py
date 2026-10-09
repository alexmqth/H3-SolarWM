"""CPU-only integrity audit of the stopped EXP-001 third-chunk outputs."""

import hashlib
import json
from fractions import Fraction
from pathlib import Path

import av
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUTPUT = ROOT / "H3-World/outputs/EXP-001_v2b_124"
PATHS = ("AA", "DD", "AD", "DA")


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def rgb_sha(frames):
    return hashlib.sha256(frames.tobytes(order="C")).hexdigest()


def video_info(path, expected_frames):
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        frames = list(container.decode(video=0))
        pts = [frame.pts for frame in frames]
        assert len(frames) == expected_frames, (path, len(frames))
        assert all(point is not None for point in pts), path
        assert all(b > a for a, b in zip(pts, pts[1:])), path
        assert stream.average_rate == Fraction(24, 1), (path, stream.average_rate)
        assert (stream.width, stream.height) == (832, 480), path
        assert stream.codec_context.name == "h264", path
        assert all((frame.width, frame.height) == (832, 480) for frame in frames)
        return {
            "path": str(path.relative_to(ROOT)),
            "sha256": sha(path),
            "codec": stream.codec_context.name,
            "fps": str(stream.average_rate),
            "frame_count": len(frames),
            "resolution": [stream.width, stream.height],
            "first_pts": pts[0],
            "last_pts": pts[-1],
            "time_base": str(stream.time_base),
        }


def main():
    result = {"task_id": "EXP-001", "method": "CPU PyAV complete decode and SHA-256", "paths": {}}
    for name in PATHS:
        directory = OUTPUT / name
        row = json.loads((directory / "chunk_17_22.json").read_text())
        state = json.loads((directory / "state.json").read_text())
        published = np.load(directory / "published.npy", mmap_mode="r")
        assert published.shape == (73, 480, 832, 3), (name, published.shape)
        assert published.dtype == np.uint8
        assert rgb_sha(published[:56]) == row["prior_published_rgb_sha256"], name
        assert rgb_sha(published) == row["published_rgb_sha256"], name
        assert sha(directory / "published.npy") == row["published_file_sha256"], name
        assert sha(directory / "chunk_17_22.pt") == row["endpoint_file_sha256"], name
        assert state["published_rgb_sha256"] == row["published_rgb_sha256"], name
        assert state["next_start"] == 22, name
        videos = {}
        for filename, count, key in (
            ("rollout_73.mp4", 73, "rollout_mp4_sha256"),
            ("chunk_17_22.mp4", 17, "current_mp4_sha256"),
            ("boundary_17_22.mp4", 2, None),
        ):
            info = video_info(directory / filename, count)
            if key:
                assert info["sha256"] == row[key], (name, filename)
            videos[filename] = info
        result["paths"][name] = {
            "published_shape": list(published.shape),
            "prior_56_rgb_sha256": row["prior_published_rgb_sha256"],
            "published_73_rgb_sha256": row["published_rgb_sha256"],
            "published_file_sha256": row["published_file_sha256"],
            "endpoint_file_sha256": row["endpoint_file_sha256"],
            "videos": videos,
            "integrity": "PASS",
        }
    output = HERE / "artifacts/video_integrity_audit.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(output)
    print("PASS: four complete 73f rollouts, four 17f chunks, four 2f boundaries")


if __name__ == "__main__":
    main()
