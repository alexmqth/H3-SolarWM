"""CPU-only source-frame spot check for the fixed real56 clips.

For each selected output position, compare its decoded image with the
predicted source frame and two neighboring source frames on either side.
The PyAV RGB rescale is approximate to FFmpeg's YUV filter path, so the
ranking is evidence of alignment; the pixel MAD is not an exact identity.
"""
from __future__ import annotations

import json
from pathlib import Path

import av
import numpy as np

from build_real56 import HERE, ART, rows, write_json

SAMPLES = (0, 38, 39, 55)


def inspect() -> list[dict]:
    result = []
    for item in rows():
        scene = item["candidate"]["episode_id"][:8]
        with av.open(str(ART / scene / "real56.mp4")) as output:
            output.streams.video[0].thread_count = 1
            frames = [f.to_ndarray(format="rgb24") for f in output.decode(video=0)]
        assert len(frames) == 56
        checks = []
        with av.open(str(item["video"])) as source:
            stream = source.streams.video[0]
            stream.thread_count = 1
            assert stream.width == 1920 and stream.height == 1080
            assert int(stream.average_rate) == 30
            for j in SAMPLES:
                center = item["indices"][j]
                source.seek((center - 2) * 512, stream=stream, backward=True)
                distances = {}
                for frame in source.decode(video=0):
                    assert frame.pts is not None and frame.pts % 512 == 0
                    index = frame.pts // 512
                    if index > center + 2:
                        break
                    if index < center - 2:
                        continue
                    scaled = frame.reformat(width=854, height=480, format="rgb24").to_ndarray()
                    crop = scaled[:, 11:843]
                    distances[index - center] = float(np.abs(
                        crop.astype(np.int16) - frames[j].astype(np.int16)).mean())
                assert set(distances) == {-2, -1, 0, 1, 2}, (scene, j, distances)
                best = min(distances, key=distances.get)
                checks.append({"output_frame": j, "predicted_source_index": center,
                               "neighbor_offsets_MAD": {str(k): v for k, v in sorted(distances.items())},
                               "best_offset": best})
        result.append({"scene": scene, "checks": checks,
                       "all_predicted_indices_best": all(x["best_offset"] == 0 for x in checks)})
        print(scene, [(x["output_frame"], x["best_offset"]) for x in checks], flush=True)
    return result


if __name__ == "__main__":
    checks = inspect()
    write_json(HERE / "SOURCE_ALIGNMENT_PIXEL_SPOTCHECK.json",
               {"method": "PyAV source seek by 30fps PTS; RGB854x480 center crop approximates FFmpeg scaler",
                "GPU_calls": 0, "samples_per_scene": list(SAMPLES), "scenes": checks})
    if not all(x["all_predicted_indices_best"] for x in checks):
        print("WARNING: approximate RGB pixel ranking is ambiguous for one scene; verify exact decoded PTS separately")
