"""Independent CPU audit of RGB media, GPU ledger, checkpoints and sources."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import av
import torch

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def sha(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in ("", "-1"):
        raise RuntimeError("CPU audit only")
    source = json.loads((HERE / "SOURCE_MANIFEST.json").read_text())
    assert all(sha(ROOT / path) == digest for path, digest in source.items())
    data = json.loads((HERE / "RGB_MATERIALIZATION.json").read_text())
    assert len(data["rows"]) == 36
    by_split = {"train": 0, "validation": 0}
    media_bytes = 0
    for row in data["rows"]:
        path = ROOT / row["video"]
        assert sha(path) == row["video_sha256"]
        by_split[row["split"]] += 1
        media_bytes += path.stat().st_size
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            stream.thread_count = 1
            assert stream.average_rate == 24 and (stream.width, stream.height) == (832, 480)
            pts = [frame.pts for frame in container.decode(video=0)]
        assert len(pts) == 56 and all(a < b for a,b in zip(pts,pts[1:]))
        assert row["quality"]["near_black_frames_mean_lt_3"] == 0
    assert by_split == {"train": 28, "validation": 8}
    cache = json.loads((HERE / "CACHE_EQUIVALENCE.json").read_text())
    forward = json.loads((HERE / "FORWARD_EQUIVALENCE.json").read_text())
    assert len(cache) == 150 and all(r["exact"] and r["max_abs"] == 0 for r in cache)
    assert forward["velocity"]["exact"] and forward["velocity"]["max_abs"] == 0
    budget = json.loads((HERE / "GPU_BUDGET.json").read_text())
    result = json.loads((HERE / "GPU_RESULT.json").read_text())
    assert budget["status"] == result["status"] == "COMPLETE_PENDING_JUDGE"
    assert budget["counts"] == result["counts"]
    assert budget["counts"] == {"top_level_forward": 11, "backward": 2, "update": 2,
                                "vae": 0, "text": 0, "sampling": 0}
    assert budget["elapsed_gpu_process_seconds"] < 3600
    assert budget["peak_allocated_gib"] < 44
    update = json.loads((HERE / "UPDATES.json").read_text())
    assert len(update) == 2 and [r["sigma"] for r in update] == [.5, .75]
    assert all(r["grad_l2"] > 0 and r["parameter_delta_l2"] > 0 for r in update)
    reload = json.loads((HERE / "RELOAD_CHECK.json").read_text())
    assert reload["velocity"]["exact"] and reload["cuda_rng_exact"]
    checkpoint = HERE / "step02.pt"
    assert sha(checkpoint) == reload["checkpoint_sha256"]
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert state["step"] == 2 and state["metadata"]["objective"] == "ordinary_fm"
    assert len(state["adapter"]) == 8 and len(state["optimizer"]["state"]) == 16
    assert sum(x["A"].numel()+x["B"].numel() for x in state["adapter"]) == 1720320
    sampled_rss_max = max([x["rss_gib"] for x in budget["events"] if "rss_gib" in x]
                          + [budget["rss_gib"]])
    summary = {"status": "PASS_CPU_ARTIFACT_AND_BUDGET_AUDIT",
        "videos_redecoded": 36, "frames_redecoded": 2016,
        "train_new": 28, "validation_new": 8,
        "direct_mp4_file_bytes": media_bytes,
        "cache_tensors_exact": 150, "velocity_exact": True,
        "top_level_dit_forwards": 11, "backward_calls": 2, "updates": 2,
        "vae_calls": 0, "text_calls": 0, "sampling_calls": 0,
        "gpu_wall_seconds": budget["elapsed_gpu_process_seconds"],
        "peak_allocated_gib": budget["peak_allocated_gib"],
        "peak_reserved_gib": budget["peak_reserved_gib"],
        "max_recorded_event_rss_gib": sampled_rss_max,
        "continuous_host_rss_peak": "NOT_MEASURED",
        "checkpoint_sha256": reload["checkpoint_sha256"],
        "checkpoint_reload_velocity_exact": True,
        "checkpoint_cuda_rng_exact": True,
        "activation_checkpoint_recompute_exact_count": "NOT_INSTRUMENTED; included in measured backward time",
        "scope": "engineering and artifact audit; no visual/action quality test"}
    target = HERE / "CPU_FINAL_AUDIT.json"
    if target.exists():
        raise FileExistsError(target)
    target.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
