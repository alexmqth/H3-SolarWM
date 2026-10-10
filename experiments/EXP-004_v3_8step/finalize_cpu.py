"""Copy bounded EXP-004 deliverables and validate saved artifacts; no model inference."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import av
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = ROOT / "H3-World/outputs/EXP-004_v3_8step"
OLD = ROOT / "H3-World/outputs/EXP-002_native_cached"
VIDEO = HERE / "artifacts/videos"
ROWS = HERE / "artifacts/metrics"


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def copy_once(source, target):
    assert source.is_file()
    if target.exists():
        assert sha(source) == sha(target), f"existing target differs: {target}"
    else:
        shutil.copy2(source, target)
    return dict(source=str(source), target=str(target.relative_to(HERE)),
                sha256=sha(target), bytes=target.stat().st_size)


def video_check(path, frames, width, height):
    with av.open(str(path)) as c:
        stream = c.streams.video[0]
        assert (stream.width, stream.height, str(stream.average_rate)) == (width, height, "24")
        assert stream.codec_context.name == "h264"
        decoded = list(c.decode(video=0))
    pts = [f.pts for f in decoded]
    assert len(decoded) == frames and all(x is not None for x in pts)
    assert all(a < b for a, b in zip(pts, pts[1:]))
    return dict(path=str(path.relative_to(HERE)), frames=frames, fps=24,
                width=width, height=height, first_pts=pts[0], last_pts=pts[-1],
                sha256=sha(path))


def main():
    VIDEO.mkdir(parents=True, exist_ok=True)
    ROWS.mkdir(parents=True, exist_ok=True)
    ledger = json.loads((OUT / "budget.json").read_text())
    assert (ledger["sampling"], ledger["commits"], ledger["total_forwards"],
            ledger["vae_decodes"]) == (32, 2, 34, 4)
    assert len(ledger["runs"]) == 4 and all(x["status"] == "complete" for x in ledger["runs"])
    assert ledger["gpu_seconds"] <= 1800
    copied, checks, metrics = [], [], {}
    shared = {}
    for path in ("AA", "AD"):
        metrics[path] = {}
        prev_rgb = None
        for stage, start, stop, n in (("second", 12, 17, 56), ("third", 17, 22, 73)):
            source_row = OUT / path / f"chunk_{start}_{stop}.json"
            row = json.loads(source_row.read_text())
            old = json.loads((OLD / path / f"chunk_{start}_{stop}.json").read_text())
            assert row["status"] == "complete_pending_visual_review"
            assert row["sampling_forwards"] == 8 and row["commit_forwards"] == (stage == "third")
            assert row["vae_decodes"] == 1 and len(row["sigmas"]) == 9
            assert row["sigmas"][0] == 1 and row["sigmas"][-1] == 0
            assert row["memory_cache_read_unchanged"] and row["cache_file_unchanged_during_sampling"]
            assert row["frozen_parameter_versions_unchanged"]
            for key in ("initial_noise_sha256", "anchor_sha256", "audio_sha256",
                        "position_sha256", "prompt_sha256"):
                assert row[key] == old[key], (path, stage, key)
            if stage == "second":
                for key in ("history_tensor_sha256", "first12_cache_sha256"):
                    assert row[key] == old[key], (path, key)
            else:
                assert row["second_endpoint_sha256"] == metrics[path]["second"]["endpoint_sha256"]
                assert row["prior_RGB_sha256"] == prev_rgb
            copied.append(copy_once(source_row, ROWS / f"{path}_chunk_{start}_{stop}.json"))
            source_video = OUT / path / f"rollout_{n}.mp4"
            target_video = VIDEO / f"{path}_8step_{n}.mp4"
            copied.append(copy_once(source_video, target_video))
            checks.append(video_check(target_video, n, 832, 480))
            comparison = VIDEO / f"V3_30_vs_8_{path}_{n}.mp4"
            checks.append(video_check(comparison, n, 1664, 560))
            pixels = np.load(OUT / path / f"published_{n}.npy", mmap_mode="r")
            assert pixels.shape == (n, 480, 832, 3)
            if stage == "second":
                reference = np.load(OLD / path / "published_56.npy", mmap_mode="r")
                assert np.array_equal(pixels[:39], reference[:39])
                shared[path] = pixels[:39].copy()
            else:
                prior = np.load(OUT / path / "published_56.npy", mmap_mode="r")
                assert np.array_equal(pixels[:56], prior)
            prev_rgb = row["published_RGB_sha256"]
            metrics[path][stage] = dict(
                rgb_frames=n, action=path[1], sampling_steps=8,
                sampling_seconds=row["sampling_seconds"],
                commit_seconds=row.get("second_clean_commit_seconds", 0),
                decode_seconds=row["decode_seconds"], wall_seconds=row["wall_seconds"],
                peak_allocated_MiB=row["GPU_peak_allocated_MiB"],
                history_cache_bytes=row["first12_cache"]["nbytes"] if stage == "second"
                    else row["cache_after_commit"]["nbytes"],
                horizontal_flow_px_per_frame=row["flow"]["horizontal_flow_px"]["mean"],
                baseline_30_flow_px_per_frame=old["flow"]["horizontal_flow_px"]["mean"],
                baseline_30_sampling_seconds=old["sampling_seconds"],
                boundary_gray_MAD=row["boundary_gray_MAD"],
                inside_gray_MAD=row["inside_gray_MAD"],
                endpoint_sha256=row["endpoint_sha256"],
                published_file_sha256=row["published_file_sha256"],
                published_RGB_sha256=row["published_RGB_sha256"],
                cache_sha256=row["first12_cache_sha256"] if stage == "second"
                    else row["cache_through17_sha256"],
                original_row=str(source_row))
    assert np.array_equal(shared["AA"], shared["AD"])
    aa = json.loads((OUT / "AA/chunk_12_17.json").read_text())
    ad = json.loads((OUT / "AD/chunk_12_17.json").read_text())
    for key in ("history_tensor_sha256", "initial_noise_sha256", "anchor_sha256",
                "audio_sha256", "position_sha256", "first12_cache_sha256",
                "prior_RGB_sha256", "sigmas"):
        assert aa[key] == ad[key], key
    assert aa["prompt_sha256"] != ad["prompt_sha256"]
    copied.append(copy_once(OUT / "budget.json", HERE / "artifacts/budget.json"))
    write(HERE / "artifacts/video_integrity.json", checks)
    produced = []
    for path in sorted(list(VIDEO.glob("V3_30_vs_8_*.mp4")) +
                       list(VIDEO.glob("V3_30_vs_8_*.json")) +
                       list((HERE / "artifacts").glob("*all_new_frames.jpg")) +
                       list((HERE / "artifacts").glob("*30_vs_8_samples.jpg")) +
                       list(HERE.glob("*_second.log")) + list(HERE.glob("*_third.log"))):
        produced.append(dict(path=str(path.relative_to(HERE)), sha256=sha(path),
                             bytes=path.stat().st_size))
    write(HERE / "artifact_manifest.json", dict(copied=copied, produced=produced))
    write(HERE / "metrics.json", dict(task="EXP-004/v1", execution="complete_pending_judge",
        model="V3 Original H3 + released action LoRA; native FM/Euler, not AnyFlow",
        first12_history="EXP-002 frozen 30-step generation; only new chunks use 8 steps",
        same_state_second_chunk=True, third_chunk_own_generated_history=True,
        budget=ledger, paths=metrics,
        note="30-step reference generated at a different time under shared hardware. "
             "Per-chunk sampling is not complete E2E or first-screen latency."))
    print("validated", len(checks), "H.264 videos; copied", len(copied),
          "artifacts; produced", len(produced))


if __name__ == "__main__":
    main()
