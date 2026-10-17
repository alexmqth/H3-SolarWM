"""CPU-only integrity audit and s3 2x2 assembly; never runs the model."""
from __future__ import annotations

import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(EXP))

import av
import numpy as np
import run_exp029 as frozen
from common import sha, atomic_json

ORIGINAL = ROOT / "H3-World/outputs/EXP-029_current_prefix_behavior"
RECOVERY = ORIGINAL / "recovery01"
SCENES = ("s0_43866101", "s3_b784d995")
BRANCHES = ("ON_A", "ON_D", "OFF_A", "OFF_D")


def media(path, frames):
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        count = sum(1 for _ in container.decode(video=0))
        fps = float(stream.average_rate)
    assert count == frames and fps == 24
    return dict(path=str(path.relative_to(ROOT)), sha256=sha(path),
                frames=count, fps=fps, bytes=path.stat().st_size)


def main():
    original = json.loads((HERE / "ORIGINAL_FREEZE.json").read_text())
    for entry in original["frozen_seven"].values():
        for relative, expected in entry["files"].items():
            assert sha(ROOT / relative) == expected, relative
    old_ledger = json.loads((ORIGINAL / "GPU_LEDGER.json").read_text())
    new_ledger = json.loads((RECOVERY / "GPU_LEDGER.json").read_text())
    assert sha(ORIGINAL / "GPU_LEDGER.json") == original["original_ledger_sha256"]
    assert old_ledger["status"] == "FAILED_PENDING_JUDGE"
    assert new_ledger["status"] == "COMPLETE_PENDING_JUDGE"
    assert new_ledger["counts"]["dit"] == 31 and new_ledger["counts"]["decode"] == 1

    manifest = dict(task="EXP-029/v2", status="COMPLETE_PENDING_JUDGE",
        comparison_type="same generated C1; current-prefix ON/OFF by A/D",
        prior_rgb_frames=39, newly_generated_rgb_frames=17,
        source_original_attempt=str(ORIGINAL.relative_to(ROOT)),
        source_recovery=str(RECOVERY.relative_to(ROOT)),
        original_ledger_sha256=original["original_ledger_sha256"],
        original_scheduler_sha256=original["original_scheduler_sha256"],
        recovery_ledger_sha256=sha(RECOVERY / "GPU_LEDGER.json"),
        branches={}, comparisons={})
    for scene in SCENES:
        scene_rows = {}
        frames_for_montage = {}
        for branch in BRANCHES:
            source = RECOVERY / "OFF_D" if (scene, branch) == (SCENES[1], "OFF_D") else ORIGINAL / scene / branch
            row = json.loads((source / "RESULT.json").read_text())
            assert row["status"] in ("COMPLETE_PENDING_REVIEW", "COMPLETE_PENDING_JUDGE")
            assert row["sampling_forwards"] == 30
            frames = np.load(source / "published56.npy", mmap_mode="r")
            assert frames.shape == (56, 480, 832, 3)
            source_prefix = np.load(ROOT / "H3-World/outputs/EXP-014_v3_multiscene_teacher/G1" /
                                    scene / "FM30/published_39.npy", mmap_mode="r")
            assert np.array_equal(frames[:39], source_prefix)
            generated = np.load(source / "generated17.npy", mmap_mode="r")
            assert generated.shape == (17, 480, 832, 3) and np.array_equal(frames[39:], generated)
            video = media(source / "rollout_56.mp4", 56)
            new_video = media(source / "new_17.mp4", 17)
            contact = source / "contact_18.jpg"
            assert contact.is_file()
            scene_rows[branch] = dict(source="recovery01" if source.is_relative_to(RECOVERY) else "original_attempt",
                result=str((source / "RESULT.json").relative_to(ROOT)),
                result_sha256=sha(source / "RESULT.json"),
                endpoint_sha256=sha(source / "endpoint5.pt"),
                generated17_sha256=sha(source / "generated17.npy"),
                published56_sha256=sha(source / "published56.npy"),
                video=video, new_video=new_video,
                contact=dict(path=str(contact.relative_to(ROOT)), sha256=sha(contact)),
                boundary_mad=row["boundary_mad"], adjacent_new_mad=row["adjacent_new_mad"],
                decoded_history_mad=row["decoded_history_mad"],
                sampling_seconds=row["sampling_seconds"], decode_seconds=row["decode_seconds"],
                initial_noise_sha256=row["initial_noise_sha256"],
                prompt_sha256=row["prompt_sha256"],
                position_sha256=row["position_sha256"],
                cache_sha256=row["source_cache_sha256"],
                cache_sha256_after=row["cache_sha256_after"])
            assert row["source_cache_sha256"] == row["cache_sha256_after"]
            frames_for_montage[branch] = frames
        assert len({x["initial_noise_sha256"] for x in scene_rows.values()}) == 1
        assert len({x["position_sha256"] for x in scene_rows.values()}) == 1
        assert len({x["cache_sha256"] for x in scene_rows.values()}) == 1
        manifest["branches"][scene] = scene_rows
        if scene == SCENES[0]:
            comparison = media(ORIGINAL / scene / "comparison_2x2.mp4", 56)
        else:
            comparison = frozen.montage(RECOVERY, frames_for_montage)
            assert comparison["frames"] == 56
        manifest["comparisons"][scene] = comparison
    manifest["original_counts"] = old_ledger["counts"]
    manifest["recovery_counts"] = new_ledger["counts"]
    manifest["combined_counts"] = {name: old_ledger["counts"][name]+new_ledger["counts"][name]
                                    for name in old_ledger["counts"]}
    atomic_json(HERE / "ALL_EIGHT_MANIFEST.json", manifest)
    print(manifest["status"], "8 videos + 2 comparisons", manifest["combined_counts"])


if __name__ == "__main__": main()
