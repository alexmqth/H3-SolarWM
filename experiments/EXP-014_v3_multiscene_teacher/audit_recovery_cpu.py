"""CPU-only audit of a completed EXP-014/v2 missing-AD recovery."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from audit_teacher_cpu import motion_metrics, sha, tensor_sha, video_check
from common import HERE, OUT, setup_paths, verify_code_manifest, verify_sources
from recover_ad import CONFIG, RECOVERY, verify_recovery_code, verify_snapshot


def audit(scene: str) -> dict:
    setup_paths()
    verify_sources()
    verify_code_manifest()
    verify_recovery_code()
    verify_snapshot()
    marker = json.loads((HERE / "judge/RECOVERY_APPROVED.json").read_text())
    assert marker["approved"] and scene in marker["scenes"]
    preflight = json.loads((HERE / "artifacts/recovery_v2_cpu_preflight.json").read_text())
    before = next(row for row in preflight["scenes"] if row["scene"] == scene)
    g1 = OUT / "G1" / scene / "FM30"
    g2 = OUT / "G2" / scene / "FM30"
    directory = RECOVERY / scene
    row = json.loads((directory / "AD/result.json").read_text())
    ledger = json.loads((directory / "budget.json").read_text())
    target = json.loads((directory / "target_manifest.json").read_text())
    assert row["task"] == CONFIG["task"] and row["scene"] == scene
    assert row["status"] == "complete_pending_visual_review"
    assert row["preflight"] == before
    assert row["runner_sha256"] == sha(HERE / "recover_ad.py")
    assert row["sampling_forwards"] == 30 and row["vae_decodes"] == 1
    assert row["cpu_raw_kv_bytes"] == before["cache_bytes"]
    assert row["peak_allocated_gib"] <= CONFIG["gpu_allocated_cap_gib"]
    assert row["prompt_sha256"] == before["ad_prompt_sha256"]
    assert row["position_sha256"] == before["ad_position_sha256"]
    assert row["initial_noise_sha256"] == before["ad_noise_sha256"]
    assert row["sigmas"] == before["ad_sigmas"]
    endpoint = directory / "AD/chunk_12_17.pt"
    assert sha(endpoint) == row["endpoint_sha256"]
    z = torch.load(endpoint, map_location="cpu", weights_only=True)
    assert tuple(z.shape) == (1, 24, 5, 30, 52) and torch.isfinite(z).all()
    assert tensor_sha(z) == row["endpoint_tensor_sha256"]
    assert sha(g2 / "cache_through12.pt") == before["cache_sha256"]
    assert sha(g1 / "first12.pt") == before["c1_endpoint_sha256"]
    assert sha(g1 / "published_39.npy") == before["c1_rgb_sha256"]
    assert sha(g2 / "AA/chunk_12_17.pt") == before["aa_endpoint_sha256"]
    assert sha(g2 / "AD/result.json") == before["old_ad_result_sha256"]

    rgb_file = directory / "AD/published_56.npy"
    assert sha(rgb_file) == row["published_sha256"]
    rgb = np.load(rgb_file)
    first = np.load(g1 / "published_39.npy")
    aa = np.load(g2 / "AA/published_56.npy")
    assert rgb.shape == (56, 480, 832, 3) and rgb.dtype == np.uint8
    assert np.array_equal(rgb[:39], first)
    assert np.array_equal(rgb[:39], aa[:39])
    assert not np.array_equal(rgb[39:], aa[39:])
    video = directory / "AD/rollout_56.mp4"
    assert sha(video) == row["video_sha256"]
    full_check = video_check(video, 56)
    new_check = video_check(directory / "AD/new_17.mp4", 17)
    assert ledger["sampling_forwards"] == 30 and ledger["vae_decodes"] == 1
    assert ledger["commit_forwards"] == ledger["image_vae_encodes"] == ledger["training_updates"] == 0
    assert ledger["gpu_seconds"] <= CONFIG["max_gpu_seconds_per_scene"]
    assert ledger["events"][-1]["status"] == "complete"
    assert target["quality_status"] == "pending_judge"
    assert target["teacher_C1_latent"]["sha256"] == before["c1_endpoint_sha256"]
    assert target["teacher_C2_endpoint"]["AA"]["sha256"] == before["aa_endpoint_sha256"]
    assert target["teacher_C2_endpoint"]["AD"]["sha256"] == row["endpoint_sha256"]
    return {
        "task": CONFIG["task"], "scene": scene,
        "cpu_protocol_audit": "PASS", "visual_assessment": "SEPARATE",
        "old_first_39_rgb_exactly_unchanged": True,
        "old_cache_and_partial_ad_unchanged": True,
        "sampling_forwards": 30, "vae_decodes": 1, "commit_forwards": 0,
        "gpu_seconds": ledger["gpu_seconds"], "peak_allocated_gib": row["peak_allocated_gib"],
        "full_video": full_check, "new_video": new_check,
        "aa_ad_new_rgb_abs_diff_mean": float(np.abs(aa[39:].astype(np.float32)-rgb[39:].astype(np.float32)).mean()),
        "aa_motion": motion_metrics(aa), "ad_motion": motion_metrics(rgb),
        "recovered_ad_video_sha256": row["video_sha256"],
        "recovered_ad_endpoint_sha256": row["endpoint_sha256"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", choices=CONFIG["scenes"], required=True)
    args = parser.parse_args()
    result = audit(args.scene)
    target = HERE / "artifacts/recovery_v2_audits" / f"{args.scene}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(target)
    target.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key not in ("aa_motion", "ad_motion")}, indent=2))
