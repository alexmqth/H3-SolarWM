"""Audit C1 and the completed AA branch after an interrupted T2 AD branch.

This does not turn a partial scene into an accepted paired teacher target.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from audit_teacher_cpu import motion_metrics, sha, tensor_sha, video_check
from common import CFG, HERE, OUT, ROOT, setup_paths, verify_code_manifest, verify_sources


def audit(scene: str) -> dict:
    assert scene in CFG["scenes"] and scene != CFG["t1_scene"]
    setup_paths()
    verify_sources()
    verify_code_manifest()
    from causal.h3_cached import H3ChunkCache
    from causal.local_topology import visible_inputs

    marker = json.loads((HERE / "judge/T2_APPROVED.json").read_text())
    assert marker["approved"] and scene in marker["scenes"]
    p1 = json.loads((OUT / "P1_result.json").read_text())
    fixture_path = ROOT / p1["fixtures"][scene]["path"]
    assert sha(fixture_path) == p1["fixtures"][scene]["sha256"]
    fixture = torch.load(fixture_path, map_location="cpu", weights_only=True)
    layout, cropped = visible_inputs(fixture["packed"], fixture["prompts"]["A"], 17, 390)

    c1 = OUT / "G1" / scene / "FM30"
    c1_result = json.loads((c1 / "result.json").read_text())
    assert c1_result["status"] == "complete_pending_visual_review"
    assert c1_result["sampling_forwards"] == 30 and c1_result["vae_decodes"] == 1
    assert sha(c1 / "first12.pt") == c1_result["endpoint_sha256"]
    assert sha(c1 / "published_39.npy") == c1_result["published_sha256"]
    first_rgb = np.load(c1 / "published_39.npy")
    assert first_rgb.shape == (39, 480, 832, 3)
    c1_video = video_check(c1 / "first39.mp4", 39)

    g2 = OUT / "G2" / scene / "FM30"
    parent = json.loads((g2 / "result.json").read_text())
    assert parent["cache_sha256"] == sha(g2 / "cache_through12.pt")
    assert parent["first_endpoint_sha256"] == c1_result["endpoint_sha256"]
    cache = torch.load(g2 / "cache_through12.pt", map_location="cpu", weights_only=False)
    assert isinstance(cache, H3ChunkCache)
    assert len(cache.layers) == 50 and cache.commits == 50
    assert all([entry.index for entry in cache.history(layer, 1)] == [0]
               for layer in range(50))
    cache_bytes = cache.nbytes
    del cache

    aa_dir = g2 / "AA"
    aa = json.loads((aa_dir / "result.json").read_text())
    assert aa["status"] == "complete_pending_visual_review"
    assert aa["branch"] == "AA" and aa["sampling_forwards"] == 30
    assert aa["source_cache_sha256"] == parent["cache_sha256"]
    assert aa["first_endpoint_sha256"] == c1_result["endpoint_sha256"]
    assert aa["cpu_raw_kv_bytes"] == cache_bytes
    assert aa["prompt_sha256"] == tensor_sha(cropped)
    assert aa["position_sha256"] == tensor_sha(layout["img_position_ids"])
    assert aa["initial_noise_sha256"] == tensor_sha(fixture["initial_noise"][:, :, 12:17].float())
    assert sha(aa_dir / "chunk_12_17.pt") == aa["endpoint_sha256"]
    endpoint = torch.load(aa_dir / "chunk_12_17.pt", map_location="cpu", weights_only=True)
    assert tuple(endpoint.shape) == (1, 24, 5, 30, 52) and torch.isfinite(endpoint).all()
    assert tensor_sha(endpoint) == aa["endpoint_tensor_sha256"]
    assert sha(aa_dir / "published_56.npy") == aa["published_sha256"]
    rgb = np.load(aa_dir / "published_56.npy")
    assert rgb.shape == (56, 480, 832, 3) and rgb.dtype == np.uint8
    assert np.array_equal(rgb[:39], first_rgb)
    assert sha(aa_dir / "rollout_56.mp4") == aa["video_sha256"]
    aa_video = video_check(aa_dir / "rollout_56.mp4", 56)
    aa_new_video = video_check(aa_dir / "new_17.mp4", 17)

    ad_dir = g2 / "AD"
    ad = json.loads((ad_dir / "result.json").read_text())
    assert ad["status"] == "sampling" and not (ad_dir / "rollout_56.mp4").exists()
    ledger = json.loads((OUT / "teacher" / scene / "budget.json").read_text())
    assert ledger["scene"] == scene and ledger["stage"] == "T2"
    assert ledger["events"][-1]["event"] == "reserve"
    assert not any(event["event"] == "stage_stop" for event in ledger["events"])
    assert ledger["commit_forwards"] == 1 and ledger["vae_decodes"] == 2
    # The write-ahead ledger reserves a forward before the model call. The AD
    # result row was last saved before the process received SIGTERM.
    assert ledger["sampling_forwards"] == 61 + ad["sampling_forwards"]
    assert not (OUT / "teacher" / scene / "target_manifest.json").exists()
    return {
        "task": "EXP-014/v1", "stage": "T2", "scene": scene,
        "status": "INTERRUPTED_AD_NOT_A_PAIRED_TEACHER_TARGET",
        "cpu_partial_protocol_audit": "PASS_C1_AND_AA_ONLY",
        "c1_video": c1_video, "aa_video": aa_video, "aa_new_video": aa_new_video,
        "first_39_rgb_exactly_unchanged": True,
        "cache_50_layers_index0": True, "cache_sha256": parent["cache_sha256"],
        "cpu_raw_kv_bytes": cache_bytes,
        "aa_endpoint_sha256": aa["endpoint_sha256"], "aa_video_sha256": aa["video_sha256"],
        "aa_motion": motion_metrics(rgb),
        "reserved_sampling_forwards": ledger["sampling_forwards"],
        "completed_c1_aa_sampling_forwards": 60,
        "reserved_ad_sampling_forwards": ledger["sampling_forwards"] - 60,
        "ad_result_persisted_sampling_forwards": ad["sampling_forwards"],
        "commit_forwards": ledger["commit_forwards"], "vae_decodes": ledger["vae_decodes"],
        "last_reservation_utc": ledger["events"][-1]["at"],
        "ad_output": "MISSING_INTERRUPTED",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", choices=[s for s in CFG["scenes"] if s != CFG["t1_scene"]], required=True)
    args = parser.parse_args()
    result = audit(args.scene)
    target = HERE / "artifacts/teacher_audits" / f"{args.scene}_partial.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(target)
    target.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "aa_motion"}, indent=2))
