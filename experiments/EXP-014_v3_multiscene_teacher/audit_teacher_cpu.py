"""CPU-only protocol/video audit of a completed EXP-014 teacher fork."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import av
import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import OUT, ROOT, setup_paths, verify_code_manifest, verify_sources


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(str((tuple(value.shape), str(value.dtype))).encode() +
                          value.view(torch.uint8).numpy().tobytes()).hexdigest()


def video_check(path: Path, count: int) -> dict:
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        assert (stream.width, stream.height) == (832, 480)
        assert stream.average_rate == 24
        frames = list(container.decode(video=0))
        assert len(frames) == count
        pts = [frame.pts for frame in frames]
        assert all(b > a for a, b in zip(pts, pts[1:]))
        return {"frames": len(frames), "fps": str(stream.average_rate),
                "resolution": [stream.width, stream.height], "unique_pts": len(set(pts))}


def motion_metrics(rgb: np.ndarray) -> dict:
    # Auxiliary diagnostics only: scene and character motion cannot be inferred
    # from the sign of global optical flow on these previously untested scenes.
    gray = [cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in rgb]
    mad = [float(np.abs(a.astype(np.float32) - b.astype(np.float32)).mean())
           for a, b in zip(rgb, rgb[1:])]
    fx = []
    for a, b in zip(gray[39:55], gray[40:56]):
        flow = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        fx.append(float(np.median(flow[80:400, 160:672, 0])))
    return {"boundary_mad_38_39": mad[38],
            "within_c2_mad_mean": float(np.mean(mad[39:55])),
            "within_c2_mad_max": float(np.max(mad[39:55])),
            "central_median_horizontal_flow_sum_c2": float(np.sum(fx)),
            "central_median_horizontal_flow_per_transition": fx}


def audit(scene: str) -> dict:
    setup_paths()
    verify_sources()
    verify_code_manifest()
    from causal.local_topology import visible_inputs
    from causal.h3_cached import H3ChunkCache
    from common import CFG
    stage = "T1" if scene == CFG["t1_scene"] else "T2"
    marker = json.loads((HERE / "judge" / f"{stage}_APPROVED.json").read_text())
    assert marker["approved"] and scene in marker["scenes"]
    p1 = json.loads((OUT / "P1_result.json").read_text())
    fixture_path = ROOT / p1["fixtures"][scene]["path"]
    assert sha(fixture_path) == p1["fixtures"][scene]["sha256"]
    if stage == "T1":
        assert sha(fixture_path) == marker["fixture_sha256"]
    fixture = torch.load(fixture_path, map_location="cpu", weights_only=True)
    prompt_aa = fixture["prompts"]["A"].clone()
    prompt_ad = prompt_aa.clone()
    donor = fixture["prompts"]["D"]
    spans = fixture["packed"]["action_text_spans_local"]
    for lo, hi in spans[12:17]:
        prompt_ad[lo:hi] = donor[lo:hi]
    allowed = torch.zeros(prompt_aa.shape[0], dtype=torch.bool)
    for lo, hi in spans[12:17]:
        allowed[lo:hi] = True
    assert torch.equal(prompt_aa[~allowed], prompt_ad[~allowed])
    assert not torch.equal(prompt_aa[allowed], prompt_ad[allowed])
    layout_aa, cropped_aa = visible_inputs(fixture["packed"], prompt_aa, 17, 390)
    layout_ad, cropped_ad = visible_inputs(fixture["packed"], prompt_ad, 17, 390)
    assert tensor_sha(layout_aa["img_position_ids"]) == tensor_sha(layout_ad["img_position_ids"])
    g1 = OUT / "G1" / scene / "FM30"
    g1_row = json.loads((g1 / "result.json").read_text())
    assert g1_row["status"] == "complete_pending_visual_review"
    assert g1_row["sampling_forwards"] == 30 and g1_row["vae_decodes"] == 1
    assert sha(g1 / "first12.pt") == g1_row["endpoint_sha256"]
    assert sha(g1 / "published_39.npy") == g1_row["published_sha256"]
    first_rgb = np.load(g1 / "published_39.npy")
    assert first_rgb.shape == (39, 480, 832, 3)
    assert video_check(g1 / "first39.mp4", 39)["frames"] == 39
    directory = OUT / "G2" / scene / "FM30"
    row = json.loads((directory / "result.json").read_text())
    assert row["status"] == "complete_pending_visual_review"
    assert row["scene"] == scene and row["method"] == "FM30"
    assert row["runner_sha256"] == sha(HERE / "run_teacher.py")
    assert row["commit_forwards"] == 1 and row["sampling_forwards"] == 60
    assert row["vae_decodes"] == 2 and row["cpu_raw_kv_bytes"] > 0
    assert row["precision"]["native_fp32_weights_restored"]
    assert row["peak_allocated_gib"] <= 44
    assert sha(directory / "cache_through12.pt") == row["cache_sha256"]
    cache = torch.load(directory / "cache_through12.pt", map_location="cpu", weights_only=False)
    assert isinstance(cache, H3ChunkCache)
    assert len(cache.layers) == 50 and cache.commits == 50
    assert all([entry.index for entry in cache.history(layer, 1)] == [0]
               for layer in range(50))
    assert cache.nbytes == row["cpu_raw_kv_bytes"]
    del cache
    results = {}
    for branch in ("AA", "AD"):
        sub = directory / branch
        part = json.loads((sub / "result.json").read_text())
        assert part["status"] == "complete_pending_visual_review"
        assert part["branch"] == branch and part["source_cache_sha256"] == row["cache_sha256"]
        assert part["first_endpoint_sha256"] == row["first_endpoint_sha256"]
        assert part["sampling_forwards"] == 30
        assert part["vae_decodes"] == 1 and part["cpu_raw_kv_bytes"] == row["cpu_raw_kv_bytes"]
        assert sha(sub / "chunk_12_17.pt") == part["endpoint_sha256"]
        current = torch.load(sub / "chunk_12_17.pt", map_location="cpu", weights_only=True)
        assert tuple(current.shape) == (1, 24, 5, 30, 52) and torch.isfinite(current).all()
        assert tensor_sha(current) == part["endpoint_tensor_sha256"]
        assert tensor_sha(fixture["initial_noise"][:, :, 12:17].float()) == part["initial_noise_sha256"]
        expected_prompt = cropped_aa if branch == "AA" else cropped_ad
        assert tensor_sha(expected_prompt) == part["prompt_sha256"]
        assert tensor_sha(layout_aa["img_position_ids"]) == part["position_sha256"]
        assert sha(sub / "published_56.npy") == part["published_sha256"]
        rgb = np.load(sub / "published_56.npy")
        assert rgb.shape == (56, 480, 832, 3) and rgb.dtype == np.uint8
        assert np.array_equal(rgb[:39], first_rgb)
        assert sha(sub / "rollout_56.mp4") == part["video_sha256"]
        full_video = video_check(sub / "rollout_56.mp4", 56)
        new_video = video_check(sub / "new_17.mp4", 17)
        results[branch] = {
            "result_sha256": sha(sub / "result.json"), "endpoint_sha256": part["endpoint_sha256"],
            "published_sha256": part["published_sha256"], "video_sha256": part["video_sha256"],
            "first_39_rgb_exactly_unchanged": True, "sampling_forwards": part["sampling_forwards"],
            "sampling_seconds": part["sampling_seconds"], "decode_seconds": part["decode_seconds"],
            "peak_allocated_gib": part["peak_allocated_gib"], "full_video": full_video,
            "new_video": new_video, "motion": motion_metrics(rgb),
            "prompt_sha256": part["prompt_sha256"], "position_sha256": part["position_sha256"],
            "initial_noise_sha256": part["initial_noise_sha256"], "sigmas": part["sigmas"]}
    assert results["AA"]["initial_noise_sha256"] == results["AD"]["initial_noise_sha256"]
    assert results["AA"]["position_sha256"] == results["AD"]["position_sha256"]
    assert results["AA"]["sigmas"] == results["AD"]["sigmas"]
    assert results["AA"]["prompt_sha256"] != results["AD"]["prompt_sha256"]
    aa = np.load(directory / "AA/published_56.npy")
    ad = np.load(directory / "AD/published_56.npy")
    assert np.array_equal(aa[:39], ad[:39])
    assert not np.array_equal(aa[39:], ad[39:])
    new_rgb_abs_diff = float(np.abs(aa[39:].astype(np.float32) - ad[39:].astype(np.float32)).mean())
    ledger = json.loads((OUT / "teacher" / scene / "budget.json").read_text())
    assert ledger["events"][-1]["status"] == "complete"
    assert ledger["sampling_forwards"] == 90 and ledger["commit_forwards"] == 1
    assert ledger["vae_decodes"] == 3 and ledger["gpu_seconds"] <= 1350
    target_manifest = json.loads((OUT / "teacher" / scene / "target_manifest.json").read_text())
    assert target_manifest["quality_status"] == "pending_judge"
    assert target_manifest["teacher_C1_latent"]["sha256"] == g1_row["endpoint_sha256"]
    assert all(target_manifest["teacher_C2_endpoint"][branch]["sha256"] ==
               results[branch]["endpoint_sha256"] for branch in ("AA", "AD"))
    return {"task": "EXP-014/v1", "stage": stage, "scene": scene, "method": "FM30",
            "cpu_protocol_audit": "PASS", "visual_assessment": "SEPARATE",
            "teacher_gpu_seconds": ledger["gpu_seconds"],
            "C1_sampling_forwards": 30, "C1_vae_decodes": 1,
            "cache_50_layers_index0": True,
            "row_sha256": sha(directory / "result.json"), "cache_sha256": row["cache_sha256"],
            "cpu_raw_kv_bytes": row["cpu_raw_kv_bytes"], "commit_forwards": 1,
            "sampling_forwards": row["sampling_forwards"], "vae_decodes": 2,
            "wall_seconds": row["wall_seconds"], "peak_allocated_gib": row["peak_allocated_gib"],
            "AA_AD_new_rgb_abs_diff_mean": new_rgb_abs_diff, "branches": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    from common import CFG
    parser.add_argument("--scene", choices=list(CFG["scenes"]), required=True)
    args = parser.parse_args()
    result = audit(args.scene)
    target = HERE / "artifacts" / "teacher_audits" / f"{args.scene}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(target)
    target.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "branches"}, indent=2))
