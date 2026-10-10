"""Independent CPU audit of one completed AF8 scene; visuals judged separately."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import av
import cv2
import numpy as np
import torch

from common import HERE, OUT, ROOT, setup_paths, sha, verify_code_manifest, verify_sources


def tensor_sha(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(str((tuple(value.shape), str(value.dtype))).encode() +
                          value.view(torch.uint8).numpy().tobytes()).hexdigest()


def video_check(path: Path, count: int) -> dict:
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        assert (stream.width, stream.height, stream.average_rate) == (832, 480, 24)
        frames = list(container.decode(video=0))
        assert len(frames) == count
        assert len({frame.pts for frame in frames}) == count
    return {"frames": count, "fps": 24, "width": 832, "height": 480}


def motion(rgb: np.ndarray) -> dict:
    gray = [cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in rgb]
    mad = [float(np.abs(a.astype(np.float32) - b.astype(np.float32)).mean())
           for a, b in zip(rgb, rgb[1:])]
    dx = []
    for a, b in zip(gray[39:55], gray[40:56]):
        flow = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        dx.append(float(np.median(flow[80:400, 160:672, 0])))
    return {"boundary_mad_38_39": mad[38],
            "within_c2_mad_mean": float(np.mean(mad[39:55])),
            "within_c2_mad_max": float(np.max(mad[39:55])),
            "central_median_horizontal_flow_sum_c2": float(np.sum(dx))}


def audit(scene: str) -> dict:
    setup_paths()
    verify_sources()
    verify_code_manifest()
    from causal.local_topology import visible_inputs
    sources = json.loads((HERE / "source_manifest.json").read_text())["sources"]
    row_dir = OUT / "G1" / scene
    row = json.loads((row_dir / "result.json").read_text())
    assert row["status"] == "complete_pending_visual_review"
    assert row["scene"] == scene and row["stage"] == "G1"
    assert row["checkpoint"]["step"] == 32
    assert row["checkpoint"]["qkv_sha256"] == sources["af2_qkv"]["sha256"]
    assert row["checkpoint"]["target_sha256"] == sources["af2_target"]["sha256"]
    assert row["commit_forwards"] == 1 and row["sampling_forwards"] == 16 and row["vae_decodes"] == 2
    assert row["peak_allocated_gib"] <= 44
    assert row["precision"]["native_fp32_weights_restored"]
    fixture_path = ROOT / sources[f"fixture_{scene}"]["path"]
    fixture = torch.load(fixture_path, map_location="cpu", weights_only=True)
    first_path = ROOT / sources[f"g1_{scene}_first12"]["path"]
    rgb_path = ROOT / sources[f"g1_{scene}_rgb39"]["path"]
    first = torch.load(first_path, map_location="cpu", weights_only=True)
    old_rgb = np.load(rgb_path)
    assert sha(first_path) == row["first_endpoint_sha256"]
    assert sha(rgb_path) == row["first_published_sha256"]
    assert tensor_sha(first.float()) == row["first_tensor_sha256"]
    cache_path = row_dir / "af_cache_through12.pt"
    assert sha(cache_path) == row["cache_sha256"]
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    assert len(cache.layers) == 50
    assert all(len(entries) == 1 and entries[0].index == 0 for entries in cache.layers.values())
    assert cache.nbytes == row["cpu_raw_kv_bytes"] == 6799104000
    del cache
    spans = fixture["packed"]["action_text_spans_local"]
    prompt_aa = fixture["prompts"]["A"].clone()
    prompt_ad = prompt_aa.clone()
    allowed = torch.zeros(prompt_aa.shape[0], dtype=torch.bool)
    for lo, hi in spans[12:17]:
        allowed[lo:hi] = True
        prompt_ad[lo:hi] = fixture["prompts"]["D"][lo:hi]
    assert torch.equal(prompt_aa[~allowed], prompt_ad[~allowed])
    assert not torch.equal(prompt_aa[allowed], prompt_ad[allowed])
    layout_aa, visible_aa = visible_inputs(fixture["packed"], prompt_aa, 17, 390)
    layout_ad, visible_ad = visible_inputs(fixture["packed"], prompt_ad, 17, 390)
    assert torch.equal(layout_aa["img_position_ids"], layout_ad["img_position_ids"])
    sigmas = json.loads((HERE / "P0_CPU_AUDIT.json").read_text())["native8_sigmas"]
    results = {}
    for branch, visible in (("AA", visible_aa), ("AD", visible_ad)):
        folder = row_dir / branch
        part = json.loads((folder / "result.json").read_text())
        assert part["status"] == "complete_pending_visual_review"
        assert part["sampling_forwards"] == 8 and part["vae_decodes"] == 1
        assert part["source_cache_sha256"] == row["cache_sha256"]
        assert part["first_endpoint_sha256"] == row["first_endpoint_sha256"]
        assert part["sigmas"] == sigmas
        assert part["initial_noise_sha256"] == tensor_sha(fixture["initial_noise"][:, :, 12:17].float())
        assert part["prompt_sha256"] == tensor_sha(visible)
        assert part["position_sha256"] == tensor_sha(layout_aa["img_position_ids"])
        current_path = folder / "chunk_12_17.pt"
        assert sha(current_path) == part["endpoint_sha256"]
        current = torch.load(current_path, map_location="cpu", weights_only=True)
        assert tuple(current.shape) == (1, 24, 5, 30, 52) and torch.isfinite(current).all()
        assert tensor_sha(current) == part["endpoint_tensor_sha256"]
        rgb_path_new = folder / "published_56.npy"
        assert sha(rgb_path_new) == part["published_sha256"]
        rgb = np.load(rgb_path_new)
        assert rgb.shape == (56, 480, 832, 3) and rgb.dtype == np.uint8
        assert np.array_equal(rgb[:39], old_rgb)
        assert sha(folder / "rollout_56.mp4") == part["video_sha256"]
        full = video_check(folder / "rollout_56.mp4", 56)
        new = video_check(folder / "new_17.mp4", 17)
        results[branch] = {"result_sha256": sha(folder / "result.json"),
                           "endpoint_sha256": part["endpoint_sha256"],
                           "video_sha256": part["video_sha256"],
                           "first39_rgb_exactly_unchanged": True,
                           "full_video": full, "new_video": new,
                           "sampling_seconds": part["sampling_seconds"],
                           "decode_seconds": part["decode_seconds"],
                           "motion": motion(rgb)}
    aa = np.load(row_dir / "AA/published_56.npy")
    ad = np.load(row_dir / "AD/published_56.npy")
    assert np.array_equal(aa[:39], ad[:39]) and not np.array_equal(aa[39:], ad[39:])
    baseline = json.loads((ROOT / "submission/experiments/EXP-011_v3_scene_transfer/artifacts/G2/audits" /
                           f"{scene}_FM8.json").read_text())
    return {"task": "EXP-012/v1", "scene": scene,
            "cpu_protocol_audit": "PASS", "visual_assessment": "SEPARATE",
            "row_sha256": sha(row_dir / "result.json"),
            "cache_sha256": row["cache_sha256"],
            "cpu_raw_kv_bytes": row["cpu_raw_kv_bytes"],
            "commit_forwards": 1, "sampling_forwards": 16, "vae_decodes": 2,
            "wall_seconds": row["wall_seconds"],
            "peak_allocated_gib": row["peak_allocated_gib"],
            "AA_AD_new_rgb_abs_diff_mean": float(np.abs(aa[39:].astype(np.float32)
                                                      - ad[39:].astype(np.float32)).mean()),
            "baseline_FM8_flow_sum": {b: baseline["branches"][b]["motion"]["central_median_horizontal_flow_sum_c2"]
                                      for b in ("AA", "AD")},
            "branches": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", choices=["industrial", "village"], required=True)
    args = parser.parse_args()
    result = audit(args.scene)
    target = HERE / "artifacts/audits" / f"{args.scene}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(target)
    target.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "branches"}, indent=2))
