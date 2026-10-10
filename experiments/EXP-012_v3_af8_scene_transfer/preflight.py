"""CPU-only EXP-012 P0 audit. Does not load 33B weights or start inference."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import torch

from common import CFG, HERE, OUT, ROOT, setup_paths, sha, verify_sources
from checkpoint import load_checkpoint


def tensor_sha(tensor: torch.Tensor) -> str:
    import hashlib
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(str((tuple(value.shape), str(value.dtype))).encode() +
                          value.view(torch.uint8).numpy().tobytes()).hexdigest()


def audit() -> dict:
    setup_paths()
    verify_sources()
    from causal.anyflow import install_anyflow, finite_map_step
    from causal.anyflow_sampling import configure_video_schedule
    from causal.h3_cached import H3ChunkCache
    from causal.local_topology import visible_inputs
    from causal.pretrained_lora import install_adapters
    from current_prefix import current_prefix_feedback
    from diffsynth.diffusion.flow_match import FlowMatchScheduler
    from interval_student import interval_student
    from benchmark import write_video
    import infer
    assert all(callable(item) for item in (install_anyflow, finite_map_step,
        configure_video_schedule, H3ChunkCache, visible_inputs, install_adapters,
        current_prefix_feedback, interval_student, write_video, infer.load_pipeline))
    qkv, target, state = load_checkpoint()
    assert len(qkv["lora_A"]) == len(qkv["lora_B"]) == 8
    assert all(torch.isfinite(value).all() for value in qkv["lora_A"] + qkv["lora_B"])
    assert all(torch.isfinite(value).all() for value in target["weights"].values())
    sigmas = configure_video_schedule(FlowMatchScheduler("MiniMax-H3"),
        steps=8, grid="native", flow_shift=CFG["flow_shift"])
    assert len(sigmas) == 9 and sigmas[0] == 1 and sigmas[-1] == 0
    assert all(a > b for a, b in zip(sigmas, sigmas[1:]))
    sources = json.loads((HERE / "source_manifest.json").read_text())["sources"]
    p1 = json.loads((ROOT / sources["exp011_p1_result"]["path"]).read_text())
    scenes = {}
    for scene in CFG["scenes"]:
        fixture_path = ROOT / sources[f"fixture_{scene}"]["path"]
        assert sha(fixture_path) == p1["fixtures"][scene]["sha256"]
        fixture = torch.load(fixture_path, map_location="cpu", weights_only=True)
        assert fixture["format"] == "exp011_native_single_i0_full37_v1"
        assert fixture["scene"] == scene
        assert fixture["initial_noise"].shape == (1, 24, 37, 30, 52)
        assert fixture["anchor"].shape == (390, 96)
        packed = fixture["packed"]
        assert len(packed["action_text_spans_local"]) == 37
        spans = packed["action_text_spans_local"]
        prompt_a = fixture["prompts"]["A"].clone()
        prompt_ad = prompt_a.clone()
        donor = fixture["prompts"]["D"]
        mask = torch.zeros(prompt_a.shape[0], dtype=torch.bool)
        for lo, hi in spans[12:17]:
            mask[lo:hi] = True
            prompt_ad[lo:hi] = donor[lo:hi]
        assert torch.equal(prompt_a[~mask], prompt_ad[~mask])
        assert not torch.equal(prompt_a[mask], prompt_ad[mask])
        layout12, visible12 = visible_inputs(packed, prompt_a, 12, 390)
        layout_a, visible_a = visible_inputs(packed, prompt_a, 17, 390)
        layout_ad, visible_ad = visible_inputs(packed, prompt_ad, 17, 390)
        assert len(layout12["action_text_rows"]) == 12
        assert len(layout_a["action_text_rows"]) == len(layout_ad["action_text_rows"]) == 17
        assert layout_a["seq_len"] == layout_a["action_video_start"] + 17 * 390
        assert layout12["seq_len"] == layout12["action_video_start"] + 12 * 390
        assert torch.equal(layout_a["img_position_ids"], layout_ad["img_position_ids"])
        # Cropping from 17 to 12 removes five action spans before the video,
        # so physical packed offsets shift. Compare the native I0/video rows.
        start12 = layout12["action_video_start"]
        start17 = layout_a["action_video_start"]
        assert torch.equal(layout12["img_position_ids"][:, start12 - 390:start12],
                           layout_a["img_position_ids"][:, start17 - 390:start17])
        assert torch.equal(layout12["img_position_ids"][:, start12:start12 + 12 * 390],
                           layout_a["img_position_ids"][:, start17:start17 + 12 * 390])
        assert len(visible_a) == len(layout_a["text_pos"])
        assert len(visible_ad) == len(layout_ad["text_pos"])
        assert len(visible12) == len(layout12["text_pos"])
        row_path = ROOT / sources[f"g1_{scene}_row"]["path"]
        first_path = ROOT / sources[f"g1_{scene}_first12"]["path"]
        rgb_path = ROOT / sources[f"g1_{scene}_rgb39"]["path"]
        g1 = json.loads(row_path.read_text())
        assert g1["method"] == "FM8" and g1["scene"] == scene
        assert g1["status"] == "complete_pending_visual_review"
        assert sha(first_path) == g1["endpoint_sha256"]
        assert sha(rgb_path) == g1["published_sha256"]
        assert g1["sigmas"] == sigmas
        first = torch.load(first_path, map_location="cpu", weights_only=True)
        rgb = np.load(rgb_path, mmap_mode="r")
        assert first.shape == (1, 24, 12, 30, 52)
        assert rgb.shape == (39, 480, 832, 3) and rgb.dtype == np.uint8
        assert torch.isfinite(first).all()
        # The runner separately commits this first12 under AF weights at
        # sigma=target_sigma=0; there is no FM8 cache input in source inventory.
        scenes[scene] = {"fixture_sha256": sha(fixture_path),
                         "g1_row_sha256": sha(row_path),
                         "first12_sha256": sha(first_path),
                         "rgb39_sha256": sha(rgb_path),
                         "first_tensor_sha256": tensor_sha(first),
                         "c2_initial_noise_sha256": tensor_sha(fixture["initial_noise"][:, :, 12:17].float()),
                         "C1_visible_actions": 12, "C2_visible_actions": 17,
                         "AA_AD_only_C2_action_spans_differ": True,
                         "global_position_sha256": tensor_sha(layout_a["img_position_ids"])}
    assert CFG["total_calls"] == {"sampling_forwards": 32, "commit_forwards": 2, "vae_decodes": 4}
    assert CFG["per_scene_calls"] == {"sampling_forwards": 16, "commit_forwards": 1, "vae_decodes": 2}
    assert CFG["max_gpu_seconds"] == 1260 and CFG["deadline_hkt"] == "2026-10-11T09:00:00+08:00"
    result = {"task": CFG["task"], "status": "CPU_PASS",
              "gpu_calls": 0, "checkpoint_step": state["step"],
              "qkv_sha256": sources["af2_qkv"]["sha256"],
              "target_sha256": sources["af2_target"]["sha256"],
              "native8_sigmas": sigmas,
              "finite_map_pairs": [[a, b] for a, b in zip(sigmas[:-1], sigmas[1:])],
              "scenes": scenes}
    return result


if __name__ == "__main__":
    result = audit()
    path = HERE / "P0_CPU_AUDIT.json"
    if path.exists():
        if json.loads(path.read_text()) != result:
            raise RuntimeError("P0 CPU audit changed after freeze")
    else:
        path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "gpu_calls": 0,
                      "checkpoint_step": result["checkpoint_step"],
                      "scenes": list(result["scenes"])}, indent=2))
