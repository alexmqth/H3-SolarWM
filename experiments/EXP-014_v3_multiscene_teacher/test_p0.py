"""EXP-014 P0 CPU-only checks for newly introduced source/stage/budget wiring."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile

import common
from source_audit import inspect_sources


def rejects(fn, exc=Exception):
    try:
        fn()
    except exc:
        return
    raise AssertionError("expected rejection did not occur")


def main():
    import torch
    from run_teacher import cache_signature, tensor_sha

    common.setup_paths()
    common.verify_sources()
    common.verify_code_manifest()
    src = inspect_sources()
    assert src == json.loads((common.HERE / "source_manifest.json").read_text())
    assert list(src["scenes"]) == list(common.CFG["scenes"])
    assert set(src["train_episode_ids"]).isdisjoint(src["excluded_validation_episode_ids"])
    assert all(x["split"] == "train" for x in src["scenes"].values())
    assert common.CFG["p1"]["text_encoder_calls"] == 3 * len(src["scenes"]) == 12
    assert common.CFG["p1"]["image_vae_encodes"] == len(src["scenes"]) == 4
    assert common.CFG["teacher_per_scene"]["sampling_forwards"] == 3 * 30 == 90
    assert common.CFG["teacher_per_scene"]["commit_forwards"] == 1
    assert common.CFG["teacher_per_scene"]["vae_decodes"] == 3
    assert common.CFG["teacher_total"]["sampling_forwards"] == 360
    assert common.CFG["teacher_total"]["commit_forwards"] == 4
    assert common.CFG["teacher_total"]["vae_decodes"] == 12

    # Reuse only a previously accepted native fixture as a CPU topology
    # fixture. Four new scene tensors do not exist until separately gated P1.
    prior = common.ROOT / "H3-World/outputs/EXP-011_v3_scene_transfer/fixtures/industrial.pt"
    fixture = torch.load(prior, map_location="cpu", weights_only=True)
    packed = fixture["packed"]
    assert fixture["initial_noise"].shape == (1, 24, 37, 30, 52)
    assert fixture["anchor"].shape == (390, 96)
    assert packed["img_pos"].numel() == 38 * 390
    assert len(packed["action_text_spans_local"]) == 37
    from causal.local_topology import visible_inputs
    from causal.h3_cached import ChunkAttention, H3ChunkCache
    for stop, index, start in ((12, 0, 0), (17, 1, 12)):
        layout, prompt = visible_inputs(packed, fixture["prompts"]["A"], stop, 390)
        assert len(layout["action_text_rows"]) == stop
        assert layout["img_pos"].numel() == (stop + 1) * 390
        assert layout["seq_len"] == layout["action_video_start"] + stop * 390
        assert len(prompt) < len(fixture["prompts"]["A"])
        control = ChunkAttention(H3ChunkCache(5, "cpu"), index,
                                 int(layout["action_video_start"]), 390,
                                 start, layout["action_text_rows"], "own", True)
        _, mask = control.masks(390, 0, "cpu")
        own_lo, own_hi = layout["action_text_rows"][start].tolist()
        assert bool(mask[0, own_lo:own_hi].all())
        for k, (lo, hi) in enumerate(layout["action_text_rows"].tolist()):
            if k != start:
                assert not bool(mask[0, lo:hi].any())
    assert common.CFG["seed"] == 13 and common.CFG["partition"] == [12, 5]
    assert common.CFG["num_frames"] == 124 and common.CFG["latent_frames"] == 37

    # Current denoising input is FP32 even when full37 initial noise is BF16.
    native = torch.arange(16, dtype=torch.bfloat16).reshape(1, 1, 1, 4, 4)
    assert tensor_sha(native.to(torch.float32).clone()) == tensor_sha(native.to(torch.float32))
    assert tensor_sha(native) != tensor_sha(native.to(torch.float32))
    cache = H3ChunkCache(5, "cpu")
    with torch.no_grad():
        for layer in range(50):
            item = torch.zeros((2, 1), dtype=torch.float32)
            cache.commit(layer, 0, item, item, item)
    assert len(cache.layers) == 50 and all([e.index for e in cache.history(layer, 1)] == [0]
                                           for layer in range(50))
    before = cache_signature(cache)
    cache.layers[0][0].key.add_(1)
    assert cache_signature(cache) != before

    old_out, old_approvals = common.OUT, common.APPROVALS
    with tempfile.TemporaryDirectory(prefix="exp014_p0_") as temp:
        common.OUT = Path(temp)
        common.APPROVALS = Path(temp) / "judge"
        rejects(lambda: common.require_marker("P1"), PermissionError)
        rejects(lambda: common.require_marker("T1", common.CFG["t1_scene"]), PermissionError)
        rejects(lambda: common.require_marker("T2", list(common.CFG["scenes"])[1]), PermissionError)
        p1 = common.Ledger(0, "P1")
        for _ in range(12):
            p1.reserve("text_encoder_calls")
        for _ in range(4):
            p1.reserve("image_vae_encodes")
        for field in ("text_encoder_calls", "image_vae_encodes", "video_vae_encodes",
                      "sampling_forwards", "commit_forwards", "vae_decodes"):
            rejects(lambda field=field: p1.reserve(field), RuntimeError)
        p1.close("cpu_test")
        scene = common.CFG["t1_scene"]
        t1 = common.Ledger(0, "T1", scene)
        for _ in range(90):
            t1.reserve("sampling_forwards")
        t1.reserve("commit_forwards")
        for _ in range(3):
            t1.reserve("vae_decodes")
        for field in ("sampling_forwards", "commit_forwards", "vae_decodes",
                      "text_encoder_calls", "image_vae_encodes"):
            rejects(lambda field=field: t1.reserve(field), RuntimeError)
        t1.close("cpu_test")
        # A different scene can hold its own journal/lock concurrently.
        next_scene = list(common.CFG["scenes"])[1]
        t2 = common.Ledger(1, "T2", next_scene)
        assert t2.path != common.OUT / "teacher" / scene / "budget.json"
        t2.close("cpu_test")
    common.OUT, common.APPROVALS = old_out, old_approvals
    print(json.dumps({"task": common.CFG["task"], "P0_cpu": "PASS",
                      "source_scene_count": 4, "native_full37_reference": "PASS",
                      "future_rows_trimmed": True, "own_action_only": True,
                      "cache_50_layer_index0": True, "budget_rejection": True,
                      "no_marker_rejection": True, "gpu_calls": 0}))


if __name__ == "__main__":
    main()
