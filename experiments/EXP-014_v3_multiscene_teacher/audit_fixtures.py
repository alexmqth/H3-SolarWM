"""CPU audit of four newly encoded native Single-I0 P1 fixtures."""
from __future__ import annotations

import json

from common import CFG, HERE, OUT, ROOT, atomic_json, setup_paths, sha, verify_sources


def main():
    import torch
    setup_paths()
    verify_sources()
    from causal.local_topology import visible_inputs
    from causal.h3_cached import ChunkAttention, H3ChunkCache
    result = json.loads((OUT / "P1_result.json").read_text())
    if result["status"] != "complete_pending_cpu_audit":
        raise RuntimeError("P1 encoding is incomplete")
    report = {"task": CFG["task"], "stage": "P1_cpu_audit", "status": "auditing",
              "fixture_sha256": {}, "scenes": {}, "gpu_calls": 0}
    atomic_json(OUT / "P1_cpu_audit.json", report)
    source = json.loads((HERE / "source_manifest.json").read_text())
    common_noise = None
    for name, entry in result["fixtures"].items():
        path = ROOT / entry["path"]
        assert sha(path) == entry["sha256"]
        fixture = torch.load(path, map_location="cpu", weights_only=True)
        assert fixture["format"] == "exp014_native_single_i0_full37_v1"
        assert fixture["scene"] == name
        assert fixture["source_png_sha256"] == source["scenes"][name]["png_sha256"]
        assert fixture["scene_static"] == source["scenes"][name]["scene_static"]
        packed = fixture["packed"]
        assert fixture["anchor"].shape == (390, 96)
        assert packed["img_pos"].numel() == 38 * 390
        assert len(packed["action_text_spans_local"]) == 37
        assert packed["action_video_start"] == packed["img_pos"][390].item()
        if common_noise is None:
            common_noise = (fixture["initial_noise"], fixture["audio_noise"])
        else:
            assert torch.equal(fixture["initial_noise"], common_noise[0])
            assert torch.equal(fixture["audio_noise"], common_noise[1])
        a, d = fixture["prompts"]["A"], fixture["prompts"]["D"]
        spans = packed["action_text_spans_local"]
        assert a.shape == d.shape
        is_action = torch.zeros(len(a), dtype=torch.bool)
        for lo, hi in spans:
            is_action[lo:hi] = True
            assert not torch.equal(a[lo:hi], d[lo:hi])
        assert torch.equal(a[~is_action], d[~is_action])
        visibility = {}
        for stop in (12, 17):
            layouts = []
            for prompt in (a, d):
                layout, cropped = visible_inputs(packed, prompt, stop, 390)
                assert len(layout["action_text_rows"]) == stop
                assert layout["seq_len"] == layout["action_video_start"] + stop * 390
                assert layout["img_pos"].numel() == (stop + 1) * 390
                layouts.append(layout)
            for key in layouts[0]:
                x, y = layouts[0][key], layouts[1][key]
                assert torch.equal(x, y) if torch.is_tensor(x) else x == y, key
            cache = H3ChunkCache(max_history=5, storage_device="cpu")
            start = 0 if stop == 12 else 12
            control = ChunkAttention(cache, 0 if start == 0 else 1,
                                     int(layouts[0]["action_video_start"]), 390,
                                     start, layouts[0]["action_text_rows"], "own", True)
            _, mask = control.masks(390, 0, "cpu")
            own_lo, own_hi = layouts[0]["action_text_rows"][start].tolist()
            assert bool(mask[0, own_lo:own_hi].all())
            for k, (lo, hi) in enumerate(layouts[0]["action_text_rows"].tolist()):
                if k != start:
                    assert not bool(mask[0, lo:hi].any()), (stop, k)
            visibility[str(stop)] = {"future_actions_removed": 37-stop,
                                      "current_video_own_action_only": True,
                                      "position_layout_A_equals_D": True}
        report["scenes"][name] = {"source_png_sha256": fixture["source_png_sha256"],
                                  "single_i0": True, "same_nonaction_prompt": True,
                                  "same_noise_and_layout_for_A_D": True,
                                  "visibility": visibility}
        report["fixture_sha256"][name] = sha(path)
        atomic_json(OUT / "P1_cpu_audit.json", report)
    report["status"] = "PASS"
    report["same_initial_video_audio_noise_between_scenes"] = True
    atomic_json(OUT / "P1_cpu_audit.json", report)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
