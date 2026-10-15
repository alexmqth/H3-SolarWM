"""CPU source, native layout and training-wrapper contract checks for EXP-022."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
FROZEN = ROOT / "H3-World/outputs/2026-10-09-22/chunk_partition_cb/runtime"
sys.path[:0] = [str(FROZEN / "code"), str(FROZEN / "code/causal"),
                str(FROZEN / "DiffSynth-Studio-h3-v2"),
                str(ROOT / "submission/reports/stage1_anyflow/02_causal_diagnostics/current_prefix_candidate"),
                str(ROOT / "submission/experiments/EXP-002_native_cached")]

import torch
from causal.local_topology import visible_inputs
from causal.h3_cached import H3ChunkCache, ChunkAttention, slice_packed
from current_prefix import current_prefix_feedback
from interval_train import interval_train


def sha(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in ("", "-1"):
        raise RuntimeError("CPU preflight only")
    known = json.loads((ROOT / "submission/experiments/EXP-019_v3_mixed_action_validation/SOURCE_MANIFEST.json").read_text())
    source = [
        "submission/experiments/EXP-002_native_cached/interval_cached.py",
        "submission/reports/stage1_anyflow/02_causal_diagnostics/current_prefix_candidate/current_prefix.py",
        "H3-World/outputs/2026-10-09-22/chunk_partition_cb/runtime/code/causal/h3_cached.py",
        "H3-World/outputs/2026-10-09-22/chunk_partition_cb/runtime/code/causal/local_topology.py",
        "H3-World/outputs/2026-10-09-22/chunk_partition_cb/runtime/code/causal/h3_precision.py",
        "H3-World/outputs/EXP-016_v3_real56_vae/b784d995/full17.pt",
        "H3-World/outputs/EXP-016_v3_real56_vae/b784d995/prefix12.pt",
    ]
    for path in source:
        assert sha(ROOT / path) == known[path], path
    epath = ROOT / "H3-World/outputs/EXP-019_v3_mixed_action_validation/E/b784d995.pt"
    ehash = "0347bbdd9c7860bef593f8861eca495fe91a0708cfdbee59f45409640eaae9cc"
    assert sha(epath) == ehash
    f = torch.load(epath, map_location="cpu", weights_only=True)
    clean = torch.load(ROOT / source[-2], map_location="cpu", weights_only=True)
    prefix = torch.load(ROOT / source[-1], map_location="cpu", weights_only=True)
    assert clean.shape == (1, 24, 17, 30, 52)
    assert torch.equal(clean[:, :, :12], prefix)
    assert f["format"] == "exp019_mixed_native_v1" and len(f["script"]) == 17
    assert f["anchor"].shape == (390, 96)
    vis = {}
    for stop in (12, 17):
        p, pr = visible_inputs(f["packed"], f["prompt"], stop, 390)
        assert len(p["action_text_rows"]) == stop
        assert int(p["seq_len"]) == int(p["action_video_start"]) + stop * 390
        assert len(pr) == len(p["text_pos"])
        q = slice_packed(p, 0 if stop == 12 else 12, stop, 390)
        assert torch.equal(q["img_position_ids"][:, - (12 if stop == 12 else 5) * 390:],
                           p["img_position_ids"][:, - (12 if stop == 12 else 5) * 390:])
        vis[stop] = (p, pr)
    # Changing an unknown future action embedding cannot alter C1 visible inputs.
    changed = f["prompt"].clone()
    future_text = f["packed"]["action_text_spans_local"][12][0]
    changed[future_text:] = 0
    alt_p, alt_pr = visible_inputs(f["packed"], changed, 12, 390)
    assert torch.equal(alt_pr, vis[12][1])
    for field in ("img_position_ids", "token_tags", "action_text_rows"):
        assert torch.equal(alt_p[field], vis[12][0][field])
    with current_prefix_feedback():
        c = ChunkAttention(H3ChunkCache(5, "cpu"), 1,
                           int(vis[17][0]["action_video_start"]), 390, 12,
                           vis[17][0]["action_text_rows"], "own", True,
                           allow_grad_read=True)
        p_mask, v_mask = c.masks(5 * 390, 12 * 390, "cpu")
        assert p_mask.shape[0] == int(vis[17][0]["action_video_start"])
        assert v_mask.shape == (5 * 390, p_mask.shape[0] + 17 * 390)
        assert ChunkAttention.attend.__module__ == "current_prefix"
    assert interval_train.__module__ == "interval_train"
    info = {"status": "PASS_CPU_SOURCE_LAYOUT_ONLY", "gpu_calls": 0,
            "sources": {x: known[x] for x in source},
            "fixture": {"path": str(epath.relative_to(ROOT)), "sha256": ehash},
            "clean_prefix12_exact": True, "future_action_embedding_C1_isolated": True,
            "native_partition": [12, 5], "frame_rows": 390,
            "new_wrapper": {"path": str((HERE / "interval_train.py").relative_to(ROOT)),
                            "sha256": sha(HERE / "interval_train.py")},
            "scope": "CPU structural checks; 33B forward/KV equivalence pending authorized GPU"}
    output = HERE / "CPU_PREFLIGHT.json"
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps({k: v for k, v in info.items() if k != "sources"}, indent=2))


if __name__ == "__main__":
    main()
