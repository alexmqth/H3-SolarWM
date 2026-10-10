"""CPU validation of the single frozen AF2 step32 checkpoint pair."""
from __future__ import annotations

import json

from common import HERE, ROOT, sha


def load_checkpoint():
    import torch

    manifest = json.loads((HERE / "source_manifest.json").read_text())
    entries = manifest["sources"]
    paths = {key: ROOT / entries[key]["path"]
             for key in ("af2_qkv", "af2_target", "af2_trainer_state")}
    for key, path in paths.items():
        if sha(path) != entries[key]["sha256"]:
            raise RuntimeError(f"AF2 checkpoint changed: {key}")
    qkv = torch.load(paths["af2_qkv"], map_location="cpu", weights_only=True)
    target = torch.load(paths["af2_target"], map_location="cpu", weights_only=True)
    state = torch.load(paths["af2_trainer_state"], map_location="cpu", weights_only=True)
    assert qkv["format"] == "h3_causal_tail_qkv_v2"
    assert qkv["rank"] == 8 and qkv["block_indices"] == list(range(42, 50))
    assert qkv["scale"] == 0.125
    assert target["format"] == "h3world_anyflow_time_v1"
    assert target["gate"] == 0.25 and target["velocity_convention"] == "noise-clean"
    assert state["task"] == qkv["metadata"]["task"] == target["metadata"]["task"] == "EXP-007/v3-AF2"
    assert state["step"] == qkv["metadata"]["step"] == target["metadata"]["step"] == 32
    assert state["metadata"] == qkv["metadata"] == target["metadata"]
    assert state["qkv_sha256"] == entries["af2_qkv"]["sha256"]
    assert state["target_sha256"] == entries["af2_target"]["sha256"]
    assert state["metadata"]["protocol"] == "V3 strict causal/global/current-prefix/Single I0/12+5/sigma0 student KV"
    assert state["metadata"]["precision"] == "h3_fp32"
    # AF3 source inventory is an independent record of this same accepted pair.
    parent = json.loads((ROOT / entries["exp007_af3_source_manifest"]["path"]).read_text())
    assert parent["sources"]["af2_qkv"]["sha256"] == entries["af2_qkv"]["sha256"]
    assert parent["sources"]["af2_target"]["sha256"] == entries["af2_target"]["sha256"]
    assert parent["sources"]["af2_state"]["sha256"] == entries["af2_trainer_state"]["sha256"]
    return qkv, target, state
