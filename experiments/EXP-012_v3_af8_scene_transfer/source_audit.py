"""Independent source inventory for EXP-012 P0; CPU and filesystem only."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
AF2 = ROOT / "H3-World/outputs/EXP-007_v3_anyflow_af2"
EXP007 = ROOT / "submission/experiments/EXP-007_v3_anyflow"
EXP011 = ROOT / "submission/experiments/EXP-011_v3_scene_transfer"
O11 = ROOT / "H3-World/outputs/EXP-011_v3_scene_transfer"
FROZEN = ROOT / "H3-World/outputs/2026-10-09-22/chunk_partition_cb"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def paths() -> dict[str, Path]:
    p1 = json.loads((O11 / "P1_result.json").read_text())
    result = {
        "taskbook": HERE / "taskbook_v1.md",
        "exp011_source_manifest": EXP011 / "source_manifest.json",
        "exp011_code_manifest": EXP011 / "code_manifest.json",
        "exp011_p1_result": O11 / "P1_result.json",
        "exp011_final_judge": EXP011 / "judge/FINAL_REVIEW.md",
        "exp007_af3_source_manifest": EXP007 / "source_manifest_v3_af3.json",
        "exp007_interval_student": EXP007 / "interval_student.py",
        "exp007_af3_runner": EXP007 / "run_af3.py",
        "exp007_af2_result": AF2 / "result.json",
        "af2_qkv": AF2 / "step_32/qkv.pt",
        "af2_target": AF2 / "step_32/target_time.pt",
        "af2_trainer_state": AF2 / "step_32/trainer_state.pt",
        "released_action_lora": FROZEN / "runtime/checkpoints/H3-World/step-10000.safetensors",
        "anyflow": FROZEN / "runtime/code/causal/anyflow.py",
        "anyflow_sampling": FROZEN / "runtime/code/causal/anyflow_sampling.py",
        "pretrained_lora": FROZEN / "runtime/code/causal/pretrained_lora.py",
        "h3_precision": FROZEN / "runtime/code/causal/h3_precision.py",
        "h3_cached": FROZEN / "runtime/code/causal/h3_cached.py",
        "local_topology": FROZEN / "runtime/code/causal/local_topology.py",
        "current_prefix": ROOT / "submission/reports/stage1_anyflow/02_causal_diagnostics/current_prefix_candidate/current_prefix.py",
        "infer_loader": FROZEN / "runtime/code/abot/infer.py",
        "video_writer": FROZEN / "runtime/code/causal/benchmark.py",
    }
    for scene in ("industrial", "village"):
        item = p1["fixtures"][scene]
        result[f"fixture_{scene}"] = ROOT / item["path"]
        folder = O11 / "G1" / scene / "FM8"
        result[f"g1_{scene}_row"] = folder / "result.json"
        result[f"g1_{scene}_first12"] = folder / "first12.pt"
        result[f"g1_{scene}_rgb39"] = folder / "published_39.npy"
        result[f"fm8_{scene}_aa_video"] = O11 / "G2" / scene / "FM8/AA/rollout_56.mp4"
        result[f"fm8_{scene}_ad_video"] = O11 / "G2" / scene / "FM8/AD/rollout_56.mp4"
    return result


def inspect_sources() -> dict:
    entries = {}
    for label, path in paths().items():
        if not path.is_file():
            raise FileNotFoundError(path)
        entries[label] = {"path": str(path.relative_to(ROOT)), "sha256": sha(path),
                          "bytes": path.stat().st_size}
    return {"task": "EXP-012/v1", "checkpoint_step": 32, "sources": entries}


if __name__ == "__main__":
    target = HERE / "source_manifest.json"
    data = inspect_sources()
    if target.exists():
        if json.loads(target.read_text()) != data:
            raise RuntimeError("source manifest changed after freeze")
    else:
        target.write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps({"task": data["task"], "sources": len(data["sources"]),
                      "manifest": str(target)}, indent=2))
