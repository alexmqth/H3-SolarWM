"""Freeze hashes of the accepted EXP-006 code, conditions and model adapter."""
from pathlib import Path
import hashlib
import json

here = Path(__file__).resolve().parent
root = here.parents[2]
frozen = Path("H3-World/outputs/2026-10-09-22/chunk_partition_cb")
sources = {
    "runner": "submission/experiments/EXP-006_v3_fm8_full/run_fm8.py",
    "config": "submission/experiments/EXP-006_v3_fm8_full/config.json",
    "taskbook": "submission/experiments/EXP-006_v3_fm8_full/taskbook_v1.md",
    "parking_A": str(frozen / "source_coarse/inputs/parking_A.pt"),
    "parking_D": str(frozen / "source_coarse/inputs/parking_D.pt"),
    "released_action_lora": str(frozen / "runtime/checkpoints/H3-World/step-10000.safetensors"),
    "accepted_interval": "submission/experiments/EXP-002_native_cached/interval_cached.py",
    "accepted_current_prefix": "submission/reports/stage1_anyflow/02_causal_diagnostics/current_prefix_candidate/current_prefix.py",
    "raw_cache": str(frozen / "runtime/code/causal/h3_cached.py"),
    "precision": str(frozen / "runtime/code/causal/h3_precision.py"),
    "visible_inputs": str(frozen / "runtime/code/causal/local_topology.py"),
    "schedule_helper": str(frozen / "runtime/code/causal/anyflow_sampling.py"),
    "flow_scheduler": str(frozen / "runtime/DiffSynth-Studio-h3-v2/diffsynth/diffusion/flow_match.py"),
    "h3_dit": str(frozen / "runtime/DiffSynth-Studio-h3-v2/diffsynth/models/minimax_h3_dit.py"),
    "h3_pipeline": str(frozen / "runtime/DiffSynth-Studio-h3-v2/diffsynth/pipelines/minimax_h3_audio_video.py"),
    "infer_loader": str(frozen / "runtime/code/abot/infer.py"),
    "video_writer": str(frozen / "runtime/code/causal/benchmark.py"),
    "flow_metric": str(frozen / "runtime/code/causal/evaluate_action_control.py"),
    "rollout_contract": "submission/experiments/EXP-001_v2b_124/rollout_contract.py",
    "frozen_8step_schedule": "H3-World/outputs/EXP-004_v3_8step/AA/chunk_12_17.json",
}


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


rows = {name: {"path": rel, "sha256": sha(root / rel),
               "bytes": (root / rel).stat().st_size}
        for name, rel in sources.items()}
(here / "source_manifest.json").write_text(
    json.dumps({"task": "EXP-006/v1", "sources": rows}, indent=2) + "\n")
print(f"frozen {len(rows)} sources; runner={rows['runner']['sha256']}")
