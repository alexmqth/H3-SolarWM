"""CPU-only manifest generation for EXP-014 P1/T1/T2 code contract."""
from __future__ import annotations

import json
from pathlib import Path

from common import HERE, ROOT, FROZEN, atomic_json, sha


def main():
    own = [HERE / name for name in (
        "config.json", "taskbook_v1.md", "source_audit.py", "common.py",
        "encode_native.py", "run_teacher.py", "audit_fixtures.py", "test_p0.py")]
    frozen = [FROZEN / rel for rel in (
        "runtime/code/abot/action_script.py", "runtime/code/abot/abot_action.py",
        "runtime/code/abot/infer.py", "runtime/code/causal/h3_cached.py",
        "runtime/code/causal/local_topology.py", "runtime/code/causal/h3_precision.py",
        "runtime/code/causal/anyflow_sampling.py", "runtime/code/causal/benchmark.py",
        "runtime/code/causal/evaluate_action_control.py",
        "runtime/DiffSynth-Studio-h3-v2/diffsynth/pipelines/minimax_h3_audio_video.py",
        "runtime/DiffSynth-Studio-h3-v2/diffsynth/models/minimax_h3_dit.py",
        "runtime/DiffSynth-Studio-h3-v2/diffsynth/models/minimax_h3_text_encoder.py",
        "runtime/DiffSynth-Studio-h3-v2/diffsynth/models/minimax_h3_video_vae.py",
        "runtime/DiffSynth-Studio-h3-v2/diffsynth/diffusion/base_pipeline.py",
        "runtime/DiffSynth-Studio-h3-v2/diffsynth/diffusion/flow_match.py")]
    dependencies = [ROOT / rel for rel in (
        "submission/experiments/EXP-002_native_cached/interval_cached.py",
        "submission/reports/stage1_anyflow/02_causal_diagnostics/current_prefix_candidate/current_prefix.py",
        "submission/experiments/EXP-001_v2b_124/rollout_contract.py")]
    paths = own + frozen + dependencies
    manifest = {"task": "EXP-014/v1", "scope": "P1/T1/T2 code freeze; no weights copied",
                "files": {str(path.relative_to(ROOT)): sha(path) for path in paths}}
    atomic_json(HERE / "code_manifest.json", manifest)
    print(f"froze {len(paths)} code/config files; manifest SHA {sha(HERE / 'code_manifest.json')}")


if __name__ == "__main__":
    main()
