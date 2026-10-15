"""CPU-only official H3 Stage2-fix candidate and fail-closed input inventory."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOLAR = ROOT / "SolarWM"
ORIGINAL = SOLAR / "configs/examples/minimax_h3/infer-stage2-158f-sp4.yaml"
DATA = ROOT / "data/SolarWM-Data/releases-v1"
MODEL = ROOT / "models/SolarWM/SolarWM-h3-33B-base"
STAGE2 = ROOT / "models/SolarWM/SolarWM-h3-33B-sgf-stage2-158f-fix"
GENERATION = "minimax-h3-158f-768p-nomind-v1"
RECIPE = f"recipes/clean-158f-h3/latent-wds/{GENERATION}"
SUPPORT = DATA / "latent-wds" / GENERATION / "support"
INDEX = DATA / RECIPE / "test-index.jsonl.gz"
SILENCE = SUPPORT / "h3_silence_153_158_170.safetensors"
CONTRACT = SUPPORT / "encoder_contract.json"
HF_MAIN_REVISION = "60c85899ee5303bb3ccdd5add7b5d606c54ddaab"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write(name: str, value):
    (HERE / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def main():
    source = yaml.safe_load(ORIGINAL.read_text())
    candidate = yaml.safe_load(ORIGINAL.read_text())
    changes = {
        "model.checkpoint_path": str(MODEL),
        "data.index_root": str(DATA),
        "data.transport.root": str(DATA),
        "data.silence_latents_path": str(SILENCE),
        "data.encoder_contract_path": str(CONTRACT),
        "checkpoint.resume_from": str(STAGE2),
        "checkpoint.weight_source": "ema",
        "distributed.sequence_parallel_size": 8,
        "validation.sample_count": 1,
        "validation.balance_by_dataset": False,
        "runtime.output_dir": str(ROOT / "outputs/EXP-026_solar_stage2_reference"),
    }
    diff = []
    for dotted, new in changes.items():
        keys = dotted.split(".")
        original_node, candidate_node = source, candidate
        for key in keys[:-1]:
            original_node = original_node[key]
            candidate_node = candidate_node[key]
        previous = original_node[keys[-1]]
        candidate_node[keys[-1]] = new
        if previous != new:
            diff.append({"path": dotted, "official": previous, "candidate": new})
    # Only a resource-driven SP8/DP1 topology can schedule precisely one case
    # with world_size=8; this does NOT make its full 33B weights fit on L40.
    assert candidate["distributed"]["world_size"] == 8
    assert candidate["distributed"]["sequence_parallel_size"] == 8
    assert candidate["validation"]["sample_count"] == 1
    path = HERE / "candidate_single_case.yaml"
    path.write_text(yaml.safe_dump(candidate, sort_keys=False, allow_unicode=True))
    write("OFFICIAL_CONFIG_DIFF.json", {
        "official_path": str(ORIGINAL.relative_to(ROOT)),
        "official_sha256": sha(ORIGINAL),
        "candidate_path": str(path.relative_to(ROOT)),
        "candidate_sha256": sha(path),
        "changes": diff,
        "warning": "SP8/DP1 permits one logical sample but does not shard the 61.7GiB base across ranks"})

    import sys
    sys.path.insert(0, str(SOLAR / "src"))
    from solarwm.config.loader import load_config
    from solarwm.data import resolve_index_path
    from solarwm.backends.minimax_h3.optional import require_h3_runtime
    parsed = load_config(path)
    runtime_torch, _, _ = require_h3_runtime()
    if runtime_torch.cuda.is_initialized():
        raise RuntimeError("CPU preflight unexpectedly initialized CUDA")
    assert parsed.values == candidate
    assert resolve_index_path(candidate["data"], "test_index") == INDEX
    repo_sha = subprocess.check_output(["git", "-C", str(SOLAR), "rev-parse", "HEAD"], text=True).strip()
    repo_diff = subprocess.check_output(["git", "-C", str(SOLAR), "status", "--short"], text=True).strip()
    weight_index = MODEL / "transformer/diffusion_pytorch_model.safetensors.index.json"
    weight_bytes = json.loads(weight_index.read_text())["metadata"]["total_size"]
    ema = STAGE2 / "ema.pt"
    expected_ema = "92e8f4b07aa92b0453a2f63954db1421caa3b8cac83ece706535db1a91493a05"
    inventory = json.loads((ROOT / "submission/experiments/EXP-019_v3_mixed_action_validation/judge/SOLAR_REFERENCE_INVENTORY.json").read_text())
    assert inventory["ema_sha256"] == expected_ema
    assert ema.is_file() and ema.stat().st_size == inventory["bytes"]
    required = {
        "test_index": INDEX,
        "silence_support": SILENCE,
        "encoder_contract": CONTRACT,
        "selected_payload_shard": None,
    }
    missing = [name for name, item in required.items() if item is None or not item.is_file()]
    write("INPUT_MANIFEST.json", {
        "status": "BLOCKED_ACCESS", "official_checkout": repo_sha,
        "main_dataset_repo": "https://huggingface.co/datasets/junchaoh-cs/SolarWM-Data",
        "main_dataset_revision": HF_MAIN_REVISION,
        "latent_dataset_repo": "https://modelscope.cn/datasets/junchao2003/SolarWM-Data_Latent-WDS_minimax-h3-158f-768p-nomind-v1",
        "main_model_repo": "https://huggingface.co/junchaoh-cs/SolarWM-H3-33B",
        "test_index_remote_bytes": 414621,
        "access_errors": [
            {"provider": "Hugging Face", "target": "main dataset recipe/test-index", "http": 401,
             "class": "GatedRepoError", "detail": "repo listing available, file download requires authenticated accepted access"},
            {"provider": "ModelScope", "target": "main dataset and H3 latent generation tree", "http": 403,
             "class": "PermissionDeniedError", "detail": "dataset tree denied without account access"},
        ],
        "sample_id": None, "selected_index_row": None,
        "payload_shard_size": None, "source_path": None,
        "selection": "Official H3 validation_candidates: one unbalanced slot, seed42, first SHA256(42:sample_id)-ordered distinct source with valid camera; not yet executable without index",
        "required": {name: ({"path": str(item), "exists": item.is_file(),
                              "bytes": item.stat().st_size if item.is_file() else None,
                              "sha256": sha(item) if item.is_file() else None} if item is not None
                             else {"path": None, "exists": False, "bytes": None, "sha256": None})
                     for name, item in required.items()},
        "downloaded_bytes": 0, "new_payload_bytes": 0,
        "missing": missing,
        "base_transformer_weights_bytes": weight_bytes,
        "base_transformer_weights_gib": weight_bytes / 2**30,
        "stage2_ema_path": str(ema), "stage2_ema_bytes": ema.stat().st_size,
        "stage2_ema_sha256_prior_verified": expected_ema,
    })
    versions = {}
    for name in ("torch", "diffusers", "transformers", "safetensors", "accelerate", "imageio", "solarwm"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    write("CPU_PREFLIGHT.json", {
        "status": "BLOCKED_INPUT_AND_GPU_MEMORY", "gpu_initialized": runtime_torch.cuda.is_initialized(),
        "h3_runtime_imports": True,
        "official_checkout": repo_sha, "official_worktree_diff": repo_diff,
        "candidate_config_parses": True,
        "candidate_resolved_digest": parsed.resolved_digest,
        "resolved_test_index": str(resolve_index_path(candidate["data"], "test_index")),
        "all_required_input_present": not missing,
        "sample_reader_executed": False,
        "reason_sample_reader_not_executed": "index, support, and selected latent shard are inaccessible",
        "base_model_transformer_weights_gib": weight_bytes / 2**30,
        "per_rank_l40_vram_gib": 46068 / 1024,
        "single_sample_dp_world_size": 1,
        "initial_full_model_load_fit_l40": weight_bytes < 46068 * 2**20,
        "environment_versions": versions,
        "input_manifest_sha256": sha(HERE / "INPUT_MANIFEST.json"),
    })
    print("CPU_CONFIG_PARSE_PASS_INPUT_BLOCKED", flush=True)


if __name__ == "__main__":
    main()
