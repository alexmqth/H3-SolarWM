"""CPU-only real ABot 56-frame window inventory; does not encode or train."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import av
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "H3-World/code/abot"))
sys.path.insert(0, str(ROOT / "H3-World/code/causal"))
import abot_action as A  # noqa: E402
import action_script as S  # noqa: E402
from real_transition_data import bounded_keys9  # noqa: E402


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def save(path: Path, value) -> None:
    assert not path.exists(), f"refuse to overwrite {path}"
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def combinations(keys: np.ndarray) -> list[str]:
    return ["+".join(name for i, name in enumerate(A.KEY_COLS) if row[i] > .5) or "none"
            for row in keys]


def one_window(ep: dict, sample_id: str, split: str, start: int,
               scale: float, source: dict, materialized: Path | None = None) -> dict:
    indices = [start + x for x in A.window_offsets(56)]
    assert len(indices) == 56 and indices[-1] < ep["total_frames"]
    raw = A.window_action_matrix(ep, start, 56, scale)
    pooled = A.bin_to_latent(raw, 17)
    keys_full = bounded_keys9(pooled, stops=(12, 17))
    keys_prefix = bounded_keys9(pooled[:12], stops=(12,))
    assert np.array_equal(keys_full[:12], keys_prefix)
    script = S.annotate_from_keys9(keys_full)
    assert len(script) == 17 and raw.shape == (56, 17) and pooled.shape == (17, 17)
    assert np.isfinite(raw).all() and np.isfinite(pooled).all()
    comb = combinations(raw[:, :A.NUM_KEYS])
    switches = sum(a != b for a, b in zip(comb, comb[1:]))
    unsupported = {key: int(raw[:, A.KEY_COLS.index(key)].sum()) for key in ("Q", "E", "Space")}
    conflict = any(np.any(raw[:, A.KEY_COLS.index(a)] * raw[:, A.KEY_COLS.index(b)] > 0)
                   for a, b in (("W", "S"), ("A", "D"), ("I", "K"), ("J", "L")))
    category = "constant" if len(set(comb)) == 1 and comb[0] != "none" else "switch"
    out = {
        "clip_id": f"{sample_id}_src{start:04d}_rgb56", "episode_id": sample_id,
        "split": split, "source_start": start, "source_indices": indices,
        "source_video": source["video"], "source_annotations": source["annotations"],
        "source_video_sha256": source["video_sha256"],
        "source_annotations_sha256": source["annotations_sha256"],
        "scene_static_sha256": source["scene_static_sha256"],
        "category": category, "frame_key_combinations": dict(sorted(Counter(comb).items())),
        "frame_switch_count": switches, "unsupported_key_counts": unsupported,
        "opposing_keys_same_frame": bool(conflict),
        "action_matrix_sha256": hashlib.sha256(raw.tobytes()).hexdigest(),
        "pooled17_sha256": hashlib.sha256(pooled.tobytes()).hexdigest(),
        "keys9_17_sha256": hashlib.sha256(keys_full.tobytes()).hexdigest(),
        "action_script17": script,
        "action_script_sha256": hashlib.sha256(json.dumps(script, ensure_ascii=False).encode()).hexdigest(),
        "prefix12_action_exact_with_full17": True,
        "materialized_real56": str(materialized.relative_to(ROOT)) if materialized else None,
        "materialization_status": "verified_existing_56f" if materialized else "source_window_only_needs_RGB_and_VAE_encoding",
    }
    if materialized:
        old = np.load(materialized.parent / "raw56.npy")
        assert np.array_equal(raw, old)
        out["materialized_real56_sha256"] = sha(materialized)
        out["existing_action_matrix_exact"] = True
    return out


def spread_pick(options: list[dict], selected: list[dict], category: str, count: int) -> None:
    for _ in range(count):
        pool = [o for o in options if o["category"] == category and
                all(abs(o["source_start"] - x["source_start"]) >= A.window_span(56) for x in selected)]
        if not pool:
            raise RuntimeError(f"insufficient non-overlap {category} windows")
        choice = max(pool, key=lambda o: (min(abs(o["source_start"] - x["source_start"]) for x in selected)
                                          if selected else 1800,
                                          -o["frame_switch_count"] if category == "constant" else o["frame_switch_count"],
                                          -o["source_start"]))
        selected.append(choice)


def main() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in ("", "-1"):
        raise RuntimeError("CPU task: set CUDA_VISIBLE_DEVICES='' ")
    started = time.monotonic()
    old = ROOT / "submission/experiments/EXP-013_v3_multiscene_data_plan/source_manifest.json"
    source_manifest = json.loads(old.read_text())["sources"]
    baseline = {r["episode_id"]: r["src_start"] for r in
                json.loads((ROOT / "submission/experiments/EXP-013_v3_multiscene_data_plan/candidate_manifest.json").read_text())["selected"]}
    episodes = []
    train, dev = [], []
    src_checks = []
    for sample_id, prior in sorted(source_manifest.items()):
        split = prior["split"][0]
        source_dir = ROOT / "H3-World/data/abot_bridge/raw/data" / sample_id[:2] / sample_id
        video, annotations = source_dir / "video.mp4", source_dir / "annotations.tar"
        assert video.is_file() and annotations.is_file()
        assert sha(video) == prior["source_video_sha256"]
        assert sha(annotations) == prior["annotations_sha256"]
        with av.open(str(video)) as container:
            stream = container.streams.video[0]
            assert stream.average_rate == 30 and stream.frames == 1800
            assert (stream.width, stream.height) == (1920, 1080)
        ep = A.read_episode(str(annotations))
        assert ep["total_frames"] == 1800 and ep["fps"] == 30
        assert ep["control_scheme"] == "WASD_QE_locomotion_IJKL_rotation"
        caption = ep["caption"]["scene_static"]
        assert hashlib.sha256(caption.encode()).hexdigest() == prior["source_caption_scene_static_sha256"]
        scale = A.episode_translation_scale(ep)
        source = {"video": str(video.relative_to(ROOT)), "annotations": str(annotations.relative_to(ROOT)),
                  "video_sha256": prior["source_video_sha256"],
                  "annotations_sha256": prior["annotations_sha256"],
                  "scene_static_sha256": prior["source_caption_scene_static_sha256"]}
        options = [one_window(ep, sample_id, split, start, scale, source)
                   for start in range(0, ep["total_frames"] - A.window_span(56) + 1, A.window_span(56))]
        options = [o for o in options if not any(o["unsupported_key_counts"].values()) and not o["opposing_keys_same_frame"]]
        chosen = []
        if split == "train":
            start = baseline[sample_id]
            materialized = ROOT / "submission/experiments/EXP-015_v3_real56_data/artifacts" / sample_id[:8] / "real56.mp4"
            assert materialized.is_file()
            chosen.append(one_window(ep, sample_id, split, start, scale, source, materialized))
            target_constant = 4 - sum(x["category"] == "constant" for x in chosen)
            target_switch = 4 - sum(x["category"] == "switch" for x in chosen)
            spread_pick(options, chosen, "constant", target_constant)
            spread_pick(options, chosen, "switch", target_switch)
            assert len(chosen) == 8
            train.extend(sorted(chosen, key=lambda x: x["source_start"]))
        else:
            spread_pick(options, chosen, "constant", 2)
            spread_pick(options, chosen, "switch", 2)
            assert len(chosen) == 4
            dev.extend(sorted(chosen, key=lambda x: x["source_start"]))
        episodes.append({"episode_id": sample_id, "split": split, "available_nonoverlap_windows": len(options),
                         "selected": len(chosen), "selected_constant": sum(x["category"] == "constant" for x in chosen),
                         "selected_switch": sum(x["category"] == "switch" for x in chosen)})
        src_checks.append({"episode_id": sample_id, "video_sha256": source["video_sha256"],
                           "annotations_sha256": source["annotations_sha256"], "frame_count": 1800,
                           "source_fps": 30, "action_pose_coverage": 1800,
                           "source_window_span_for_56_rgb": A.window_span(56),
                           "translation_scale": scale})
    train_ids = {x["episode_id"] for x in train}
    dev_ids = {x["episode_id"] for x in dev}
    assert len(train) == 32 and len(dev) == 8 and len(train_ids) == 4 and len(dev_ids) == 2
    assert not train_ids & dev_ids
    assert len({x["clip_id"] for x in train + dev}) == 40
    manifest = {"task": "EXP-021/v1", "status": "CPU_CANDIDATES_ONLY",
                "source_manifest": str(old.relative_to(ROOT)), "source_manifest_sha256": sha(old),
                "rgb_frames": 56, "latent_frames": 17, "partition": [12, 5],
                "source_fps": 30, "output_fps": 24, "frame_selection": "drop every fifth source frame",
                "selection_rule": "per episode 4 constant-active and 4 switch train windows; baseline real56 retained; new source windows nonoverlap each other and baseline; validation 2+2 per episode",
                "train_candidates": train, "dev_candidates": dev,
                "episode_summary": episodes,
                "blind_test_status": "NOT_AVAILABLE: only two previously observed validation episodes locally; need 4-6 new episodes for 6-8 episode-blind scenes"}
    save(HERE / "DATA_MANIFEST.json", manifest)
    checks = {"status": "PASS_CPU_SOURCE_ACTION_ONLY", "elapsed_seconds": time.monotonic() - started,
              "gpu_calls": 0, "train_updates": 0, "source_checks": src_checks,
              "counts": {"train_candidate_windows": 32, "train_materialized_56f": 4,
                         "train_needs_materialization": 28, "dev_candidate_windows": 8,
                         "train_episodes": 4, "dev_episodes": 2,
                         "train_constant": sum(x["category"] == "constant" for x in train),
                         "train_switch": sum(x["category"] == "switch" for x in train),
                         "dev_constant": sum(x["category"] == "constant" for x in dev),
                         "dev_switch": sum(x["category"] == "switch" for x in dev)},
              "checks": {"source_video_sha": True, "annotation_sha": True, "full_1800_frame_action_pose": True,
                         "source_time_and_56_frame_indices": True, "nonoverlap_within_episode": True,
                         "episode_disjoint_train_dev": True, "native_17_latent_action_bins": True,
                         "prefix12_actions_equal_full17_prefix": True,
                         "existing_four_raw56_matrices_exact": True},
              "not_checked": ["new 28 train + 8 dev RGB windows not materialized/decoded",
                              "new window VAE latents/I0/prompt/KV not encoded",
                              "visual data quality of new windows not reviewed",
                              "no blind episodes beyond observed 2 validation episodes"]}
    save(HERE / "CPU_CHECKS.json", checks)
    print(json.dumps(checks["counts"], indent=2), flush=True)


if __name__ == "__main__":
    main()
