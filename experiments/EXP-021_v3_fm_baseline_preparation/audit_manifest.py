"""Independent structural audit of EXP-021's frozen candidate manifest (CPU)."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPAN = 69  # 56 output frames sampled at 24 fps from 30 fps source


def audit(manifest: dict) -> dict:
    train = manifest["train_candidates"]
    dev = manifest["dev_candidates"]
    assert len(train) == 32 and len(dev) == 8
    assert manifest["partition"] == [12, 5]
    assert manifest["rgb_frames"] == 56 and manifest["latent_frames"] == 17
    assert {r["episode_id"] for r in train}.isdisjoint({r["episode_id"] for r in dev})
    assert len({r["clip_id"] for r in train + dev}) == 40
    by_episode = defaultdict(list)
    categories = Counter()
    materialized = 0
    for row in train + dev:
        assert row["split"] in ("train", "validation")
        assert len(row["source_indices"]) == 56
        assert row["source_indices"][0] == row["source_start"]
        assert row["source_indices"][-1] == row["source_start"] + SPAN - 1
        assert all(0 <= x < 1800 for x in row["source_indices"])
        assert len(row["action_script17"]) == 17
        assert row["prefix12_action_exact_with_full17"] is True
        assert not any(row["unsupported_key_counts"].values())
        assert row["opposing_keys_same_frame"] is False
        assert row["category"] in ("constant", "switch")
        if row["category"] == "constant":
            assert len(row["frame_key_combinations"]) == 1
            assert "none" not in row["frame_key_combinations"]
            assert row["frame_switch_count"] == 0
        if row["materialized_real56"]:
            assert row in train and row["materialization_status"] == "verified_existing_56f"
            materialized += 1
        else:
            assert row["materialization_status"] == "source_window_only_needs_RGB_and_VAE_encoding"
        by_episode[row["episode_id"]].append(row["source_start"])
        categories[(row["split"], row["category"])] += 1
    for starts in by_episode.values():
        assert all(b - a >= SPAN for a, b in zip(sorted(starts), sorted(starts)[1:]))
    assert materialized == 4
    assert categories == {("train", "constant"): 16, ("train", "switch"): 16,
                          ("validation", "constant"): 4, ("validation", "switch"): 4}
    return {"status": "PASS_CPU_MANIFEST_STRUCTURE", "total": 40,
            "train_source_windows": 32, "train_materialized": materialized,
            "train_unmaterialized": 32 - materialized, "dev_unmaterialized": 8,
            "episodes": len(by_episode), "per_episode": {k: len(v) for k, v in sorted(by_episode.items())},
            "checks": ["episode-disjoint train/dev", "nonoverlapping source spans per episode",
                       "56-frame source indices and native 12+5 action layout",
                       "constant/switch counts", "no unsupported or opposing key flags"],
            "scope": "structural manifest audit; not independent RGB/VAE or 33B validation"}


if __name__ == "__main__":
    result = audit(json.loads((HERE / "DATA_MANIFEST.json").read_text()))
    output = HERE / "MANIFEST_AUDIT.json"
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
