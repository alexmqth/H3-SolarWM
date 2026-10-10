"""CPU provenance freeze for four predetermined EXP-013 train PNGs; no model loading."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CFG = json.loads((HERE / "config.json").read_text())
DATA = ROOT / "H3-World/data/abot_bridge"
CANDIDATE = ROOT / "submission/experiments/EXP-013_v3_multiscene_data_plan/candidate_manifest.json"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def inspect_sources() -> dict:
    assert sha(CANDIDATE) == CFG["candidate_manifest_sha256"]
    candidates = json.loads(CANDIDATE.read_text())["selected"]
    assert len(candidates) == len(CFG["scenes"]) == 4
    anns = {r["sample_id"]: r for r in json.loads((DATA / "annotation_pilot.json").read_text())}
    clips = {r["clip_id"]: r for r in (json.loads(line) for line in (DATA / "clips.jsonl").read_text().splitlines())}
    prepared = json.loads((DATA / "preparation.json").read_text())
    downloads = {r["sample_id"]: r for r in prepared["downloads"]}
    old = json.loads((DATA / "encoded_manifest.json").read_text())
    assert old["anchor_protocol"] == "global_retimed_rgb_prefix_last_image_dual_v2"
    validation = {r["sample_id"] for r in clips.values() if r["split"] == "validation"}
    train = {r["sample_id"] for r in clips.values() if r["split"] == "train"}
    assert train.isdisjoint(validation) and len(train) == 4 and len(validation) == 2
    result = {"task": CFG["task"], "candidate_manifest_sha256": sha(CANDIDATE),
              "prior_encoded_protocol": old["anchor_protocol"],
              "prior_encoded_files_used": False,
              "train_episode_ids": sorted(train), "excluded_validation_episode_ids": sorted(validation),
              "scenes": {}, "input_files": {}}
    for (name, spec), candidate in zip(CFG["scenes"].items(), candidates):
        sid, stem = spec["sample_id"], spec["stem"]
        cid = f"{sid}_{stem}"
        assert candidate["episode_id"] == sid and candidate["clip_id"] == cid
        assert sid in train and sid not in validation
        row = clips[cid]
        assert row["split"] == candidate["split"] == "train"
        png = Path(candidate["first_frame"])
        assert png == Path(row["first_frame"]) and png.is_file()
        raw = png.read_bytes()
        assert raw[:8] == b"\x89PNG\r\n\x1a\n"
        size = struct.unpack(">II", raw[16:24])
        assert size == (CFG["width"], CFG["height"])
        assert sha(png) == row["first_frame_sha256"] == candidate["first_frame_sha256"]
        static = anns[sid]["caption"]["scene_static"]
        assert static == row["prompt"] == candidate["static_prompt"]
        assert hashlib.sha256(static.encode()).hexdigest() == candidate["static_prompt_sha256"]
        assert row["source_video_sha256"] == downloads[sid]["sha256"]
        assert row["source_annotations_sha256"] == anns[sid]["sha256"]
        assert row["RGB_frames"] == 39 and row["latent_frames"] == 12
        assert len(row["source_frame_indices"]) == 39
        assert candidate["future_counterfactual_actions"] == ["A", "D"]
        assert candidate["counterfactual_is_not_recorded_ground_truth"] is True
        result["scenes"][name] = {
            "clip_id": cid, "sample_id": sid, "split": "train", "src_start": row["src_start"],
            "png": str(png.relative_to(ROOT)), "png_sha256": sha(png),
            "png_bytes": len(raw), "png_width_height": list(size),
            "scene_static": static,
            "scene_static_sha256": hashlib.sha256(static.encode()).hexdigest(),
            "recorded_frame_key_combinations": candidate["recorded_frame_key_combinations"],
            "counterfactual_actions": ["A", "D"],
            "source_video_url": downloads[sid]["source_url"],
            "source_video_sha256": row["source_video_sha256"],
            "source_annotations_sha256": row["source_annotations_sha256"],
        }
    for rel in ("submission/experiments/EXP-013_v3_multiscene_data_plan/candidate_manifest.json",
                "H3-World/data/abot_bridge/clips.jsonl",
                "H3-World/data/abot_bridge/annotation_pilot.json",
                "H3-World/data/abot_bridge/encoded_manifest.json",
                "H3-World/data/abot_bridge/preparation.json"):
        path = ROOT / rel
        result["input_files"][rel] = {"sha256": sha(path), "bytes": path.stat().st_size}
    return result


if __name__ == "__main__":
    out = HERE / "source_manifest.json"
    out.write_text(json.dumps(inspect_sources(), ensure_ascii=False, indent=2) + "\n")
    print(f"wrote {out}")
