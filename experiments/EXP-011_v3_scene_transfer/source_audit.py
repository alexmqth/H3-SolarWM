"""EXP-011 CPU-only provenance audit. Never opens encoded .pt inputs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CFG = json.loads((HERE / "config.json").read_text())
DATA = ROOT / "H3-World/data/abot_bridge"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_sources() -> dict:
    annotations = {row["sample_id"]: row for row in json.loads((DATA / "annotation_pilot.json").read_text())}
    encoded = json.loads((DATA / "encoded_manifest.json").read_text())
    prepared = json.loads((DATA / "preparation.json").read_text())
    assert encoded["anchor_protocol"] == "global_retimed_rgb_prefix_last_image_dual_v2"
    clips = {row["clip_id"]: row for row in prepared["clips"]}
    prior = {row["clip_id"]: row for row in encoded["clips"]}
    downloads = {row["sample_id"]: row for row in prepared["downloads"]}
    result = {"task": CFG["task"], "prior_encoded_protocol": encoded["anchor_protocol"],
              "prior_encoded_files_used": False, "scenes": {}, "input_files": {}}
    for name, spec in CFG["scenes"].items():
        sid, stem = spec["sample_id"], spec["stem"]
        cid = f"{sid}_{stem}"
        png = DATA / "clips/validation" / sid / f"{stem}.png"
        raw = png.read_bytes()
        assert raw[:8] == b"\x89PNG\r\n\x1a\n"
        width, height = struct.unpack(">II", raw[16:24])
        assert (width, height) == (CFG["width"], CFG["height"])
        clip, old, ann, down = clips[cid], prior[cid], annotations[sid], downloads[sid]
        assert clip["split"] == old["split"] == "validation"
        assert clip["first_frame_sha256"] == sha(png)
        assert clip["source_video_sha256"] == down["sha256"]
        assert clip["source_annotations_sha256"] == ann["sha256"]
        assert clip["sample_id"] == old["sample_id"] == sid
        assert len(clip["source_frame_indices"]) == 39
        scene_static = ann["caption"]["scene_static"]
        assert scene_static == clip["prompt"] and scene_static.strip()
        result["scenes"][name] = {
            "clip_id": cid, "sample_id": sid, "split": "validation",
            "png": str(png.relative_to(ROOT)), "png_sha256": sha(png),
            "png_bytes": len(raw), "png_width_height": [width, height],
            "scene_static": scene_static, "scene_static_sha256": hashlib.sha256(scene_static.encode()).hexdigest(),
            "source_video_url": down["source_url"], "source_video_sha256": down["sha256"],
            "source_annotations_sha256": ann["sha256"],
            "prior_encoded_file": old["encoded_file"], "prior_encoded_protocol": encoded["anchor_protocol"],
            "first_frame_matches_preparation": True,
        }
    for rel in ("H3-World/data/abot_bridge/annotation_pilot.json",
                "H3-World/data/abot_bridge/encoded_manifest.json",
                "H3-World/data/abot_bridge/preparation.json"):
        path = ROOT / rel
        result["input_files"][rel] = {"sha256": sha(path), "bytes": path.stat().st_size}
    return result


if __name__ == "__main__":
    result = inspect_sources()
    out = HERE / "source_manifest.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(f"wrote {out}; {len(result['scenes'])} fixed validation scenes")
