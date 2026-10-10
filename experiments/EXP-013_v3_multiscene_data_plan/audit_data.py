#!/usr/bin/env python3
"""EXP-013 CPU-only, reproducible audit of the 24 local ABot clips.

Reads original data; writes only small manifests and a contact sheet beside this file.
Run from the GWM root with .venvs/h3world/bin/python. No model or GPU imports.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import av
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "H3-World/data/abot_bridge"
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "H3-World/code/abot"))
import abot_action as A  # noqa: E402

KEYS = A.KEY_COLS
FIXED_VALIDATION = {
    "118eb5d8b75e1b8ac23a4e9ae77af9a9",
    "dfec8ed3237860eba14d67c089ecd041",
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dump(name: str, data: object) -> None:
    (OUT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def decode_clip(path: Path, png: Path) -> dict:
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        pts = []
        size = set()
        first = None
        for frame in container.decode(stream):
            if first is None:
                first = frame.to_ndarray(format="rgb24")
            pts.append(float(frame.pts * frame.time_base) if frame.pts is not None else None)
            size.add((frame.width, frame.height))
        diffs = np.diff(pts) if len(pts) > 1 and None not in pts else np.array([])
        return {
            "codec": stream.codec_context.name,
            "fps": float(stream.average_rate) if stream.average_rate else None,
            "frame_count_decoded": len(pts),
            "resolution_set": sorted([list(x) for x in size]),
            "pts_first_last": [pts[0], pts[-1]] if pts else [],
            "pts_strict_increasing": bool(len(diffs) and np.all(diffs > 0)),
            "pts_delta_min_max": [float(diffs.min()), float(diffs.max())] if len(diffs) else [],
            "first_rgb_shape": list(first.shape) if first is not None else None,
            "first_frame_png_pixel_equal": bool(first is not None and np.array_equal(
                first, np.asarray(Image.open(png).convert("RGB")))),
        }


def combos(mat: np.ndarray) -> dict:
    labels = []
    for row in mat[:, :A.NUM_KEYS]:
        active = [name for i, name in enumerate(KEYS) if row[i] > .5]
        labels.append("+".join(active) if active else "none")
    return dict(sorted(Counter(labels).items()))


def source_first_frame_shift_scores(source_video: Path, index: int, png: Path) -> dict:
    """Independent raw-video spot check; compare source n-1/n/n+1 to clip PNG.

    Source and clip use different codecs/scaling, so only the minimum's shift
    is meaningful. The absolute MAD is not a video-quality metric.
    """
    target = np.asarray(Image.open(png).convert("RGB"), dtype=np.float32)
    scores = {}
    with av.open(str(source_video)) as container:
        stream = container.streams.video[0]
        start = int(max(index - 2, 0) / 30 / float(stream.time_base))
        container.seek(start, stream=stream, any_frame=False, backward=True)
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            src_index = round(float(frame.pts * frame.time_base) * 30)
            if src_index < index - 1:
                continue
            if src_index > index + 1:
                break
            raw = Image.fromarray(frame.to_ndarray(format="rgb24"))
            scaled_w = int(round(raw.width * 480 / raw.height / 2)) * 2
            scaled = raw.resize((scaled_w, 480), Image.Resampling.BILINEAR)
            left = (scaled_w - 832) // 2
            crop = np.asarray(scaled, dtype=np.float32)[:, left:left + 832]
            scores[str(src_index - index)] = float(np.mean(np.abs(crop - target)))
    return {"MAD_by_source_frame_shift": scores,
            "best_shift": min(scores, key=scores.get) if len(scores) == 3 else None,
            "source_frame_exact_best": len(scores) == 3 and
                                       scores["0"] < scores["-1"] and scores["0"] < scores["1"]}


def rebuild_clip_probe(rec: dict) -> dict:
    """Reproduce the original FFmpeg select/scale/encode recipe for an ambiguous spot check."""
    start = rec["src_start"]
    span = A.window_span(39)
    filt = (f"select='between(n\\,{start}\\,{start + span - 1})"
            f"*not(eq(mod(n-{start}\\,5)\\,4))',"
            "setpts=N/24/TB,scale=-2:480,crop=832:480")
    with tempfile.TemporaryDirectory(prefix="exp013_rebuild_") as temp:
        target = Path(temp) / "rebuilt.mp4"
        subprocess.run([
            imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-v", "error", "-threads", "2",
            "-i", rec["source_video"], "-vf", filt, "-r", "24", "-frames:v", "39",
            "-c:v", "libx264", "-crf", "14", "-preset", "veryfast",
            "-pix_fmt", "yuv420p", "-an", "-movflags", "+faststart", str(target),
        ], check=True, capture_output=True)
        rebuilt_sha = sha(target)
        return {"clip_id": rec["clip_id"], "rebuild_sha256": rebuilt_sha,
                "existing_sha256": rec["video_sha256"],
                "exact_byte_match": rebuilt_sha == rec["video_sha256"],
                "producer_filter": filt,
                "encoder": "libx264 crf14 preset=veryfast yuv420p"}


def main() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    rows = [json.loads(line) for line in (DATA / "clips.jsonl").read_text().splitlines() if line.strip()]
    ann = {r["sample_id"]: r for r in json.loads((DATA / "annotation_pilot.json").read_text())}
    old = json.loads((DATA / "encoded_manifest.json").read_text())
    preparation = json.loads((DATA / "preparation.json").read_text())
    encoding = json.loads((DATA / "encoding.json").read_text())
    episodes = defaultdict(list)
    for row in rows:
        episodes[row["sample_id"]].append(row)
    source = {}
    source_ep = {}
    for sid, group in sorted(episodes.items()):
        v = DATA / "raw/data" / sid[:2] / sid / "video.mp4"
        t = v.with_name("annotations.tar")
        source_ep[sid] = A.read_episode(str(t))
        with av.open(str(v)) as raw_container:
            raw_stream = raw_container.streams.video[0]
            raw_video_probe = {
                "codec": raw_stream.codec_context.name,
                "fps": float(raw_stream.average_rate) if raw_stream.average_rate else None,
                "reported_frames": raw_stream.frames,
                "width_height": [raw_stream.width, raw_stream.height],
                "duration_seconds": float(raw_stream.duration * raw_stream.time_base)
                    if raw_stream.duration is not None else None,
            }
        source[sid] = {
            "episode_id": sid, "split": sorted({r["split"] for r in group}),
            "source_video": str(v), "source_video_sha256": sha(v),
            "annotations": str(t), "annotations_sha256": sha(t),
            "source_fps": source_ep[sid]["fps"],
            "source_frames": source_ep[sid]["total_frames"],
            "source_control_scheme": source_ep[sid]["control_scheme"],
            "source_video_probe": raw_video_probe,
            "source_annotation_matches_pilot_sha256": sha(t) == ann[sid]["sha256"],
            "source_caption_scene_static_sha256": hashlib.sha256(
                source_ep[sid]["caption"].get("scene_static", "").encode()).hexdigest(),
            "clip_count": len(group),
        }
    metadata_sids = set()
    with (DATA / "source_metadata/metadata.jsonl").open() as f:
        for line in f:
            if line.strip():
                sid = json.loads(line).get("sample_id")
                if sid in episodes:
                    metadata_sids.add(sid)
    lineage = {
        "file_sha256": {name: sha(DATA / name) for name in (
            "clips.jsonl", "annotation_pilot.json", "preparation.json",
            "encoding.json", "encoded_manifest.json", "source_metadata/metadata.jsonl")},
        "all_episodes_in_source_metadata": metadata_sids == set(episodes),
        "preparation_download_ids_match":
            {r["sample_id"] for r in preparation["downloads"]} == set(episodes),
        "preparation_clip_ids_match":
            {r["clip_id"] for r in preparation["clips"]} == {r["clip_id"] for r in rows},
        "encoding_clip_ids_match":
            {r["clip_id"] for r in encoding["clips"]} == {r["clip_id"] for r in rows},
        "old_encoded_clip_ids_match":
            {r["clip_id"] for r in old["clips"]} == {r["clip_id"] for r in rows},
        "encoding_anchor_protocol": encoding.get("anchor_protocol"),
    }

    inventory = []
    problems = []
    for row in rows:
        sid = row["sample_id"]
        ep = source_ep[sid]
        mp4, npy, png = (Path(row[k]) for k in ("video", "action", "first_frame"))
        vid = decode_clip(mp4, png)
        mat = np.load(npy, allow_pickle=False)
        with Image.open(png) as im:
            png_size = list(im.size)
            png_mode = im.mode
        expected_idx = [row["src_start"] + i for i in A.window_offsets(39)]
        idx = row["source_frame_indices"]
        scale = A.episode_translation_scale(ep)
        expected_action = A.window_action_matrix(ep, row["src_start"], 39, scale)
        action_max_abs = float(np.max(np.abs(mat - expected_action))) if mat.shape == expected_action.shape else None
        key_counts = {key: int(np.count_nonzero(mat[:, i] > .5)) for i, key in enumerate(KEYS)}
        observed = combos(mat)
        static = ep["caption"].get("scene_static", "").strip()
        rec = {
            "clip_id": row["clip_id"], "sample_id": sid, "split": row["split"],
            "target_filename_label": row["target"], "src_start": row["src_start"],
            "source_video": source[sid]["source_video"],
            "source_video_sha256": source[sid]["source_video_sha256"],
            "source_annotations": source[sid]["annotations"],
            "source_annotations_sha256": source[sid]["annotations_sha256"],
            "source_frame_indices": idx, "source_fps": row["source_fps"],
            "video_path": str(mp4), "video_sha256": sha(mp4), "video_probe": vid,
            "first_frame_path": str(png), "first_frame_sha256": sha(png),
            "first_frame_size": png_size, "first_frame_mode": png_mode,
            "action_path": str(npy), "action_sha256": sha(npy),
            "action_shape": list(mat.shape), "action_dtype": str(mat.dtype),
            "action_finite": bool(np.isfinite(mat).all()),
            "key_columns_binary": bool(np.isin(mat[:, :A.NUM_KEYS], [0, 1]).all()),
            "continuous_columns_min_max": [[float(mat[:, i].min()), float(mat[:, i].max())]
                                          for i in range(A.NUM_KEYS, A.ACTION_DIM)],
            "key_counts": key_counts, "frame_key_combinations": observed,
            "source_index_formula_matches": idx == expected_idx,
            "action_recomputed_from_source_max_abs": action_max_abs,
            "prompt_origin": "annotation_pilot.caption.scene_static",
            "prompt_sha256": hashlib.sha256(row["prompt"].encode()).hexdigest(),
            "prompt_matches_annotation_static": row["prompt"] == ann[sid]["caption"]["scene_static"],
            "prompt_matches_raw_static": row["prompt"] == static,
            "manifest_sha_matches_files": {
                "video": sha(mp4) == row["video_sha256"],
                "action": sha(npy) == row["action_sha256"],
                "first_frame": sha(png) == row["first_frame_sha256"],
                "source_video": source[sid]["source_video_sha256"] == row["source_video_sha256"],
                "annotations": source[sid]["annotations_sha256"] == row["source_annotations_sha256"],
            },
        }
        checks = {
            "decoded39": vid["frame_count_decoded"] == 39,
            "png_is_decoded_first_frame": vid["first_frame_png_pixel_equal"],
            "fps24": abs(vid["fps"] - 24) < 1e-6,
            "pts_regular": vid["pts_strict_increasing"] and
                           max(abs(x - 1 / 24) for x in vid["pts_delta_min_max"]) < .001,
            "resolution832x480": vid["resolution_set"] == [[832, 480]] and png_size == [832, 480],
            "shape39x17": mat.shape == (39, 17),
            "action_source_exact": action_max_abs is not None and action_max_abs < 1e-5,
            "source_indices": rec["source_index_formula_matches"],
            "prompt_static": rec["prompt_matches_annotation_static"] and rec["prompt_matches_raw_static"],
            "sha": all(rec["manifest_sha_matches_files"].values()),
        }
        rec["checks"] = checks
        problems.extend({"clip_id": row["clip_id"], "check": k} for k, ok in checks.items() if not ok)
        inventory.append(rec)
        print(row["clip_id"], observed, "PASS" if all(checks.values()) else "FAIL", flush=True)

    (OUT / "inventory.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in inventory))
    dump("inventory.json", inventory)
    dump("source_manifest.json", {"task": "EXP-013/v1", "source_root": str(DATA / "raw"),
                                  "episode_count": len(source), "sources": source})
    by_id = {r["clip_id"]: r for r in inventory}
    selected = []
    for sid, group in sorted(episodes.items()):
        if group[0]["split"] != "train":
            continue
        target_a = [r for r in group if r["target"] == "A"]
        pick = min(target_a, key=lambda r: (r["src_start"], r["clip_id"])) if target_a else min(group, key=lambda r: r["clip_id"])
        obs = by_id[pick["clip_id"]]
        selected.append({
            "episode_id": sid, "split": "train", "clip_id": pick["clip_id"],
            "selection_rule": "minimum src_start among target=A, tie clip_id" if target_a else "no A; first clip_id",
            "src_start": pick["src_start"], "first_frame": obs["first_frame_path"],
            "first_frame_sha256": obs["first_frame_sha256"],
            "static_prompt": pick["prompt"], "static_prompt_sha256": obs["prompt_sha256"],
            "recorded_frame_key_combinations": obs["frame_key_combinations"],
            "recorded_key_counts": obs["key_counts"],
            "future_counterfactual_actions": ["A", "D"],
            "counterfactual_is_not_recorded_ground_truth": True,
        })
    source_spotchecks = {}
    for rec in inventory:
        source_spotchecks[rec["clip_id"]] = source_first_frame_shift_scores(
            Path(rec["source_video"]), rec["src_start"], Path(rec["first_frame_path"]))
    ambiguous = [rec for rec in inventory if not source_spotchecks[rec["clip_id"]]["source_frame_exact_best"]]
    rebuild_probes = [rebuild_clip_probe(rec) for rec in ambiguous]
    problems.extend({"clip_id": p["clip_id"], "check": "source_rebuild_sha"}
                    for p in rebuild_probes if not p["exact_byte_match"])
    dump("candidate_manifest.json", {"task": "EXP-013/v1", "candidate_count": len(selected), "selected": selected})
    train = {sid for sid, group in episodes.items() if group[0]["split"] == "train"}
    val = set(episodes) - train
    train_hash = {source[s]["source_video_sha256"] for s in train}
    val_hash = {source[s]["source_video_sha256"] for s in val}
    overlap = []
    for sid, group in episodes.items():
        for i, a in enumerate(group):
            for b in group[i + 1:]:
                common = sorted(set(a["source_frame_indices"]) & set(b["source_frame_indices"]))
                if common:
                    overlap.append({"episode_id": sid, "clip_a": a["clip_id"], "clip_b": b["clip_id"],
                                    "shared_source_frame_count": len(common), "first_last": [common[0], common[-1]]})
    audit = {
        "task": "EXP-013/v1", "clips": len(rows), "episodes": len(episodes),
        "train_episodes": sorted(train), "validation_episodes": sorted(val),
        "fixed_observed_validation_episodes": sorted(FIXED_VALIDATION),
        "validation_exactly_fixed": val == FIXED_VALIDATION,
        "episode_split_disjoint": train.isdisjoint(val),
        "source_video_hash_split_disjoint": train_hash.isdisjoint(val_hash),
        "source_annotations_hash_split_disjoint":
            {source[s]["annotations_sha256"] for s in train}.isdisjoint(
                {source[s]["annotations_sha256"] for s in val}),
        "within_episode_source_frame_overlap": overlap,
        "sources": source, "clip_check_failures": problems,
        "manifest_lineage": lineage,
        "source_video_first_frame_spotchecks": source_spotchecks,
        "ambiguous_pixel_shift_exact_ffmpeg_rebuilds": rebuild_probes,
        "action_column_names": A.ACTION_COLS,
        "old_encoded_anchor_protocol": old.get("anchor_protocol"),
        "old_encoded_suitable_for_V3_Single_I0": False,
        "existing_39RGB_12latent_covers_only_C1": True,
    }
    dump("source_split_action_audit.json", audit)
    w, h = 416, 275
    canvas = Image.new("RGB", (2 * w, 2 * h), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    except OSError:
        font = ImageFont.load_default()
    for i, item in enumerate(selected):
        im = Image.open(item["first_frame"]).convert("RGB")
        im.thumbnail((w, 240))
        x, y = (i % 2) * w, (i // 2) * h
        canvas.paste(im, (x, y))
        clip_suffix = "_".join(item["clip_id"].split("_")[-2:])
        draw.text((x + 4, y + 240), f"{item['episode_id'][:8]} | train | {clip_suffix}",
                  fill="black", font=font)
        observed = ", ".join(f"{k} {v}/39" for k, v in item["recorded_frame_key_combinations"].items())
        draw.text((x + 4, y + 256), f"Recorded keys: {observed}", fill="black", font=font)
    canvas.save(OUT / "candidate_contact_sheet.jpg", quality=90)
    print(json.dumps({"clips": len(rows), "episodes": len(episodes), "selected": len(selected),
                      "failures": len(problems), "train_val_hash_disjoint": train_hash.isdisjoint(val_hash)}))


if __name__ == "__main__":
    main()
