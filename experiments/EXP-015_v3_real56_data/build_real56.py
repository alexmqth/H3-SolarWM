"""CPU-only EXP-015: extend four fixed ABot train windows to real 56 RGB frames."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import av
import imageio_ffmpeg
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DATA = ROOT / "H3-World/data/abot_bridge"
OLD = ROOT / "submission/experiments/EXP-013_v3_multiscene_data_plan"
ART = HERE / "artifacts"
CFG = json.loads((HERE / "config.json").read_text())
sys.path.insert(0, str(ROOT / "H3-World/code/abot"))
sys.path.insert(0, str(ROOT / "H3-World/code/causal"))
import abot_action as A  # noqa: E402
import action_script as S  # noqa: E402
from real_transition_data import bounded_keys9  # noqa: E402


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: object) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def check_budget(began: float, initial_free: int) -> None:
    if time.monotonic() - began > CFG["max_wall_seconds"]:
        raise TimeoutError("20-minute EXP-015 wall budget")
    if datetime.now(timezone.utc) >= datetime.fromisoformat(CFG["stop_new_work_hkt"]):
        raise TimeoutError("08:50 HKT stop-new-work cutoff")
    free = shutil.disk_usage(HERE).free
    if free / 2**30 < CFG["minimum_free_disk_gib"]:
        raise OSError("disk free floor")
    if initial_free - free > CFG["max_new_bytes"]:
        raise OSError("EXP-015 new data size cap")


def rows() -> list[dict]:
    candidate_path = OLD / "candidate_manifest.json"
    assert sha(candidate_path) == CFG["candidate_manifest_sha256"]
    selected = json.loads(candidate_path.read_text())["selected"]
    assert [r["episode_id"] for r in selected] == CFG["episodes_in_order"]
    clips = {r["clip_id"]: r for r in map(json.loads, (DATA / "clips.jsonl").read_text().splitlines())}
    source = json.loads((OLD / "source_manifest.json").read_text())["sources"]
    out = []
    for item in selected:
        assert item["split"] == "train"
        clip = clips[item["clip_id"]]
        assert clip["split"] == "train" and clip["src_start"] == item["src_start"]
        assert clip["sample_id"] == item["episode_id"]
        src = source[item["episode_id"]]
        assert src["split"] == ["train"] and src["source_fps"] == CFG["source_fps"]
        video, ann = Path(src["source_video"]), Path(src["annotations"])
        assert video.is_file() and ann.is_file()
        assert sha(video) == src["source_video_sha256"] == clip["source_video_sha256"]
        assert sha(ann) == src["annotations_sha256"] == clip["source_annotations_sha256"]
        assert sha(Path(clip["first_frame"])) == item["first_frame_sha256"]
        indices = [item["src_start"] + o for o in A.window_offsets(CFG["rgb_frames"])]
        assert len(indices) == 56 and len(set(indices)) == 56 and indices == sorted(indices)
        assert indices[-1] < src["source_frames"]
        assert indices[:39] == clip["source_frame_indices"]
        assert item["static_prompt"] == clip["prompt"]
        assert hashlib.sha256(clip["prompt"].encode()).hexdigest() == item["static_prompt_sha256"]
        out.append({"candidate": item, "clip": clip, "source": src,
                    "video": video, "annotations": ann, "indices": indices})
    return out


def preflight() -> list[dict]:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("set CUDA_VISIBLE_DEVICES='' for CPU-only EXP-015")
    entries = rows()
    result = []
    for item in entries:
        src = item["source"]
        with av.open(str(item["video"])) as container:
            stream = container.streams.video[0]
            stream.thread_count = 1
            assert stream.average_rate == CFG["source_fps"]
            assert stream.frames >= item["indices"][-1] + 1
        result.append({"episode_id": item["candidate"]["episode_id"],
                       "src_start": item["candidate"]["src_start"],
                       "source_first_last": [item["indices"][0], item["indices"][-1]],
                       "source_video_sha256": src["source_video_sha256"],
                       "annotations_sha256": src["annotations_sha256"],
                       "split": "train", "check": "PASS_CPU"})
    return result


def video_probe(path: Path) -> tuple[dict, list[np.ndarray]]:
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_count = 1
        assert stream.average_rate == CFG["output_fps"]
        frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
        assert len(frames) == CFG["rgb_frames"]
        assert all(x.shape == (480, 832, 3) for x in frames)
    # Reopen to check PTS without relying on a generator after decoding.
    with av.open(str(path)) as container:
        container.streams.video[0].thread_count = 1
        pts = [f.pts for f in container.decode(video=0)]
    assert len(pts) == 56 and len(set(pts)) == 56
    assert all(a < b for a, b in zip(pts, pts[1:]))
    return {"frames": len(frames), "resolution": [832, 480], "fps": 24,
            "unique_pts": len(set(pts)), "first_last_pts": [pts[0], pts[-1]]}, frames


def combos(mat: np.ndarray) -> dict[str, int]:
    labels = []
    for row in mat[:, :A.NUM_KEYS]:
        labels.append("+".join(k for i, k in enumerate(A.KEY_COLS) if row[i] > .5) or "none")
    return dict(sorted(Counter(labels).items()))


def build_one(item: dict, ffmpeg: str, began: float, initial_free: int) -> dict:
    check_budget(began, initial_free)
    selected, clip = item["candidate"], item["clip"]
    scene = selected["episode_id"][:8]
    dest = ART / scene
    expected = ("real56.mp4", "I0.png", "raw56.npy", "pooled17.npy",
                "keys9_17.npy", "action_script17.json")
    completed = dest.is_dir() and all((dest / name).is_file() for name in expected)
    if dest.exists() and not completed:
        # Only a zero-byte partial MP4 from this interrupted run is recoverable.
        remaining = list(dest.iterdir())
        if not (len(remaining) == 1 and remaining[0].name == "real56.partial.mp4"
                and remaining[0].stat().st_size == 0):
            raise FileExistsError(f"incomplete output requires manual audit: {dest}")
        remaining[0].unlink()
    dest.mkdir(parents=True, exist_ok=True)
    start, span = selected["src_start"], A.window_span(56)
    vf = (f"select='between(n\\,{start}\\,{start+span-1})"
          f"*not(eq(mod(n-{start}\\,5)\\,4))',"
          "setpts=N/24/TB,scale=-2:480,crop=832:480")
    video = dest / "real56.mp4"
    temp = dest / "real56.partial.mp4"
    cmd = [ffmpeg, "-y", "-v", "error", "-threads", "1", "-filter_threads", "1",
           "-i", str(item["video"]), "-vf", vf, "-r", "24", "-frames:v", "56",
           "-c:v", "libx264", "-threads:v", "1", "-crf", "14", "-preset", "veryfast",
           "-pix_fmt", "yuv420p", "-an", "-movflags", "+faststart", str(temp)]
    original_cmd = [ffmpeg, "-y", "-v", "error", "-threads", "2",
                    "-i", str(item["video"]), "-vf", vf, "-r", "24", "-frames:v", "56",
                    "-c:v", "libx264", "-threads:v", "2", "-crf", "14", "-preset", "veryfast",
                    "-pix_fmt", "yuv420p", "-an", "-movflags", "+faststart", str(temp)]
    if not completed:
        encoded = subprocess.run(cmd, capture_output=True, text=True)
        if encoded.returncode:
            raise RuntimeError(f"FFmpeg failed ({encoded.returncode}): {encoded.stderr.strip()}")
        temp.rename(video)
    probe, frames = video_probe(video)
    i0 = dest / "I0.png"
    if not completed:
        Image.fromarray(frames[0]).save(i0)
    assert np.array_equal(np.asarray(Image.open(i0).convert("RGB")), frames[0])
    old_i0 = np.asarray(Image.open(clip["first_frame"]).convert("RGB"))
    old_video = Path(clip["video"])
    with av.open(str(old_video)) as container:
        container.streams.video[0].thread_count = 1
        old_frames = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
    assert len(old_frames) == 39
    prefix_mad = [float(np.abs(frames[i].astype(np.float32)-old_frames[i].astype(np.float32)).mean())
                  for i in range(39)]
    i0_mad = float(np.abs(frames[0].astype(np.float32)-old_i0.astype(np.float32)).mean())

    ep = A.read_episode(str(item["annotations"]))
    assert ep["total_frames"] >= item["indices"][-1]+1
    assert ep["control_scheme"] == "WASD_QE_locomotion_IJKL_rotation"
    assert ep["caption"]["scene_static"].strip() == selected["static_prompt"]
    scale = A.episode_translation_scale(ep)
    raw = A.window_action_matrix(ep, start, 56, scale)
    assert raw.shape == (56, 17) and np.isfinite(raw).all()
    assert np.array_equal(raw[:, :A.NUM_KEYS], ep["keys"][item["indices"]])
    old_raw = np.load(clip["action"], allow_pickle=False)
    assert old_raw.shape == (39, 17)
    raw_prefix_max_error = float(np.max(np.abs(raw[:39]-old_raw)))
    assert raw_prefix_max_error < 1e-5
    spans = A.frame_spans(17)
    assert spans[11][1] == 39 and spans[12][0] == 39 and spans[-1][1] == 56
    pooled = A.bin_to_latent(raw, 17)
    old_pooled = A.bin_to_latent(old_raw, 12)
    assert np.array_equal(pooled[:12], old_pooled)
    widths = [e-s for s,e in spans]
    manual = np.zeros_like(pooled)
    for k,(lo,hi) in enumerate(spans):
        manual[k,:A.NUM_KEYS]=raw[lo:hi,:A.NUM_KEYS].max(axis=0)
        manual[k,A.NUM_KEYS:A.NUM_KEYS+3]=np.clip(raw[lo:hi,A.NUM_KEYS:A.NUM_KEYS+3].sum(axis=0)/A.ROT_SCALE,-A.ROT_CLIP,A.ROT_CLIP)
        manual[k,A.NUM_KEYS+3:]=np.clip(raw[lo:hi,A.NUM_KEYS+3:].sum(axis=0)/A.TRA_SCALE,-A.TRA_CLIP,A.TRA_CLIP)
    assert np.allclose(pooled,manual,rtol=0,atol=1e-6)
    keys9 = bounded_keys9(pooled, stops=(12,17))
    script = S.annotate_from_keys9(keys9)
    full_script_keys9 = S.keys9(pooled)
    altered = raw.copy(); altered[39:] = 0
    altered_pooled = A.bin_to_latent(altered,17)
    assert np.array_equal(altered_pooled[:12],pooled[:12])
    assert np.array_equal(bounded_keys9(altered_pooled,stops=(12,17))[:12],keys9[:12])
    assert S.annotate_from_keys9(bounded_keys9(altered_pooled,stops=(12,17)))[:12] == script[:12]
    conflicts=[]
    for k,(lo,hi) in enumerate(spans):
        on={name:bool(pooled[k,A.KEY_COLS.index(name)]>0) for name in A.KEY_COLS}
        for a,b in (("W","S"),("A","D"),("J","L"),("I","K")):
            if on[a] and on[b]: conflicts.append({"span":k,"pair":[a,b],"RGB":[lo,hi]})
    unsupported={name:int(np.count_nonzero(raw[:,A.KEY_COLS.index(name)])) for name in ("Q","E","Space")}
    for name,array in (("raw56.npy",raw),("pooled17.npy",pooled),("keys9_17.npy",keys9)):
        if completed:
            assert np.array_equal(np.load(dest/name,allow_pickle=False),array)
        else:
            np.save(dest/name,array)
    if completed:
        assert json.loads((dest/"action_script17.json").read_text()) == script
    else:
        write_json(dest/"action_script17.json",script)
    output_files={name:{"path":str(dest/name),"sha256":sha(dest/name),"bytes":(dest/name).stat().st_size}
                  for name in ("real56.mp4","I0.png","raw56.npy","pooled17.npy","keys9_17.npy","action_script17.json")}
    return {"episode_id":selected["episode_id"],"scene":scene,"split":"train",
            "src_start":start,"source_frame_indices":item["indices"],
            "source_video_sha256":item["source"]["source_video_sha256"],
            "annotations_sha256":item["source"]["annotations_sha256"],
            "old_clip_id":clip["clip_id"],"old39_video_sha256":clip["video_sha256"],
            "old39_I0_sha256":selected["first_frame_sha256"],
            "old39_action_sha256":clip["action_sha256"],
            "static_prompt_sha256":selected["static_prompt_sha256"],
            "ffmpeg_command":original_cmd if completed else cmd,
            "ffmpeg_filter":vf,"video_probe":probe,
            "I0_new_vs_old_MAD":i0_mad,"I0_new_sha_equals_old":sha(i0)==selected["first_frame_sha256"],
            "new_vs_old39_RGB_MAD_mean":float(np.mean(prefix_mad)),
            "new_vs_old39_RGB_MAD_max":float(np.max(prefix_mad)),
            "new_vs_old39_RGB_MAD_per_frame":prefix_mad,
            "raw_action_shape":[56,17],"raw_action_dtype":str(raw.dtype),
            "raw_prefix39_max_abs_error":raw_prefix_max_error,
            "episode_translation_scale":scale,
            "frame_key_combinations":combos(raw),
            "key_counts":{k:int(np.count_nonzero(raw[:,i])) for i,k in enumerate(A.KEY_COLS)},
            "unsupported_Q_E_Space_counts":unsupported,
            "continuous_min_max":[[float(raw[:,i].min()),float(raw[:,i].max())] for i in range(A.NUM_KEYS,17)],
            "native_frame_spans":spans,"native_span_widths":widths,
            "first12_span_end_RGB":39,"C2_spans":[list(x) for x in spans[12:]],
            "manual_pool_max_abs_error":float(np.max(np.abs(pooled-manual))),
            "pooled_prefix12_equal_old39":True,
            "keys9_columns":S.KEYS9,
            "keys9_F_from_yaw_not_Space":True,
            "keys9_bounded_vs_global_different_spans":np.flatnonzero(np.any(keys9!=full_script_keys9,axis=1)).tolist(),
            "opposing_key_conflicts_purified_in_text":conflicts,
            "short_span_rate_neighbor_map":{str(k):(k+1 if k+1<(12 if k<12 else 17) else k-1) for k,w in enumerate(widths) if w<4},
            "C1_future_raw_mutation_preserves_pooled_keys_and_script":True,
            "script_C1":script[:12],"script_C2":script[12:],
            "output_files":output_files}


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight",action="store_true")
    parser.add_argument("--build",action="store_true")
    args=parser.parse_args()
    if args.preflight==args.build: parser.error("choose exactly one mode")
    began=time.monotonic(); initial_free=shutil.disk_usage(HERE).free
    checks=preflight()
    if args.preflight:
        write_json(HERE/"PREFLIGHT.json",{"task":CFG["task"],"scenes":checks,"GPU_calls":0})
        print("preflight PASS",len(checks),flush=True)
        return
    if not (HERE/"PREFLIGHT.json").is_file(): raise RuntimeError("run preflight first")
    ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
    ffver=subprocess.check_output([ffmpeg,"-version"],text=True).splitlines()[0]
    output=[]
    for item in rows():
        output.append(build_one(item,ffmpeg,began,initial_free))
        check_budget(began,initial_free)
        print(item["candidate"]["episode_id"],"56f PASS",flush=True)
    write_json(HERE/"OUTPUT_MANIFEST.json",{"task":CFG["task"],"scene_count":4,
        "rgb_frames_total":224,"GPU_calls":0,"model_encodes":0,"training_updates":0,
        "ffmpeg_version":ffver,"code_sha256":sha(Path(__file__)),
        "config_sha256":sha(HERE/"config.json"),
        "candidate_manifest_sha256":CFG["candidate_manifest_sha256"],
        "wall_seconds":time.monotonic()-began,
        "new_bytes":initial_free-shutil.disk_usage(HERE).free,
        "scenes":output})
    print("all four real56 complete",flush=True)


if __name__=="__main__":
    main()
