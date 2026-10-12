"""EXP-018 frozen single-row launcher; one fresh 124-frame run per process.

The legacy V0/V1/V2a engine is reused without changing model math. V2b/V3/V4
use the native-interval runner in this directory. This launcher owns the
cross-row budget, atomic status, cold wall, and 120-frame export.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = ROOT / "H3-World/outputs/EXP-018_unified_5s_benchmark"
ROWS = ("v0", "v1", "v2a", "v2b", "v3_30", "v3_8", "v4")
BENCH = ROOT / "H3-World/code/causal/benchmark.py"
PYTHON = ROOT / ".venvs/h3world/bin/python"
CKPT = ROOT / "H3-World/outputs/2026-10-06-09/visual_online_rgb_tail16_endpoint_ad2"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".partial")
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")
    os.replace(temp, path)


def sources() -> dict:
    paths = {
        "first_rgb": ROOT / "H3-World/examples/first_frame.png",
        "released_action_lora": ROOT / "models/H3-World/step-10000.safetensors",
        "benchmark": BENCH,
        "native_runner": HERE / "native_row.py",
        "v2a_visual": CKPT / "causal_adapter.pt",
        "v2a_action": CKPT / "action_adapter.pt",
        "dmd_result": ROOT / "H3-World/outputs/EXP-008_v3_dmd_train_v2/result.json",
        "v3_fixture": ROOT / "H3-World/outputs/2026-10-09-22/chunk_partition_cb/source_coarse/inputs/parking_A.pt",
    }
    dmd = json.loads(paths["dmd_result"].read_text())
    assert dmd["last_complete_cycle"] == 8
    receipt = next(x for x in dmd["checkpoint_paths"] if x["cycle"] == 8)
    for name in ("student_qkv.pt", "student_target.pt"):
        paths["v4_" + name] = ROOT / receipt["path"] / name
    return {name: {"path": str(path.relative_to(ROOT)), "sha256": sha(path)}
            for name, path in paths.items()}


def preflight() -> dict:
    assert len(ROWS) == 7 and BENCH.is_file() and PYTHON.is_file()
    import imageio_ffmpeg
    assert Path(imageio_ffmpeg.get_ffmpeg_exe()).is_file()
    import shutil
    assert shutil.disk_usage(ROOT).free >= 40 * 2**30
    manifest = {
        "task": "EXP-018/v1", "created_at": dt.datetime.now().astimezone().isoformat(),
        "source": sources(), "rows": list(ROWS), "seed": 13,
        "rgb_frames_generated": 124, "rgb_frames_display": 120,
        "resolution": [832, 480], "fps": 24, "flow_shift": 2.22,
        "gpu_hour_cap": 4, "forward_cap": 850, "row_wall_cap_seconds": 2700,
        "no_existing_endpoint_or_kv": True,
    }
    path = HERE / "SOURCE_MANIFEST.json"
    if path.exists():
        previous = json.loads(path.read_text())
        assert previous["source"] == manifest["source"], "frozen source changed"
        return previous
    atomic(path, manifest)
    return manifest


def command(row: str, row_dir: Path) -> list[str]:
    if row in {"v0", "v1", "v2a"}:
        cmd = [str(PYTHON), str(BENCH), "--modes", "baseline" if row == "v0" else "cached",
               "--num-frames", "124", "--seed", "13", "--action-preset", "A",
               "--flow-shift", "2.22", "--baseline-steps", "30", "--steps", "8",
               "--chunk-frames", "5", "--history-chunks", "5", "--cache-device", "cpu",
               "--out-dir", str(row_dir)]
        if row == "v1":
            cmd += ["--anchor-mode", "dynamic_last_frame_dual", "--action-prefix-mode", "own"]
        if row == "v2a":
            cmd += ["--anchor-mode", "dynamic_last_frame_rgb_dual", "--action-prefix-mode", "causal",
                    "--action-feedback", "--causal-adapter", str(CKPT / "causal_adapter.pt"),
                    "--causal-action-adapter", str(CKPT / "action_adapter.pt")]
        return cmd
    return [str(PYTHON), str(HERE / "native_row.py"), "--row", row, "--out", str(row_dir)]


def ffprobe(path: Path) -> dict:
    import av
    with av.open(str(path)) as container:
        stream=container.streams.video[0]
        frames=sum(1 for _ in container.decode(video=0))
        rate=stream.average_rate
        return {"codec_name":stream.codec_context.name,"width":stream.width,
            "height":stream.height,"r_frame_rate":f"{rate.numerator}/{rate.denominator}",
            "nb_frames":frames,"duration_seconds":float(stream.duration*stream.time_base) if stream.duration else None}


def run(row: str, gpu: int) -> None:
    import fcntl
    manifest = preflight()
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        index_file = HERE / "results_index.json"
        index = json.loads(index_file.read_text()) if index_file.exists() else {"task": "EXP-018/v1", "rows": {}}
        if row in index["rows"] or (OUT / row).exists():
            raise RuntimeError(f"row {row} has a prior attempt; no silent overwrite")
        gpu_rows = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.free,memory.total,uuid",
            "--format=csv,noheader,nounits"], text=True).strip().splitlines()
        free, total, uuid = [s.strip() for s in gpu_rows[gpu].split(",")][1:]
        if int(free) < 44000:
            raise RuntimeError(f"GPU {gpu} is occupied: free={free} MiB")
        prior_wall = sum(float(x.get("cold_total_seconds", 0)) for x in index["rows"].values())
        prior_wall += sum(float(x.get("cold_total_seconds", 0))
                          for attempts in index.get("attempts",{}).values() for x in attempts)
        prior_fw = sum(int(x.get("total_forward_count", 0)) for x in index["rows"].values())
        expected = {"v0":30,"v1":72,"v2a":72,"v2b":180,"v3_30":186,"v3_8":54,"v4":54}[row]
        if prior_wall + 2700 > 4*3600 or prior_fw + expected > 850:
            raise RuntimeError("remaining experiment budget insufficient for worst-case row")
        row_dir = OUT / row
        row_dir.mkdir()
        cmd = command(row, row_dir)
        atomic(row_dir / "launch.json", {"command": cmd, "gpu": gpu, "gpu_uuid": uuid,
            "initial_free_mib": int(free), "total_mib": int(total),
            "source_manifest_sha256": sha(HERE / "SOURCE_MANIFEST.json"),
            "started_at": dt.datetime.now().astimezone().isoformat(), "status": "running"})
        index["rows"][row] = {"status": "running", "gpu": gpu,
            "metrics": str((row_dir / "metrics.json").relative_to(ROOT))}
        atomic(index_file, index)
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=str(gpu), ABOT_VRAM_RESERVE_GIB="18",
                   HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", DIFFSYNTH_SKIP_DOWNLOAD="True",
                   TOKENIZERS_PARALLELISM="false", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
        start = time.perf_counter()
        with (row_dir / "run.log").open("w") as log:
            try:
                p = subprocess.run(cmd, cwd=ROOT / "H3-World" if row in {"v0","v1","v2a"} else ROOT,
                    env=env, stdout=log, stderr=subprocess.STDOUT, timeout=2700)
                rc = p.returncode
            except subprocess.TimeoutExpired:
                rc = 124
        cold = time.perf_counter() - start
        atomic(row_dir / "process_timing.json", {"child_process_exit_seconds": cold,
            "boundary": "launch of child through process exit; excludes 120-frame display export and metadata checks"})
        base = "baseline" if row == "v0" else "cached"
        raw = row_dir / f"{base}.mp4" if row in {"v0","v1","v2a"} else row_dir / "full_124.mp4"
        origin = row_dir / f"{base}.json" if row in {"v0","v1","v2a"} else row_dir / "native_metrics.json"
        metrics = json.loads(origin.read_text()) if origin.exists() else {}
        setup = json.loads((row_dir / "setup.json").read_text()) if (row_dir / "setup.json").exists() else {}
        summary = {"row": row, "status": "failed", "returncode": rc,
                   "cold_total_seconds": cold, "gpu": gpu, "gpu_uuid": uuid,
                   "initial_free_mib": int(free), "command": cmd,
                   "source_manifest_sha256_at_launch": json.loads((row_dir / "launch.json").read_text())["source_manifest_sha256"],
                   "source_manifest_sha256_at_finalize": sha(HERE / "SOURCE_MANIFEST.json"),
                   "raw_metrics": str(origin.relative_to(ROOT)) if origin.exists() else None}
        if raw.exists():
            launch_epoch=dt.datetime.fromisoformat(json.loads((row_dir / "launch.json").read_text())["started_at"]).timestamp()
            summary["process_to_full_file_seconds_mtime"] = raw.stat().st_mtime-launch_epoch
            summary["process_to_full_file_source"] = "MP4 mtime minus launch timestamp; approximate to filesystem timestamp resolution"
        if rc == 0 and raw.exists():
            info = ffprobe(raw)
            if (info["codec_name"], int(info["nb_frames"]), int(info["width"]), int(info["height"])) != ("h264",124,832,480):
                raise RuntimeError(f"unexpected 124-frame stream {info}")
            display = row_dir / "display_120.mp4"
            import imageio_ffmpeg
            ffmpeg_bin=imageio_ffmpeg.get_ffmpeg_exe()
            subprocess.run([ffmpeg_bin,"-v","error","-threads","4","-i",str(raw),"-frames:v","120",
                "-c:v","libx264","-preset","medium","-crf","18","-pix_fmt","yuv420p",
                "-movflags","+faststart",str(display)],check=True)
            dinfo = ffprobe(display)
            assert int(dinfo["nb_frames"]) == 120 and dinfo["r_frame_rate"] == "24/1"
            summary.update(status="complete_pending_judge", full_video=str(raw.relative_to(ROOT)),
                display_video=str(display.relative_to(ROOT)), full_sha256=sha(raw), display_sha256=sha(display),
                full_ffprobe=info, display_ffprobe=dinfo)
        summary["sampling_nfe_total"] = metrics.get("denoiser_forwards", metrics.get("sampling_nfe_total", 0))
        summary["commit_forward_total"] = metrics.get("commit_forwards", metrics.get("commit_forward_total", 0))
        summary["prefill_forward_total"] = metrics.get("prefill_forward_total", 0)
        summary["total_forward_count"] = sum(int(summary[k]) for k in ("sampling_nfe_total","commit_forward_total","prefill_forward_total"))
        summary["sampling_steps_per_chunk"] = 30 if row in {"v0","v2b","v3_30"} else 8
        summary["peak_allocated_gib"] = metrics.get("peak_allocated_gib", metrics.get("memory_entire_run_peak_MiB",0)/1024)
        summary["peak_reserved_gib"] = metrics.get("peak_reserved_gib", max(
            setup.get("memory",{}).get("reserved_peak_MiB",0),
            metrics.get("memory_after_setup",{}).get("reserved_peak_MiB",0))/1024)
        summary["cpu_kv_peak_gib"] = metrics.get("cpu_kv_peak_gib",metrics.get("kv_cache_peak_MiB",0)/1024)
        summary["ready_to_video_seconds"] = metrics.get("ready_to_video_seconds",metrics.get("wall_after_conditioning_seconds"))
        summary["load_seconds"] = metrics.get("load_seconds",setup.get("load_models_and_lora_seconds"))
        summary["conditioning_seconds"] = metrics.get("conditioning_seconds",setup.get("conditioning_seconds"))
        summary["sampling_loop_inclusive_seconds"] = metrics.get("sampling_loop_inclusive_seconds",metrics.get("sampling_seconds"))
        summary["decode_seconds"] = metrics.get("decode_seconds",metrics.get("vae_decode_seconds"))
        summary["encode_seconds"] = metrics.get("encode_seconds",metrics.get("encode_mp4_seconds"))
        summary["video_noise_sha256"] = metrics.get("video_noise_sha256",setup.get("input_fingerprints",{}).get("video_noise_sha256"))
        atomic(row_dir / "metrics.json", summary)
        index["rows"][row] = summary
        atomic(index_file,index)
        if summary["status"] != "complete_pending_judge":
            raise RuntimeError(f"row {row} failed, rc={rc}; see {row_dir/'run.log'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--row", choices=ROWS)
    ap.add_argument("--gpu", type=int, default=5)
    a = ap.parse_args()
    if a.preflight:
        print(json.dumps(preflight(), ensure_ascii=False, indent=2))
    else:
        if not a.row: ap.error("--row required")
        run(a.row,a.gpu)
