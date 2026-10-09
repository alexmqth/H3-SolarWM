"""EXP-001: resume the four frozen V2b paths, one 5-latent chunk per run.

No training, adapter change, reset, hidden KV, or automatic quality acceptance.
Every invocation writes one checkpoint and one 17-frame extension for review.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CONFIG = json.loads((HERE / "config_v3.yaml").read_text())
FROZEN = Path(CONFIG["frozen_source"]).resolve()
OUTPUT = Path(CONFIG["output_root"]).resolve()
STATE_FILES = Path(CONFIG["initial_states"]).resolve()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    tmp = Path(str(path) + ".partial")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(tmp, path)


def tensor_hash(x):
    import torch
    x = x.detach().cpu().contiguous()
    return hashlib.sha256(str((tuple(x.shape), str(x.dtype))).encode() +
                          x.view(torch.uint8).numpy().tobytes()).hexdigest()


def now():
    return time.time()


def gpu_limit_at(epoch):
    cutoff = datetime.fromisoformat(CONFIG["gpu_cutoff_hkt"]).timestamp()
    return (CONFIG["max_concurrent_GPUs_before_09"] if epoch < cutoff
            else CONFIG["max_concurrent_GPUs_after_09"])


def budget_usage_at(ledger, epoch):
    active = [r for r in ledger["runs"] if r["status"] == "running"]
    return ledger["gpu_seconds"] + sum(max(0, epoch - r["start"]) for r in active)


def validate_v3_reservation(ledger, path_id, gpu, epoch, diagnostics, sampling):
    active = [r for r in ledger["runs"] if r["status"] == "running"]
    assert path_id in CONFIG["active_paths"]
    assert diagnostics == 0 and sampling == 30, "v3 permits one undiagnosed chunk"
    assert epoch - ledger["first_gpu_start"] < CONFIG["max_elapsed_hours"] * 3600
    assert budget_usage_at(ledger, epoch) + 600 < CONFIG["max_GPU_hours"] * 3600
    assert ledger["sampling_reserved"] + sampling <= CONFIG["max_sampling_forwards"]
    assert ledger["sampling_reserved"] - 120 + sampling <= CONFIG["max_new_sampling_forwards"]
    assert ledger["diagnostic_reserved"] == 12, "v1 diagnostics must not be reset"
    assert (ledger["sampling_reserved"] + ledger["diagnostic_reserved"] +
            ledger["original_reserved"] + sampling <= CONFIG["max_total_forwards_v3"])
    assert len(active) < gpu_limit_at(epoch)
    assert gpu not in [r["gpu"] for r in active]
    assert path_id not in [r["path"] for r in active]


def verify_sources():
    recorded = json.loads((HERE / "source_manifest_v3.json").read_text())
    for name, expected in recorded.get("worker_files", {}).items():
        assert sha(HERE / name) == expected, ("worker_file_changed", name)
    for name, expected in recorded["source_files"].items():
        assert sha(ROOT / name) == expected, ("source_changed", name)
    old_protocol = json.loads((FROZEN / "protocol.json").read_text())
    for name, expected in old_protocol["inputs_sha256"].items():
        assert sha(ROOT / name) == expected, ("input_changed", name)
    for name, expected in json.loads((FROZEN / "source_coarse/runtime_manifest.json").read_text()).items():
        assert sha(FROZEN / "runtime" / name) == expected, ("runtime_changed", name)
    assert sha(HERE / "interval_forward.py") == recorded["source_files"][
        str((FROZEN / "interval_forward.py").relative_to(ROOT))]
    source_states = json.loads((ROOT / "submission/experiments/11_causal_12_then5_selfhistory/manifest.json").read_text())
    for item in source_states["state_files"]:
        assert sha(STATE_FILES / Path(item["path"]).name) == item["sha256"]
    assert sha(FROZEN / "runtime/checkpoints/H3-World/step-10000.safetensors") == recorded["released_lora_sha256"]


@contextmanager
def budget_reservation(path_id, gpu, diagnostics, sampling):
    """Reserve all planned forwards before loading; even failed calls count."""
    lock = OUTPUT / "budget.lock"
    with lock.open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        ledger_path = OUTPUT / "budget.json"
        ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else dict(
            first_gpu_start=None, gpu_seconds=0, sampling_reserved=0,
            diagnostic_reserved=0, original_reserved=0, runs=[])
        t = now()
        if ledger["first_gpu_start"] is None:
            ledger["first_gpu_start"] = t
        validate_v3_reservation(ledger, path_id, gpu, t, diagnostics, sampling)
        active = [r for r in ledger["runs"] if r["status"] == "running"]
        # A dead worker is a failure, not an invitation to silently restart.
        for r in active:
            assert Path(f'/proc/{r["pid"]}').exists(), ("prior_worker_disappeared", r)
        run = dict(id=f"{path_id}_{int(t*1000)}", path=path_id, pid=os.getpid(), gpu=gpu,
                   start=t, status="running", diagnostics_reserved=diagnostics,
                   sampling_reserved=sampling, actual_diagnostic_calls=0,
                   actual_sampling_calls=0)
        ledger["runs"].append(run)
        ledger["sampling_reserved"] += sampling
        ledger["diagnostic_reserved"] += diagnostics
        atomic_json(ledger_path, ledger)
        fcntl.flock(handle, fcntl.LOCK_UN)
    try:
        yield run
    finally:
        with lock.open("a+") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            ledger = json.loads(ledger_path.read_text())
            target = next(r for r in ledger["runs"] if r["id"] == run["id"])
            target.update(status=run.get("final_status", "failed"), end=now(),
                          actual_diagnostic_calls=run["actual_diagnostic_calls"],
                          actual_sampling_calls=run["actual_sampling_calls"])
            target["gpu_seconds"] = target["end"] - target["start"]
            ledger["gpu_seconds"] += target["gpu_seconds"]
            atomic_json(ledger_path, ledger)
            fcntl.flock(handle, fcntl.LOCK_UN)


def frame_array(frames):
    import numpy as np
    return np.stack([np.asarray(im.convert("RGB"), dtype=np.uint8) for im in frames])


def pil_frames(arr):
    from PIL import Image
    return [Image.fromarray(x, mode="RGB") for x in arr]


def contact_sheet(frames, first, path):
    from PIL import Image, ImageDraw
    import math
    sheet = Image.new("RGB", (5 * 416, math.ceil(len(frames) / 5) * 260), "#161c25")
    draw = ImageDraw.Draw(sheet)
    for j, f in enumerate(frames):
        x, y = (j % 5) * 416, (j // 5) * 260
        sheet.paste(f.resize((416, 240)), (x, y + 20))
        draw.text((x + 6, y + 3), f"RGB {first+j}", fill="white")
    sheet.save(path)


def load_and_check_data():
    import torch
    data = {a: torch.load(FROZEN / f"source_coarse/inputs/parking_{a}.pt",
                          map_location="cpu", weights_only=True) for a in "AD"}
    for key in ("initial_noise", "audio_noise", "anchor"):
        assert torch.equal(data["A"][key], data["D"][key]), key
    for key in ("img_position_ids", "action_text_rows", "text_pos"):
        assert torch.equal(data["A"]["packed"][key], data["D"]["packed"][key]), key
    return data


def prior_history(path_id, start):
    import torch
    first, later = path_id
    files = [STATE_FILES / f"first12_{first}.pt",
             STATE_FILES / f"history{first}_current{later}_next5.pt"]
    for lo, hi in ((17, 22), (22, 27), (27, 32)):
        if hi <= start:
            files.append(OUTPUT / path_id / f"chunk_{lo}_{hi}.pt")
    parts = [torch.load(p, map_location="cpu", weights_only=True).float() for p in files]
    history = torch.cat(parts, dim=2)
    assert history.shape == (1, 24, start, 30, 52), (files, history.shape)
    return history, [{"path": str(p), "sha256": sha(p), "tensor_sha256": tensor_hash(x)}
                     for p, x in zip(files, parts)]


def decode(pipe, latents):
    import torch
    with torch.no_grad():
        rgb = pipe.video_vae.decode_video(latents, dtype=pipe.torch_dtype,
                                           process_image=False, tiled=True,
                                           tile_size=256, tile_overlap=64)
        frames = pipe.vae_output_to_video(rgb, min_value=0, max_value=1)
    del rgb
    return frame_array(frames)


def run(args, run_record):
    import av
    import numpy as np
    import torch
    from PIL import Image
    import infer as abot
    from causal.h3_precision import configure_precision
    from causal.local_topology import visible_inputs
    from causal.anyflow_sampling import configure_video_schedule
    from benchmark import write_video
    from evaluate_action_control import evaluate as flow_metrics
    from interval_forward import interval_forward
    from rollout_contract import PATHS, rgb_sha, rgb_stop, stitch_immutable, prompt_for_path

    torch.set_num_threads(4)
    torch.manual_seed(13)
    path_dir = OUTPUT / args.path
    path_dir.mkdir(exist_ok=True)
    state_file = path_dir / "state.json"
    state = json.loads(state_file.read_text()) if state_file.exists() else dict(
        path=args.path, next_start=17, results=[], published_rgb_sha256=None,
        source="own first12 + own second5 endpoints; no GT/teacher reset")
    start = state["next_start"]
    assert start in (22, 27, 32), "v3 only continues preserved 73f states"
    stop = start + 5
    assert stop <= (27 if args.stage == "gate90" else 37)
    if args.stage == "gate90":
        assert start == 22, "gate90 only generates the fourth chunk"
    if args.stage == "extend124":
        gate = json.loads((OUTPUT / "gate90.json").read_text())
        assert gate["plan_version"] == 3
        assert set(gate["active_paths"]) == set(CONFIG["active_paths"])
        assert gate["paths"][args.path] == "CONTINUE"
        if start == 27:
            assert gate["state_sha256"][args.path] == sha(OUTPUT / args.path / "state.json")
    data = load_and_check_data()
    history_cpu, history_sources = prior_history(args.path, start)
    first, current_action = PATHS[args.path]
    initial = data["A"]["initial_noise"]
    assert initial.shape[2] == 37
    initial_hash = tensor_hash(initial)
    history_hash = tensor_hash(history_cpu)
    old_state = json.loads((FROZEN / f"C_{first}/evaluation.json").read_text())
    expected_old = next(x for x in old_state["records"]
                        if x["mode"] == "sigma_noised" and x["action"] == current_action)
    assert history_sources[1]["tensor_sha256"] == expected_old["endpoint_sha256"]
    frozen_eval_hash = sha(FROZEN / f"C_{first}/evaluation.json")
    old_published = FROZEN / f"C_{first}/sigma_noised_{current_action}_rollout.mp4"
    assert old_published.is_file()

    row = dict(status="loading", path=args.path, stage=args.stage, gpu=args.gpu,
               interval=[start, stop], rgb_interval=[rgb_stop(start), rgb_stop(stop)],
               first_action=first, current_action=current_action, seed=13,
               source_sha256=sha(__file__), interval_source_sha256=sha(HERE / "interval_forward.py"),
               config_sha256=sha(HERE / "config_v3.yaml"), plan_version=3,
               history_sources=history_sources,
               history_tensor_sha256=history_hash, initial_noise_sha256=initial_hash,
               old_evaluation_sha256=frozen_eval_hash,
               old_published_video_sha256=sha(old_published),
               history_latent_MiB=history_cpu.numel()*history_cpu.element_size()/2**20,
               cpu_hidden_KV_MiB=0, optimizer_updates=0,
               diagnostic_forwards=0, sampling_forwards=0, vae_decodes=0)
    result_file = path_dir / f"chunk_{start}_{stop}.json"
    atomic_json(result_file, row)
    began = time.perf_counter()
    try:
        pipe = abot.load_pipeline("cuda:0")
        released = FROZEN / "runtime/checkpoints/H3-World/step-10000.safetensors"
        pipe.load_lora(pipe.dit, state_dict=abot.load_checkpoint_lora(released), hotload=True)
        model = pipe.dit.eval().requires_grad_(False)
        row["released_LoRA_sha256"] = sha(released)
        row["precision"] = configure_precision(model, "h3_fp32",
            native_transformer_dir=FROZEN / "runtime/DiffSynth-Studio-h3-v2/models/MiniMax/MiniMax-H3/FL2VA/transformer")
        pipe.load_models_to_device(["dit"])
        torch.cuda.synchronize()
        row["load_seconds"] = time.perf_counter() - began
        versions = [(p, p._version) for p in model.parameters()]
        def move(x):
            if torch.is_tensor(x): return x.to("cuda:0")
            if isinstance(x, dict): return {k: move(v) for k, v in x.items()}
            return x
        noise = initial.to("cuda:0", torch.float32)
        history = history_cpu.to("cuda:0", torch.float32)
        audio = data[first]["audio_noise"].to("cuda:0")
        anchor = data[first]["anchor"].to("cuda:0")
        packed = move(data[first]["packed"])
        prompt = prompt_for_path(data, args.path, stop).to("cuda:0")
        layout, prompt = visible_inputs(packed, prompt, stop, 390)
        assert len(layout["action_text_rows"]) == stop
        row["prompt_sha256"] = tensor_hash(prompt)
        row["position_sha256"] = tensor_hash(layout["img_position_ids"])
        row["anchor_sha256"] = tensor_hash(anchor)
        row["audio_sha256"] = tensor_hash(audio)
        sigmas = configure_video_schedule(pipe.scheduler, steps=30, grid="native", flow_shift=2.22)
        row["sigmas"] = sigmas
        assert sigmas == old_state["sigmas"]
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            if start == 17:
                # The first saved solver state provides a frozen numerical reference.
                probe = torch.load(FROZEN / "source_coarse/coarse_A/window0_solver_states.pt",
                                   map_location="cpu", weights_only=True)[1]
                base_prompt = data["A"]["pairs"][0]["prompts"]["A"].to("cuda:0")
                first_layout, first_prompt = visible_inputs(move(data["A"]["packed"]), base_prompt, 12, 390)
                empty = noise[:, :, :0]
                reference = probe["velocity_A"].to("cuda:0")
                observed = interval_forward(model, probe["state"].to("cuda:0"), start=0,
                    history=empty, history_noise=empty, mode="clean",
                    full_packed=first_layout, prompt=first_prompt,
                    anchor=anchor, audio=audio, sigma=probe["sigma"])
                run_record["actual_diagnostic_calls"] += 1
                row["diagnostic_forwards"] += 1
                diff = (observed - reference).float()
                rr = float(diff.square().mean().sqrt() /
                           reference.float().square().mean().sqrt().clamp_min(1e-12))
                row["first12_replay_relative_RMS"] = rr
                assert rr <= 1e-6, ("first12 replay mismatch", rr)
                # Original second-window API and new explicit interval must agree.
                from history_conditioning import history_forward
                prev_text = prompt_for_path(data, args.path, 17).to("cuda:0")
                prev_layout, prev_text = visible_inputs(packed, prev_text, 17, 390)
                old_field = history_forward(model, noise[:, :, 12:17].clone(),
                    history=history[:, :, :12], history_noise=noise[:, :, :12], mode="N",
                    full_packed=prev_layout, prompt=prev_text, anchor=anchor,
                    audio=audio, sigma=sigmas[0], index=1, chunk_frames=12,
                    history_chunks=5)
                run_record["actual_diagnostic_calls"] += 1
                row["diagnostic_forwards"] += 1
                new_field = interval_forward(model, noise[:, :, 12:17].clone(),
                    start=12, history=history[:, :, :12], history_noise=noise[:, :, :12],
                    mode="sigma_noised", full_packed=prev_layout, prompt=prev_text,
                    anchor=anchor, audio=audio, sigma=sigmas[0])
                run_record["actual_diagnostic_calls"] += 1
                row["diagnostic_forwards"] += 1
                diff = (old_field - new_field).float()
                rr = float(diff.square().mean().sqrt() /
                           old_field.float().square().mean().sqrt().clamp_min(1e-12))
                row["second5_legacy_API_relative_RMS"] = rr
                assert rr <= 1e-6, ("second5 protocol mismatch", rr)
                del old_field, new_field, observed, reference, diff
            row["status"] = "sampling"
            atomic_json(result_file, row)
            current = noise[:, :, start:stop].clone()
            current_noise_hash = tensor_hash(current)
            history_before = tensor_hash(history)
            torch.cuda.synchronize()
            tick = time.perf_counter()
            for i, t in enumerate(pipe.scheduler.timesteps):
                ledger = json.loads((OUTPUT / "budget.json").read_text())
                if now() - ledger["first_gpu_start"] > CONFIG["max_elapsed_hours"]*3600:
                    raise TimeoutError("EXP-001 8-hour deadline")
                if budget_usage_at(ledger, now()) >= CONFIG["max_GPU_hours"]*3600:
                    raise TimeoutError("EXP-001 8 GPU-hour budget")
                velocity = interval_forward(model, current, start=start,
                    history=history, history_noise=noise[:, :, :start],
                    mode="sigma_noised", full_packed=layout, prompt=prompt,
                    anchor=anchor, audio=audio, sigma=float(t)/1000)
                run_record["actual_sampling_calls"] += 1
                row["sampling_forwards"] += 1
                current = pipe.scheduler.step(velocity, t, current)
                if not torch.isfinite(current).all(): raise FloatingPointError("nonfinite current latent")
                if torch.cuda.max_memory_allocated()/2**30 > CONFIG["max_allocated_GiB"]:
                    raise MemoryError("allocated VRAM over 44GiB")
                if (i+1) % 5 == 0:
                    row["step_completed"] = i+1
                    atomic_json(result_file, row)
                    print(f"[{args.path}] [{start},{stop}) {i+1}/30", flush=True)
            torch.cuda.synchronize()
            row["sampling_seconds"] = time.perf_counter() - tick
            assert tensor_hash(history) == history_before
            assert tensor_hash(noise) == initial_hash or tensor_hash(noise) == tensor_hash(initial.float())
            assert tensor_hash(noise[:, :, start:stop]) == current_noise_hash
            combined = torch.cat((history, current), dim=2)
            assert combined.shape[2] == stop
            endpoint = path_dir / f"chunk_{start}_{stop}.pt"
            temporary = Path(str(endpoint) + ".partial")
            torch.save(current.detach().cpu(), temporary)
            os.replace(temporary, endpoint)
            row["endpoint_file_sha256"] = sha(endpoint)
            row["endpoint_tensor_sha256"] = tensor_hash(current)
            row["history_and_noise_unchanged"] = True
            del velocity
            pipe.load_models_to_device(["video_vae"])
            torch.cuda.synchronize()
            decode_tick = time.perf_counter()
            if start == 17:
                first12 = decode(pipe, history[:, :, :12]); row["vae_decodes"] += 1
                full17 = decode(pipe, history); row["vae_decodes"] += 1
                assert len(first12) == 39 and len(full17) == 56
                published = np.concatenate((first12, full17[39:56]), axis=0)
                # MP4 has lossy H.264 compression; report differences separately.
                with av.open(str(old_published)) as video:
                    compressed = np.stack([frame.to_ndarray(format="rgb24")
                                           for frame in video.decode(video=0)])
                assert compressed.shape == published.shape
                row["reconstructed56_vs_oldMP4_MAD"] = float(np.abs(
                    published.astype(np.float32)-compressed.astype(np.float32)).mean())
                assert row["reconstructed56_vs_oldMP4_MAD"] < 10.0, "old published history mismatch"
            else:
                old_frame_file = path_dir / "published.npy"
                assert sha(old_frame_file) == state["published_file_sha256"]
                published = np.load(old_frame_file)
                assert rgb_sha(published) == state["published_rgb_sha256"]
                assert len(published) == rgb_stop(start)
            prepublished_sha = rgb_sha(published)
            full = decode(pipe, combined); row["vae_decodes"] += 1
            assert len(full) == rgb_stop(stop)
            row["past5_redecode_MAD"] = float(np.abs(
                full[rgb_stop(start)-5:rgb_stop(start)].astype(np.float32) -
                published[-5:].astype(np.float32)).mean())
            appended = stitch_immutable(published, full, start, stop)
            assert rgb_sha(appended[:len(published)]) == prepublished_sha
            row["prior_published_rgb_sha256"] = prepublished_sha
            row["published_rgb_sha256"] = rgb_sha(appended)
            tmp = path_dir / "published.partial.npy"
            np.save(tmp, appended)
            os.replace(tmp, path_dir / "published.npy")
            row["published_file_sha256"] = sha(path_dir / "published.npy")
            row["decode_seconds"] = time.perf_counter() - decode_tick
            new = pil_frames(appended[rgb_stop(start):])
            prior = pil_frames(appended)
            current_mp4 = path_dir / f"chunk_{start}_{stop}.mp4"
            rollout_mp4 = path_dir / f"rollout_{rgb_stop(stop)}.mp4"
            write_video(new, current_mp4)
            write_video(prior, rollout_mp4)
            boundary_mp4 = path_dir / f"boundary_{start}_{stop}.mp4"
            write_video(pil_frames(appended[rgb_stop(start)-1:rgb_stop(start)+1]), boundary_mp4)
            contact_sheet(new, rgb_stop(start), path_dir / f"chunk_{start}_{stop}_sheet.jpg")
            contact_sheet(pil_frames(appended[rgb_stop(start)-3:rgb_stop(start)+4]),
                          rgb_stop(start)-3, path_dir / f"boundary_{start}_{stop}_sheet.jpg")
            row["flow"] = flow_metrics(current_mp4)
            row["boundary_flow"] = flow_metrics(boundary_mp4)
            gray = np.stack([np.asarray(x.convert("L"), dtype=np.float32) for x in new])
            row["inside_gray_MAD"] = float(np.abs(np.diff(gray,axis=0)).mean())
            row["boundary_gray_MAD"] = float(np.abs(
                np.asarray(new[0].convert("L"),dtype=np.float32) -
                np.asarray(Image.fromarray(published[-1]).convert("L"),dtype=np.float32)).mean())
            row["current_mp4_sha256"] = sha(current_mp4)
            row["rollout_mp4_sha256"] = sha(rollout_mp4)
            row["GPU_peak_allocated_MiB"] = torch.cuda.max_memory_allocated()/2**20
            row["GPU_peak_reserved_MiB"] = torch.cuda.max_memory_reserved()/2**20
            row["frozen_parameter_versions_unchanged"] = all(p._version == version for p, version in versions)
            assert row["frozen_parameter_versions_unchanged"]
            row["status"] = "complete_pending_visual_review"
            row["wall_seconds"] = time.perf_counter() - began
            atomic_json(result_file, row)
            state["results"].append({"interval": [start, stop], "result": str(result_file),
                                     "result_sha256": sha(result_file),
                                     "endpoint": str(endpoint), "endpoint_sha256": sha(endpoint),
                                     "published_rgb_sha256": row["published_rgb_sha256"]})
            state.update(next_start=stop, published_rgb_sha256=row["published_rgb_sha256"],
                         published_file_sha256=row["published_file_sha256"])
            atomic_json(state_file, state)
            run_record["final_status"] = "complete_pending_visual_review"
            print(json.dumps({"path": args.path, "interval": [start, stop],
                              "flow": row["flow"]["horizontal_flow_px"]["mean"],
                              "video": str(rollout_mp4), "status": row["status"]}), flush=True)
    except BaseException as exc:
        row.update(status="failed", error=repr(exc), wall_seconds=time.perf_counter()-began)
        atomic_json(result_file, row)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=["gate90", "extend124"], required=True)
    parser.add_argument("--path", choices=CONFIG["active_paths"], required=True)
    parser.add_argument("--gpu", type=int, required=True)
    args = parser.parse_args()
    assert args.config.resolve() == (HERE / "config_v3.yaml").resolve()
    verify_sources()
    state_file = OUTPUT / args.path / "state.json"
    next_start = json.loads(state_file.read_text())["next_start"] if state_file.exists() else 17
    assert next_start in (22,27,32), "v3 only continues AA/DD saved 73f states"
    assert not (OUTPUT / args.path / f"chunk_{next_start}_{next_start+5}.json").exists(), \
        "Prior attempt exists; inspect and explicitly resolve it before another model call"
    if args.stage == "gate90": assert next_start < 27
    if args.stage == "extend124": assert next_start >= 27
    free = int(subprocess.check_output(["nvidia-smi", f"--id={args.gpu}",
              "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True).strip())
    assert free >= 40000, ("GPU is not idle", args.gpu, free)
    path_dir = OUTPUT / args.path
    path_dir.mkdir(exist_ok=True)
    with (path_dir / "worker.lock").open("a+") as worker_lock:
        fcntl.flock(worker_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu), ABOT_VRAM_RESERVE_GIB="18",
                          HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                          DIFFSYNTH_SKIP_DOWNLOAD="True", TOKENIZERS_PARALLELISM="false",
                          PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
        sys.path[:0] = [str(FROZEN / "runtime/code"), str(FROZEN / "runtime/code/abot"),
                        str(FROZEN / "runtime/code/causal"),
                        str(FROZEN / "runtime/DiffSynth-Studio-h3-v2"),
                        str(FROZEN / "source_history")]
        diagnostic_reserve = 3 if next_start == 17 else 0
        with budget_reservation(args.path, args.gpu, diagnostic_reserve, 30) as record:
            run(args, record)


if __name__ == "__main__":
    main()
