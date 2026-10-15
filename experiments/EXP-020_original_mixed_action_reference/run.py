"""One bounded Original H3 mixed-action reference; no causal runtime or training."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
FROZEN = ROOT / "H3-World/outputs/2026-10-09-22/chunk_partition_cb/runtime"
sys.path[:0] = [str(FROZEN / "code/abot"), str(FROZEN / "DiffSynth-Studio-h3-v2")]


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")
    os.replace(tmp, path)


def inputs() -> tuple[dict, list[str], Path, Path, Path]:
    source = ROOT / "submission/experiments/EXP-014_v3_multiscene_teacher/source_manifest.json"
    entry = json.loads(source.read_text())["scenes"]["s3_b784d995"]
    image = ROOT / entry["png"]
    script_path = ROOT / "submission/experiments/EXP-015_v3_real56_data/artifacts/b784d995/action_script17.json"
    script = json.loads(script_path.read_text())
    lora = FROZEN / "checkpoints/H3-World/step-10000.safetensors"
    return entry, script, image, script_path, lora


def preflight() -> None:
    import inspect
    import torch
    from PIL import Image
    import infer as abot
    import diffsynth.models.minimax_h3_dit as dit
    import diffsynth.pipelines.minimax_h3_audio_video as h3pipe

    entry, script, image, script_path, lora = inputs()
    auth_path = HERE / "judge/AUTHORIZATION.json"
    taskbook = HERE / "taskbook_v1.md"
    auth = json.loads(auth_path.read_text())
    assert auth["approved"] and auth["scene"] == "b784d995"
    assert auth["taskbook_sha256"] == sha(taskbook)
    assert len(script) == 17 and len(set(script)) == 3
    assert script[:13] == ["the man strafes left, camera follows him"] * 13
    assert script[13] == "the man stands still, camera holds steady"
    assert script[14:] == ["the man walks forward, camera follows him"] * 3
    assert image.is_file() and lora.is_file() and script_path.is_file()
    assert sha(image) == entry["png_sha256"]
    assert Image.open(image).size == (832, 480)
    assert "-v2/" in dit.__file__ and "-v2/" in h3pipe.__file__
    assert "leak_out" in inspect.getsource(dit._build_action_block_masks)
    assert "action_text_spans_local" in inspect.getsource(h3pipe)
    source_call = inspect.getsource(h3pipe.MiniMaxH3Pipeline.__call__)
    assert "self.scheduler_audio.set_timesteps" in source_call
    assert "self.audio_vae.decode_audio" in source_call
    assert "self.video_vae.decode_video" in source_call
    assert "causal_chunk_size" in source_call
    assert not hasattr(abot, "configure_precision")
    root_weights = FROZEN / "DiffSynth-Studio-h3-v2/models/MiniMax/MiniMax-H3/FL2VA"
    weight_files = sorted(root_weights.glob("transformer/model*.safetensors"))
    assert weight_files and all(p.is_file() for p in weight_files)
    manifest = {
        "task": "EXP-020/v1", "scene": "b784d995", "num_frames": 56,
        "latent_frames": 17, "num_action_sentences": len(script),
        "unique_action_sentences": sorted(set(script)),
        "input_scene_manifest": "submission/experiments/EXP-014_v3_multiscene_teacher/source_manifest.json",
        "scene_static_sha256": hashlib.sha256(entry["scene_static"].encode()).hexdigest(),
        "files": {}, "transformer_weight_files": [],
        "python": sys.version, "torch": torch.__version__,
        "native_pipeline_module": h3pipe.__file__,
        "native_dit_module": dit.__file__,
    }
    for name, p in {"taskbook": taskbook, "runner": HERE / "run.py", "infer": Path(abot.__file__),
                    "pipeline": Path(h3pipe.__file__), "dit": Path(dit.__file__),
                    "image": image, "script": script_path, "lora": lora}.items():
        manifest["files"][name] = {"path": str(p.relative_to(ROOT)), "sha256": sha(p), "bytes": p.stat().st_size}
    for p in weight_files:
        manifest["transformer_weight_files"].append({"path": str(p.relative_to(ROOT)), "bytes": p.stat().st_size})
    atomic(HERE / "SOURCE_MANIFEST.json", manifest)
    atomic(HERE / "CPU_PREFLIGHT.json", {"status": "PASS", "source_manifest_sha256": sha(HERE / "SOURCE_MANIFEST.json"),
                                      "auth_sha256": sha(auth_path), "script": script,
                                      "model_origin": "Original H3 + released step-10000 Action LoRA",
                                      "inference": "native MiniMaxH3Pipeline.__call__, causal_chunk_size=None, cfg=1, 30NFE, flow_shift=2.22, audio_shift=3.0"})
    print("CPU preflight PASS", flush=True)


def run() -> None:
    import torch
    import infer as abot
    from diffsynth.utils.data.audio_video import write_video_audio

    assert (HERE / "CPU_PREFLIGHT.json").exists()
    prior = json.loads((HERE / "CPU_PREFLIGHT.json").read_text())
    assert prior["source_manifest_sha256"] == sha(HERE / "SOURCE_MANIFEST.json")
    assert json.loads((HERE / "judge/AUTHORIZATION.json").read_text())["approved"]
    out = ROOT / "H3-World/outputs/EXP-020_original_mixed_action_reference/b784d995"
    out.mkdir(parents=True, exist_ok=True)
    assert not (out / "original56.mp4").exists(), "no overwrite/retry"
    entry, script, image, _, lora = inputs()
    start = time.perf_counter()
    state = {"task": "EXP-020/v1", "status": "loading", "scene": "b784d995",
             "source_manifest_sha256": sha(HERE / "SOURCE_MANIFEST.json"),
             "gpu_visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
             "config": {"num_frames": 56, "steps": 30, "seed": 13, "cfg_scale": 1.0,
                        "flow_shift": 2.22, "audio_flow_shift": 3.0,
                        "height": 480, "width": 832, "causal_chunk_size": None,
                        "action_script": script},
             "counts": {"dit": 0, "text": 0, "image_encode": 0, "video_decode": 0, "audio_decode": 0}}
    result_path = out / "result.json"
    atomic(result_path, state)
    torch.set_num_threads(4)
    torch.manual_seed(13)
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats()
    try:
        pipe = abot.load_pipeline("cuda:0")
        lora_state = abot.load_checkpoint_lora(lora)
        n_pairs = sum(".lora_A." in k for k in lora_state)
        before = sum(len(m.lora_A_weights) for m in pipe.dit.modules() if hasattr(m, "lora_A_weights"))
        pipe.load_lora(pipe.dit, state_dict=lora_state, hotload=True)
        after = sum(len(m.lora_A_weights) for m in pipe.dit.modules() if hasattr(m, "lora_A_weights"))
        assert after - before == n_pairs == 104
        state["lora_pairs"] = n_pairs
        state["model_load_seconds"] = time.perf_counter() - start
        state["dtype"] = str(pipe.torch_dtype)
        state["attention_backend"] = "native frozen v2 directed FlexAttention"

        def counted(obj, attr: str, key: str, limit: int):
            original = getattr(obj, attr)
            def wrapper(*args, **kwargs):
                state["counts"][key] += 1
                atomic(result_path, state)
                if state["counts"][key] > limit:
                    raise RuntimeError(f"{key} over approved count {limit}")
                return original(*args, **kwargs)
            setattr(obj, attr, wrapper)

        counted(pipe.dit, "forward", "dit", 30)
        counted(pipe.text_encoder, "forward", "text", 4)
        counted(pipe.video_vae, "encode_video", "image_encode", 1)
        counted(pipe.video_vae, "decode_video", "video_decode", 1)
        counted(pipe.audio_vae, "decode_audio", "audio_decode", 1)
        state["status"] = "generation"
        atomic(result_path, state)
        with torch.no_grad():
            video, audio = pipe(prompt=entry["scene_static"], negative_prompt=" ",
                                height=480, width=832, num_frames=56, num_inference_steps=30,
                                seed=13, rand_device="cpu", cfg_scale=1.0,
                                flow_shift=2.22, audio_flow_shift=3.0,
                                keyframes=[abot.load_first_frame(image)], keyframe_indices=[0],
                                action_script=script, causal_chunk_size=None,
                                progress_bar_cmd=lambda x: x)
        torch.cuda.synchronize()
        state["generation_seconds_including_condition_decode"] = time.perf_counter() - start - state["model_load_seconds"]
        assert len(video) == 56
        assert state["counts"] == {"dit": 30, "text": 4, "image_encode": 1,
                                   "video_decode": 1, "audio_decode": 1}
        state["video_sigmas"] = [float(x) for x in pipe.scheduler.sigmas.detach().cpu()]
        state["audio_sigmas"] = [float(x) for x in pipe.scheduler_audio.sigmas.detach().cpu()]
        state["status"] = "encoding"
        atomic(result_path, state)
        tmp = out / "original56.partial.mp4"
        write_video_audio(video=video, audio=audio, output_path=str(tmp), fps=24, audio_sample_rate=32000)
        os.replace(tmp, out / "original56.mp4")
        state["output_sha256"] = sha(out / "original56.mp4")
        state["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
        state["peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 2**30
        state["wall_seconds"] = time.perf_counter() - start
        assert state["peak_allocated_gib"] <= 44 and state["wall_seconds"] <= 1200
        state["status"] = "COMPLETE_PENDING_JUDGE"
        atomic(result_path, state)
        print("EXP-020 complete", state["counts"], state["wall_seconds"], flush=True)
    except BaseException as exc:
        state["status"] = "FAILED"
        state["error"] = repr(exc)
        state["wall_seconds"] = time.perf_counter() - start
        if torch.cuda.is_initialized():
            state["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
        atomic(result_path, state)
        raise


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--preflight":
        preflight()
    elif len(sys.argv) == 2 and sys.argv[1] == "--run":
        run()
    else:
        raise SystemExit("use --preflight or --run")
