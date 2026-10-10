"""EXP-014 P1: encode four fixed train PNGs into native full37 Single-I0 fixtures.

No video GT, old encoded tensors, denoiser, decoder, or training path is read.
An explicit Judge P1 marker is required for --encode. --preflight is CPU only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

from common import CFG, FROZEN, HERE, OUT, ROOT, Ledger, atomic_json, check_machine, require_marker, setup_paths, sha, verify_sources
from source_audit import inspect_sources


def preflight():
    import torch
    setup_paths()
    verify_sources()
    import infer as abot
    import action_script as script
    from causal.local_topology import visible_inputs
    from diffsynth.pipelines.minimax_h3_audio_video import (
        MiniMaxH3Pipeline, MiniMaxH3Unit_PromptEmbedder, MiniMaxH3Unit_PackedSequenceBuilder)
    from causal.h3_cached import H3ChunkCache, ChunkAttention
    from interval_cached import interval_cached
    from current_prefix import current_prefix_feedback
    assert CFG["num_frames"] == 124 and CFG["latent_frames"] == 37
    assert CFG["partition"] == [12, 5] and CFG["seed"] == 13
    assert CFG["p1"]["max_gpu_seconds"] == 600
    assert CFG["p1"]["text_encoder_calls"] == 12
    assert CFG["p1"]["image_vae_encodes"] == 4
    assert CFG["p1"]["video_vae_encodes"] == CFG["p1"]["sampling_forwards"] == 0
    assert CFG["p1"]["commit_forwards"] == CFG["p1"]["vae_decodes"] == 0
    assert inspect_sources() == json.loads((HERE / "source_manifest.json").read_text())
    assert len(script.annotate_from_keys9(__import__("numpy").zeros((37, len(script.KEYS9))))) == 37
    assert all(x is not None for x in (abot.load_pipeline, MiniMaxH3Pipeline,
                                        MiniMaxH3Unit_PromptEmbedder, MiniMaxH3Unit_PackedSequenceBuilder,
                                        visible_inputs, H3ChunkCache, ChunkAttention,
                                        interval_cached, current_prefix_feedback))
    print(json.dumps({"task": CFG["task"], "P0_cpu_import_preflight": "PASS",
                      "torch": torch.__version__, "frozen_runtime": str(FROZEN),
                      "prior_encoded_used": False, "gpu_calls": 0}))


def load_encoding_pipeline():
    """Load only H3 text encoder, source VAE and processor, never DiT."""
    import torch
    from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Pipeline, ModelConfig
    dtype = torch.bfloat16
    vram = dict(offload_dtype=dtype, offload_device="cpu", onload_dtype=dtype,
                onload_device="cpu", preparing_dtype=dtype, preparing_device="cuda:0",
                computation_dtype=dtype, computation_device="cuda:0")
    pipe = MiniMaxH3Pipeline.from_pretrained(
        torch_dtype=dtype, device="cuda:0", model_configs=[
            ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/text_encoder/model*.safetensors", **vram),
            ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/video_vae/source/model.safetensors", **vram),
        ], processor_config=ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/processor/"),
        vram_limit=41.0)
    if pipe.dit is not None or pipe.audio_vae is not None:
        raise RuntimeError("P1 unexpectedly loaded denoiser or audio VAE")
    return pipe


def script_for(direction: str):
    import numpy as np
    import action_script as script
    keys = np.zeros((CFG["latent_frames"], len(script.KEYS9)), dtype=np.int64)
    keys[:, script.KEYS9.index(direction)] = 1
    sentences = script.annotate_from_keys9(keys)
    assert len(sentences) == 37 and len(set(sentences)) == 1
    return sentences


def encode(gpu: int):
    setup_paths()
    verify_sources()
    require_marker("P1")
    import torch
    from PIL import Image
    from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Unit_PromptEmbedder
    from causal.local_topology import visible_inputs
    torch.set_num_threads(4)
    torch.manual_seed(CFG["seed"])
    source = inspect_sources()
    ledger = Ledger(gpu, "P1")
    started = time.perf_counter()
    row = {"task": CFG["task"], "stage": "P1", "status": "initializing",
           "gpu": gpu, "fixtures": {}}
    try:
        check_machine(gpu, "P1")
        row.update(status="loading",
                   marker_sha256=sha(HERE / "judge/P1_APPROVED.json"),
                   source_manifest_sha256=sha(HERE / "source_manifest.json"),
                   runner_sha256=sha(__file__))
        atomic_json(OUT / "P1_result.json", row)
        pipe = load_encoding_pipeline()
        assert pipe.text_encoder is not None and pipe.video_vae is not None
        def text_hook(_module, _args):
            ledger.reserve("text_encoder_calls")
        hook = pipe.text_encoder.register_forward_pre_hook(text_hook)
        raw_encode = pipe.video_vae.encode_video
        def metered_encode(*args, **kwargs):
            if kwargs.get("process_image") is not True:
                ledger.reserve("video_vae_encodes")  # forbidden in P1
                raise RuntimeError("P1 forbids video VAE encode")
            ledger.reserve("image_vae_encodes")
            return raw_encode(*args, **kwargs)
        pipe.video_vae.encode_video = metered_encode
        row["status"] = "encoding"; atomic_json(OUT / "P1_result.json", row)
        with torch.no_grad():
            for name, src in source["scenes"].items():
                target = OUT / "fixtures" / f"{name}.pt"
                if target.exists():
                    raise FileExistsError(f"fixture already exists; no overwrite: {target}")
                image = Image.open(ROOT / src["png"]).convert("RGB")
                assert image.size == (CFG["width"], CFG["height"])
                shared = {"cfg_scale": 1.0, "height": CFG["height"], "width": CFG["width"],
                          "num_frames": CFG["num_frames"], "seed": CFG["seed"], "rand_device": "cpu",
                          "keyframes": [image], "keyframe_indices": [0],
                          "imgvid_cond_noise_aug": 0.999, "audio_cond_noise_aug": 1.0}
                positive = {"prompt": src["scene_static"], "action_script": script_for("A")}
                negative = {}
                for unit in pipe.units:
                    ledger.check_time()
                    shared, positive, negative = pipe.unit_runner(unit, pipe, shared, positive, negative)
                head = positive["prompt_embeds"]
                packed = positive["packed"]
                spans = packed["action_text_spans_local"]
                assert len(spans) == CFG["latent_frames"]
                assert packed["action_video_start"] == packed["img_pos"][390].item()
                assert packed["img_pos"].numel() == (37 + 1) * 390
                assert shared["keyframe_cond_anchor"].shape == (390, 96)
                # Encode the one distinct D sentence through the same frozen
                # PromptEmbedder; enforce action-content-independent full37 positions.
                pipe.load_models_to_device(["text_encoder"])
                rows, d_spans = MiniMaxH3Unit_PromptEmbedder()._encode_action_script(
                    pipe, script_for("D"), int(spans[0][0]), pipe.torch_dtype)
                assert d_spans == spans, "A/D token span mismatch; requires Judge protocol review"
                prompt_d = head.clone()
                for (lo, hi), encoded in zip(spans, rows):
                    prompt_d[lo:hi] = encoded
                prompt_a = head
                for stop in (12, 17):
                    for prompt in (prompt_a, prompt_d):
                        visible, text = visible_inputs(packed, prompt, stop, 390)
                        assert len(visible["action_text_rows"]) == stop
                        assert visible["seq_len"] == visible["action_video_start"] + stop * 390
                        assert len(text) < len(prompt)
                assert torch.equal(prompt_a[:spans[0][0]], prompt_d[:spans[0][0]])
                assert not torch.equal(prompt_a[spans[12][0]:spans[12][1]],
                                       prompt_d[spans[12][0]:spans[12][1]])
                assert shared["video_latents"].shape == (1, 24, 37, 30, 52)
                assert torch.isfinite(prompt_a).all() and torch.isfinite(prompt_d).all()
                fixture = {"format": "exp014_native_single_i0_full37_v1", "scene": name,
                           "source_png_sha256": src["png_sha256"], "scene_static": src["scene_static"],
                           "action_sentences": {"A": script_for("A")[0], "D": script_for("D")[0]},
                           "initial_noise": shared["video_latents"].cpu(),
                           "audio_noise": shared["audio_latents"].cpu(),
                           "anchor": shared["keyframe_cond_anchor"].cpu(),
                           "packed": {k: v.cpu() if torch.is_tensor(v) else v for k, v in packed.items()},
                           "prompts": {"A": prompt_a.cpu(), "D": prompt_d.cpu()}}
                target.parent.mkdir(parents=True, exist_ok=True)
                temp = target.with_name(target.name + ".partial")
                torch.save(fixture, temp)
                temp.replace(target)
                row["fixtures"][name] = {"path": str(target.relative_to(ROOT)),
                                         "sha256": sha(target), "bytes": target.stat().st_size,
                                         "action_spans": len(spans),
                                         "action_sentences": fixture["action_sentences"]}
                atomic_json(OUT / "P1_result.json", row)
        hook.remove()
        counts = {key: ledger.data[key] for key in ("text_encoder_calls", "image_vae_encodes",
                                                     "video_vae_encodes", "sampling_forwards",
                                                     "commit_forwards", "vae_decodes")}
        assert counts == {"text_encoder_calls": 12, "image_vae_encodes": 4,
                          "video_vae_encodes": 0, "sampling_forwards": 0,
                          "commit_forwards": 0, "vae_decodes": 0}, counts
        row.update(status="complete_pending_cpu_audit", actual_calls=counts,
                   elapsed_seconds=time.perf_counter() - started,
                   peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30)
        if row["peak_allocated_gib"] > CFG["gpu_allocated_cap_gib"]:
            raise MemoryError("P1 allocated VRAM cap exceeded")
        atomic_json(OUT / "P1_result.json", row)
        ledger.close("complete")
    except BaseException as error:
        row.update(status="failed", error=repr(error), elapsed_seconds=time.perf_counter() - started)
        atomic_json(OUT / "P1_result.json", row)
        ledger.close("failed")
        raise


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--encode", action="store_true")
    ap.add_argument("--gpu", type=int)
    args = ap.parse_args()
    if args.preflight and not args.encode:
        preflight()
    elif args.encode and not args.preflight and args.gpu is not None:
        encode(args.gpu)
    else:
        ap.error("use --preflight OR --encode --gpu ID")
