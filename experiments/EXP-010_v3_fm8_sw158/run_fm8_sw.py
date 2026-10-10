"""EXP-010 staged FM8 + SW-G continuation. CPU preflight precedes Judge GPU release.

Reuses the published EXP-006 AA 73-frame endpoint and its own C1/C2 cache.
Stages: commit_c3, C4, C5, C6, C7 A/D, C8 A/D. No training or AF modules.
"""
from __future__ import annotations

from datetime import datetime
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CFG = json.loads((HERE / "config.json").read_text())
OUT = ROOT / CFG["output_root"]
FM8 = ROOT / "H3-World/outputs/EXP-006_v3_fm8_full"
FROZEN = ROOT / "H3-World/outputs/2026-10-09-22/chunk_partition_cb"
SW = HERE.parent / "EXP-005_v3_sliding_window"
SW2 = SW / "stage2"
ROUTER = ROOT / "submission/reports/stage1_anyflow/02_causal_diagnostics/current_prefix_candidate"
EXP006 = HERE.parent / "EXP-006_v3_fm8_full"
CONTRACT = HERE.parent / "EXP-001_v2b_124"
FIXTURE = ROOT / "H3-World/outputs/EXP-005_v3_sliding_window/inputs/long47.pt"
MANIFEST = HERE / "source_manifest.json"


def sha(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(str(path) + ".partial")
    temp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    os.replace(temp, path)


def setup_paths() -> None:
    sys.path[:0] = [str(FROZEN / "runtime/code"), str(FROZEN / "runtime/code/abot"),
                    str(FROZEN / "runtime/code/causal"),
                    str(FROZEN / "runtime/DiffSynth-Studio-h3-v2"),
                    str(SW), str(SW2), str(ROUTER), str(EXP006), str(CONTRACT), str(HERE)]


def source_paths() -> dict[str, Path]:
    runtime = FROZEN / "runtime"
    model = runtime / "DiffSynth-Studio-h3-v2"
    return {
        "runner": HERE / "run_fm8_sw.py", "config": HERE / "config.json",
        "taskbook": HERE / "taskbook_v1.md",
        "long47": FIXTURE, "long47_metadata": FIXTURE.with_suffix(".json"),
        "parking_A": FROZEN / "source_coarse/inputs/parking_A.pt",
        "parking_D": FROZEN / "source_coarse/inputs/parking_D.pt",
        "released_action_lora": runtime / "checkpoints/H3-World/step-10000.safetensors",
        "fm8_first12": FM8 / "first12.pt",
        "fm8_AA_C2": FM8 / "AA/chunk_12_17.pt",
        "fm8_AA_C3": FM8 / "AA/chunk_17_22.pt",
        "fm8_AA_cache17": FM8 / "AA/cache_through17.pt",
        "fm8_AA_rgb73": FM8 / "AA/published_73.npy",
        "fm8_C1_metrics": FM8 / "chunk_0_12.json",
        "fm8_AA_C2_metrics": FM8 / "AA/chunk_12_17.json",
        "fm8_AA_C3_metrics": FM8 / "AA/chunk_17_22.json",
        "fm8_runner": EXP006 / "run_fm8.py",
        "rollout_contract": CONTRACT / "rollout_contract.py",
        "sw_interval": SW / "interval_sw.py",
        "sw_stage2_interval": SW2 / "interval_stage2.py",
        "sw_long_builder": SW2 / "long_fixture.py",
        "sw_plan": SW / "chunk_plan.py", "sw_positions": SW / "position_sw.py",
        "prefix_feedback": ROUTER / "current_prefix.py",
        "cached_attention": runtime / "code/causal/h3_cached.py",
        "local_topology": runtime / "code/causal/local_topology.py",
        "video_schedule": runtime / "code/causal/anyflow_sampling.py",
        "precision": runtime / "code/causal/h3_precision.py",
        "benchmark_video": runtime / "code/causal/benchmark.py",
        "action_flow": runtime / "code/causal/evaluate_action_control.py",
        "abot_infer": runtime / "code/abot/infer.py",
        "h3_pipeline": model / "diffsynth/pipelines/minimax_h3_audio_video.py",
        "h3_dit": model / "diffsynth/models/minimax_h3_dit.py",
        "h3_scheduler": model / "diffsynth/diffusion/flow_match.py",
        "transformer_config": model / "models/MiniMax/MiniMax-H3/FL2VA/transformer/config.json",
        "transformer_index": model / "models/MiniMax/MiniMax-H3/FL2VA/transformer/model.safetensors.index.json",
    }


def manifest_data() -> dict:
    return {"task": CFG["task"], "sources": {
        name: {"path": str(path.relative_to(ROOT)), "sha256": sha(path)}
        for name, path in source_paths().items()}}


def verify_manifest(*, create: bool = False) -> dict:
    data = manifest_data()
    if MANIFEST.exists():
        if json.loads(MANIFEST.read_text()) != data:
            raise RuntimeError("EXP-010 frozen source mismatch")
    elif create:
        atomic_json(MANIFEST, data)
    else:
        raise RuntimeError("EXP-010 CPU preflight/source freeze absent")
    return data


def rgb_stop(latent_stop: int) -> int:
    if latent_stop < 12 or (latent_stop - 12) % 5:
        raise ValueError(latent_stop)
    return 39 + 17 * ((latent_stop - 12) // 5)


def preflight() -> None:
    import numpy as np
    import torch
    import infer as abot
    from benchmark import write_video
    from causal.anyflow_sampling import configure_video_schedule
    from causal.h3_precision import configure_precision
    from causal.local_topology import visible_inputs
    from current_prefix import current_prefix_feedback
    from diffsynth.diffusion.flow_match import FlowMatchScheduler
    from chunk_plan import PLAN, cache_identity, validate_cache
    from interval_stage2 import interval_stage2
    from long_fixture import same_layout, tensor_sha
    from run_fm8 import decode_video, contact_sheet
    from evaluate_action_control import evaluate as flow_metrics
    from rollout_contract import rgb_sha

    assert CFG["max_history"] == 5 and CFG["steps_per_chunk"] == 8
    assert (CFG["max_forwards"], CFG["max_sampling"], CFG["max_commits"], CFG["max_vae"]) == (62, 56, 6, 7)
    assert CFG["flow_shift"] == 2.22 and CFG["max_gpu_hours"] == .75
    entries = manifest_data()  # Hash before any output is frozen.
    fixture_meta = json.loads(FIXTURE.with_suffix(".json").read_text())
    assert fixture_meta["certified_cpu"] and fixture_meta["latent_stop"] == 47
    assert fixture_meta["noise_extension_seed"] == 130005
    assert entries["sources"]["long47"]["sha256"] == fixture_meta["fixture_sha256"]
    assert entries["sources"]["sw_long_builder"]["sha256"] == fixture_meta["builder_sha256"]
    for action in "AD":
        assert entries["sources"][f"parking_{action}"]["sha256"] == fixture_meta["source"][action]["sha256"]

    c1 = json.loads((FM8 / "chunk_0_12.json").read_text())
    c2 = json.loads((FM8 / "AA/chunk_12_17.json").read_text())
    c3 = json.loads((FM8 / "AA/chunk_17_22.json").read_text())
    assert all(row["status"] == "complete_pending_visual_review" for row in (c1, c2, c3))
    assert c2["cache_after_sha256"] == c3["cache_file_sha256"] == entries["sources"]["fm8_AA_cache17"]["sha256"]
    assert c2["endpoint_sha256"] == entries["sources"]["fm8_AA_C2"]["sha256"]
    assert c3["endpoint_sha256"] == entries["sources"]["fm8_AA_C3"]["sha256"]
    assert c3["published_file_sha256"] == entries["sources"]["fm8_AA_rgb73"]["sha256"]
    assert c3["old_RGB_unchanged"] and c3["cache_read_unchanged"]

    fixture = torch.load(FIXTURE, map_location="cpu", weights_only=True)
    old = torch.load(FROZEN / "source_coarse/inputs/parking_A.pt", map_location="cpu", weights_only=True)
    original_d = torch.load(FROZEN / "source_coarse/inputs/parking_D.pt", map_location="cpu", weights_only=True)
    assert torch.equal(fixture["initial_noise"][:, :, :37], old["initial_noise"])
    assert torch.equal(fixture["anchor"], old["anchor"])
    assert torch.equal(fixture["audio_noise"], old["audio_noise"])
    assert torch.equal(old["initial_noise"], original_d["initial_noise"])
    assert fixture["metadata"]["default_full_length_builder_equivalent"] is False
    assert tensor_sha(fixture["initial_noise"]) == fixture_meta["extended_noise_sha256"]
    assert tensor_sha(fixture["initial_noise"][:, :, 37:]) == fixture_meta["tail_noise_sha256"]
    original_text = old["pairs"][0]["prompts"]["A"]
    old_layout = old["packed"]
    for stop in (12, 17, 22, 27, 32, 37):
        reference, original_prompt = visible_inputs(old_layout, original_text, stop, 390)
        for action in "AD":
            candidate, prompt = visible_inputs(fixture["packed"], fixture["prompts"][action], stop, 390)
            same_layout(reference, candidate)
            assert torch.equal(prompt, original_prompt)
    for stop in (42, 47):
        a, pa = visible_inputs(fixture["packed"], fixture["prompts"]["A"], stop, 390)
        d, pd = visible_inputs(fixture["packed"], fixture["prompts"]["D"], stop, 390)
        same_layout(a, d)
        assert not torch.equal(pa, pd)
        assert a["seq_len"] == int(a["action_video_start"]) + stop * 390

    pieces = [torch.load(FM8 / part, map_location="cpu", weights_only=True) for part in
              ("first12.pt", "AA/chunk_12_17.pt", "AA/chunk_17_22.pt")]
    history = torch.cat(pieces, dim=2)
    assert history.shape == (1, 24, 22, 30, 52)
    from run_fm8 import tensor_sha as fm8_tensor_sha
    assert fm8_tensor_sha(history[:, :, :17]) == c3["history_tensor_sha256"]
    assert np.load(FM8 / "AA/published_73.npy", mmap_mode="r").shape == (73, 480, 832, 3)
    cache = torch.load(FM8 / "AA/cache_through17.pt", map_location="cpu", weights_only=False)
    assert validate_cache(cache, plan=PLAN, index=2, frame_rows=390, expected_layers=50)["ancestors"] == [0, 1]
    scheduler = FlowMatchScheduler("MiniMax-H3")
    sigmas = configure_video_schedule(scheduler, steps=8, grid="native", flow_shift=2.22)
    assert sigmas == c3["sigmas"] and len(sigmas) == 9
    assert [stage_spec(stage,path) for stage,path in
            (("commit_c3",None),("C4",None),("C5",None),("C6",None),
             ("C7","A"),("C7","D"),("C8","A"),("C8","D"))] == [
            (2,22,"shared"),(3,27,"shared"),(4,32,"shared"),(5,37,"shared"),
            (6,42,"A"),(6,42,"D"),(7,47,"A"),(7,47,"D")]
    assert 1+3*(8+1)==28 and 2*(8+1)+2*8==34
    assert list(PLAN.ancestors(6)) == [1,2,3,4,5]
    assert list(PLAN.ancestors(7)) == [2,3,4,5,6]
    verify_manifest(create=True)
    result = dict(task=CFG["task"], status="CPU_PASS_NO_GPU_AUTHORIZATION",
                  source_manifest_sha256=sha(MANIFEST), source_count=len(entries["sources"]),
                  first73_rgb_sha256=entries["sources"]["fm8_AA_rgb73"]["sha256"],
                  cache17_sha256=entries["sources"]["fm8_AA_cache17"]["sha256"],
                  long47_sha256=entries["sources"]["long47"]["sha256"],
                  native_sigmas=sigmas, old_prefix_stops_checked=[12,17,22,27,32,37],
                  branch_stops_checked=[42,47], expected_first_eviction="C6 commit: C1 -> C2-C6",
                  gpu_calls=0)
    atomic_json(HERE / "cpu_preflight.json", result)
    print(json.dumps(result))


def stage_spec(stage: str, action: str | None) -> tuple[int, int, str]:
    if stage == "commit_c3":
        if action is not None: raise ValueError("shared commit takes no action path")
        return 2, 22, "shared"
    if stage in ("C4", "C5", "C6"):
        if action is not None: raise ValueError("shared C4-C6 takes no path")
        return {"C4": (3,27,"shared"), "C5": (4,32,"shared"), "C6": (5,37,"shared")}[stage]
    if stage in ("C7", "C8") and action in ("A", "D"):
        return (6,42,action) if stage == "C7" else (7,47,action)
    raise ValueError("C7/C8 require A or D")


def run(stage: str, action: str | None, gpu: int, ledger: dict) -> None:
    import numpy as np
    import torch
    from PIL import Image
    import infer as abot
    from benchmark import write_video
    from causal.anyflow_sampling import configure_video_schedule
    from causal.h3_precision import configure_precision
    from causal.local_topology import visible_inputs
    from current_prefix import current_prefix_feedback
    from chunk_plan import PLAN, cache_identity, validate_cache
    from interval_stage2 import interval_stage2
    from run_fm8 import tensor_sha, decode_video, contact_sheet
    from evaluate_action_control import evaluate as flow_metrics
    from rollout_contract import rgb_sha

    torch.set_num_threads(4)
    torch.manual_seed(CFG["seed"])
    index, stop, branch = stage_spec(stage, action)
    start = PLAN.span(index)[0]
    assert PLAN.span(index) == ((17,22) if stage == "commit_c3" else (start,stop))
    directory = OUT / branch
    directory.mkdir(parents=True, exist_ok=True)
    row_path = directory / f"{stage}{'' if action is None else '_' + action}.json"
    if row_path.exists(): raise RuntimeError(f"prior stage exists; no retry: {row_path}")
    began = time.perf_counter()
    row = dict(task=CFG["task"],status="loading",stage=stage,action=action,
               index=index,interval=[start,stop],gpu=gpu,
               source_manifest_sha256=sha(MANIFEST),runner_sha256=sha(__file__),
               sampling_forwards=0,commit_forwards=0,vae_calls=0)
    atomic_json(row_path,row)

    def check_limits() -> None:
        now = time.time()
        if (ledger["gpu_seconds"] + now-ledger["active_start"] >= CFG["max_gpu_hours"]*3600
                or now >= datetime.fromisoformat(CFG["deadline_hkt"]).timestamp()):
            raise TimeoutError("EXP-010 budget or 09:00 cutoff")
    def reserve(kind: str) -> None:
        check_limits()
        if kind == "vae":
            if ledger["vae"] >= CFG["max_vae"]: raise RuntimeError("VAE budget exhausted")
            ledger["vae"] += 1
        else:
            key = "sampling" if kind == "sampling" else "commits"
            cap = CFG["max_sampling"] if kind == "sampling" else CFG["max_commits"]
            if ledger["forwards"] >= CFG["max_forwards"] or ledger[key] >= cap:
                raise RuntimeError(f"{kind} budget exhausted")
            ledger[key] += 1; ledger["forwards"] += 1
        atomic_json(OUT / "budget.json", ledger)
    def peak() -> float:
        value = torch.cuda.max_memory_allocated()/2**30
        if value > CFG["max_allocated_gib"]: raise MemoryError("EXP-010 allocated VRAM cap")
        return value
    def move(value):
        if torch.is_tensor(value): return value.to("cuda:0")
        if isinstance(value,dict): return {k: move(v) for k,v in value.items()}
        return value
    def parent_file(folder: Path, name: str, expected_sha_key: str, parent: Path) -> Path:
        f = folder / name
        data = json.loads(parent.read_text())
        assert data["status"] == "complete_pending_visual_review"
        assert data["source_manifest_sha256"] == sha(MANIFEST)
        assert sha(f) == data[expected_sha_key]
        return f

    try:
        fixture = torch.load(FIXTURE,map_location="cpu",weights_only=True)
        full = move(fixture["packed"])
        prompt = fixture["prompts"][action or "A"].to("cuda:0")
        layout,prompt = visible_inputs(full,prompt,stop,390)
        assert len(layout["action_text_rows"]) == stop
        anchor = fixture["anchor"].to("cuda:0")
        audio = fixture["audio_noise"].to("cuda:0")
        frozen_reference = move(fixture["frozen_reference"])
        frozen_prompt = fixture["frozen_reference_prompt"].to("cuda:0")
        initial = fixture["initial_noise"]
        pieces = [torch.load(FM8 / f,map_location="cpu",weights_only=True) for f in
                  ("first12.pt","AA/chunk_12_17.pt","AA/chunk_17_22.pt")]
        if stage == "commit_c3":
            cache_file = FM8 / "AA/cache_through17.pt"
            current = pieces[2].to("cuda:0",torch.float32)
            history = torch.cat(pieces[:2],dim=2).to("cuda:0",torch.float32)
            published = None
        else:
            if stage == "C4":
                cache_file = parent_file(OUT/"shared","cache_through22.pt","cache_after_sha256",
                                         OUT/"shared/commit_c3.json")
            elif stage in ("C5","C6"):
                prior = {"C5":("C4",27),"C6":("C5",32)}[stage]
                cache_file = parent_file(OUT/"shared",f"cache_through{prior[1]}.pt","cache_after_sha256",
                                         OUT/"shared"/f"{prior[0]}.json")
            elif stage == "C7":
                cache_file = parent_file(OUT/"shared","cache_through37.pt","cache_after_sha256",
                                         OUT/"shared/C6.json")
            else:
                cache_file = parent_file(directory,"cache_through42.pt","cache_after_sha256",
                                         directory/f"C7_{action}.json")
            history_parts = list(pieces)
            core_intervals = [(22,27),(27,32),(32,37)]
            for lo,hi in core_intervals:
                if lo >= start: break
                endpoint = OUT/"shared"/f"chunk_{lo}_{hi}.pt"
                parent = OUT/"shared"/f"C{4+(lo-22)//5}.json"
                history_parts.append(torch.load(parent_file(endpoint.parent,endpoint.name,"endpoint_sha256",parent),
                                                map_location="cpu",weights_only=True))
            if stage == "C8":
                endpoint = directory/"chunk_37_42.pt"
                history_parts.append(torch.load(parent_file(directory,endpoint.name,"endpoint_sha256",
                                                              directory/f"C7_{action}.json"),
                                                map_location="cpu",weights_only=True))
            history = torch.cat(history_parts,dim=2).to("cuda:0",torch.float32)
            assert history.shape[2] == start
            if stage == "C4":
                published_file = FM8/"AA/published_73.npy"
            elif stage in ("C5","C6","C7"):
                prior = {"C5":("C4",90),"C6":("C5",107),"C7":("C6",124)}[stage]
                published_file = parent_file(OUT/"shared",f"published_{prior[1]}.npy",
                                             "published_file_sha256",OUT/"shared"/f"{prior[0]}.json")
            else:
                published_file = parent_file(directory,"published_141.npy","published_file_sha256",
                                             directory/f"C7_{action}.json")
            published = np.load(published_file)
            assert published.shape == (rgb_stop(start),480,832,3)
            current = initial[:,:,start:stop].to("cuda:0",torch.float32).clone()
            row["source_published_sha256"] = sha(published_file)
            row["prior_RGB_sha256"] = rgb_sha(published)
            row["initial_noise_sha256"] = tensor_sha(current)
        cache = torch.load(cache_file,map_location="cpu",weights_only=False)
        row["source_cache_sha256"] = sha(cache_file)
        row["history_tensor_sha256"] = tensor_sha(history)
        row["prompt_sha256"] = tensor_sha(prompt)
        row["cache_before"] = validate_cache(cache,plan=PLAN,index=index,frame_rows=390,expected_layers=50)

        pipe = abot.load_pipeline("cuda:0")
        released = FROZEN/"runtime/checkpoints/H3-World/step-10000.safetensors"
        pipe.load_lora(pipe.dit,state_dict=abot.load_checkpoint_lora(released),hotload=True)
        model = pipe.dit.eval().requires_grad_(False)
        row["precision"] = configure_precision(model,"h3_fp32",native_transformer_dir=
            FROZEN/"runtime/DiffSynth-Studio-h3-v2/models/MiniMax/MiniMax-H3/FL2VA/transformer")
        pipe.load_models_to_device(["dit"])
        sigmas = configure_video_schedule(pipe.scheduler,steps=8,grid="native",flow_shift=2.22)
        assert sigmas == json.loads((FM8/"AA/chunk_17_22.json").read_text())["sigmas"]
        row["sigmas"] = sigmas
        row["model_ready_seconds"] = time.perf_counter()-began

        def forward(sigma: float, kind: str, commit: bool = False):
            reserve(kind)
            with current_prefix_feedback(),torch.no_grad():
                output = interval_stage2(model,current,start=start,index=index,cache=cache,
                    full_packed=layout,prompt=prompt,anchor=anchor,audio=audio,
                    sigma=sigma,commit=commit,mode="global",expected_layers=50,
                    frozen_reference=frozen_reference,frozen_reference_prompt=frozen_prompt,
                    certified_long_fixture=True)
            peak()
            return output

        signature = cache_identity(cache)
        history_hash = tensor_sha(history)
        if stage == "commit_c3":
            assert sha(cache_file) == json.loads((FM8/"AA/chunk_17_22.json").read_text())["cache_file_sha256"]
            tick=time.perf_counter(); _=forward(0,"commit",commit=True)
            torch.cuda.synchronize()
            row["commit_seconds"] = time.perf_counter()-tick
            row["commit_forwards"] = 1
        else:
            noise_hash=tensor_sha(current)
            tick=time.perf_counter()
            for j,(sigma,_) in enumerate(zip(sigmas[:-1],sigmas[1:])):
                velocity=forward(sigma,"sampling")
                current=pipe.scheduler.step(velocity,pipe.scheduler.timesteps[j],current)
                if not torch.isfinite(current).all(): raise FloatingPointError("EXP-010 nonfinite latent")
                row["step_completed"]=j+1
                atomic_json(row_path,row)
            torch.cuda.synchronize()
            row["sampling_seconds"] = time.perf_counter()-tick
            row["sampling_forwards"] = 8
            assert cache_identity(cache) == signature
            assert tensor_sha(history) == history_hash
            assert tensor_sha(initial[:,:,start:stop].to("cuda:0",torch.float32)) == noise_hash
            row["cache_read_unchanged"] = True
            endpoint=directory/f"chunk_{start}_{stop}.pt"
            temp=Path(str(endpoint)+".partial")
            torch.save(current.detach().cpu(),temp);os.replace(temp,endpoint)
            row["endpoint_sha256"] = sha(endpoint)
            if stage != "C8":
                tick=time.perf_counter(); _=forward(0,"commit",commit=True)
                torch.cuda.synchronize()
                row["commit_seconds"] = time.perf_counter()-tick
                row["commit_forwards"] = 1
        if stage != "C8":
            after=validate_cache(cache,plan=PLAN,index=index+1,frame_rows=390,expected_layers=50)
            assert after["ancestors"] == list(PLAN.ancestors(index+1))
            row["cache_after"] = after
            cache_out=directory/f"cache_through{stop}.pt"
            temp=Path(str(cache_out)+".partial")
            torch.save(cache,temp);os.replace(temp,cache_out)
            row["cache_after_sha256"] = sha(cache_out)
        assert sha(cache_file) == row["source_cache_sha256"]
        if stage != "commit_c3":
            assert tensor_sha(history) == history_hash
            row["source_cache_unchanged"] = True
            pipe.load_models_to_device(["video_vae"])
            reserve("vae")
            tick=time.perf_counter()
            decoded=decode_video(pipe,torch.cat((history,current),dim=2))
            peak()
            row["decode_seconds"] = time.perf_counter()-tick
            row["vae_calls"] = 1
            assert decoded.shape == (rgb_stop(stop),480,832,3)
            appended=np.concatenate((published,decoded[rgb_stop(start):]),axis=0)
            assert rgb_sha(appended[:len(published)]) == row["prior_RGB_sha256"]
            row["old_RGB_unchanged"] = True
            rgb_file=directory/f"published_{rgb_stop(stop)}.npy"
            np.save(rgb_file,appended)
            row["published_file_sha256"] = sha(rgb_file)
            row["published_RGB_sha256"] = rgb_sha(appended)
            video=directory/f"rollout_{rgb_stop(stop)}.mp4"
            write_video([Image.fromarray(frame) for frame in appended],video)
            active=directory/f"chunk_{start}_{stop}.mp4"
            new=appended[rgb_stop(start):]
            write_video([Image.fromarray(frame) for frame in new],active)
            contact_sheet(new,directory/f"chunk_{start}_{stop}_all_frames.jpg")
            row["video_sha256"] = sha(video)
            row["flow"] = flow_metrics(active)
            gray=np.stack([np.asarray(Image.fromarray(f).convert("L"),dtype=np.float32)
                           for f in appended[rgb_stop(start)-1:]])
            row["boundary_gray_MAD"] = float(np.abs(gray[1]-gray[0]).mean())
            row["inside_gray_MAD"] = float(np.abs(np.diff(gray[1:],axis=0)).mean())
        row["peak_allocated_gib"] = torch.cuda.max_memory_allocated()/2**30
        row["peak_reserved_gib"] = torch.cuda.max_memory_reserved()/2**30
        row["peak_cpu_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
        row["seconds"] = time.perf_counter()-began
        row["status"] = "complete_pending_visual_review"
        atomic_json(row_path,row)
    except BaseException as exc:
        row["status"]="failed";row["error"]=repr(exc)
        row["seconds"]=time.perf_counter()-began
        atomic_json(row_path,row)
        raise


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight",action="store_true")
    parser.add_argument("--stage",choices=("commit_c3","C4","C5","C6","C7","C8"))
    parser.add_argument("--path",choices=("A","D"))
    parser.add_argument("--gpu",type=int,default=CFG["gpu"])
    args=parser.parse_args()
    setup_paths()
    if args.preflight:
        preflight();return
    if args.stage is None: parser.error("GPU stage required")
    stage_spec(args.stage,args.path)
    verify_manifest()
    phase="core" if args.stage in ("commit_c3","C4","C5","C6") else "branch"
    marker=HERE/f"GPU_AUTHORIZATION_{phase.upper()}.json"
    if not marker.exists(): raise RuntimeError(f"EXP-010 Judge {phase} GPU marker absent")
    permit=json.loads(marker.read_text())
    if not (permit.get("approved") and permit.get("source_manifest_sha256")==sha(MANIFEST)
            and args.stage in permit.get("allowed_stages",[])
            and args.gpu==permit.get("gpu")):
        raise RuntimeError("EXP-010 GPU release/source/stage mismatch")
    if phase=="branch":
        review_path=HERE/"CORE_JUDGE_REVIEW.json"
        if not review_path.exists():
            raise RuntimeError("EXP-010 C7/C8 needs Judge C4-C6 review marker")
        review=json.loads(review_path.read_text())
        if not (review.get("approved") is True and review.get("stage")=="core"
                and review.get("source_manifest_sha256")==sha(MANIFEST)
                and review.get("C6_row_sha256")==sha(OUT/"shared/C6.json")
                and review.get("cache37_sha256")==sha(OUT/"shared/cache_through37.pt")
                and review.get("published124_sha256")==sha(OUT/"shared/published_124.npy")):
            raise RuntimeError("EXP-010 core review marker mismatch")
    free=int(subprocess.check_output(["nvidia-smi",f"--id={args.gpu}",
        "--query-gpu=memory.free","--format=csv,noheader,nounits"],text=True).strip())
    if free<44000: raise RuntimeError(f"GPU{args.gpu} not idle enough: {free}MiB")
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/"worker.lock").open("a+") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        budget=OUT/"budget.json"
        ledger=json.loads(budget.read_text()) if budget.exists() else dict(
            task=CFG["task"],first_start=time.time(),gpu_seconds=0.,
            forwards=0,sampling=0,commits=0,vae=0,runs=[])
        started=time.time();ledger["active_start"]=started
        atomic_json(budget,ledger)
        remaining=min(CFG["max_gpu_hours"]*3600-ledger["gpu_seconds"],
                      datetime.fromisoformat(CFG["deadline_hkt"]).timestamp()-started)
        if remaining<=0: raise TimeoutError("EXP-010 GPU budget/deadline")
        def alarm(_signum,_frame): raise TimeoutError("EXP-010 absolute alarm")
        signal.signal(signal.SIGALRM,alarm)
        signal.setitimer(signal.ITIMER_REAL,remaining)
        os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu),ABOT_VRAM_RESERVE_GIB="18",
            HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",DIFFSYNTH_SKIP_DOWNLOAD="True",
            TOKENIZERS_PARALLELISM="false",PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
        status="failed"
        try:
            run(args.stage,args.path,args.gpu,ledger)
            status="complete_pending_visual_review"
        finally:
            signal.setitimer(signal.ITIMER_REAL,0)
            ledger["gpu_seconds"]+=time.time()-started
            ledger.pop("active_start",None)
            ledger["runs"].append(dict(stage=args.stage,path=args.path,gpu=args.gpu,
                                       start=started,end=time.time(),status=status))
            atomic_json(budget,ledger)


if __name__=="__main__": main()
