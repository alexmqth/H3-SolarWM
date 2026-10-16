"""EXP-029: one cached generated C1, four matched C2 behavior branches per scene.

No training. The only ON/OFF change is the frozen current_prefix_feedback
context around C2 sampling. C1 is committed ON once, then its raw KV is read
unchanged by all four branches.
"""
from __future__ import annotations

from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
EXP14 = ROOT / "submission/experiments/EXP-014_v3_multiscene_teacher"
OUT14 = ROOT / "H3-World/outputs/EXP-014_v3_multiscene_teacher"
OUT = ROOT / "H3-World/outputs/EXP-029_current_prefix_behavior"
sys.path.insert(0, str(EXP14))

import numpy as np
import torch
from PIL import Image, ImageDraw

from common import setup_paths, sha, atomic_json

SCENES = ("s0_43866101", "s3_b784d995")
BRANCHES = (("ON_A", True, "A"), ("ON_D", True, "AD"),
            ("OFF_A", False, "A"), ("OFF_D", False, "AD"))
CAP = 3600
SOFT_STOP = datetime.fromisoformat("2026-10-17T00:59:45+00:00")


def tensor_sha(t):
    v = t.detach().cpu().contiguous()
    return hashlib.sha256(str((tuple(v.shape),str(v.dtype))).encode() +
                          v.view(torch.uint8).numpy().tobytes()).hexdigest()


def cache_digest(cache):
    h = hashlib.sha256()
    assert cache.commits == 50 and len(cache.layers) == 50
    for layer in sorted(cache.layers):
        entries = cache.layers[layer]
        assert len(entries) == 1 and entries[0].index == 0
        for value in (entries[0].key, entries[0].value, entries[0].rope):
            tensor = value.detach().cpu().contiguous()
            h.update(str((layer,tuple(tensor.shape),str(tensor.dtype))).encode())
            h.update(memoryview(tensor.view(torch.uint8).numpy()))
    return h.hexdigest()


def rss_gib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20


def disk_bytes():
    return sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())


class Ledger:
    caps = dict(dit=242, commit=2, sampling=240, decode=8,
                backward=0, update=0, text_encode=0, vae_encode=0)

    def __init__(self):
        self.started = time.monotonic()
        self.counts = {k:0 for k in self.caps}
        self.events = []
        self.status = "RUNNING"
        self.save()

    def save(self, **extra):
        atomic_json(OUT / "GPU_LEDGER.json", dict(task="EXP-029/v1",status=self.status,
            elapsed_in_process_seconds=time.monotonic()-self.started,
            counts=self.counts,limits=self.caps,
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
            max_rss_gib=rss_gib(), disk_bytes=disk_bytes(), events=self.events, **extra))

    def reserve(self, kind, label):
        if kind not in ("commit","sampling","decode"):
            raise ValueError("non-authorized call")
        if datetime.now(timezone.utc) >= SOFT_STOP or time.monotonic()-self.started >= CAP-20:
            raise TimeoutError("EXP-029 deadline/wall cap")
        if self.counts[kind] >= self.caps[kind]:
            raise RuntimeError(f"{kind} cap")
        if kind in ("commit","sampling") and self.counts["dit"] >= self.caps["dit"]:
            raise RuntimeError("DiT cap")
        self.counts[kind] += 1
        if kind in ("commit","sampling"):
            self.counts["dit"] += 1
        self.events.append(dict(kind=kind,label=label,elapsed=time.monotonic()-self.started))
        self.save()

    def gate(self):
        torch.cuda.synchronize()
        if torch.cuda.max_memory_allocated()/2**30 >= 40 or rss_gib() >= 96:
            raise MemoryError("allocated GPU/RSS cap")
        if disk_bytes() >= 5*2**30:
            raise OSError("5GiB disk growth cap")
        if datetime.now(timezone.utc) >= SOFT_STOP or time.monotonic()-self.started >= CAP-20:
            raise TimeoutError("EXP-029 deadline/wall cap")
        self.save()


def write_mp4(frames, path, labels=None):
    import av
    path.parent.mkdir(parents=True,exist_ok=True)
    temp = path.with_suffix(".partial.mp4")
    with av.open(str(temp),"w",options={"movflags":"+faststart"}) as container:
        stream=container.add_stream("libx264",rate=24)
        stream.width,stream.height=frames[0].shape[1],frames[0].shape[0]
        stream.pix_fmt="yuv420p"; stream.options={"crf":"18","preset":"medium"}
        for arr in frames:
            for packet in stream.encode(av.VideoFrame.from_ndarray(np.asarray(arr),format="rgb24")):
                container.mux(packet)
        for packet in stream.encode(): container.mux(packet)
    temp.replace(path)
    with av.open(str(path)) as container:
        video=container.streams.video[0]
        count=sum(1 for _ in container.decode(video=0))
        fps=float(video.average_rate)
    assert count==len(frames) and abs(fps-24)<1e-6
    return dict(path=str(path.relative_to(ROOT)),sha256=sha(path),frames=count,fps=fps,
                width=frames[0].shape[1],height=frames[0].shape[0],bytes=path.stat().st_size)


def contact_sheet(boundary, generated, path):
    frames=np.concatenate((boundary[None],generated),axis=0)
    cell_w,cell_h,cols=208,120,6
    sheet=Image.new("RGB",(cols*cell_w,((len(frames)+cols-1)//cols)*(cell_h+25)),"black")
    draw=ImageDraw.Draw(sheet)
    for i,frame in enumerate(frames):
        x=(i%cols)*cell_w; y=(i//cols)*(cell_h+25)
        sheet.paste(Image.fromarray(frame).resize((cell_w,cell_h)),(x,y))
        draw.text((x+3,y+cell_h+2),"prior frame 38" if i==0 else f"new {i-1:02d}",fill="white")
    sheet.save(path,quality=92)
    return dict(path=str(path.relative_to(ROOT)),sha256=sha(path),frames_shown=len(frames))


def decode_native(pipe, history, current, ledger, label):
    pipe.load_models_to_device(["video_vae"])
    ledger.reserve("decode",label)
    torch.cuda.synchronize(); started=time.monotonic()
    with torch.no_grad():
        rgb=pipe.video_vae.decode_video(torch.cat((history,current),dim=2),
            dtype=pipe.torch_dtype,process_image=False,tiled=True,tile_size=256,tile_overlap=64)
        frames=pipe.vae_output_to_video(rgb,min_value=0,max_value=1)
    torch.cuda.synchronize()
    arr=np.stack([np.asarray(frame.convert("RGB"),dtype=np.uint8) for frame in frames])
    assert arr.shape==(56,480,832,3)
    seconds=time.monotonic()-started
    pipe.load_models_to_device(["dit"])
    return arr,seconds


def montage(scene_dir, branch_frames):
    labels=("ON / A","ON / D","OFF / A","OFF / D")
    names=("ON_A","ON_D","OFF_A","OFF_D")
    assert all(len(branch_frames[name])==56 for name in names)
    frames=[]
    for frame in range(56):
        canvas=Image.new("RGB",(1664,1000),"black")
        draw=ImageDraw.Draw(canvas)
        for i,name in enumerate(names):
            x=(i%2)*832; y=(i//2)*500
            draw.text((x+8,y+4),labels[i]+f" | frame {frame:02d} | 30 NFE",fill="white")
            canvas.paste(Image.fromarray(branch_frames[name][frame]),(x,y+20))
        frames.append(np.asarray(canvas))
    return write_mp4(frames,scene_dir/"comparison_2x2.mp4")


def main():
    gate = HERE / "START_GATE.json"
    for _ in range(600):
        if gate.exists(): break
        time.sleep(.1)
    if not gate.exists(): raise TimeoutError("process registration gate missing")
    registration=json.loads(gate.read_text())
    assert registration["python_pid"] == os.getpid()
    auth=json.loads((HERE/"AUTHORIZATION.json").read_text())
    assert int(os.environ["CUDA_VISIBLE_DEVICES"]) in auth["allowed_gpu"]
    assert registration["physical_gpu"] == int(os.environ["CUDA_VISIBLE_DEVICES"])
    assert auth["task_id"]=="EXP-029" and auth["max_top_level_dit_attempts"]==242
    cpu=json.loads((HERE/"CPU_PREFLIGHT.json").read_text())
    assert cpu["status"]=="PASS_CPU"
    for relative,expected in cpu["sha256"].items():
        if sha(ROOT/relative)!=expected:raise RuntimeError(f"source/input changed: {relative}")
    setup_paths()
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats()
    OUT.mkdir(parents=True,exist_ok=False)
    ledger=Ledger()
    signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError("worker wall alarm")))
    signal.alarm(3590)
    try:
        import run_teacher as old
        from causal.h3_cached import H3ChunkCache
        from causal.anyflow_sampling import configure_video_schedule
        from current_prefix import current_prefix_feedback
        from interval_cached import interval_cached
        pipe,model,precision=old.load_model()
        all_scenes=[]
        for scene in SCENES:
            fixture,entry=old.load_fixture(scene)
            g1_dir=OUT14/"G1"/scene/"FM30"
            g1=json.loads((g1_dir/"result.json").read_text())
            first_file=g1_dir/"first12.pt"; published_file=g1_dir/"published_39.npy"
            assert sha(first_file)==g1["endpoint_sha256"]
            assert sha(published_file)==g1["published_sha256"]
            published=np.load(published_file)
            assert published.shape==(39,480,832,3)
            first=torch.load(first_file,map_location="cpu",weights_only=True).to("cuda:0",torch.float32)
            audio=fixture["audio_noise"].to("cuda:0")
            anchor=fixture["anchor"].to("cuda:0")
            scene_dir=OUT/scene; scene_dir.mkdir()
            cache=H3ChunkCache(max_history=5,storage_device="cpu")
            layout,prompt=old.visible(fixture,"A",12)
            ledger.reserve("commit",scene)
            with torch.no_grad(),current_prefix_feedback():
                interval_cached(model,first,start=0,index=0,cache=cache,
                    full_packed=layout,prompt=prompt,anchor=anchor,audio=audio,sigma=0.,commit=True)
            assert cache.commits==50
            frozen_digest=cache_digest(cache)
            scene_row=dict(scene=scene,status="RUNNING",fixture_sha256=entry["sha256"],
                generated_first12_sha256=sha(first_file),published39_sha256=sha(published_file),
                cpu_kv_bytes=cache.nbytes,cache_sha256=frozen_digest,branches={})
            atomic_json(scene_dir/"RESULT.json",scene_row)
            ledger.gate()
            branch_frames={}
            for name,on,action in BRANCHES:
                branch_dir=scene_dir/name;branch_dir.mkdir()
                layout,prompt=old.visible(fixture,action,17)
                noise=fixture["initial_noise"][:,:,12:17].to("cuda:0",torch.float32)
                initial_sha=tensor_sha(noise)
                current=noise.clone()
                sigmas=configure_video_schedule(pipe.scheduler,steps=30,grid="native",flow_shift=2.22)
                assert len(sigmas)==31 and sigmas[0]==1 and sigmas[-1]==0
                row=dict(scene=scene,branch=name,action="A" if action=="A" else "D",
                    prefix_feedback=on,status="SAMPLING",sigmas=sigmas,
                    prompt_sha256=tensor_sha(prompt),position_sha256=tensor_sha(layout["img_position_ids"]),
                    initial_noise_sha256=initial_sha,source_cache_sha256=frozen_digest,
                    sampling_forwards=0)
                atomic_json(branch_dir/"RESULT.json",row)
                torch.cuda.synchronize();started=time.monotonic()
                context=current_prefix_feedback() if on else nullcontext()
                with torch.no_grad(),context:
                    for step,timestep in enumerate(pipe.scheduler.timesteps,1):
                        ledger.reserve("sampling",f"{scene}/{name}/{step}")
                        velocity=interval_cached(model,current,start=12,index=1,cache=cache,
                            full_packed=layout,prompt=prompt,anchor=anchor,audio=audio,
                            sigma=float(timestep)/1000)
                        current=pipe.scheduler.step(velocity,timestep,current)
                        if not bool(torch.isfinite(current).all()):raise FloatingPointError("nonfinite current latent")
                        row["sampling_forwards"]=step
                        ledger.gate()
                torch.cuda.synchronize()
                row["sampling_seconds"]=time.monotonic()-started
                assert tensor_sha(noise)==initial_sha
                assert cache_digest(cache)==frozen_digest
                endpoint=branch_dir/"endpoint5.pt"
                torch.save(current.detach().cpu(),endpoint)
                row["endpoint_sha256"]=sha(endpoint)
                decoded,row["decode_seconds"]=decode_native(pipe,first,current,ledger,f"{scene}/{name}")
                row["decoded_history_mad"]=float(np.abs(decoded[:39].astype(np.int16)-published.astype(np.int16)).mean())
                generated=decoded[39:].copy()
                assert generated.shape==(17,480,832,3)
                frozen=np.concatenate((published,generated),axis=0)
                assert np.array_equal(frozen[:39],published)
                np.save(branch_dir/"generated17.npy",generated)
                np.save(branch_dir/"published56.npy",frozen)
                row["boundary_mad"]=float(np.abs(published[-1].astype(np.int16)-generated[0].astype(np.int16)).mean())
                row["adjacent_new_mad"]=float(np.abs(generated[1:].astype(np.int16)-generated[:-1].astype(np.int16)).mean())
                row["rollout_video"]=write_mp4(frozen,branch_dir/"rollout_56.mp4")
                row["new_video"]=write_mp4(generated,branch_dir/"new_17.mp4")
                row["contact_sheet"]=contact_sheet(published[-1],generated,branch_dir/"contact_18.jpg")
                row["cache_sha256_after"]=cache_digest(cache)
                assert row["cache_sha256_after"]==frozen_digest
                row.update(status="COMPLETE_PENDING_REVIEW",peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
                atomic_json(branch_dir/"RESULT.json",row)
                scene_row["branches"][name]=str((branch_dir/"RESULT.json").relative_to(ROOT))
                atomic_json(scene_dir/"RESULT.json",scene_row)
                branch_frames[name]=frozen
                del current,noise,velocity,decoded,generated
                ledger.gate()
            scene_row["comparison"]=montage(scene_dir,branch_frames)
            scene_row["status"]="COMPLETE_PENDING_REVIEW"
            atomic_json(scene_dir/"RESULT.json",scene_row)
            all_scenes.append(scene_row)
            ledger.gate()
            del cache,fixture,first,branch_frames
        assert ledger.counts["dit"]==242 and ledger.counts["sampling"]==240
        assert ledger.counts["commit"]==2 and ledger.counts["decode"]==8
        ledger.status="COMPLETE_PENDING_JUDGE"
        ledger.save(precision=precision,scenes=[r["scene"] for r in all_scenes])
    except BaseException as exc:
        ledger.status="FAILED_PENDING_JUDGE"
        ledger.save(error=repr(exc),traceback=traceback.format_exc())
        raise
    finally:
        signal.alarm(0)


if __name__=="__main__":main()
