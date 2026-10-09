"""EXP-003: resume accepted EXP-002 AA/AD raw KV to 124 RGB frames.

One path per process; each new chunk pauses for visual review before the next.
No training, changed topology, ancestor rebuild, or final unnecessary commit.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CFG = json.loads((HERE/"config.json").read_text())
OUT = Path(CFG["output_root"])
PREV = ROOT/"H3-World/outputs/EXP-002_native_cached"
PREV_CODE = ROOT/"submission/experiments/EXP-002_native_cached"
FROZEN = ROOT/"H3-World/outputs/2026-10-09-22/chunk_partition_cb"
OLD = ROOT/"submission/experiments/EXP-001_v2b_124"
CANDIDATE = ROOT/"submission/reports/stage1_anyflow/02_causal_diagnostics/current_prefix_candidate"
FIRST = ROOT/"submission/experiments/11_causal_12_then5_selfhistory/states/first12_A.pt"


def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for block in iter(lambda:f.read(4<<20),b""):h.update(block)
    return h.hexdigest()


def th(x):
    import torch
    x=x.detach().cpu().contiguous()
    return hashlib.sha256(str((tuple(x.shape),str(x.dtype))).encode()+
                          x.view(torch.uint8).numpy().tobytes()).hexdigest()


def atomic_json(path,value):
    tmp=Path(str(path)+".partial")
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+"\n")
    os.replace(tmp,path)


def source_manifest(path):
    manifest=json.loads((OLD/"source_manifest_v3.json").read_text())
    files={"first12":FIRST,
           "released_action_lora":FROZEN/"runtime/checkpoints/H3-World/step-10000.safetensors",
           "accepted_interval":PREV_CODE/"interval_cached.py",
           "accepted_router":CANDIDATE/"current_prefix.py",
           "accepted_config":PREV_CODE/"config.json",
           "accepted_budget":PREV/"budget.json"}
    for p in ("AA","AD"):
        files.update({f"{p}_second_endpoint":PREV/p/"chunk_12_17.pt",
                      f"{p}_third_endpoint":PREV/p/"chunk_17_22.pt",
                      f"{p}_cache_through17":PREV/p/"cache_through17.pt",
                      f"{p}_published73":PREV/p/"published_73.npy",
                      f"{p}_second_json":PREV/p/"chunk_12_17.json",
                      f"{p}_third_json":PREV/p/"chunk_17_22.json"})
    hashes={name:sha(file) for name,file in files.items()}
    assert hashes["first12"]=="242a1db06bc3423fe21207af5d2ccf59c9f27f07c9d88f22ce9f77921f6b0eea"
    assert hashes["released_action_lora"]==manifest["released_lora_sha256"]
    for p in ("AA","AD"):
        second=json.loads(files[f"{p}_second_json"].read_text())
        third=json.loads(files[f"{p}_third_json"].read_text())
        assert hashes[f"{p}_second_endpoint"]==second["endpoint_sha256"]
        assert hashes[f"{p}_third_endpoint"]==third["endpoint_sha256"]
        assert hashes[f"{p}_cache_through17"]==third["cache_through17_sha256"]
        assert hashes[f"{p}_published73"]==third["published_file_sha256"]
        assert third["status"]=="complete_pending_visual_review"
    return {name:{"path":str(files[name]),"sha256":digest} for name,digest in hashes.items()}


def reserve_call(ledger,kind):
    now=time.time()
    if now-ledger["first_start"]>=CFG["max_elapsed_hours"]*3600:
        raise TimeoutError("EXP-003 elapsed budget")
    if ledger["gpu_seconds"]+now-ledger["active_start"]>=CFG["max_gpu_hours"]*3600:
        raise TimeoutError("EXP-003 GPU-hour budget")
    if ledger["total_forwards"]>=CFG["max_forwards"]:
        raise RuntimeError("EXP-003 total forward budget")
    if kind=="sampling":
        if ledger["sampling"]>=CFG["max_sampling"]:raise RuntimeError("sampling budget")
        ledger["sampling"]+=1
    elif kind=="commit":
        if ledger["commits"]>=CFG["max_commits"]:raise RuntimeError("commit budget")
        ledger["commits"]+=1
    else:raise ValueError(kind)
    ledger["total_forwards"]+=1  # before call, failed attempts included
    atomic_json(OUT/"budget.json",ledger)


def cache_info(cache):
    assert len(cache.layers)==50
    values=[[(e.index,int(e.key.shape[0])) for e in entries]
            for entries in cache.layers.values()]
    assert all(x==values[0] for x in values)
    return dict(layer_count=len(values),entries_per_layer=values[0],
                tokens_per_layer=sum(n for _,n in values[0]),
                nbytes=cache.nbytes,peak_bytes=cache.peak_bytes,
                commit_calls_per_layer=cache.commits//len(values))


def cache_identity(cache):
    return (cache.commits,tuple((layer,tuple(
        (e.index,id(e),tuple((id(t),t.data_ptr(),t._version,tuple(t.shape))
             for t in (e.key,e.value,e.rope))) for e in entries))
        for layer,entries in sorted(cache.layers.items())))


def peak(torch):
    value=torch.cuda.max_memory_allocated()/2**20
    if value>CFG["max_allocated_gib"]*1024:
        raise MemoryError(f"GPU allocated peak {value:.1f} MiB exceeds 44 GiB")
    return value


def decode(pipe,history):
    import numpy as np
    import torch
    with torch.no_grad():
        rgb=pipe.video_vae.decode_video(history,dtype=pipe.torch_dtype,
            process_image=False,tiled=True,tile_size=256,tile_overlap=64)
        frames=pipe.vae_output_to_video(rgb,min_value=0,max_value=1)
    return np.stack([np.asarray(im.convert("RGB"),dtype=np.uint8) for im in frames])


def log(path,msg):
    stamped=f"{datetime.now().astimezone().isoformat()} {msg}"
    print(stamped,flush=True)
    with open(path,"a") as f:f.write(stamped+"\n")


def run(args,ledger,sources):
    import numpy as np
    import torch
    from PIL import Image
    import infer as abot
    from benchmark import write_video
    from causal.h3_cached import H3ChunkCache
    from causal.h3_precision import configure_precision
    from causal.local_topology import visible_inputs
    from causal.anyflow_sampling import configure_video_schedule
    from evaluate_action_control import evaluate as flow_metrics
    from metered_prefix import current_prefix_feedback, METER
    from interval_cached import interval_cached
    from rollout_contract import prompt_for_path,rgb_sha,stitch_immutable,rgb_stop

    torch.set_num_threads(4);torch.manual_seed(13)
    path_dir=OUT/args.path;path_dir.mkdir(exist_ok=True)
    events=path_dir/"events.log"
    assert not (path_dir/"chunk_22_27.json").exists(),"prior attempt requires separate audit"
    began=time.perf_counter()
    data={a:torch.load(FROZEN/f"source_coarse/inputs/parking_{a}.pt",
                       map_location="cpu",weights_only=True) for a in "AD"}
    for k in ("initial_noise","audio_noise","anchor"):
        assert torch.equal(data["A"][k],data["D"][k]),k
    pipe=abot.load_pipeline("cuda:0")
    released=FROZEN/"runtime/checkpoints/H3-World/step-10000.safetensors"
    pipe.load_lora(pipe.dit,state_dict=abot.load_checkpoint_lora(released),hotload=True)
    model=pipe.dit.eval().requires_grad_(False)
    precision=configure_precision(model,"h3_fp32",native_transformer_dir=
        FROZEN/"runtime/DiffSynth-Studio-h3-v2/models/MiniMax/MiniMax-H3/FL2VA/transformer")
    pipe.load_models_to_device(["dit"]);torch.cuda.synchronize()
    load_seconds=time.perf_counter()-began
    torch.cuda.reset_peak_memory_stats()
    versions=[(p,p._version) for p in model.parameters()]
    def move(x):
        if torch.is_tensor(x):return x.to("cuda:0")
        if isinstance(x,dict):return {k:move(v) for k,v in x.items()}
        return x
    initial=data["A"]["initial_noise"].to("cuda:0",torch.float32)
    audio=data["A"]["audio_noise"].to("cuda:0")
    anchor=data["A"]["anchor"].to("cuda:0")
    full=move(data["A"]["packed"])
    previous=json.loads((PREV/args.path/"chunk_17_22.json").read_text())
    assert th(initial)==previous["initial_noise_sha256"]
    assert th(anchor)==previous["anchor_sha256"] and th(audio)==previous["audio_sha256"]
    parts=[torch.load(f,map_location="cpu",weights_only=True).to("cuda:0",torch.float32)
           for f in (FIRST,PREV/args.path/"chunk_12_17.pt",PREV/args.path/"chunk_17_22.pt")]
    history=torch.cat(parts,dim=2)
    assert history.shape==(1,24,22,30,52)
    cache_load_tick=time.perf_counter()
    cache=torch.load(PREV/args.path/"cache_through17.pt",map_location="cpu",weights_only=False)
    assert isinstance(cache,H3ChunkCache) and cache.storage_device=="cpu" and cache.max_history==5
    assert cache_info(cache)["entries_per_layer"]==[(0,4680),(1,1950)]
    cache_load_seconds=time.perf_counter()-cache_load_tick
    published=np.load(PREV/args.path/"published_73.npy")
    assert published.shape==(73,480,832,3)
    assert rgb_sha(published)==previous["published_RGB_sha256"]
    assert sha(PREV/args.path/"published_73.npy")==previous["published_file_sha256"]
    sigmas=configure_video_schedule(pipe.scheduler,steps=30,grid="native",flow_shift=2.22)
    assert sigmas==previous["sigmas"]
    log(events,f"loaded path={args.path} cache={cache_info(cache)} load_s={load_seconds:.3f} cache_load_s={cache_load_seconds:.3f}")
    source_history_sha=th(history)
    third=parts[-1]
    third_prompt=prompt_for_path(data,args.path,22).to("cuda:0")
    third_layout,third_prompt=visible_inputs(full,third_prompt,22,390)
    assert th(third_prompt)==previous["prompt_sha256"]
    METER.reset();reserve_call(ledger,"commit")
    tick=time.perf_counter()
    with current_prefix_feedback(),torch.no_grad():
        _=interval_cached(model,third,start=17,index=2,cache=cache,
            full_packed=third_layout,prompt=third_prompt,anchor=anchor,audio=audio,
            sigma=0,commit=True)
    third_transfer=METER.finish();torch.cuda.synchronize()
    third_commit_seconds=time.perf_counter()-tick
    assert cache_info(cache)["entries_per_layer"]==[(0,4680),(1,1950),(2,1950)]
    assert th(history)==source_history_sha
    third_peak=peak(torch)
    tick=time.perf_counter();initial_cache=path_dir/"cache_through22.pt"
    torch.save(cache,Path(str(initial_cache)+".partial"));os.replace(str(initial_cache)+".partial",initial_cache)
    third_cache_save_seconds=time.perf_counter()-tick
    initial_cache_sha=sha(initial_cache)
    log(events,f"third5 committed once; cache_through22_sha={initial_cache_sha} commit_s={third_commit_seconds:.3f} save_s={third_cache_save_seconds:.3f}")
    prev_stop=22
    for index,start in enumerate((22,27,32),start=3):
        chunk_begin=time.perf_counter()
        stop=start+5
        assert start==prev_stop and history.shape[2]==start
        assert all(len(v)==index for v in cache.layers.values())
        row_path=path_dir/f"chunk_{start}_{stop}.json"
        assert not row_path.exists(),"prior chunk attempt exists; no automatic retry"
        row=dict(task="EXP-003/v1",path=args.path,interval=[start,stop],index=index,
            rgb_interval=[rgb_stop(start),rgb_stop(stop)],gpu=args.gpu,status="sampling",
            model="Original H3 + released action LoRA; no new training",
            source_manifest_sha256=sha(HERE/"source_manifest.json"),
            source_third_commit_cache_sha256=initial_cache_sha,
            runner_sha256=sha(__file__),interval_sha256=sha(HERE/"interval_cached.py"),
            meter_router_sha256=sha(HERE/"metered_prefix.py"),
            accepted_router_sha256=sources["accepted_router"]["sha256"],
            precision=precision,load_seconds_shared_process=load_seconds,
            cache_initial_disk_load_seconds_shared_process=cache_load_seconds,
            first_new_commit_seconds_shared_process=third_commit_seconds,
            first_new_commit_transfer=third_transfer,
            first_new_cache_save_seconds_shared_process=third_cache_save_seconds,
            history_tensor_sha256=th(history),history_cache_before=cache_info(cache),
            cache_file_sha256_before=(initial_cache_sha if start==22 else row_last["cache_file_sha256_after"]),
            prior_rgb_sha256=rgb_sha(published),prior_rgb_frames=len(published),
            initial_noise_sha256=th(initial),anchor_sha256=th(anchor),audio_sha256=th(audio),
            sampling_forwards=0,commit_forwards=0,vae_decodes=0,
            source_checkpoint_sha256=sources["released_action_lora"]["sha256"])
        prompt=prompt_for_path(data,args.path,stop).to("cuda:0")
        layout,prompt=visible_inputs(full,prompt,stop,390)
        assert len(layout["action_text_rows"])==stop
        row["prompt_sha256"]=th(prompt);row["position_sha256"]=th(layout["img_position_ids"])
        row["sigmas"]=sigmas
        current=initial[:,:,start:stop].clone()
        noise_sha=th(current);history_sha=th(history)
        signature=cache_identity(cache)
        torch.cuda.reset_peak_memory_stats()
        tick=time.perf_counter();meter_total={k:0.0 for k in (
            "cache_read_cpu_seconds","cache_transfer_enqueue_seconds",
            "cache_transfer_gpu_seconds","cache_transfer_bytes")}
        atomic_json(row_path,row)
        try:
            with current_prefix_feedback(),torch.no_grad():
                for j,t in enumerate(pipe.scheduler.timesteps):
                    reserve_call(ledger,"sampling")
                    METER.reset()
                    velocity=interval_cached(model,current,start=start,index=index,cache=cache,
                        full_packed=layout,prompt=prompt,anchor=anchor,audio=audio,sigma=float(t)/1000)
                    measurement=METER.finish()
                    for k,v in measurement.items():meter_total[k]+=v
                    current=pipe.scheduler.step(velocity,t,current)
                    row["sampling_forwards"]+=1
                    if not torch.isfinite(current).all():raise FloatingPointError("nonfinite latent")
                    row["GPU_peak_allocated_MiB"]=peak(torch)
                    if (j+1)%5==0:
                        row["step_completed"]=j+1;atomic_json(row_path,row)
                        log(events,f"{args.path} [{start},{stop}) {j+1}/30")
            torch.cuda.synchronize();row["sampling_seconds"]=time.perf_counter()-tick
            row.update(meter_total)
            row["cache_read_unchanged"]=cache_identity(cache)==signature
            assert row["cache_read_unchanged"]
            assert th(history)==history_sha and th(initial[:,:,start:stop])==noise_sha
            endpoint=path_dir/f"chunk_{start}_{stop}.pt"
            tick=time.perf_counter();torch.save(current.detach().cpu(),Path(str(endpoint)+".partial"))
            os.replace(str(endpoint)+".partial",endpoint)
            row["endpoint_save_seconds"]=time.perf_counter()-tick
            row["endpoint_sha256"]=sha(endpoint);row["endpoint_tensor_sha256"]=th(current)
            if stop<37:
                METER.reset();reserve_call(ledger,"commit")
                tick=time.perf_counter()
                with current_prefix_feedback(),torch.no_grad():
                    _=interval_cached(model,current,start=start,index=index,cache=cache,
                        full_packed=layout,prompt=prompt,anchor=anchor,audio=audio,sigma=0,commit=True)
                row["commit_cache_transfer"]=METER.finish();torch.cuda.synchronize()
                row["clean_commit_seconds"]=time.perf_counter()-tick
                row["commit_forwards"]=1
                row["history_cache_after"]=cache_info(cache)
                assert all(len(v)==index+1 for v in cache.layers.values())
                cache_path=path_dir/f"cache_through{stop}.pt"
                tick=time.perf_counter();torch.save(cache,Path(str(cache_path)+".partial"))
                os.replace(str(cache_path)+".partial",cache_path)
                row["cache_save_seconds"]=time.perf_counter()-tick
                row["cache_file_sha256_after"]=sha(cache_path)
            else:
                row["clean_commit_seconds"]=0.0
                row["commit_forwards"]=0
                row["history_cache_after"]=cache_info(cache)
                row["cache_save_seconds"]=0.0
                row["cache_file_sha256_after"]=row["cache_file_sha256_before"]
            row["GPU_peak_allocated_MiB"]=peak(torch)
            del velocity
            pipe.load_models_to_device(["video_vae"]);torch.cuda.synchronize()
            tick=time.perf_counter()
            assert ledger["vae_decodes"]<CFG["max_decode"]
            ledger["vae_decodes"]+=1;atomic_json(OUT/"budget.json",ledger)
            decoded=decode(pipe,torch.cat((history,current),dim=2))
            row["vae_decodes"]=1
            assert len(decoded)==rgb_stop(stop)
            row["vae_seconds"]=time.perf_counter()-tick
            row["GPU_peak_allocated_MiB"]=peak(torch)
            row["past5_redecode_MAD"]=float(np.abs(decoded[rgb_stop(start)-5:rgb_stop(start)].astype(np.float32)-
                                                published[-5:].astype(np.float32)).mean())
            old_rgb_sha=rgb_sha(published)
            extended=stitch_immutable(published,decoded,start,stop)
            assert rgb_sha(extended[:rgb_stop(start)])==old_rgb_sha
            rgb_file=path_dir/f"published_{rgb_stop(stop)}.npy"
            np.save(rgb_file,extended)
            row["published_file_sha256"]=sha(rgb_file)
            row["published_rgb_sha256"]=rgb_sha(extended)
            row["past_rgb_unchanged"]=True
            video_file=path_dir/f"rollout_{rgb_stop(stop)}.mp4"
            current_video=path_dir/f"chunk_{start}_{stop}.mp4"
            tick=time.perf_counter()
            write_video([Image.fromarray(f) for f in extended],video_file)
            write_video([Image.fromarray(f) for f in extended[rgb_stop(start):]],current_video)
            row["render_seconds"]=time.perf_counter()-tick
            row["video_sha256"]=sha(video_file);row["active_video_sha256"]=sha(current_video)
            row["flow"]=flow_metrics(current_video)
            gray=np.stack([np.asarray(Image.fromarray(f).convert("L"),dtype=np.float32)
                           for f in extended[rgb_stop(start)-1:]])
            row["boundary_gray_MAD"]=float(np.abs(gray[1]-gray[0]).mean())
            row["inside_gray_MAD"]=float(np.abs(np.diff(gray[1:],axis=0)).mean())
            row["GPU_peak_allocated_MiB"]=peak(torch)
            row["frozen_parameters_unchanged"]=all(p._version==v for p,v in versions)
            assert row["frozen_parameters_unchanged"]
            row["status"]="complete_pending_visual_review"
            row["chunk_wall_seconds_before_review"]=time.perf_counter()-chunk_begin
            row["wall_seconds_since_process_start"]=time.perf_counter()-began
            atomic_json(row_path,row)
            log(events,f"DONE {args.path} RGB{rgb_stop(stop)} flow={row['flow']['horizontal_flow_px']['mean']:.4f} video={video_file}")
            row_last=row
            published=extended
            history=torch.cat((history,current),dim=2)
            prev_stop=stop
            if stop<37:
                log(events,f"REVIEW_GATE RGB{rgb_stop(stop)}: inspect video; type CONTINUE or STOP")
                review_tick=time.perf_counter()
                choice=input().strip().upper()
                row["review_wait_seconds"]=time.perf_counter()-review_tick
                if choice!="CONTINUE":
                    atomic_json(row_path,row)
                    log(events,f"stopped by review gate after RGB{rgb_stop(stop)}")
                    return "stopped_after_review"
                reload_tick=time.perf_counter()
                pipe.load_models_to_device(["dit"]);torch.cuda.synchronize()
                row["reload_dit_seconds_before_next_chunk"]=time.perf_counter()-reload_tick
                row["GPU_peak_allocated_MiB_including_reload"]=peak(torch)
                atomic_json(row_path,row)
        except BaseException as exc:
            row.update(status="failed",error=repr(exc),
                       wall_seconds_since_process_start=time.perf_counter()-began)
            atomic_json(row_path,row)
            raise
    return "complete_pending_visual_review"


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",type=Path,required=True)
    p.add_argument("--path",choices=["AA","AD"],required=True)
    p.add_argument("--gpu",type=int,required=True)
    args=p.parse_args()
    assert args.config.resolve()==(HERE/"config.json").resolve()
    assert not (OUT/args.path/"chunk_22_27.json").exists()
    sources=source_manifest(args.path)
    atomic_json(HERE/"source_manifest.json",sources)
    free=int(subprocess.check_output(["nvidia-smi",f"--id={args.gpu}",
        "--query-gpu=memory.free","--format=csv,noheader,nounits"],text=True).strip())
    assert free>=44000,("GPU not idle",args.gpu,free)
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/"worker.lock").open("a+") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        budget_file=OUT/"budget.json"
        ledger=json.loads(budget_file.read_text()) if budget_file.exists() else dict(
            first_start=time.time(),gpu_seconds=0,total_forwards=0,sampling=0,
            commits=0,vae_decodes=0,runs=[])
        assert time.time()-ledger["first_start"]<CFG["max_elapsed_hours"]*3600
        assert ledger["gpu_seconds"]<CFG["max_gpu_hours"]*3600
        started=time.time();ledger["active_start"]=started
        atomic_json(budget_file,ledger)
        os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu),ABOT_VRAM_RESERVE_GIB="18",
            HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",DIFFSYNTH_SKIP_DOWNLOAD="True",
            TOKENIZERS_PARALLELISM="false",PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
        sys.path[:0]=[str(FROZEN/"runtime/code"),str(FROZEN/"runtime/code/abot"),
            str(FROZEN/"runtime/code/causal"),str(FROZEN/"runtime/DiffSynth-Studio-h3-v2"),
            str(OLD),str(CANDIDATE)]
        try:status=run(args,ledger,sources)
        except BaseException:
            status="failed";raise
        finally:
            ledger["gpu_seconds"]+=time.time()-started
            ledger["runs"].append(dict(path=args.path,gpu=args.gpu,start=started,
                end=time.time(),status=status))
            ledger.pop("active_start",None)
            atomic_json(budget_file,ledger)


if __name__=="__main__":main()
