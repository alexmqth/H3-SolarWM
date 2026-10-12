"""Finish V0 metadata after the first wrapper's FPS-text-only assertion.

The 33B benchmark returned code 0. Both real MP4s already exist and decode.
This script makes no GPU call and does not alter either video or raw result.
"""
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(HERE))
from run_row import ffprobe

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda:f.read(4<<20),b''):h.update(part)
    return h.hexdigest()

def atomic(path,value):
    tmp=path.with_suffix(path.suffix+'.partial')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    os.replace(tmp,path)

out=ROOT/'H3-World/outputs/EXP-018_unified_5s_benchmark/v0'
full=out/'baseline.mp4';display=out/'display_120.mp4'
assert not (out/'metrics.json').exists()
launch=json.loads((out/'launch.json').read_text())
raw=json.loads((out/'baseline.json').read_text())
setup=json.loads((out/'setup.json').read_text())
assert raw['status']=='complete' and raw['denoiser_forwards']==30
f=ffprobe(full);d=ffprobe(display)
assert (f['codec_name'],f['nb_frames'],f['width'],f['height'],f['r_frame_rate'])==('h264',124,832,480,'24/1')
assert (d['codec_name'],d['nb_frames'],d['width'],d['height'],d['r_frame_rate'],d['duration_seconds'])==('h264',120,832,480,'24/1',5.)
start=dt.datetime.fromisoformat(launch['started_at']).timestamp()
approx=full.stat().st_mtime-start
assert 0<approx<2700
summary=dict(row='v0',status='complete_pending_judge',returncode=0,gpu=launch['gpu'],
    gpu_uuid=launch['gpu_uuid'],initial_free_mib=launch['initial_free_mib'],
    command=launch['command'],source_manifest_sha256_at_launch=launch['source_manifest_sha256'],
    source_manifest_sha256_at_finalize=sha(HERE/'SOURCE_MANIFEST.json'),
    raw_metrics=str((out/'baseline.json').relative_to(ROOT)),
    full_video=str(full.relative_to(ROOT)),display_video=str(display.relative_to(ROOT)),
    full_sha256=sha(full),display_sha256=sha(display),full_ffprobe=f,display_ffprobe=d,
    cold_total_seconds=approx,
    cold_total_source='full_MP4_mtime minus launch.json started_at; ~1-second filesystem precision after wrapper FPS-string postprocess assertion',
    postprocess_recovery='PyAV had returned 24 instead of string 24/1 in the in-memory first wrapper; video and model result already complete; no GPU rerun',
    sampling_nfe_total=30,commit_forward_total=0,prefill_forward_total=0,total_forward_count=30,
    sampling_steps_per_chunk=30,
    peak_allocated_gib=raw['memory_entire_run_peak_MiB']/1024,
    peak_reserved_gib=max(setup['memory']['reserved_peak_MiB'],raw['memory_after_setup']['reserved_peak_MiB'])/1024,
    cpu_kv_peak_gib=0.,ready_to_video_seconds=raw['wall_after_conditioning_seconds'],
    load_seconds=setup['load_models_and_lora_seconds'],conditioning_seconds=setup['conditioning_seconds'],
    sampling_loop_inclusive_seconds=raw['sampling_seconds'],decode_seconds=raw['vae_decode_seconds'],
    encode_seconds=raw['encode_mp4_seconds'],
    video_noise_sha256=setup['input_fingerprints']['video_noise_sha256'])
atomic(out/'metrics.json',summary)
index_path=HERE/'results_index.json';index=json.loads(index_path.read_text());index['rows']['v0']=summary
atomic(index_path,index)
print(json.dumps({'status':summary['status'],'cold_total_seconds':approx,
    'full_frames':f['nb_frames'],'display_frames':d['nb_frames'],'full_sha256':summary['full_sha256']}))
