"""CPU-only EXP-023 immutable selection/provenance and raw RGB protocol audit."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

import av
import numpy as np
from PIL import Image

from common import (HERE, ROOT, EXP021, EXP022, VAE_OLD, COND_OLD,
                    inputs, new_rows, raw_source_rgb, sha, write)


def main():
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in ("", "-1"):
        raise RuntimeError("CPU-only preflight")
    m,s,a,train,dev,roles=inputs()
    pilot,rest=new_rows()
    assert len(pilot)==2 and len(rest)==32
    old=[r for r in train.values() if r["materialized_real56"]]
    assert len(old)==4
    # EXP-016's existing four were encoded from EXP-015's re-encoded real56.
    # Addendum 01 requires fresh source-RGB encoding for all selected clips.
    for r in old:
        scene=r["episode_id"][:8]
        assert (VAE_OLD/scene/"full17.pt").is_file()
        assert (COND_OLD/f"{scene}.pt").is_file()
    source_by_episode={}
    all_new=pilot+rest
    for r in all_new:
        if r["episode_id"] not in source_by_episode:
            p=ROOT/r["source_video"]
            assert sha(p)==r["source_video_sha256"]
            source_by_episode[r["episode_id"]]=r["source_video_sha256"]
        assert len(r["source_indices"])==56 and len(r["action_script17"])==17
        materialized=(ROOT/r["materialized_real56"] if r["materialized_real56"] else
                      EXP022/"artifacts/data"/r["clip_id"]/"real56.mp4")
        assert materialized.is_file()
    rgb=raw_source_rgb(pilot[0])
    with av.open(str(EXP022/"artifacts/data"/pilot[0]["clip_id"]/"real56.mp4")) as c:
        c.streams.video[0].thread_count=1
        h264=np.stack([f.to_ndarray(format="rgb24") for f in c.decode(video=0)])
    assert h264.shape==rgb.shape
    mad=float(np.abs(rgb.astype(np.float32)-h264.astype(np.float32)).mean())
    # CPU verify the known train synthetic canonical native layout exactly
    # reconstructs EXP-019 E, so the same construction is applicable to dev.
    frozen=ROOT/"H3-World/outputs/2026-10-09-22/chunk_partition_cb/runtime"
    sys.path.insert(0,str(ROOT/"submission/experiments/EXP-017_v3_fixed_action_layout"))
    sys.path.insert(0,str(frozen/"DiffSynth-Studio-h3-v2"))
    import torch
    from fixed_layout import frozen_cpu_functions,candidate,action_ids,same_layout,TOKENIZER_DIR
    from transformers import AutoTokenizer
    b,_=frozen_cpu_functions();tok=AutoTokenizer.from_pretrained(str(TOKENIZER_DIR),local_files_only=True)
    f=torch.load(COND_OLD/"b784d995.pt",map_location="cpu",weights_only=True)
    oldf=torch.load(ROOT/"H3-World/outputs/EXP-014_v3_multiscene_teacher/fixtures/s3_b784d995.pt",map_location="cpu",weights_only=True)
    a_sentence=oldf["action_sentences"]["A"]
    head=f["packed"]["action_text_spans_local"][0][0]
    alen=len(action_ids(tok,[a_sentence])[0]);spans=[(head+i*alen,head+(i+1)*alen) for i in range(37)]
    canon=b._build_packed_fl2va(head+37*alen,37,30,52,len(f["packed"]["audio_pos"])//2,[0],action_text_spans=spans)
    canon["token_tags"][:head]=f["packed"]["token_tags"][:head]
    p,_=candidate(b,canon,action_ids(tok,f["script"]+[a_sentence]*20))
    assert same_layout(p,f["packed"])
    selection_sha=sha(HERE/"SELECTION.json")
    sources={str((HERE/x).relative_to(ROOT)):sha(HERE/x) for x in ("AUTHORIZATION.json","taskbook_v1.md","SELECTION.json")}
    for x in ("ADDENDUM_01_SOURCE_RGB.md","AUTHORIZATION_ADDENDUM_01.json"):
        sources[str((HERE/x).relative_to(ROOT))]=sha(HERE/x)
    sources[str((EXP021/"DATA_MANIFEST.json").relative_to(ROOT))]=sha(EXP021/"DATA_MANIFEST.json")
    sources[str((EXP022/"RGB_MATERIALIZATION.json").relative_to(ROOT))]=sha(EXP022/"RGB_MATERIALIZATION.json")
    sources[str((frozen/"code/causal/h3_cached.py").relative_to(ROOT))]=sha(frozen/"code/causal/h3_cached.py")
    sources[str((frozen/"DiffSynth-Studio-h3-v2/diffsynth/pipelines/minimax_h3_audio_video.py").relative_to(ROOT))]=sha(frozen/"DiffSynth-Studio-h3-v2/diffsynth/pipelines/minimax_h3_audio_video.py")
    write(HERE/"SOURCE_MANIFEST.json",sources)
    write(HERE/"CPU_PREFLIGHT.json",{"status":"PASS_CPU_SELECTION_AND_LAYOUT","gpu_calls":0,
        "train":32,"dev_selected":2,"new_train":28,"new_dev":2,"old_train_source_reencoded":4,
        "pilot_clip_ids":[r["clip_id"] for r in pilot],
        "rest_count":32,"episode_disjoint":True,"canonical_b784d995_reconstructed_exact":True,
        "pilot_source_vs_h264_rgb_mad":mad,
        "source_protocol_note":"all selected 34 directly decoded from source under addendum 01; old four EXP-016 H264-derived latents retained only for separate regression, not training",
        "selection_sha256":selection_sha,"source_manifest_sha256":sha(HERE/"SOURCE_MANIFEST.json")})
    print("PREFLIGHT PASS",[r["clip_id"] for r in pilot],"source/H264 MAD",mad,flush=True)


if __name__=="__main__":main()
