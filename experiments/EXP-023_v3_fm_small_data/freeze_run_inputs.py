"""CPU-only freeze of the exact 34 encoded direct-source clips and runner."""
from __future__ import annotations

import json
from pathlib import Path

from common import HERE, ROOT, ENCODED, inputs, sha, write


def main():
    _,selection,_,_,_,roles=inputs()
    addendum2=json.loads((HERE/"AUTHORIZATION_ADDENDUM_02.json").read_text())
    assert addendum2["approved"] and addendum2["task"]=="EXP-023/v1"
    assert sha(HERE/"ADDENDUM_02_CHECKPOINT_CONTEXT.md")==addendum2["addendum_sha256"]
    assert sha(HERE/"AUTHORIZATION_ADDENDUM_01.json")==addendum2["source_rgb_addendum_authorization_sha256"]
    budgets=[HERE/"ENCODE_PILOT_BUDGET.json",HERE/"ENCODE_REST_BUDGET.json"]
    if any(json.loads(p.read_text())["status"]!="COMPLETE_PENDING_JUDGE" for p in budgets):
        raise RuntimeError("encoding incomplete")
    clips=selection["train_clip_ids"]+[roles[x]["clip_id"] for x in ("dev_constant","dev_switch")]
    assert len(clips)==34 and len(set(clips))==34
    _,_,_,train,dev,_=inputs()
    pilot=json.loads((HERE/"ENCODE_PILOT_RESULT.json").read_text())["clips"]
    assert len(pilot)==2
    import torch
    for clip in pilot:
        full=torch.load(ENCODED/clip/"full17.pt",map_location="cpu",weights_only=True)
        prefix=torch.load(ENCODED/clip/"prefix12.pt",map_location="cpu",weights_only=True)
        assert torch.equal(full[:,:,:12],prefix)
    for clip in clips:
        record=json.loads((ENCODED/clip/"record.json").read_text())
        selected=(train if clip in train else dev)[clip]
        assert record["source_rgb_direct"] and not record["old_h264_latents_reused"]
        assert record["source_video_sha256"]==selected["source_video_sha256"]
        assert record["action_script_sha256"]==selected["action_script_sha256"]
        assert record["future_action_isolation"]=={"12":True,"17":True}
        if clip in pilot: assert record["pilot_prefix_exact"] is True
    paths=[HERE/n for n in ("AUTHORIZATION.json","AUTHORIZATION_ADDENDUM_01.json",
             "AUTHORIZATION_ADDENDUM_02.json","ADDENDUM_01_SOURCE_RGB.md",
             "ADDENDUM_02_CHECKPOINT_CONTEXT.md","taskbook_v1.md","SELECTION.json",
             "CPU_PREFLIGHT.json","SOURCE_MANIFEST.json","RUN_SOURCE_MANIFEST.json",
             "GRADIENT_PROTOCOL_CPU.json","CHECKPOINT_CONTEXT_CPU.json",
             "fm_run.py","interval_train.py","common.py","freeze_run_inputs.py")]
    paths += [ENCODED/clip/name for clip in clips for name in
              ("I0.pt","full17.pt","condition.pt","record.json")]
    paths += [ROOT/n for n in (
             "submission/reports/stage1_anyflow/02_causal_diagnostics/current_prefix_candidate/current_prefix.py",
             "submission/experiments/EXP-002_native_cached/interval_cached.py",
             "H3-World/outputs/2026-10-09-22/chunk_partition_cb/runtime/code/causal/h3_cached.py",
             "H3-World/outputs/2026-10-09-22/chunk_partition_cb/runtime/code/causal/local_topology.py")]
    manifest={str(p.relative_to(ROOT)):sha(p) for p in paths}
    path=HERE/"TRAIN_SOURCE_MANIFEST.json"
    if path.exists(): raise FileExistsError(path)
    write(path,manifest)
    write(HERE/"ENCODED_DATASET_AUDIT.json",dict(status="PASS_34_SOURCE_RGB",
          clips=clips,train_count=32,dev_count=2,
          pilot_prefix12_exact=pilot,old_h264_latents_reused=0,
          cumulative_counts=json.loads((HERE/"ENCODE_REST_BUDGET.json").read_text())["cumulative_counts"],
          source_manifest_sha256=sha(path)))
    print("FROZEN",len(clips),"clips",sha(path))


if __name__=="__main__":main()
