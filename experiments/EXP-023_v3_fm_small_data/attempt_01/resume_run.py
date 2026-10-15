"""Append-only recovery entry after the recorded missing-ffmpeg pre-eval failure.

This file leaves fm_run.py, its frozen source manifest, checkpoint0 and the
failed EVAL_PRE_BUDGET untouched. GPU execution requires Judge addendum 03.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import fm_run as fm
from common import HERE, ROOT, sha, write


class RecoveryBudget(fm.Budget):
    ORDER=("encode_pilot","encode_rest","eval_pre","eval_pre_retry","train","eval_post")
    CAPS={**fm.Budget.CAPS,"vae_decode":13}

    def __init__(self,phase,gpu):
        self.phase,self.gpu=phase,gpu
        self.path=HERE/f"{phase.upper()}_BUDGET.json"
        if self.path.exists():raise FileExistsError(self.path)
        self.previous=[]
        for p in self.ORDER[:self.ORDER.index(phase)]:
            item=json.loads((HERE/f"{p.upper()}_BUDGET.json").read_text())
            if p=="eval_pre":
                assert item["status"]=="FAILED_STOPPED"
                assert item["counts"]["dit"]==13 and item["counts"]["vae_decode"]==1
                assert item["counts"]["backward"]==item["counts"]["update"]==0
            elif item["status"]!="COMPLETE_PENDING_JUDGE":
                raise RuntimeError(f"previous phase {p}: {item['status']}")
            self.previous.append(item)
        self.started=time.monotonic()
        self.wall_cap={"eval_pre_retry":1800,"train":6000,"eval_post":1800}[phase]
        self.counts={k:0 for k in self.CAPS}
        self.events=[];self.status="RUNNING"
        self.save()

    def reserve(self,kind,note):
        if self.phase in ("eval_pre_retry","eval_post"):
            spent=time.monotonic()-self.started+sum(x["wall_seconds"] for x in self.previous
                if x["phase"] in ("eval_pre","eval_pre_retry","eval_post"))
            if spent>=3570:
                raise TimeoutError("combined evaluation GPU-seconds budget")
        return super().reserve(kind,note)


def verify_recovery():
    # Preserve the original frozen source revision as attempt_00, while the
    # live fm_run.py may differ only in media persistence/writing.
    old_manifest=json.loads((HERE/"TRAIN_SOURCE_MANIFEST.json").read_text())
    old_runner_key=str((HERE/"fm_run.py").relative_to(ROOT))
    assert sha(HERE/"attempt_00/fm_run.py")==old_manifest[old_runner_key]
    for relative,expected in old_manifest.items():
        if relative==old_runner_key:continue
        if sha(ROOT/relative)!=expected:
            raise RuntimeError(f"original training source changed: {relative}")
    a=HERE/"AUTHORIZATION_ADDENDUM_03.json"
    d=HERE/"ADDENDUM_03_VIDEO_IO_RECOVERY.md"
    if not a.exists() or not d.exists():
        raise RuntimeError("Judge's explicit recovery addendum missing")
    auth=json.loads(a.read_text())
    assert auth["task"]=="EXP-023/v1" and auth["approved"]
    assert sha(d)==auth["addendum_sha256"]
    assert sha(HERE/"AUTHORIZATION_ADDENDUM_02.json")==auth["previous_authorization_sha256"]
    assert auth["effective_changed_limit"]["vae_decode"]==13
    manifest=HERE/"RECOVERY_SOURCE_MANIFEST.json"
    if not manifest.exists():raise RuntimeError("recovery source manifest missing")
    for relative,expected in json.loads(manifest.read_text()).items():
        if sha(ROOT/relative)!=expected:
            raise RuntimeError(f"recovery source changed: {relative}")
    import imageio_ffmpeg
    binary=Path(imageio_ffmpeg.get_ffmpeg_exe()).resolve()
    if not binary.is_file():raise RuntimeError("imageio-ffmpeg executable missing")
    pinned=json.loads((HERE/"RECOVERY_BINARY.json").read_text())
    assert str(binary)==pinned["path"] and sha(binary)==pinned["sha256"]
    smoke=json.loads((HERE/"MEDIA_SMOKE.json").read_text())
    assert smoke["status"]=="PASS_CPU_REAL_SOURCE_CODEC_ONLY" and smoke["gpu_calls"]==0
    return sha(manifest)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--phase",choices=("pre_retry","train","post"),required=True)
    parser.add_argument("--gpu",type=int,required=True)
    args=parser.parse_args()
    phase={"pre_retry":"eval_pre_retry","train":"train","post":"eval_post"}[args.phase]
    if os.environ.get("CUDA_VISIBLE_DEVICES")!=str(args.gpu):
        raise RuntimeError("GPU selector mismatch")
    import run as prior
    prior.setup()
    import torch
    torch.set_num_threads(4)
    source_sha=verify_recovery()
    budget=RecoveryBudget(phase,args.gpu)
    try:
        pipe,model,adapters,params,precision=budget.measure("load_model",fm.setup_model)
        budget.save(precision=precision,recovery_source_manifest_sha256=source_sha)
        if args.phase=="pre_retry":
            # The failed call had not written any result or MP4. Preserve its
            # budget and checkpoint; only remove its empty output directory.
            empty=HERE/"eval_before"
            assert empty.is_dir() and not any(empty.iterdir())
            empty.rmdir()
            zero=fm.load_checkpoint(HERE/"checkpoint_000.pt",adapters)
            assert zero["step"]==0
            fm.evaluate(pipe,model,budget,"eval_pre")
        elif args.phase=="train":
            fm.train(model,adapters,params,budget)
        else:
            final=fm.load_checkpoint(HERE/"checkpoint_128.pt",adapters)
            assert final["step"]==128
            fm.evaluate(pipe,model,budget,"eval_post")
        write(HERE/f"{phase.upper()}_RESULT.json",dict(status="COMPLETE_PENDING_JUDGE",
             phase=phase,counts=budget.counts,wall_seconds=time.monotonic()-budget.started,
             failed_pre_attempt_included=True))
        budget.close("COMPLETE_PENDING_JUDGE")
        print("COMPLETE",phase,budget.counts,flush=True)
    except BaseException as error:
        budget.close("FAILED_STOPPED",repr(error))
        print("STOPPED",phase,repr(error),flush=True)
        raise


if __name__=="__main__":main()
