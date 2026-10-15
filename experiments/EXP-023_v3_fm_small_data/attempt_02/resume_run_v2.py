"""Final authorized EXP-023 recovery; append-only accounting for two failed attempts."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import fm_run as fm
from common import HERE, ROOT, sha, write


class RecoveryBudget(fm.Budget):
    ORDER=("encode_pilot","encode_rest","eval_pre","eval_pre_retry",
           "eval_pre_retry2","train","eval_post")
    CAPS={**fm.Budget.CAPS,"dit":449,"vae_decode":14}

    def __init__(self,phase,gpu):
        self.phase,self.gpu=phase,gpu
        self.path=HERE/f"{phase.upper()}_BUDGET.json"
        if self.path.exists():raise FileExistsError(self.path)
        self.previous=[]
        for p in self.ORDER[:self.ORDER.index(phase)]:
            item=json.loads((HERE/f"{p.upper()}_BUDGET.json").read_text())
            if p in ("eval_pre","eval_pre_retry"):
                assert item["status"]=="FAILED_STOPPED"
                expected=(13,1) if p=="eval_pre" else (9,1)
                assert (item["counts"]["dit"],item["counts"]["vae_decode"])==expected
                assert item["counts"]["backward"]==item["counts"]["update"]==0
            elif item["status"]!="COMPLETE_PENDING_JUDGE":
                raise RuntimeError(f"previous phase {p}: {item['status']}")
            self.previous.append(item)
        self.started=time.monotonic()
        self.wall_cap={"eval_pre_retry2":1800,"train":6000,"eval_post":1800}[phase]
        self.counts={k:0 for k in self.CAPS}
        self.events=[];self.status="RUNNING"
        self.save()

    def reserve(self,kind,note):
        if self.phase in ("eval_pre_retry2","eval_post"):
            spent=time.monotonic()-self.started+sum(x["wall_seconds"] for x in self.previous
                if x["phase"] in ("eval_pre","eval_pre_retry","eval_pre_retry2","eval_post"))
            if spent>=3570:raise TimeoutError("combined evaluation GPU-seconds budget")
        return super().reserve(kind,note)


def verify_sources():
    old=json.loads((HERE/"TRAIN_SOURCE_MANIFEST.json").read_text())
    old_key=str((HERE/"fm_run.py").relative_to(ROOT))
    assert sha(HERE/"attempt_00/fm_run.py")==old[old_key]
    for relative,expected in old.items():
        if relative!=old_key and sha(ROOT/relative)!=expected:
            raise RuntimeError(f"original source changed: {relative}")
    first=json.loads((HERE/"RECOVERY_SOURCE_MANIFEST.json").read_text())
    assert sha(HERE/"attempt_01/fm_run.py")==first[old_key]
    for relative,expected in first.items():
        if relative!=old_key and sha(ROOT/relative)!=expected:
            raise RuntimeError(f"first recovery source changed: {relative}")
    auth=json.loads((HERE/"AUTHORIZATION_ADDENDUM_04.json").read_text())
    assert auth["approved"] and auth["task"]=="EXP-023/v1"
    assert sha(HERE/"ADDENDUM_04_BFLOAT16_RECOVERY.md")==auth["addendum_sha256"]
    assert sha(HERE/"AUTHORIZATION_ADDENDUM_03.json")==auth["previous_authorization_sha256"]
    assert sha(HERE/"judge/MEDIA_RECOVERY_CPU_REVIEW.json")==auth["judge_cpu_review_sha256"]
    assert sha(HERE/"MEDIA_SMOKE_V3.json")==auth["worker_cpu_review_sha256"]
    assert auth["effective_changed_limits"]=={"top_level_forward":449,"vae_decode":14}
    manifest=HERE/"RECOVERY_SOURCE_MANIFEST_V2.json"
    if not manifest.exists():raise RuntimeError("second recovery source manifest missing")
    for relative,expected in json.loads(manifest.read_text()).items():
        if sha(ROOT/relative)!=expected:
            raise RuntimeError(f"second recovery source changed: {relative}")
    import imageio_ffmpeg
    executable=Path(imageio_ffmpeg.get_ffmpeg_exe()).resolve()
    pinned=json.loads((HERE/"RECOVERY_BINARY.json").read_text())
    assert str(executable)==pinned["path"] and sha(executable)==pinned["sha256"]
    return sha(manifest)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--phase",choices=("pre_retry2","train","post"),required=True)
    parser.add_argument("--gpu",type=int,required=True)
    args=parser.parse_args()
    phase={"pre_retry2":"eval_pre_retry2","train":"train","post":"eval_post"}[args.phase]
    if os.environ.get("CUDA_VISIBLE_DEVICES")!=str(args.gpu):
        raise RuntimeError("GPU selector mismatch")
    import run as prior
    prior.setup()
    import torch
    torch.set_num_threads(4)
    source_sha=verify_sources()
    budget=RecoveryBudget(phase,args.gpu)
    try:
        pipe,model,adapters,params,precision=budget.measure("load_model",fm.setup_model)
        budget.save(precision=precision,recovery_source_manifest_sha256=source_sha)
        if args.phase=="pre_retry2":
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
             failed_eval_attempts_included=2))
        budget.close("COMPLETE_PENDING_JUDGE")
        print("COMPLETE",phase,budget.counts,flush=True)
    except BaseException as error:
        budget.close("FAILED_STOPPED",repr(error))
        print("STOPPED",phase,repr(error),flush=True)
        raise


if __name__=="__main__":main()
