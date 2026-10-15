"""Append-only training recovery after the audit-hook ordering hard stop."""
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
           "eval_pre_retry2","train","train_retry","eval_post")
    CAPS={**fm.Budget.CAPS,"dit":449,"vae_decode":14}

    def __init__(self,phase,gpu):
        self.phase,self.gpu=phase,gpu
        self.path=HERE/f"{phase.upper()}_BUDGET.json"
        if self.path.exists():raise FileExistsError(self.path)
        self.previous=[]
        for p in self.ORDER[:self.ORDER.index(phase)]:
            item=json.loads((HERE/f"{p.upper()}_BUDGET.json").read_text())
            if p in ("eval_pre","eval_pre_retry","train"):
                assert item["status"]=="FAILED_STOPPED"
                expected={"eval_pre":(13,1,0,0),"eval_pre_retry":(9,1,0,0),
                          "train":(2,0,0,0)}[p]
                c=item["counts"]
                assert (c["dit"],c["vae_decode"],c["backward"],c["update"])==expected
            elif item["status"]!="COMPLETE_PENDING_JUDGE":
                raise RuntimeError(f"previous phase {p}: {item['status']}")
            self.previous.append(item)
        self.started=time.monotonic()
        self.wall_cap={"train_retry":6000,"eval_post":1800}[phase]
        self.counts={k:0 for k in self.CAPS}
        self.events=[];self.status="RUNNING"
        self.save()

    def reserve(self,kind,note):
        elapsed=time.monotonic()-self.started
        if self.phase=="train_retry":
            spent=elapsed+sum(x["wall_seconds"] for x in self.previous
                              if x["phase"] in ("train","train_retry"))
            if spent>=5970:raise TimeoutError("combined training GPU-seconds budget")
        if self.phase=="eval_post":
            spent=elapsed+sum(x["wall_seconds"] for x in self.previous
                    if x["phase"] in ("eval_pre","eval_pre_retry","eval_pre_retry2","eval_post"))
            if spent>=3570:raise TimeoutError("combined evaluation GPU-seconds budget")
        return super().reserve(kind,note)


def verify_sources():
    histories=(("TRAIN_SOURCE_MANIFEST.json",{"fm_run.py":"attempt_00/fm_run.py"}),
               ("RECOVERY_SOURCE_MANIFEST.json",{"fm_run.py":"attempt_01/fm_run.py"}),
               ("RECOVERY_SOURCE_MANIFEST_V2.json",
                {"fm_run.py":"attempt_02/fm_run.py",
                 "resume_run_v2.py":"attempt_02/resume_run_v2.py"}))
    for filename,replacements in histories:
        manifest=json.loads((HERE/filename).read_text())
        for relative,expected in manifest.items():
            path=ROOT/relative
            if path.parent==HERE and path.name in replacements:
                path=HERE/replacements[path.name]
            if sha(path)!=expected:
                raise RuntimeError(f"frozen source changed in {filename}: {relative}")
    auth=json.loads((HERE/"AUTHORIZATION_ADDENDUM_05.json").read_text())
    assert auth["approved"] and auth["task"]=="EXP-023/v1"
    addenda=list(HERE.glob("ADDENDUM_05_*.md"))
    assert len(addenda)==1 and sha(addenda[0])==auth["addendum_sha256"]
    assert sha(HERE/"AUTHORIZATION_ADDENDUM_04.json")==auth["previous_authorization_sha256"]
    assert sha(HERE/"judge/HOOK_SCOPE_CPU_REVIEW.json")==auth["judge_cpu_review_sha256"]
    assert sha(HERE/"HOOK_LIFECYCLE_CPU_V2.json")==auth["worker_cpu_review_sha256"]
    assert auth["effective_limits"]["dit"]==449 and auth["effective_limits"]["vae_decode"]==14
    assert json.loads((HERE/"HOOK_LIFECYCLE_CPU_V2.json").read_text())["status"]=="PASS_CPU_PRODUCTION_SCOPE_HOOK"
    manifest=HERE/"TRAIN_RECOVERY_SOURCE_MANIFEST.json"
    if not manifest.exists():raise RuntimeError("training recovery source manifest missing")
    for relative,expected in json.loads(manifest.read_text()).items():
        if sha(ROOT/relative)!=expected:
            raise RuntimeError(f"training recovery source changed: {relative}")
    return sha(manifest)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--phase",choices=("train_retry","post"),required=True)
    parser.add_argument("--gpu",type=int,required=True)
    args=parser.parse_args()
    phase="train_retry" if args.phase=="train_retry" else "eval_post"
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
        budget.save(precision=precision,train_recovery_manifest_sha256=source_sha)
        if args.phase=="train_retry":
            fm.train(model,adapters,params,budget)
        else:
            final=fm.load_checkpoint(HERE/"checkpoint_128.pt",adapters)
            assert final["step"]==128
            fm.evaluate(pipe,model,budget,"eval_post")
        write(HERE/f"{phase.upper()}_RESULT.json",dict(status="COMPLETE_PENDING_JUDGE",
             phase=phase,counts=budget.counts,wall_seconds=time.monotonic()-budget.started,
             failed_eval_attempts_included=2,failed_train_attempt_included=True))
        budget.close("COMPLETE_PENDING_JUDGE")
        print("COMPLETE",phase,budget.counts,flush=True)
    except BaseException as error:
        budget.close("FAILED_STOPPED",repr(error))
        print("STOPPED",phase,repr(error),flush=True)
        raise


if __name__=="__main__":main()
