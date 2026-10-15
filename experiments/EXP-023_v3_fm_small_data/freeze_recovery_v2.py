"""CPU freeze of final authorized media recovery without mutating earlier manifests."""
from __future__ import annotations

import json
from common import HERE, ROOT, sha, write


def main():
    auth=json.loads((HERE/"AUTHORIZATION_ADDENDUM_04.json").read_text())
    assert auth["approved"] and auth["task"]=="EXP-023/v1"
    assert sha(HERE/"ADDENDUM_04_BFLOAT16_RECOVERY.md")==auth["addendum_sha256"]
    assert sha(HERE/"AUTHORIZATION_ADDENDUM_03.json")==auth["previous_authorization_sha256"]
    assert sha(HERE/"judge/MEDIA_RECOVERY_CPU_REVIEW.json")==auth["judge_cpu_review_sha256"]
    assert sha(HERE/"MEDIA_SMOKE_V3.json")==auth["worker_cpu_review_sha256"]
    budgets=[json.loads((HERE/n).read_text()) for n in
             ("EVAL_PRE_BUDGET.json","EVAL_PRE_RETRY_BUDGET.json")]
    assert [b["status"] for b in budgets]==["FAILED_STOPPED"]*2
    assert sum(b["counts"]["dit"] for b in budgets)==22
    assert sum(b["counts"]["vae_decode"] for b in budgets)==2
    assert abs(sum(b["wall_seconds"] for b in budgets)-auth["failure_counts_to_preserve"]["gpu_seconds"])<1e-6
    names=("AUTHORIZATION_ADDENDUM_04.json","ADDENDUM_04_BFLOAT16_RECOVERY.md",
           "TRAIN_SOURCE_MANIFEST.json","RECOVERY_SOURCE_MANIFEST.json",
           "EVAL_PRE_BUDGET.json","EVAL_PRE_RETRY_BUDGET.json",
           "checkpoint_000.pt","resume_run_v2.py","freeze_recovery_v2.py",
           "fm_run.py","finalize_cpu.py","media_smoke_v3.py","MEDIA_SMOKE_V3.json",
           "judge/MEDIA_RECOVERY_CPU_REVIEW.json","MEDIA_PATCH_DIFF_V2.patch",
           "attempt_00/fm_run.py","attempt_00/README.md",
           "attempt_01/fm_run.py","attempt_01/resume_run.py","attempt_01/README.md",
           "RECOVERY_BINARY.json")
    manifest={str((HERE/n).relative_to(ROOT)):sha(HERE/n) for n in names}
    path=HERE/"RECOVERY_SOURCE_MANIFEST_V2.json"
    if path.exists():raise FileExistsError(path)
    write(path,manifest)
    print("FROZEN RECOVERY V2",sha(path))


if __name__=="__main__":main()
