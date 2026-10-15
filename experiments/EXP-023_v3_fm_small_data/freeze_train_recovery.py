"""CPU freeze of authorized EXP-023 training retry and completed pre-eval."""
from __future__ import annotations

import json
from common import HERE, ROOT, sha, write


def main():
    auth=json.loads((HERE/"AUTHORIZATION_ADDENDUM_05.json").read_text())
    assert auth["approved"] and auth["task"]=="EXP-023/v1"
    assert sha(HERE/"ADDENDUM_05_TRAIN_HOOK_RECOVERY.md")==auth["addendum_sha256"]
    assert sha(HERE/"AUTHORIZATION_ADDENDUM_04.json")==auth["previous_authorization_sha256"]
    assert sha(HERE/"judge/HOOK_SCOPE_CPU_REVIEW.json")==auth["judge_cpu_review_sha256"]
    assert sha(HERE/"HOOK_LIFECYCLE_CPU_V2.json")==auth["worker_cpu_review_sha256"]
    failed=json.loads((HERE/"TRAIN_BUDGET.json").read_text())
    assert failed["status"]=="FAILED_STOPPED"
    assert (failed["counts"]["dit"],failed["counts"]["backward"],failed["counts"]["update"])==(2,0,0)
    assert abs(failed["wall_seconds"]-auth["failed_training_costs_to_retain"]["gpu_seconds"])<1e-6
    pre=json.loads((HERE/"EVAL_PRE_RETRY2_BUDGET.json").read_text())
    assert pre["status"]=="COMPLETE_PENDING_JUDGE"
    assert (pre["counts"]["dit"],pre["counts"]["vae_decode"])==(82,6)
    checkpoint=json.loads((HERE/"checkpoint_000.json").read_text())
    assert sha(HERE/"checkpoint_000.pt")==checkpoint["sha256"]
    videos=sorted((HERE/"eval_before").glob("*.mp4"))
    assert len(videos)==6
    for video in videos:
        assert all(video.with_suffix(s).is_file() for s in
                   (".identity.json",".media.json",".rgb.npy"))
    names=("AUTHORIZATION_ADDENDUM_05.json","ADDENDUM_05_TRAIN_HOOK_RECOVERY.md",
           "judge/HOOK_SCOPE_CPU_REVIEW.json","HOOK_LIFECYCLE_CPU_V2.json",
           "hook_lifecycle_cpu.py","GRADIENT_HOOK_DIFF.patch","fm_run.py",
           "resume_run_v3.py","freeze_train_recovery.py","TRAIN_BUDGET.json",
           "TRAIN.log","EVAL_PRE_RETRY2_BUDGET.json","EVAL_PRE_RETRY2_RESULT.json",
           "TRAIN_SOURCE_MANIFEST.json","RECOVERY_SOURCE_MANIFEST.json",
           "RECOVERY_SOURCE_MANIFEST_V2.json","checkpoint_000.pt","checkpoint_000.json",
           "attempt_02/fm_run.py","attempt_02/resume_run_v2.py","attempt_02/README.md",
           "eval_before/results.json","eval_before/probes.json")
    paths=[HERE/n for n in names]
    paths += [video.with_suffix(s) for video in videos for s in
              (".mp4",".identity.json",".media.json",".rgb.npy")]
    manifest={str(path.relative_to(ROOT)):sha(path) for path in paths}
    path=HERE/"TRAIN_RECOVERY_SOURCE_MANIFEST.json"
    if path.exists():raise FileExistsError(path)
    write(path,manifest)
    print("FROZEN TRAIN RECOVERY",sha(path),"pre_videos",len(videos))


if __name__=="__main__":main()
