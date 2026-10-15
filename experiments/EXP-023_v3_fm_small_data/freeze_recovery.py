"""Freeze append-only recovery source after Judge's explicit addendum 03."""
from __future__ import annotations

import json
from common import HERE, ROOT, sha, write


def main():
    auth=json.loads((HERE/"AUTHORIZATION_ADDENDUM_03.json").read_text())
    assert auth["approved"] and auth["task"]=="EXP-023/v1"
    assert sha(HERE/"ADDENDUM_03_VIDEO_IO_RECOVERY.md")==auth["addendum_sha256"]
    assert sha(HERE/"AUTHORIZATION_ADDENDUM_02.json")==auth["previous_authorization_sha256"]
    failure=json.loads((HERE/"EVAL_PRE_BUDGET.json").read_text())
    assert failure["status"]=="FAILED_STOPPED" and failure["counts"]["vae_decode"]==1
    assert auth["effective_changed_limit"]["vae_decode"]==13
    names=("AUTHORIZATION_ADDENDUM_03.json","ADDENDUM_03_VIDEO_IO_RECOVERY.md",
           "TRAIN_SOURCE_MANIFEST.json","GRADIENT_PROTOCOL_CPU.json",
           "FROZEN_ENTRY_REGRESSION.json","EVAL_PRE_BUDGET.json",
           "checkpoint_000.pt","resume_run.py","freeze_recovery.py","fm_run.py",
           "finalize_cpu.py","attempt_00/fm_run.py","attempt_00/README.md",
           "MEDIA_PATCH_DIFF.patch","media_smoke.py","MEDIA_SMOKE.json")
    import imageio_ffmpeg
    binary=__import__("pathlib").Path(imageio_ffmpeg.get_ffmpeg_exe()).resolve()
    assert binary.is_file()
    write(HERE/"RECOVERY_BINARY.json",{"path":str(binary),"sha256":sha(binary),
          "version":__import__("subprocess").check_output([str(binary),"-version"],text=True).splitlines()[0]})
    manifest={str((HERE/n).relative_to(ROOT)):sha(HERE/n) for n in names}
    manifest[str((HERE/"RECOVERY_BINARY.json").relative_to(ROOT))]=sha(HERE/"RECOVERY_BINARY.json")
    path=HERE/"RECOVERY_SOURCE_MANIFEST.json"
    if path.exists():raise FileExistsError(path)
    write(path,manifest)
    print("FROZEN RECOVERY",sha(path))


if __name__=="__main__":main()
