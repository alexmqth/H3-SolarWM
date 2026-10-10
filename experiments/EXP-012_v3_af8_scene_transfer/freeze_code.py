"""Freeze EXP-012 P0 code/config/input audit after CPU checks."""
from __future__ import annotations

import json

from common import CFG, HERE, ROOT, sha


def manifest() -> dict:
    names = ["taskbook_v1.md", "config.json", "common.py", "source_audit.py",
             "source_manifest.json", "checkpoint.py", "run_exp012.py",
             "preflight.py", "test_p0.py", "P0_CPU_AUDIT.json", "freeze_code.py"]
    return {"task": CFG["task"], "files": {
        str((HERE / name).relative_to(ROOT)): sha(HERE / name) for name in names}}


if __name__ == "__main__":
    target = HERE / "code_manifest.json"
    data = manifest()
    if target.exists():
        if json.loads(target.read_text()) != data:
            raise RuntimeError("code changed after freeze")
    else:
        target.write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps({"task": data["task"], "files": len(data["files"]),
                      "code_manifest_sha256": sha(target)}, indent=2))
