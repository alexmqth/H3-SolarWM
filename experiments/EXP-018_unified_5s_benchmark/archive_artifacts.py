"""Copy original EXP-018 run evidence into submission and verify exact hashes.

The 120-frame presentation files live in submission/report/unified_5s/videos/ and
are deliberately not duplicated here. This script never overwrites differing data.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
SOURCE = ROOT / "H3-World/outputs/EXP-018_unified_5s_benchmark"
TARGET = HERE / "artifacts"
ORDER = (
    "v0", "v1", "v2a", "v2b", "v2b_attempt_01_condition_oom",
    "v3_30", "v3_8", "v4",
)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for part in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


records = []
for row in ORDER:
    files = sorted(path for path in (SOURCE / row).iterdir() if path.is_file())
    assert files, row
    for source in files:
        if source.name == "display_120.mp4":
            continue
        destination = TARGET / row / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        expected = sha(source)
        if destination.exists():
            if sha(destination) != expected:
                raise RuntimeError(f"Refusing to overwrite a different file: {destination}")
        else:
            shutil.copy2(source, destination)
        assert sha(destination) == expected
        records.append({
            "row": row,
            "source": str(source.relative_to(ROOT)),
            "artifact": str(destination.relative_to(HERE)),
            "sha256": expected,
            "bytes": source.stat().st_size,
        })

index = json.loads((HERE / "results_index.json").read_text())
for row, data in index["rows"].items():
    item = next((r for r in records if r["row"] == row and r["source"] == data["full_video"]), None)
    assert item is not None and item["sha256"] == data["full_sha256"], row

manifest = {
    "task": "EXP-018/v1",
    "source_root": str(SOURCE.relative_to(ROOT)),
    "artifact_root": "artifacts",
    "excluded_duplicate_display": "report/unified_5s/videos/<row>_5s.mp4",
    "files": records,
    "total_bytes": sum(record["bytes"] for record in records),
}
(HERE / "ARTIFACT_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(f"archived {len(records)} files; {manifest['total_bytes']} bytes; all SHA-256 verified")
