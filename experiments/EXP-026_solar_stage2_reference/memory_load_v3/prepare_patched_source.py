"""Create the independent small source copy that GPU load-only imports."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
SOLAR=ROOT/'SolarWM'
source=json.loads((HERE/'SOURCE_RECEIPT.json').read_text())
for name, receipt in source.items():
    actual=hashlib.sha256((SOLAR/name).read_bytes()).hexdigest()
    if actual!=receipt['base_sha256']:
        raise RuntimeError(f'official source changed: {name}')
target=HERE/'patched_src'
if target.exists():
    raise RuntimeError('patched source already exists; avoid silently replacing the load target')
staging=HERE/'patched_staging'
if staging.exists():
    raise RuntimeError('staging already exists')
shutil.copytree(SOLAR/'src/solarwm',staging/'src/solarwm')
subprocess.run(['patch','-p1','-i',str(HERE/'cpu_sharded_load.patch')],
               cwd=staging,check=True,capture_output=True,text=True)
staging.rename(target)
# Python path will use patched_src/src/solarwm.
for name, receipt in source.items():
    candidate=target/name
    if hashlib.sha256(candidate.read_bytes()).hexdigest()!=receipt['modified_sha256']:
        raise RuntimeError(f'patched source digest mismatch: {name}')
(HERE/'PATCHED_SOURCE.json').write_text(json.dumps({
    'official_checkout':'ce1da4e7705391eda8eeda6016c0fd3f614b975e',
    'patched_src':str(target),
    'patch_sha256':hashlib.sha256((HERE/'cpu_sharded_load.patch').read_bytes()).hexdigest(),
    'source_receipt':source,
},indent=2)+'\n')
print('PATCHED_SOURCE_READY',target)
