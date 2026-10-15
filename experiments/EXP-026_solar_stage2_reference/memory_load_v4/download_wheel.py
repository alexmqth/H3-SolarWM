"""Download exactly one official ABI-matched flash-attn wheel with a hard byte cap."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import urllib.request

HERE=Path(__file__).resolve().parent
NAME='flash_attn-2.8.3+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl'
URL='https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3%2Bcu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl'
EXPECTED=256007522
LIMIT=2*2**30
out=HERE/'wheel'/NAME
if out.exists():
    raise RuntimeError('wheel already present; this script does not repeat downloads')
request=urllib.request.Request(URL,headers={'User-Agent':'EXP-026-v4-official-wheel'})
sha=hashlib.sha256()
count=0
try:
    with urllib.request.urlopen(request,timeout=30) as src,out.open('xb') as dst:
        while True:
            block=src.read(4*2**20)
            if not block:break
            count+=len(block)
            if count>LIMIT:raise RuntimeError('authorized download byte cap exceeded')
            dst.write(block);sha.update(block)
    if count!=EXPECTED:raise RuntimeError(f'wheel size differs: {count} != {EXPECTED}')
except BaseException:
    out.unlink(missing_ok=True)
    raise
receipt={'official_release':'Dao-AILab/flash-attention v2.8.3','url':URL,
         'name':NAME,'bytes':count,'sha256':sha.hexdigest(),'download_cap_bytes':LIMIT,
         'source':'GitHub official release asset API, matching torch2.6/cu12/cxx11abiFALSE/cp310'}
(HERE/'WHEEL_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
print('OFFICIAL_WHEEL_DOWNLOADED',count,sha.hexdigest())
