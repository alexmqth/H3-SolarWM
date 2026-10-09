"""Complete the D-history lane after the already launched A-history probe."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

OUT=Path(__file__).resolve().parent
state=dict(status='waiting_A',pid=os.getpid(),gpu=6,active=None)
def save():
    tmp=OUT/'controller.tmp.json';tmp.write_text(json.dumps(state,indent=2)+'\n')
    tmp.replace(OUT/'controller.json')
save()
try:
    while True:
        a=json.loads((OUT/'probe_A.json').read_text())
        if a['status']=='complete':break
        if a['status'] in ('failed','not_started_gpu_busy'):raise RuntimeError(a['status'])
        os.kill(a['pid'],0)
        time.sleep(10)
    # Wait for A to actually release its CUDA context; no second concurrent GPU.
    while True:
        free=int(subprocess.check_output(['nvidia-smi','-i','6','--query-gpu=memory.free',
             '--format=csv,noheader,nounits'],text=True).strip())
        if free>=36000:break
        state.update(status='waiting_memory',free_MiB=free);save();time.sleep(10)
    command=[sys.executable,'-u',str(OUT/'probe.py'),'--gpu','6','--history','D']
    with (OUT/'probe_D.log').open('x') as log:
        child=subprocess.Popen(command,cwd=OUT.parents[2],stdout=log,stderr=subprocess.STDOUT)
        state.update(status='running_D',active=child.pid);save()
        code=child.wait()
    d=json.loads((OUT/'probe_D.json').read_text())
    if code or d['status']!='complete':raise RuntimeError(f'D failed {code}: {d["status"]}')
    subprocess.run([sys.executable,str(OUT/'analyze.py')],check=True)
    state.update(status='complete',active=None)
except BaseException as exc:
    state.update(status='failed',error=repr(exc));raise
finally:save()
