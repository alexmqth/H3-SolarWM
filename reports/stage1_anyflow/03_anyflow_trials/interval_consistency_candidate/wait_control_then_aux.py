"""Run the paired auxiliary on the same GPU after its verified live control."""
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

OUT=Path(__file__).resolve().parent
PATH=OUT/'auxiliary_queue.json'
if PATH.exists():raise SystemExit('Queue already exists')
state=dict(status='waiting_for_control',pid=os.getpid(),gpu=6,
    created_at=datetime.now().astimezone().isoformat(),active=None)


def save():
    temporary=PATH.with_suffix('.tmp.json')
    temporary.write_text(json.dumps(state,indent=2)+'\n');temporary.replace(PATH)


def alive(pid):
    try:return Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[0]!='Z'
    except FileNotFoundError:return False


save()
try:
    while True:
        control=json.loads((OUT/'run_control.json').read_text())
        running=alive(control['pid'])
        state['dependency']=dict(pid=control['pid'],status=control['status'],live=running,
                                checked_at=datetime.now().astimezone().isoformat())
        save()
        if control['status']=='complete' and not running:break
        if control['status'] not in ('preflight','running','complete') or not running:
            raise RuntimeError('Control did not finish cleanly; auxiliary not started')
        time.sleep(20)
    while True:
        used=int(subprocess.check_output(['nvidia-smi','-i','6','--query-gpu=memory.used',
            '--format=csv,noheader,nounits'],text=True).strip())
        if used<1024:break
        state.update(status='waiting_for_gpu_after_control',used_MiB=used);save();time.sleep(20)
    command=[sys.executable,'-u',str(OUT/'run_auxiliary_on_gpu6.py'),'auxiliary','6']
    with (OUT/'controller_auxiliary_gpu6.log').open('x') as log:
        child=subprocess.Popen(command,cwd=OUT,stdout=log,stderr=subprocess.STDOUT)
        state.update(status='running_auxiliary',active=dict(pid=child.pid,command=command));save()
        code=child.wait()
    if code:raise RuntimeError(f'Auxiliary exited {code}')
    state.update(status='complete',active=None)
except BaseException as exc:
    state.update(status='failed',error=repr(exc));raise
finally:save()
