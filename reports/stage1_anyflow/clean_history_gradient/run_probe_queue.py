"""Wait for the matched FM control, then run exactly one 33B gradient probe."""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
PREVIOUS=ROOT/'outputs/2026-10-08-07/stage1_fm39_fullscope_control'
RUNTIME=OUT/'runtime'
STATE=dict(status='waiting_for_fullscope_fm16',pid=os.getpid(),gpu=0,active=None,
    at=datetime.now().astimezone().isoformat(),vram_reserve_gib=6,
    scope='one real A/chunk2 gradient and memory probe, no rollout-quality claim')


def save():
    p=OUT/'run.tmp.json';p.write_text(json.dumps(STATE,indent=2)+'\n');p.replace(OUT/'run.json')


def live(pid):
    try:return Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[0]!='Z'
    except FileNotFoundError:return False


if (OUT/'run.json').exists():raise SystemExit('Refusing duplicate probe queue')
save()
try:
    while True:
        s=json.loads((PREVIOUS/'run.json').read_text())
        if s['status']=='complete' and not live(s['pid']):
            if len(s['evaluations'])!=4:raise RuntimeError('FM control incomplete')
            break
        if s['status'] not in ('running','complete','waiting_for_fullscope_anyflow16') or not live(s['pid']):
            raise RuntimeError('FM predecessor failed or exited unexpectedly')
        time.sleep(20)
    for path,sha in json.loads((OUT/'manifest.json').read_text())['files'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=sha:
            raise RuntimeError(f'Frozen input/source changed: {path}')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',ABOT_VRAM_RESERVE_GIB='6',HF_HUB_OFFLINE='1',
        DIFFSYNTH_SKIP_DOWNLOAD='True',PYTHONUNBUFFERED='1',
        ABOT_DIFFSYNTH_ROOT=str(RUNTIME/'DiffSynth-Studio-h3-v2'),
        DIFFSYNTH_ROOT=str(RUNTIME/'DiffSynth-Studio-h3-v2'))
    for key,name in [('HF_HOME','hf'),('TORCHINDUCTOR_CACHE_DIR','torchinductor'),('TRITON_CACHE_DIR','triton'),('XDG_CACHE_HOME','xdg')]:
        env[key]=str(ROOT/'.cache'/name)
    directory=OUT/'gpu_probe';directory.mkdir(exist_ok=False)
    with (directory/'run.log').open('w') as log:
        p=subprocess.Popen([sys.executable,'-u',str(OUT/'probe_gpu.py')],cwd=RUNTIME,env=env,stdout=log,stderr=subprocess.STDOUT)
        STATE.update(status='running',active=dict(pid=p.pid,label='A_chunk2_clean_history_gradient_probe',directory=str(directory)));save()
        code=p.wait()
    if code:raise RuntimeError(f'GPU gradient probe exited {code}')
    report=json.loads((directory/'probe.json').read_text())
    if report['status']!='complete':raise RuntimeError('Probe report incomplete')
    STATE.update(status='complete',active=None,completed_at=datetime.now().astimezone().isoformat(),
        result=str(directory/'probe.json'))
except BaseException as exc:
    STATE.update(status='failed',error=repr(exc));raise
finally:
    save()
