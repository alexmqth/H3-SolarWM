"""Run one isolated read-only trained-H3 equivalence probe on an idle GPU4."""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
RUNTIME = ROOT / 'outputs/2026-10-08-09/stage1_training_shift12/runtime'
if (OUT / 'run.json').exists():
    raise SystemExit('Refusing duplicate probe')
STATE = dict(status='preflight', pid=os.getpid(), gpu=4, active=None,
    started_at=datetime.now().astimezone().isoformat(), optimizer_updates=0)

def save():
    p = OUT / 'run.tmp.json'
    p.write_text(json.dumps(STATE, indent=2) + '\n')
    p.replace(OUT / 'run.json')

def verify():
    for path, sha in json.loads((OUT / 'manifest.json').read_text())['files'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != sha:
            raise RuntimeError(f'Frozen input changed: {path}')

save()
try:
    verify()
    occupancy = int(subprocess.check_output(['nvidia-smi', '-i', '4',
        '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True).strip())
    if occupancy > 256:
        raise RuntimeError(f'GPU4 is occupied: {occupancy} MiB')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='4', ABOT_VRAM_RESERVE_GIB='6',
        HF_HUB_OFFLINE='1', DIFFSYNTH_SKIP_DOWNLOAD='True', PYTHONUNBUFFERED='1',
        ABOT_DIFFSYNTH_ROOT=str(RUNTIME / 'DiffSynth-Studio-h3-v2'),
        DIFFSYNTH_ROOT=str(RUNTIME / 'DiffSynth-Studio-h3-v2'))
    for key, name in [('HF_HOME', 'hf'), ('TORCHINDUCTOR_CACHE_DIR', 'torchinductor'),
                      ('TRITON_CACHE_DIR', 'triton'), ('XDG_CACHE_HOME', 'xdg')]:
        env[key] = str(ROOT / '.cache' / name)
    directory = OUT / 'gpu_probe'
    directory.mkdir(exist_ok=True)
    command = [sys.executable, '-u', str(OUT / 'probe_gpu.py')]
    with (directory / 'run.log').open('x') as log:
        p = subprocess.Popen(command, cwd=RUNTIME, env=env, stdout=log, stderr=subprocess.STDOUT)
        STATE.update(status='running', initial_occupancy_MiB=occupancy,
            active=dict(pid=p.pid, command=command), vram_reserve_gib=6)
        save()
        code = p.wait()
    if code:
        raise RuntimeError(f'Probe exited {code}')
    report = json.loads((directory / 'probe.json').read_text())
    if report['status'] != 'complete':
        raise RuntimeError('Incomplete probe')
    verify()
    STATE.update(status='complete', active=None, result=str(directory / 'probe.json'),
        completed_at=datetime.now().astimezone().isoformat())
except BaseException as exc:
    STATE.update(status='failed', error=repr(exc))
    raise
finally:
    save()
