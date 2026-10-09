"""Launch one four-GPU read-only probe, with explicit idle and input checks."""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
RUNTIME=ROOT/'outputs/2026-10-08-09/stage1_training_shift12/runtime'
if (OUT/'run.json').exists():
    raise SystemExit('Refusing duplicate launch')
state=dict(status='preflight',pid=os.getpid(),gpus=[3,4,5,6],active=None,
    started_at=datetime.now().astimezone().isoformat(),optimizer_updates=0)

def save():
    p=OUT/'run.tmp.json';p.write_text(json.dumps(state,indent=2)+'\n');p.replace(OUT/'run.json')

def verify():
    for path,sha in json.loads((OUT/'manifest.json').read_text())['files'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=sha:
            raise RuntimeError(f'Frozen dependency changed: {path}')

save()
try:
    verify()
    occupancy={i:int(subprocess.check_output(['nvidia-smi','-i',str(i),
        '--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip()) for i in state['gpus']}
    if any(v>256 for v in occupancy.values()):
        raise RuntimeError(f'GPU occupied: {occupancy}')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='3,4,5,6',ABOT_VRAM_RESERVE_GIB='6',
        HF_HUB_OFFLINE='1',DIFFSYNTH_SKIP_DOWNLOAD='True',PYTHONUNBUFFERED='1',
        ABOT_DIFFSYNTH_ROOT=str(RUNTIME/'DiffSynth-Studio-h3-v2'),
        DIFFSYNTH_ROOT=str(RUNTIME/'DiffSynth-Studio-h3-v2'))
    for key,name in [('HF_HOME','hf'),('TORCHINDUCTOR_CACHE_DIR','torchinductor'),
                      ('TRITON_CACHE_DIR','triton'),('XDG_CACHE_HOME','xdg')]:
        env[key]=str(ROOT/'.cache'/name)
    (OUT/'gpu_probe').mkdir(exist_ok=False)
    command=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc-per-node=4',str(OUT/'probe_gpu.py')]
    with (OUT/'gpu_probe/run.log').open('x') as log:
        p=subprocess.Popen(command,cwd=RUNTIME,env=env,stdout=log,stderr=subprocess.STDOUT)
        state.update(status='running',initial_occupancy_MiB=occupancy,
            active=dict(pid=p.pid,command=command),vram_reserve_gib=6);save()
        code=p.wait()
    if code:
        raise RuntimeError(f'GPU probe exited {code}')
    reports=[json.loads((OUT/f'gpu_probe/rank{i}.json').read_text()) for i in range(4)]
    if any(r['status']!='complete' for r in reports):
        raise RuntimeError('Incomplete GPU rank reports')
    verify()
    state.update(status='complete',active=None,completed_at=datetime.now().astimezone().isoformat(),
        result=str(OUT/'gpu_probe/rank0.json'))
except BaseException as exc:
    state.update(status='failed',error=repr(exc));raise
finally:
    save()
