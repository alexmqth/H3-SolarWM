"""Launch only the preregistered density candidate; reuse completed control."""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess

BASE=Path(__file__).resolve().parent
CONTROL=BASE.parents[1]/'2026-10-08-18/stage1_real_abot_fm'
PYTHON='/home/lpeng/miniconda3/envs/h3world/bin/python'


def main():
    if (BASE/'launch.json').exists() or (BASE/'train_48').exists():
        raise RuntimeError('Experiment already exists; refusing duplicate launch')
    if json.loads((BASE/'preflight.json').read_text())['status']!='passed':
        raise RuntimeError('Preflight has not passed')
    gpu=1
    free=int(subprocess.check_output(['nvidia-smi',f'--id={gpu}',
        '--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))
    if free<40000:raise RuntimeError(f'GPU1 has {free} MiB free; need40000, no launch')
    prior=json.loads((CONTROL/'launch.json').read_text())
    cmd=list(prior['command']);cmd[2]=str(BASE/'run_density.py')
    cmd[cmd.index('--out-dir')+1]=str(BASE/'train_48')
    cmd[cmd.index('--training-timestep-shift')+1]='2.22'
    cmd+=['--training-weight-shift','12']
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='1',ABOT_VRAM_RESERVE_GIB='14',
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
        OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONUNBUFFERED='1',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    with (BASE/'training.log').open('w') as f:
        p=subprocess.Popen(cmd,env=env,cwd=BASE/'runtime',stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    stat=Path(f'/proc/{p.pid}/stat').read_text().split(') ')[-1].split()
    receipt=dict(pid=p.pid,start_ticks=stat[19],gpu=gpu,free_MiB_at_launch=free,
        at=datetime.now().astimezone().isoformat(),command=cmd,reserve_GiB=14,maximum_updates=48,
        control_reused=str(CONTROL/'train_48'),source_sha256={n:hashlib.sha256((BASE/n).read_bytes()).hexdigest()
            for n in ('run_density.py','preflight.py','protocol.json','runtime_manifest.json')})
    (BASE/'launch.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
