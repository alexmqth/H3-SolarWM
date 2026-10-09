"""Two serial probe lanes on GPU4/5, alongside the existing GPU6 training lane.

No training is launched here. A lane requires at least42000MiB free before each
probe (previous measured peak35636MiB). Other users' processes are untouched.
"""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
FORK=ROOT/'outputs/2026-10-08-15/stage1_interval_consistency_candidate'
BASE=ROOT/'outputs/2026-10-08-13/stage1_parallel_resume68_to128/train_128'
PREVIOUS=ROOT/'outputs/2026-10-08-15/stage1_finite_interval_refinement128'
LOCK=threading.Lock()
VARIANTS={'initial128':BASE, **{v+'136':FORK/v/'train_136' for v in ('control','auxiliary')}}


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def freeze(variant):
    with LOCK:
        path=OUT/f'manifest_{variant}.json'
        if path.exists():return
        files=dict(json.loads((PREVIOUS/'manifest.json').read_text())['files'])
        directory=VARIANTS[variant];step=128 if variant=='initial128' else 136
        additions=[OUT/'probe.py',OUT/'run_diagnostics.py',OUT/'cpu_self_test.json',
                   OUT/'cpu_h3_integration.json',directory/'training.json']
        additions += list((directory/f'step_{step}').glob('*.pt'))
        additions += list(PREVIOUS.glob('probe_*.json'))
        additions += list(PREVIOUS.glob('*/targets/*.pt'))
        for p in additions:files[str(p)]=sha(p)
        path.write_text(json.dumps(dict(created_at=datetime.now().astimezone().isoformat(),
            variant=variant,files=files,max_simultaneous_project_gpus=3,
            resource_plan='Two read-only lanes4/5 plus existing training/evaluation6'),indent=2)+'\n')


def lane(action,gpu):
    path=OUT/f'lane_{action}.json'
    if path.exists():raise FileExistsError(path)
    state=dict(status='waiting',action=action,gpu=gpu,active=None,completed=[])
    def save():
        tmp=path.with_suffix('.tmp.json');tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(path)
    save()
    try:
        for variant,directory in VARIANTS.items():
            step=128 if variant=='initial128' else 136
            state.update(status='waiting_for_checkpoint',variant=variant);save()
            while True:
                file=directory/'training.json'
                report=json.loads(file.read_text()) if file.exists() else {}
                if report.get('status')=='complete' and len(report['updates'])==step:break
                parent=FORK/'run_auxiliary.json'
                if variant=='auxiliary136' and parent.exists():
                    if json.loads(parent.read_text()).get('status')=='failed':
                        raise RuntimeError('Auxiliary controller failed; probe not launched')
                time.sleep(10)
            freeze(variant)
            while True:
                free=int(subprocess.check_output(['nvidia-smi','-i',str(gpu),
                    '--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
                if free>=42000:break
                state.update(status='waiting_for_memory',free_MiB=free);save();time.sleep(10)
            command=[sys.executable,'-u',str(OUT/'probe.py'),'--variant',variant,
                     '--action',action,'--gpu',str(gpu)]
            with (OUT/f'{variant}_{action}.log').open('x') as log:
                child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                state.update(status='running',active=dict(pid=child.pid,command=command));save()
                code=child.wait()
            receipt=json.loads((OUT/variant/f'probe_{action}.json').read_text())
            if code or receipt['status']!='complete':
                raise RuntimeError(f'{variant}/{action} did not complete: exit{code}, {receipt["status"]}')
            state['completed'].append(variant);state['active']=None;save()
        state.update(status='complete')
    except BaseException as exc:
        state.update(status='failed',error=repr(exc));raise
    finally:save()


if __name__=='__main__':
    for name in ('cpu_self_test.json','cpu_h3_integration.json'):
        assert json.loads((OUT/name).read_text())['status']=='passed'
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(lane,'A',4),pool.submit(lane,'D',5)]
        for future in futures:future.result()
