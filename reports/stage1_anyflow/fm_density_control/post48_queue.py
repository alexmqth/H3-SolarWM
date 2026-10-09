"""Bounded density-candidate evaluation, after the exact training PID exits.

Two GPU lanes at most; no training extension or automatic retry. Other jobs
are never stopped. Receipts alone do not end a live training dependency.
"""
from datetime import datetime
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

BASE=Path(__file__).resolve().parent
CONTROL=BASE.parents[1]/'2026-10-08-18/stage1_real_abot_fm'
CLIPS=('118eb5d8b75e1b8ac23a4e9ae77af9a9_A_1140','dfec8ed3237860eba14d67c089ecd041_D_1750')


def birth(pid):
    try:
        rest=Path(f'/proc/{pid}/stat').read_text().rsplit(') ',1)[1].split()
        return None if rest[0]=='Z' else rest[19]
    except FileNotFoundError:return None


def jobs():
    cmd=lambda script,*args:[sys.executable,'-u',str(script),*map(str,args)]
    video=[]
    for history,steps in (('gt',30),('generated',30),('generated',8)):
        for clip in CLIPS:
            out=BASE/f'eval/step_48/{history}_{steps}'/clip
            video.append(cmd(CONTROL/'evaluate_real.py','--gpu',1,'--clip',clip,
                '--mode','causal','--history',history,'--steps',steps,
                '--checkpoint',BASE/'train_48/step_48','--out',out))
    video.append(cmd(BASE/'parking_regression.py','--gpu',1,'--step',48))
    diagnostic=[cmd(BASE/'noise/probe_noise.py','--gpu',0,'--step',48)]
    for history in ('gt','step00_generated'):
        diagnostic.append(cmd(BASE/'probe_real_geometry.py','--gpu',0,'--step',48,'--history',history))
    return {1:video,0:diagnostic}


def main():
    lock=(BASE/'post48_queue.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    path=BASE/'post48_queue.json'
    if path.exists():raise FileExistsError('Prior queue exists; inspect it, never blindly restart')
    launch=json.loads((BASE/'launch.json').read_text())
    schedule=jobs();positions={g:0 for g in schedule};running={};audited=False
    scripts=[BASE/n for n in ('post48_queue.py','parking_regression.py','audit_completed_training.py',
        'probe_real_geometry.py','geometry_primitives.py','noise/probe_noise.py')]+[
            CONTROL/'evaluate_real.py',CONTROL/'geometry_primitives.py',CONTROL/'prepare_counterfactual.py']
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in scripts}
    runtime=json.loads((BASE/'runtime_manifest.json').read_text())
    control_runtime=json.loads((CONTROL/'runtime_manifest.json').read_text())
    result=dict(status='waiting_for_training',started_at=datetime.now().astimezone().isoformat(),
        training_dependency={k:launch[k] for k in ('pid','start_ticks','gpu')},allowed_gpus=[1,0],
        source_sha256=hashes,schedule=schedule,launches=[],completed=[],active={},maximum_project_gpus=2)
    def save():
        result['updated_at']=datetime.now().astimezone().isoformat()
        temp=path.with_suffix('.tmp.json');temp.write_text(json.dumps(result,indent=2)+'\n');temp.replace(path)
    save()
    try:
        while True:
            live=birth(launch['pid'])==launch['start_ticks']
            if not live and not audited:
                training=json.loads((BASE/'train_48/training.json').read_text())
                stream=json.loads((BASE/'stream_audit.json').read_text())
                if training['status']!='complete' or len(training['updates'])!=48 or stream['status']!='complete':
                    raise RuntimeError('Training process ended without complete48 receipts; no retry')
                assert stream['updates']==48 and stream['step00_tensors_equal']
                with (BASE/'completed_training_audit.log').open('x') as f:
                    subprocess.run([sys.executable,str(BASE/'audit_completed_training.py')],stdout=f,stderr=subprocess.STDOUT,check=True)
                audited=True;result['status']='evaluating'
            if audited:
                for gpu,lane in schedule.items():
                    if gpu in running:
                        p,log,command=running[gpu];code=p.poll()
                        if code is None:continue
                        log.close();del running[gpu];result['active'].pop(str(gpu))
                        result['completed'].append(dict(gpu=gpu,command=command,exit_code=code))
                        if code:raise RuntimeError(f'Evaluation onGPU{gpu} exited{code}; no retry')
                        positions[gpu]+=1
                    if positions[gpu]>=len(lane):continue
                    free=int(subprocess.check_output(['nvidia-smi',f'--id={gpu}',
                        '--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))
                    if free<34000:
                        result[f'gpu{gpu}_wait_free_MiB']=free;continue
                    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
                    assert all(hashlib.sha256((BASE/'runtime'/p).read_bytes()).hexdigest()==h for p,h in runtime.items())
                    assert all(hashlib.sha256((CONTROL/'runtime'/p).read_bytes()).hexdigest()==h for p,h in control_runtime.items())
                    command=lane[positions[gpu]];logpath=BASE/f'post48_gpu{gpu}_{positions[gpu]:02d}.log'
                    log=logpath.open('x');p=subprocess.Popen(command,cwd=BASE,stdout=log,stderr=subprocess.STDOUT)
                    running[gpu]=(p,log,command)
                    row=dict(gpu=gpu,pid=p.pid,start_ticks=birth(p.pid),command=command,log=str(logpath))
                    result['launches'].append(row);result['active'][str(gpu)]=row
                if all(positions[g]==len(lane) for g,lane in schedule.items()):
                    result['status']='complete';save();return
            save();time.sleep(20)
    except BaseException as e:
        result.update(status='failed',error=repr(e),still_running_pids=[x[0].pid for x in running.values()])
        save();raise


if __name__=='__main__':main()
