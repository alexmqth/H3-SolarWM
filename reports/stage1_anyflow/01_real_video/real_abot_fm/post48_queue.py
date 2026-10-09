"""Bounded evaluation only, after the live FM48 trainer exits successfully.

Uses the existing three project lanes 0/4/1, never launches training or C/D.
No failed task is retried. PIDs include start ticks to avoid reuse confusion.
"""
from datetime import datetime
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

BASE = Path(__file__).resolve().parent


def proc(pid):
    try:
        rest = Path(f'/proc/{pid}/stat').read_text().rsplit(') ', 1)[1].split()
        return None if rest[0] == 'Z' else rest[19]
    except FileNotFoundError:
        return None


def main():
    lock = (BASE / 'post48_queue.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    dest = BASE / 'post48_queue.json'
    if dest.exists():
        raise FileExistsError('No automatic restart of a previous queue')
    launches = {
        'training': (799534, BASE / 'train_48/training.json'),
        'parking0': (1506669, BASE / 'parking_queue_step00.json'),
        'geometry0': (1695577, BASE / 'geometry/step_00_step00_generated/probe.json'),
    }
    dependencies = {name: dict(pid=pid, start_ticks=proc(pid), receipt=str(path))
                    for name, (pid, path) in launches.items()}
    def command(script, *args):
        return [sys.executable, '-u', str(BASE / script), *map(str, args)]
    lanes = [
        dict(gpu=0, dependency='training', jobs=[
            command('evaluation_queue.py', '--gpu', 0, '--phase', 'trained')]),
        dict(gpu=4, dependency='parking0', jobs=[
            command('parking_regression.py', '--gpu', 4, '--step', 48)]),
        dict(gpu=1, dependency='geometry0', jobs=[
            command('probe_real_geometry.py', '--gpu', 1, '--step', 48, '--history', 'gt'),
            command('probe_real_geometry.py', '--gpu', 1, '--step', 48, '--history', 'step00_generated')]),
    ]
    scripts = ['audit_completed_training.py', 'evaluation_queue.py', 'evaluate_real.py',
               'parking_regression.py', 'probe_real_geometry.py', 'geometry_primitives.py']
    fingerprints = {n: hashlib.sha256((BASE / n).read_bytes()).hexdigest() for n in scripts}
    result = dict(status='waiting_for_training', started_at=datetime.now().astimezone().isoformat(),
                  dependencies=dependencies, source_hashes=fingerprints, allowed_project_gpus=[0, 4, 1],
                  launches=[], completed=[], scope='FM48 evaluation only; no training extension or Stage2')
    running = {}; positions = {x['gpu']: 0 for x in lanes}; checked_training = False
    def save():
        result['updated_at'] = datetime.now().astimezone().isoformat()
        tmp = dest.with_suffix('.tmp.json'); tmp.write_text(json.dumps(result, indent=2) + '\n'); tmp.replace(dest)
    def ready(name):
        d = dependencies[name]
        birth = proc(d['pid'])
        if birth is not None and birth == d['start_ticks']:
            return False
        data = json.loads(Path(d['receipt']).read_text())
        if data['status'] != 'complete':
            raise RuntimeError(f'{name} process exited without complete receipt; no retry')
        if name == 'training' and len(data['updates']) != 48:
            raise RuntimeError('Training budget/count mismatch')
        return True
    save()
    try:
        while True:
            training_done = ready('training')
            if training_done and not checked_training:
                with (BASE / 'completed_training_audit.log').open('x') as log:
                    subprocess.run(command('audit_completed_training.py'), stdout=log, stderr=subprocess.STDOUT, check=True)
                checked_training = True; result['status'] = 'evaluation_pending_or_running'
            for lane in lanes:
                gpu = lane['gpu']
                if gpu in running:
                    p, log, cmd = running[gpu]
                    code = p.poll()
                    if code is None:
                        continue
                    log.close(); del running[gpu]
                    result['completed'].append(dict(gpu=gpu, command=cmd, exit_code=code))
                    if code:
                        raise RuntimeError(f'GPU{gpu} evaluation exited {code}; no retry')
                    positions[gpu] += 1
                if positions[gpu] == len(lane['jobs']) or not training_done or not ready(lane['dependency']):
                    continue
                free = int(subprocess.check_output(['nvidia-smi', f'--id={gpu}', '--query-gpu=memory.free',
                                                   '--format=csv,noheader,nounits'], text=True).strip())
                if free < 34000:
                    result[f'gpu{gpu}_wait_free_MiB'] = free
                    continue
                assert all(hashlib.sha256((BASE / n).read_bytes()).hexdigest() == h for n, h in fingerprints.items())
                cmd = lane['jobs'][positions[gpu]]
                log_path = BASE / f'post48_gpu{gpu}_job{positions[gpu]}.log'
                log = log_path.open('x')
                p = subprocess.Popen(cmd, cwd=BASE, stdout=log, stderr=subprocess.STDOUT)
                running[gpu] = (p, log, cmd)
                result['launches'].append(dict(gpu=gpu, pid=p.pid, command=cmd, log=str(log_path)))
            save()
            if all(positions[x['gpu']] == len(x['jobs']) for x in lanes):
                result['status'] = 'complete'; save(); return
            time.sleep(20)
    except BaseException as exc:
        # Already launched tasks are allowed to finish; never kill GPU jobs.
        result.update(status='failed', error=repr(exc), still_running_pids=[x[0].pid for x in running.values()])
        save(); raise


if __name__ == '__main__':
    main()
