"""CPU-only reports from completed FM48 evaluations; never launches a model.

Wait for complete video pairs/groups and fixed-state probes. Reports retain
manual visual-review gates. No checkpoint selection or experiment retry.
"""
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

BASE = Path(__file__).resolve().parent
CLIPS = ['118eb5d8b75e1b8ac23a4e9ae77af9a9_A_1140', 'dfec8ed3237860eba14d67c089ecd041_D_1750']


def read(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def main():
    lock = (BASE / 'post48_reports.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    receipt = BASE / 'post48_reports.json'
    if receipt.exists(): raise FileExistsError('No automatic report-controller retry')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['OMP_NUM_THREADS'] = '4'
    def command(name, *args):
        return [sys.executable, '-u', str(BASE / name), *map(str, args)]
    jobs = []
    for history, steps in [('gt', 30), ('generated', 30), ('generated', 8)]:
        jobs.append(dict(name=f'real_{history}{steps}', kind='real',
            dependencies=[BASE / f'eval/step_48/{history}_{steps}' / c / 'evaluation.json' for c in CLIPS],
            commands=[command('report_real.py', '--label', f'trained_complete_{history}{steps}',
                              '--history', history, '--steps', steps)]))
    for steps in (30, 8):
        jobs.append(dict(name=f'parking_{steps}', kind='parking',
            dependencies=[BASE / f'parking/step_48/{steps}step/{a}/evaluation.json' for a in 'AD'],
            commands=[command('report_parking.py', '--step', 48, '--steps', steps)]))
    for history in ('gt', 'step00_generated'):
        directory = BASE / f'geometry/step_48_{history}'
        jobs.append(dict(name=f'geometry_{history}', kind='geometry', dependencies=[directory / 'probe.json'],
            commands=[command('summarize_geometry.py', '--input', directory / 'probe.json', '--out', directory),
                      command('compare_geometry.py', '--before', BASE / f'geometry/step_00_{history}/probe.json',
                              '--after', directory / 'probe.json', '--out', BASE / f'geometry/compare_00_48_{history}')]))
    scripts = ['report_real.py', 'report_parking.py', 'summarize_geometry.py', 'compare_geometry.py']
    fingerprints = {name: hashlib.sha256((BASE / name).read_bytes()).hexdigest() for name in scripts}
    result = dict(status='waiting', pid=os.getpid(), started_at=datetime.now().astimezone().isoformat(),
        scope='CPU artifact generation only, manual video acceptance remains pending',
        source_sha256=fingerprints, completed=[], active=None, pending=[job['name'] for job in jobs])
    def save():
        result['updated_at'] = datetime.now().astimezone().isoformat()
        tmp = receipt.with_suffix('.tmp.json'); tmp.write_text(json.dumps(result, indent=2) + '\n'); tmp.replace(receipt)
    save()
    try:
        while jobs:
            train = read(BASE / 'train_48/training.json')
            controller = read(BASE / 'post48_queue.json')
            if (train and train['status'] == 'failed') or (controller and controller['status'] == 'failed'):
                raise RuntimeError('Upstream failed; leave GPU work untouched and inspect before any retry')
            for job in list(jobs):
                inputs = [read(path) for path in job['dependencies']]
                if any(data is None for data in inputs): continue
                if job['kind'] == 'parking':
                    ready = all(data.get('frames') == 39 and data.get('optimizer_step') == 48 for data in inputs)
                else:
                    ready = all(data.get('status') == 'complete' for data in inputs)
                if not ready: continue
                assert train['status'] == 'complete' and len(train['updates']) == 48
                assert all(hashlib.sha256((BASE / name).read_bytes()).hexdigest() == h for name, h in fingerprints.items())
                result.update(status='building_reports', active=job['name']); save()
                log_path = BASE / f'post48_report_{job["name"]}.log'
                with log_path.open('x') as log:
                    for cmd in job['commands']:
                        subprocess.run(cmd, cwd=BASE, stdout=log, stderr=subprocess.STDOUT, check=True)
                result['completed'].append(dict(name=job['name'], commands=job['commands'], log=str(log_path),
                    dependency_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in job['dependencies']}))
                jobs.remove(job); result.update(active=None, pending=[x['name'] for x in jobs]); save()
            if jobs:
                result['status'] = 'waiting_for_remaining_results'; save(); time.sleep(30)
        result['status'] = 'complete_artifacts_manual_review_pending'; save()
    except BaseException as exc:
        result.update(status='failed', error=repr(exc)); save(); raise


if __name__ == '__main__':
    main()
