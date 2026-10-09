"""CPU reports for complete paired outputs; no GPU/model calls or retries."""
from datetime import datetime
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

BASE=Path(__file__).resolve().parent
CLIPS=('118eb5d8b75e1b8ac23a4e9ae77af9a9_A_1140','dfec8ed3237860eba14d67c089ecd041_D_1750')


def birth(pid):
    try:
        fields=Path(f'/proc/{pid}/stat').read_text().rsplit(') ',1)[1].split()
        return fields[19] if fields[0]!='Z' else None
    except FileNotFoundError:return None


def plans():
    output=[dict(label='noise',args=['--kind','noise'],
        receipts=[str(BASE/'noise/step_48/probe.json')])]
    for history in ('gt','step00_generated'):
        output.append(dict(label=f'geometry_{history}',args=['--kind','geometry','--history',history],
            receipts=[str(BASE/f'geometry/step_48_{history}/probe.json')]))
    for history,steps in (('gt',30),('generated',30),('generated',8)):
        output.append(dict(label=f'natural_{history}_{steps}',
            args=['--kind','natural','--history',history,'--steps',str(steps)],
            receipts=[str(BASE/f'eval/step_48/{history}_{steps}/{c}/evaluation.json') for c in CLIPS]))
    for steps in (30,8):
        output.append(dict(label=f'parking_{steps}',args=['--kind','parking','--steps',str(steps)],
            receipts=[str(BASE/f'parking/step_48/{steps}step/{a}/evaluation.json') for a in 'AD']))
    return output


def main():
    lock=(BASE/'post48_reports.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    dest=BASE/'post48_reports.json'
    if dest.exists():raise FileExistsError('Previous report controller exists; inspect rather than restart')
    dependency=json.loads((BASE/'post48_queue_launch.json').read_text())
    scripts=['report_density.py','post48_reports.py']
    hashes={name:hashlib.sha256((BASE/name).read_bytes()).hexdigest() for name in scripts}
    validation=json.loads((BASE/'report_validation.json').read_text())
    assert validation['status']=='passed' and validation['report_source_sha256']==hashes['report_density.py']
    result=dict(status='waiting_for_complete_pairs',started_at=datetime.now().astimezone().isoformat(),
        gpu_queue_dependency=dependency,source_sha256=hashes,completed=[],active=None)
    pending=plans()
    def save():
        result['updated_at']=datetime.now().astimezone().isoformat();result['pending']=[x['label'] for x in pending]
        temp=dest.with_suffix('.tmp.json');temp.write_text(json.dumps(result,indent=2)+'\n');temp.replace(dest)
    def ready(task):
        for name in task['receipts']:
            path=Path(name)
            if not path.exists():return False
            receipt=json.loads(path.read_text())
            if task['label'].startswith('parking'):
                if receipt.get('frames')!=39 or receipt.get('optimizer_step')!=48:return False
            elif receipt['status']!='complete':return False
        return True
    save()
    try:
        while pending:
            for task in list(pending):
                if not ready(task):continue
                assert all(hashlib.sha256((BASE/n).read_bytes()).hexdigest()==h for n,h in hashes.items())
                command=[sys.executable,'-u',str(BASE/'report_density.py'),*task['args']]
                with (BASE/f"report_{task['label']}.log").open('x') as log:
                    p=subprocess.Popen(command,cwd=BASE,stdout=log,stderr=subprocess.STDOUT)
                    result['active']=dict(label=task['label'],pid=p.pid,start_ticks=birth(p.pid),command=command);save()
                    code=p.wait()
                if code:raise RuntimeError(f"Report{task['label']} exited{code}; no automatic retry")
                result['completed'].append(dict(label=task['label'],output=str(BASE/'report'/task['label']),exit_code=code))
                result['active']=None;pending.remove(task);save()
            if pending and birth(dependency['pid'])!=dependency['start_ticks']:
                queue=json.loads((BASE/'post48_queue.json').read_text())
                raise RuntimeError(f"GPU queue ended ({queue['status']}) with unavailable reports; inspect receipts")
            if pending:time.sleep(20)
        result['status']='complete_artifacts_pending_manual_visual_review';save()
    except BaseException as e:
        result.update(status='failed',error=repr(e));save();raise


if __name__=='__main__':main()
