"""Continue preselected eval suites on one GPU, never restart an existing job."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
BASE=Path(__file__).resolve().parent


def live(r):
    p=Path(f'/proc/{r["pid"]}/stat')
    if not p.exists():return False
    stat=p.read_text().rsplit(')',1)[1].split()
    return stat[19]==r['start_ticks'] and stat[0] not in ('Z','X')


def main(args):
    file=BASE/f'eval_queue_{args.arm}.json'
    if file.exists():raise FileExistsError('Do not create duplicate queue')
    before=BASE/f'eval_{args.arm}_parking_D/evaluation.json'
    original=json.loads(before.read_text())
    state=dict(at=datetime.now().astimezone().isoformat(),pid=os.getpid(),
        start_ticks=Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')',1)[1].split()[19],
        gpu=args.gpu,arm=args.arm,status='waiting_existing_parking_D',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        previous_pid=original['pid'],previous_start_ticks=original['start_ticks'],children=[])
    def save():
        p=file.with_suffix('.tmp.json');p.write_text(json.dumps(state,indent=2)+'\n');p.replace(file)
    save()
    try:
        while live(original):time.sleep(5)
        finished=json.loads(before.read_text())
        assert (finished['pid'],finished['start_ticks'])==(original['pid'],original['start_ticks'])
        assert finished['status']=='complete_pending_visual_review',finished.get('error',finished['status'])
        for suite in ('parking_A','gt'):
            output=BASE/f'eval_{args.arm}_{suite}'
            if output.exists():raise FileExistsError(f'Already exists, refusing duplicate: {output}')
            logfile=BASE/f'eval_{args.arm}_{suite}.log'
            with logfile.open('x') as log:
                command=[sys.executable,str(BASE/'evaluate_objective.py'),'--gpu',str(args.gpu),'--arm',args.arm,'--suite',suite]
                child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,cwd=str(BASE.parents[3]))
                ticks=Path(f'/proc/{child.pid}/stat').read_text().rsplit(')',1)[1].split()[19]
                state['status']='running_'+suite;state['children'].append(dict(suite=suite,pid=child.pid,start_ticks=ticks,command=command));save()
                code=child.wait();state['children'][-1]['exit_code']=code;save()
            if code:raise RuntimeError(f'{suite} exited {code}; no retry or next suite')
            rec=json.loads((output/'evaluation.json').read_text());assert rec['status']=='complete_pending_visual_review'
            assert not live(rec)
        state['status']='complete_all_preselected_suites'
    except BaseException as exc:
        state.update(status='stopped_error',error=repr(exc));raise
    finally:save()


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--gpu',type=int,required=True);ap.add_argument('--arm',choices=['fm_only','fm_action'],required=True)
    main(ap.parse_args())
