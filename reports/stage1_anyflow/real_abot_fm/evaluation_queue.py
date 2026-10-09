"""A bounded single-GPU real-video evaluation lane; no training launches."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

BASE=Path(__file__).resolve().parent
CLIPS=['118eb5d8b75e1b8ac23a4e9ae77af9a9_A_1140','dfec8ed3237860eba14d67c089ecd041_D_1750']

def main(args):
    path=BASE/f'evaluation_queue_{args.phase}.json'
    if path.exists():raise FileExistsError(path)
    script=BASE/'evaluate_real.py'
    fingerprint=hashlib.sha256(script.read_bytes()).hexdigest()
    result=dict(status='running',gpu=args.gpu,phase=args.phase,started_at=datetime.now().astimezone().isoformat(),
        clips=CLIPS,evaluator_sha256=fingerprint,completed=[],active=None)
    def save():
        tmp=path.with_suffix('.tmp.json');tmp.write_text(json.dumps(result,indent=2)+'\n');tmp.replace(path)
    jobs=[]
    if args.phase=='baseline':
        jobs=[('original','generated',30,None,c) for c in CLIPS]
        jobs += [('causal',h,30,0,c) for c in CLIPS for h in ('gt','generated')]
    else:
        train=json.loads((BASE/'train_48/training.json').read_text())
        if train['status']!='complete' or len(train['updates'])!=48:
            raise RuntimeError('Complete bounded48 training required before trained evaluation')
        jobs=[('causal',h,steps,48,c) for c in CLIPS for h,steps in [('gt',30),('generated',30),('generated',8)]]
    save()
    try:
        for mode,history,steps,step,clip in jobs:
            if hashlib.sha256(script.read_bytes()).hexdigest()!=fingerprint:raise RuntimeError('Evaluator changed')
            tag='original' if step is None else f'step_{step:02d}'
            out=BASE/'eval'/tag/f'{history}_{steps}'/clip
            out.parent.mkdir(parents=True,exist_ok=True)
            cmd=[sys.executable,'-u',str(script),'--gpu',str(args.gpu),'--clip',clip,'--mode',mode,
                '--history',history,'--steps',str(steps),'--out',str(out)]
            if step is not None:cmd+=['--checkpoint',str(BASE/f'train_48/step_{step:02d}')]
            log=out.parent/(clip+'.log')
            with log.open('x') as f:
                p=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,cwd=BASE)
                result['active']=dict(pid=p.pid,command=cmd,output=str(out),log=str(log));save()
                code=p.wait()
            if code:raise RuntimeError(f'Evaluation exited {code}; see {log}')
            row=json.loads((out/'evaluation.json').read_text())
            if row['status']!='complete':raise RuntimeError('Evaluation incomplete')
            if mode=='original' and row['denoiser_forwards']!=30:raise RuntimeError('Original count')
            if mode=='causal' and (row['denoiser_forwards']!=3*steps or row['commit_forwards']!=3):
                raise RuntimeError('Causal count')
            original=BASE/'eval/original/generated_30'/clip/'evaluation.json'
            reference=json.loads(original.read_text())
            if row['encoded_sha256']!=reference['encoded_sha256'] or row['input_fingerprints']!=reference['input_fingerprints']:
                raise RuntimeError('Initial conditioning/noise differs from Original')
            result['completed'].append(dict(output=str(out),input_equality=True,mode=mode,history=history,steps=steps,optimizer_step=step))
            result['active']=None;save()
        result['status']='complete'
    except BaseException as exc:
        result.update(status='failed',error=repr(exc));raise
    finally:save()

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--gpu',type=int,required=True)
    ap.add_argument('--phase',choices=['baseline','trained'],required=True)
    main(ap.parse_args())
