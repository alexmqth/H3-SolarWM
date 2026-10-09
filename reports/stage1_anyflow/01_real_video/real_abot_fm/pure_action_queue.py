"""Two controlled A/D clips, one GPU, optionally after the live GT probe."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

BASE=Path(__file__).resolve().parent

def live(pid):
    p=Path(f'/proc/{pid}/stat')
    try:return p.read_text().rsplit(')',1)[1].split()[0]!='Z'
    except FileNotFoundError:return False

def main(args):
    tag='original30' if args.mode=='original' else f'causal48_{args.steps}'
    path=BASE/f'pure_action_queue_{tag}.json'
    if path.exists():raise FileExistsError(path)
    files=['evaluate_action_pair.py','evaluate_real_core.py','pure_action_context_audit.json','counterfactual/action_sentences.pt']
    hashes={f:hashlib.sha256((BASE/f).read_bytes()).hexdigest() for f in files}
    assert json.loads((BASE/'pure_action_context_audit.json').read_text())['status']=='passed'
    result=dict(status='waiting' if args.after_gt_probe else 'running',started_at=datetime.now().astimezone().isoformat(),
        gpu=args.gpu,files=hashes,completed=[],active=None,mode=args.mode,steps=args.steps)
    def save():
        tmp=path.with_suffix('.tmp.json');tmp.write_text(json.dumps(result,indent=2)+'\n');tmp.replace(path)
    save()
    try:
        if args.after_gt_probe:
            if args.mode!='original':raise ValueError('GT probe wait is for Original lane only')
            deadline=time.monotonic()+3600
            while True:
                p=json.loads((BASE/'geometry/step_00_gt/probe.json').read_text())
                is_live=live(p['pid']);result['wait_for']=dict(pid=p['pid'],live=is_live,status=p['status']);save()
                if p['status']=='complete' and not is_live:break
                if p['status']=='failed' or (p['status']!='complete' and not is_live):raise RuntimeError('GT probe did not finish successfully')
                if time.monotonic()>deadline:raise TimeoutError('GT probe still live; no restart or additional GPU launch')
                time.sleep(15)
        if args.mode=='causal':
            t=json.loads((BASE/'train_48/training.json').read_text())
            if t['status']!='complete' or len(t['updates'])!=48:raise RuntimeError('Requires completed FM48')
        result['status']='running';save()
        for action in 'AD':
            assert all(hashlib.sha256((BASE/f).read_bytes()).hexdigest()==h for f,h in hashes.items())
            out=BASE/'pure_action'/tag/action;out.parent.mkdir(parents=True,exist_ok=True)
            cmd=[sys.executable,'-u',str(BASE/'evaluate_action_pair.py'),'--gpu',str(args.gpu),
                '--action',action,'--mode',args.mode,'--steps',str(args.steps),'--out',str(out)]
            if args.mode=='causal':cmd+=['--checkpoint',str(BASE/'train_48/step_48')]
            with (out.parent/f'{action}.log').open('x') as log:
                p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,cwd=BASE)
                result['active']=dict(pid=p.pid,command=cmd,output=str(out));save();code=p.wait()
            if code:raise RuntimeError(f'{action} video failed with{code}')
            m=json.loads((out/'evaluation.json').read_text());assert m['status']=='complete'
            assert m['intervention_provenance']['action']==action and m['history_source']=='generated'
            if args.mode=='causal':
                ref=json.loads((BASE/'pure_action/original30'/action/'evaluation.json').read_text())
                assert m['input_fingerprints']==ref['input_fingerprints']
            result['completed'].append(dict(action=action,output=str(out),forwards=m['denoiser_forwards'],
                horizontal_flow=m['flow_metrics']['horizontal_flow_px']['mean']))
            result['active']=None;save()
        result['status']='complete'
    except BaseException as exc:result.update(status='failed',error=repr(exc));raise
    finally:save()

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--gpu',type=int,required=True)
    ap.add_argument('--mode',choices=['original','causal'],required=True)
    ap.add_argument('--steps',type=int,choices=[8,30],default=30);ap.add_argument('--after-gt-probe',action='store_true')
    main(ap.parse_args())
