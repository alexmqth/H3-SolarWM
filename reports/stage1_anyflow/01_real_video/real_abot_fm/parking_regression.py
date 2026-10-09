"""Known-positive action scene, separate from held-out real training/validation.

Original reference is the previously verified released H3 30-step execution
(legacy inference dtype). New causal0/48 use the same h3_fp32 policy, so their
training comparison is matched; comparison to Original is a pipeline result.
"""
from datetime import datetime
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

BASE=Path(__file__).resolve().parent;RT=BASE/'runtime'
TEACHER=BASE.parents[2]/'outputs/2026-10-02-03'

def main(args):
    sys.path.insert(0,str(RT/'code/causal'))
    from report_stage1_anyflow import audit_inputs,summarize_video
    checkpoint=BASE/f'train_48/step_{args.step:02d}'
    path=BASE/f'parking_queue_step{args.step:02d}.json'
    if path.exists():raise FileExistsError(path)
    if args.step==48:
        t=json.loads((BASE/'train_48/training.json').read_text())
        assert t['status']=='complete' and len(t['updates'])==48
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(args.gpu),ABOT_VRAM_RESERVE_GIB='18',
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
        ABOT_DIFFSYNTH_ROOT=str(RT/'DiffSynth-Studio-h3-v2'),
        OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    result=dict(status='running',step=args.step,gpu=args.gpu,started_at=datetime.now().astimezone().isoformat(),
        scope='known-positive parking action regression, not real-data training targets',
        precision_note='causal0 and causal48 h3_fp32 matched; old Original30 is released legacy dtype, not an isolated-precision ablation',
        completed=[],active=None)
    def save():
        tmp=path.with_suffix('.tmp.json');tmp.write_text(json.dumps(result,indent=2)+'\n');tmp.replace(path)
    save()
    try:
        for steps in (30,8):
            for action in 'AD':
                free=int(subprocess.check_output(['nvidia-smi',f'--id={args.gpu}','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
                if free<34000:raise RuntimeError(f'GPU{args.gpu} busy: only{free}MiB free')
                out=BASE/f'parking/step_{args.step:02d}/{steps}step'/action
                out.parent.mkdir(parents=True,exist_ok=True)
                cmd=[sys.executable,'-u',str(RT/'code/causal/benchmark.py'),
                    '--modes','cached','--num-frames','39','--steps',str(steps),'--seed','13',
                    '--flow-shift','2.22','--precision-profile','h3_fp32','--chunk-frames','5',
                    '--history-chunks','5','--cache-device','cpu','--action-preset',action,
                    '--action-prefix-mode','causal','--action-feedback',
                    '--anchor-mode','dynamic_last_frame_rgb_dual','--history-source','generated',
                    '--causal-adapter',str(checkpoint/'causal_adapter.pt'),
                    '--stage1-lora',str(checkpoint/'stage1_lora.pt'),'--out-dir',str(out),'--save-latents']
                with (out.parent/f'{action}.log').open('x') as log:
                    p=subprocess.Popen(cmd,env=env,cwd=RT,stdout=log,stderr=subprocess.STDOUT)
                    result['active']=dict(pid=p.pid,command=cmd,output=str(out));save();code=p.wait()
                if code:raise RuntimeError(f'Parking{action}/{steps} failed: {code}')
                row=summarize_video(out,'cached',label=f'real-FM{args.step} parking{steps}',action=action,
                    history='generated',step=args.step,nfe=steps)
                assert row and row['noisy_forwards']==steps*3 and row['clean_commits']==3
                row['input_audit']=audit_inputs(out,TEACHER/f'action_{action}_teacher_39')
                assert row['input_audit']['exactly_equal']
                (out/'evaluation.json').write_text(json.dumps(row,indent=2)+'\n')
                result['completed'].append(row);result['active']=None;save()
        result['status']='complete'
    except BaseException as exc:result.update(status='failed',error=repr(exc));raise
    finally:save()

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--gpu',type=int,required=True)
    ap.add_argument('--step',type=int,choices=[0,48],required=True);main(ap.parse_args())
