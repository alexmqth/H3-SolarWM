"""Isolated no-training parking rollout for the current-prefix candidate.

Uses the frozen B benchmark and step00 zero adapters. This changes exactly
the routing intervention tested by probe_candidate.py. No production defaults.
"""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys

BASE=Path(__file__).resolve().parent
BRIDGE=BASE.parents[1]/'2026-10-08-18/stage1_real_abot_fm'
RT=BRIDGE/'runtime'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--gpu',type=int,required=True)
    ap.add_argument('--action',choices=['A','D'],required=True)
    ap.add_argument('--steps',type=int,choices=[30],default=30)
    a=ap.parse_args()
    free=int(subprocess.check_output(['nvidia-smi',f'--id={a.gpu}','--query-gpu=memory.free',
        '--format=csv,noheader,nounits'],text=True).strip())
    if free<34000:raise RuntimeError(f'GPU{a.gpu} needs34000MiB free, got{free}')
    preflight=json.loads((BASE/'cpu_preflight.json').read_text())
    assert preflight['status']=='passed'
    assert all(sha(BASE/n)==h for n,h in preflight['sources'].items())
    out=BASE/'parking'/f'{a.steps}step'/a.action
    if out.exists():raise FileExistsError('No automatic overwrite or retry')
    out.mkdir(parents=True)
    os.environ.update(CUDA_VISIBLE_DEVICES=str(a.gpu),ABOT_VRAM_RESERVE_GIB='18',HF_HUB_OFFLINE='1',
        TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',ABOT_DIFFSYNTH_ROOT=str(RT/'DiffSynth-Studio-h3-v2'),
        OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    sys.path[:0]=[str(RT/'code'),str(RT/'code/abot'),str(RT/'DiffSynth-Studio-h3-v2'),str(RT/'code/causal')]
    from current_prefix import current_prefix_feedback
    from report_stage1_anyflow import audit_inputs,summarize_video
    checkpoint=BRIDGE/'train_48/step_00';script=RT/'code/causal/benchmark.py'
    argv=[str(script),'--modes','cached','--num-frames','39','--steps',str(a.steps),'--seed','13',
        '--flow-shift','2.22','--precision-profile','h3_fp32','--chunk-frames','5','--history-chunks','5',
        '--cache-device','cpu','--action-preset',a.action,'--action-prefix-mode','own','--action-feedback',
        '--anchor-mode','dynamic_last_frame_rgb_dual','--history-source','generated',
        '--causal-adapter',str(checkpoint/'causal_adapter.pt'),'--stage1-lora',str(checkpoint/'stage1_lora.pt'),
        '--out-dir',str(out),'--save-latents']
    receipt=dict(status='running',pid=os.getpid(),gpu=a.gpu,optimizer_step=0,action=a.action,
        started_at=datetime.now().astimezone().isoformat(),argv=argv,
        routing='own action + common prefix reads current video; history raw KV unchanged',
        source_sha256={str(p):sha(p) for p in [Path(__file__),BASE/'current_prefix.py',script]},
        checkpoint_sha256={n:sha(checkpoint/n) for n in ('causal_adapter.pt','stage1_lora.pt')})
    dest=out/'candidate_protocol.json'
    def save():
        tmp=dest.with_suffix('.tmp.json');tmp.write_text(json.dumps(receipt,indent=2)+'\n');tmp.replace(dest)
    save()
    try:
        sys.argv=argv
        with current_prefix_feedback():runpy.run_path(str(script),run_name='__main__')
        row=summarize_video(out,'cached',label=f'Original weights current-prefix {a.steps}/chunk',
            action=a.action,history='generated',step=0,nfe=a.steps)
        assert row and row['noisy_forwards']==3*a.steps and row['clean_commits']==3
        row['input_audit']=audit_inputs(out,BRIDGE.parents[2]/f'outputs/2026-10-02-03/action_{a.action}_teacher_39')
        assert row['input_audit']['exactly_equal']
        row['routing']=receipt['routing']
        (out/'evaluation.json').write_text(json.dumps(row,indent=2)+'\n')
        receipt['status']='complete'
    except BaseException as exc:
        receipt.update(status='failed',error=repr(exc));raise
    finally:save()


if __name__=='__main__':main()
