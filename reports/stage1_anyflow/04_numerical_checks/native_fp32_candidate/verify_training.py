"""CPU optimizer/RNG compatibility and native-precision training integration."""
from pathlib import Path
import json
import subprocess
import sys

import torch

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
NEW = OUT / 'runtime/code/causal/train_stage1_anyflow.py'
OLD = ROOT / 'outputs/2026-10-08-04/stage1_anyflow39_frozen_time/runtime/code/causal/train_stage1_anyflow.py'

def train(name, trainer, steps, profile=None, resume=None, expected_success=True):
    dest = OUT / name
    if dest.exists():
        raise RuntimeError(f'Refusing to overwrite {dest}')
    dest.mkdir()
    command = [sys.executable, str(trainer), '--smoke', '--objective', 'anyflow',
        '--steps', str(steps), '--checkpoint-every', '2', '--no-train-target-time',
        '--out-dir', str(dest)]
    if profile:
        command += ['--precision-profile', profile]
    if resume:
        command += ['--resume-from', str(resume)]
    with (dest / 'run.log').open('w') as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    if (result.returncode == 0) != expected_success:
        raise RuntimeError((dest / 'run.log').read_text()[-2500:])
    if not expected_success:
        assert 'Resume protocol mismatch' in (dest / 'run.log').read_text()
        return dest
    report = json.loads((dest / 'training.json').read_text())
    assert report['status'] == 'complete' and report['qkv_parameters_changed']
    assert report['target_time_trainable_parameters'] == 0 and not report['target_time_parameters_changed']
    return dest

def equal(a,b):
    if torch.is_tensor(a):
        return torch.is_tensor(b) and a.dtype==b.dtype and torch.equal(a,b)
    if isinstance(a,dict):
        return isinstance(b,dict) and a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):
        return isinstance(b,type(a)) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b

def compare(a,b):
    for file in ('causal_adapter.pt','anyflow_adapter.pt'):
        x,y=[torch.load(p/file,map_location='cpu',weights_only=True) for p in (a,b)]
        x.pop('metadata');y.pop('metadata')
        assert equal(x,y), file
    x,y=[torch.load(p/'trainer_state.pt',map_location='cpu',weights_only=True) for p in (a,b)]
    for key in ('optimizer','optimizer_step','logical_rng_state','cpu_rng_state','cuda_rng_state'):
        assert equal(x[key],y[key]), key
    for xrow,yrow in zip(x['updates'],y['updates']):
        xx,yy=dict(xrow),dict(yrow);xx.pop('seconds');yy.pop('seconds')
        assert equal(xx,yy),'loss/gradient update history'

old4 = train('cpu_old4', OLD, 4)
old2 = train('cpu_old2', OLD, 2)
legacy_resume = train('cpu_legacy_resume4', NEW, 4, 'legacy', old2/'step_02')
compare(old4,legacy_resume)
new4 = train('cpu_fp32_full4', NEW, 4, 'h3_fp32')
new2 = train('cpu_fp32_part2', NEW, 2, 'h3_fp32')
new_resume = train('cpu_fp32_resume4', NEW, 4, 'h3_fp32', new2/'step_02')
compare(new4,new_resume)
train('cpu_reject_precision_switch', NEW, 4, 'h3_fp32', old2/'step_02', expected_success=False)
result = dict(status='passed', scope='CPU small H3 only; not 33B quality or GPU resume proof',
    old_checkpoint_legacy_resume_bitwise=True, fp32_uninterrupted_vs_resume_bitwise=True,
    checked=['adapter values','Adam','RNG','per-sample losses','QKV gradients','frozen time MLP'],
    precision_change_rejected_on_exact_resume=True)
(OUT/'cpu_training_validation.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
