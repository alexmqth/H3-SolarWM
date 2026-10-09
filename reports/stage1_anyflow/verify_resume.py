"""Real small-H3 CPU: uninterrupted4 == checkpoint2 + resumed2."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import subprocess
import sys

import torch

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--trainer', type=Path, default=ROOT / 'code/causal/train_stage1_anyflow.py')
ap.add_argument('--work-dir', type=Path, required=True, help='New directory; existing evidence is never overwritten')
args = ap.parse_args()
TRAINER = args.trainer.resolve()
WORK = args.work_dir.resolve()


def run(name, objective, train_time, steps, resume=None, lr='.001', error=None):
    target = WORK / name
    command = [sys.executable, str(TRAINER), '--smoke', '--steps', str(steps),
        '--checkpoint-every', '2', '--lr', lr, '--objective', objective,
        '--out-dir', str(target)]
    if train_time:
        command.append('--train-target-time')
    if resume is not None:
        command += ['--resume-from', str(resume)]
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    (WORK / f'{name}.log').write_text(result.stdout + result.stderr)
    if error:
        assert result.returncode != 0 and error in result.stderr, result.stderr
    else:
        assert result.returncode == 0, result.stderr
    return target


def equal_tree(a, b):
    if torch.is_tensor(a):
        return torch.is_tensor(b) and a.dtype == b.dtype and torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal_tree(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(equal_tree(x, y) for x, y in zip(a, b))
    return a == b


def verify_case(case):
    label, objective, train_time = case
    full = run(label + '_full4', objective, train_time, 4)
    half = run(label + '_half2', objective, train_time, 2)
    resumed = run(label + '_resumed4', objective, train_time, 4, half / 'step_02')
    load = lambda p: torch.load(p, map_location='cpu', weights_only=True)
    names = ['causal_adapter.pt'] + (['anyflow_adapter.pt'] if objective == 'anyflow' else [])
    for name in names:
        x, y = load(full / name), load(resumed / name)
        x.pop('metadata'); y.pop('metadata')
        assert equal_tree(x, y), (label, name)
    x, y = load(full / 'trainer_state.pt'), load(resumed / 'trainer_state.pt')
    for key in ('optimizer', 'logical_rng_state', 'cpu_rng_state'):
        assert equal_tree(x[key], y[key]), (label, key)
    assert x['optimizer_step'] == y['optimizer_step'] == 4
    for i in range(4):
        a, b = dict(x['updates'][i]), dict(y['updates'][i])
        a.pop('seconds'); b.pop('seconds')
        assert equal_tree(a, b), (label, 'update', i)
    return dict(case=label, full_updates=4, resumed_from=2,
        all_adapter_tensors_bitwise_equal=True, adam_state_bitwise_equal=True,
        logical_and_global_rng_equal=True, all_loss_and_gradient_records_equal=True)


if WORK.exists():
    raise SystemExit('Refusing to overwrite resume validation')
WORK.mkdir()
with ThreadPoolExecutor(max_workers=3) as pool:
    cases = list(pool.map(verify_case, [('frozen', 'anyflow', False),
                                      ('trainable', 'anyflow', True), ('fm', 'fm', False)]))
reference = WORK / 'frozen_half2/step_02'
run('reject_lr', 'anyflow', False, 4, reference, lr='.002', error='Resume protocol mismatch')
run('reject_step', 'anyflow', False, 2, reference, error='target TOTAL update count')
bad = WORK / 'mismatched_weights'
shutil.copytree(reference, bad)
shutil.copy2(WORK / 'trainable_half2/step_02/anyflow_adapter.pt', bad / 'anyflow_adapter.pt')
run('reject_mixed', 'anyflow', False, 4, bad, error='Checkpoint adapter hash mismatch')
missing = WORK / 'without_optimizer'
missing.mkdir()
run('reject_missing', 'anyflow', False, 4, missing, error='Checkpoint lacks optimizer/RNG state')
result = dict(scope='small actual-H3 CPU continuation; not pretrained CUDA/video acceptance',
    cases=cases, rejected=['changed_learning_rate', 'nonincreasing_total_steps',
                          'mixed_adapter_weights', 'missing_optimizer_and_rng'])
(WORK / 'resume_validation.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
