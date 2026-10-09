"""Check actual pre-update checkpoint restoration, including Adam and RNG."""
from datetime import datetime
import json
from pathlib import Path

import torch


def equal(a, b):
    if torch.is_tensor(a) or torch.is_tensor(b):
        return (torch.is_tensor(a) and torch.is_tensor(b) and a.dtype == b.dtype
                and a.shape == b.shape and torch.equal(a, b))
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b


def audit(source, restored, output):
    source, restored, output = map(Path, (source, restored, output))
    load = lambda p: torch.load(p, map_location='cpu', weights_only=True)
    before, after = load(source / 'trainer_state.pt'), load(restored / 'trainer_state.pt')
    checks = {}
    for name in ('causal_adapter.pt', 'action_adapter.pt', 'stage1_lora.pt', 'anyflow_adapter.pt'):
        a, b = load(source / name), load(restored / name)
        checks[name] = equal({k:v for k,v in a.items() if k != 'metadata'},
                             {k:v for k,v in b.items() if k != 'metadata'})
        checks[name + ':step'] = a['metadata']['optimizer_step'] == b['metadata']['optimizer_step']
    for key in ('format', 'optimizer', 'optimizer_step', 'updates', 'teacher_artifacts',
                'logical_rng_state', 'cpu_rng_state', 'cuda_rng_state'):
        checks[key] = equal(before[key], after[key])
    mutable = {'out_dir', 'steps', 'checkpoint_every', 'resume_from', 'device'}
    checks['fixed_config'] = equal({k:v for k,v in before['config'].items() if k not in mutable},
                                  {k:v for k,v in after['config'].items() if k not in mutable})
    row = dict(at=datetime.now().astimezone().isoformat(), source=str(source.resolve()),
        reloaded=str(restored.resolve()), checks=checks,
        scope='Actual saved pre-update state; excludes mutable metadata. This does not assert bitwise deterministic serial/parallel CUDA optimization.')
    output.write_text(json.dumps(row, indent=2) + '\n')
    if not all(checks.values()):
        raise RuntimeError(f'Restored checkpoint differs: {[k for k,v in checks.items() if not v]}')
    return row
