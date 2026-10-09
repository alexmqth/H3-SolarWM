"""Read-only receipt for the bounded real-data FM48 experiment; no GPU."""
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
import sys

import torch

BASE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare_banks(bank0, bank48):
    assert bank0['targets'] == bank48['targets']
    changed = {}
    for name, tensors in bank48['weights'].items():
        old = bank0['weights'][name]
        assert tensors.keys() == old.keys()
        assert all(len(tensors[k]) == len(old[k]) for k in tensors)
        pairs = [(a, b) for k in tensors for a, b in zip(tensors[k], old[k])]
        changed[name] = any(not torch.equal(a, b) for a, b in pairs)
        assert all(torch.isfinite(a).all() for a, _ in pairs)
    return changed


def main():
    data = json.loads((BASE / 'train_48/training.json').read_text())
    assert data['status'] == 'complete' and len(data['updates']) == 48
    checkpoint = BASE / 'train_48/step_48'
    sys.path.insert(0, str(BASE / 'runtime/code'))
    from causal.training_state import load_training_state
    config = dict(data['config'], steps=49)
    state = load_training_state(checkpoint, config)
    assert state['optimizer_step'] == 48 and state['updates'] == data['updates']
    assert config['objective'] == 'fm' and config['history_gradient_mode'] == 'full'
    assert config['causal_adapter'] is None and config['action_adapter'] is None
    assert config['real_manifest_sha256'] == sha(Path(config['real_data_manifest']))
    assert set(state['weight_sha256']) == {'causal_adapter.pt', 'stage1_lora.pt'}
    start = BASE / 'train_48/step_00'
    load = lambda p: torch.load(p, map_location='cpu', weights_only=True)
    visual0, visual48 = load(start / 'causal_adapter.pt'), load(checkpoint / 'causal_adapter.pt')
    assert visual0['block_indices'] == visual48['block_indices']
    for key in ('lora_A', 'lora_B'):
        assert all(torch.equal(x, y) for x, y in zip(visual0[key], visual48[key]))
    assert all(torch.count_nonzero(x) == 0 for x in visual48['lora_B'])
    bank0, bank48 = load(start / 'stage1_lora.pt'), load(checkpoint / 'stage1_lora.pt')
    changed = compare_banks(bank0, bank48)
    assert all(changed.values())
    train = [x for x in state['teacher_artifacts'] if x['split'] == 'train']
    assert len(train) == 16
    counts = Counter()
    for i, update in enumerate(data['updates']):
        label = train[i % 16]['label']
        chunk = (i // 16 + i % 16) % 3
        assert update['step'] == i + 1 and update['action'] == label and update['chunk'] == chunk
        counts[(label, chunk)] += 1
    assert len(counts) == 48 and set(counts.values()) == {1}
    runtime = json.loads((BASE / 'runtime_manifest.json').read_text())
    assert all(sha(BASE / 'runtime' / p) == h for p, h in runtime.items())
    bins = lambda s: 'low' if s <= .2407808991 else ('mid' if s <= .6894410401 else 'high')
    validation = []
    summaries = defaultdict(lambda: {'before': [], 'after': []})
    for before, after in zip(data['validation_before'], data['validation_after']):
        assert before['action'] == after['action'] and before['chunk'] == after['chunk']
        assert len(before['samples']) == len(after['samples']) == 4
        for x, y in zip(before['samples'], after['samples']):
            assert all(x[k] == y[k] for k in ('sigma', 'target_sigma', 'weight', 'sample_type'))
            row = dict(clip=before['action'], chunk=before['chunk'], sigma=x['sigma'],
                       noise_bin=bins(x['sigma']), raw_before=x['raw_loss'], raw_after=y['raw_loss'])
            validation.append(row)
            summaries[row['noise_bin']]['before'].append(x['raw_loss'])
            summaries[row['noise_bin']]['after'].append(y['raw_loss'])
    assert len(validation) == 16
    by_noise = {k: dict(count=len(v['before']), before_mean=sum(v['before']) / len(v['before']),
                       after_mean=sum(v['after']) / len(v['after'])) for k, v in summaries.items()}
    out = dict(status='passed', audited_at=datetime.now().astimezone().isoformat(), optimizer_steps=48,
               real_manifest_sha256=config['real_manifest_sha256'], frozen_runtime_files=len(runtime),
               visual_frozen_zero_output=True, bank_modules_changed=sum(changed.values()),
               all_train_case_chunks_seen_once=True, checkpoint_hashes_verified=True,
               no_anyflow_or_old_action_residual=True,
               train_sigma_counts=dict(Counter(bins(s['sigma']) for u in data['updates'] for s in u['samples'])),
               validation=validation, validation_by_noise=by_noise,
               measured_update_seconds=sum(u['seconds'] for u in data['updates']),
               forward_counts_excluding_validation_and_checkpoint_recompute={
                   k: sum(u[k] for u in data['updates']) for k in
                   ('model_evaluations', 'clean_commit_forwards', 'differentiable_history_forwards')},
               limitations='Four fixed held-out cases, four sigmas each; absent bins are not inferred. '
                           'Raw losses are not video quality. Logged sigma/weight equality verified; '
                           'actual validation noise tensor hashes were not saved.')
    (BASE / 'completed_training_audit.json').write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps({k: v for k, v in out.items() if k != 'validation'}, indent=2))


if __name__ == '__main__':
    main()
