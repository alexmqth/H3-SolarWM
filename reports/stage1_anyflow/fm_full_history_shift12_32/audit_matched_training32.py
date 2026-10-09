"""Compare completed, real FM32/AnyFlow32 training logs without CUDA work."""
from datetime import datetime
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
ANY = ROOT / 'outputs/2026-10-08-10/stage1_shift12_duration64'
logs = {label: json.loads((directory / 'train_32/training.json').read_text())
        for label, directory in [('FM', OUT), ('AnyFlow', ANY)]}
for label, log in logs.items():
    if log['status'] != 'complete' or len(log['updates']) != 32:
        raise SystemExit(f'{label}: wait for complete 32-update training, do not infer results')

fm, af = logs['FM'], logs['AnyFlow']
shared_fields = ('actions', 'teacher_dir', 'causal_adapter', 'action_adapter',
                 'precision_profile', 'adapter_scope', 'bank_rank', 'bank_alpha',
                 'history_gradient_mode', 'train_target_time', 'logical_batch',
                 'lr', 'chunk_frames', 'history_chunks', 'anchor_mode',
                 'action_prefix_mode', 'action_feedback', 'flow_shift',
                 'training_timestep_shift', 'validation_timestep_shift',
                 'seed', 'validation_seed', 'validation_cases')
checks = {f'config.{k}': fm['config'][k] == af['config'][k] for k in shared_fields}
checks['same_trainable_parameter_count'] = (
    fm['stage1_lora']['trainable_parameters'] == af['stage1_lora']['trainable_parameters'] == 43237376)
checks['action_chunk_and_current_sigmas'] = all(
    f['action'] == a['action'] and f['chunk'] == a['chunk']
    and len(f['samples']) == len(a['samples']) == 4
    and all(x['sigma'] == y['sigma'] for x, y in zip(f['samples'], a['samples']))
    for f, a in zip(fm['updates'], af['updates']))
checks['FM_targets_are_diagonal'] = all(
    x['target_sigma'] == x['sigma'] for u in fm['updates'] for x in u['samples'])
checks['full_history_forward_counts'] = all(
    u['differentiable_history_forwards'] == 4 * u['chunk']
    for log in logs.values() for u in log['updates'])
if not all(checks.values()):
    raise RuntimeError(f'Matched-control contract failed: {checks}')

dose = {}
for label, log in logs.items():
    dose[label] = {k: sum(u[k] for u in log['updates']) for k in
                  ('model_evaluations', 'clean_commit_forwards', 'differentiable_history_forwards')}
    dose[label]['optimizer_updates'] = len(log['updates'])
    dose[label]['loss_samples'] = sum(len(u['samples']) for u in log['updates'])

common_validation = []
for action in 'AD':
    fv = next(x for x in fm['validation_after'] if x['action'] == action)
    av = next(x for x in af['validation_after'] if x['action'] == action)
    for i in (0, 1):
        f, a = fv['samples'][i], av['samples'][i]
        if not (f['sigma'] == a['sigma'] == f['target_sigma'] == a['target_sigma']):
            raise RuntimeError('Validation comparison is not on the same diagonal')
        common_validation.append(dict(action=action, sample_index=i, sigma=f['sigma'],
                                      FM_raw_loss=f['raw_loss'], AnyFlow_raw_loss=a['raw_loss']))

result = dict(at=datetime.now().astimezone().isoformat(), checks=checks,
    scope='Actual completed training logs, shared config, action/chunk/sigma sequence and forward counts.',
    initial_tensor_and_rng_audit='gpu_matched_initialization.json and gpu_initial_audit_early.json',
    actual_training_noise_hashes_compared=False, dose=dose, common_validation=common_validation,
    caveats=['Same updates and trainable parameter count are not equal compute.',
             'Training noise hashes were predicted in the CPU schedule, not logged in the GPU trainer.',
             'Only the first two common r=t validation samples per action are compared.',
             'AnyFlow32 contains resumed history; script wall times include different launch/validation overhead.',
             'No visual or action acceptance follows from this audit.'])
(OUT / 'matched_training32_audit.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
