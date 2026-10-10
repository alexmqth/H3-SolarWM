"""Independent CPU audit of the terminal real-model DMD pilot."""
from pathlib import Path
import hashlib
import json
import math
import torch

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[2]
OUT = ROOT / 'H3-World/outputs/EXP-008_v3_dmd_pilot'

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(4 << 20), b''):
            h.update(b)
    return h.hexdigest()

def finite_tree(x):
    if torch.is_tensor(x):
        return bool(torch.isfinite(x).all())
    if isinstance(x, dict):
        return all(finite_tree(v) for v in x.values())
    if isinstance(x, (tuple, list)):
        return all(finite_tree(v) for v in x)
    return True

result = json.loads((OUT / 'result.json').read_text())
budget = json.loads((OUT / 'budget.json').read_text())
assert result['status'] == 'complete_pending_judge' and 'end' in budget
manifest_path = EXP / 'dmd_cpu/source_manifest_pilot.json'
manifest = json.loads(manifest_path.read_text())
checks = {}
checks['manifest_and_all_sources_unchanged'] = result['source_manifest_sha256'] == sha(manifest_path) and all(
    sha(ROOT / e['path']) == e['sha256'] for e in manifest['sources'].values())
checks['budget_and_three_card_accounting'] = (
    (budget['forwards'], budget['backward'], budget['updates'], budget['vae']) == (17, 3, 3, 0)
    and budget['physical_gpu_ids'] == [0, 2, 5]
    and 0 < budget['wall_seconds'] <= 1800
    and abs(budget['conservative_three_gpu_hours'] - budget['wall_seconds'] * 3 / 3600) < 1e-9
    and budget['conservative_three_gpu_hours'] <= 1.5)
events = budget['events']
fwd = [x for x in events if x['kind'] == 'forward']
back = [x for x in events if x['kind'] == 'backward']
upd = [x for x in events if x['kind'] == 'update']
checks['exact_role_sequence'] = (
    [x['role'] for x in fwd] == ['teacher', 'fake', 'student'] + ['student'] * 8 + ['fake'] * 4 + ['teacher', 'fake']
    and [x['role'] for x in back] == ['fake', 'fake', 'student']
    and [x['role'] for x in upd] == ['fake', 'fake', 'student']
    and all(x['ordinal'] == i + 1 for group in [fwd, back, upd] for i, x in enumerate(group)))
sigmas = json.loads((ROOT / 'H3-World/outputs/EXP-006_v3_fm8_full/chunk_0_12.json').read_text())['sigmas']
checks['eight_continuous_maps_and_matched_scores'] = result['student_eight_map_graph_live'] and all(
    e['detail'] == dict(index=1, sigma=t, r=r, commit=False, grad=True)
    for e, t, r in zip(fwd[3:11], sigmas[:-1], sigmas[1:])) and all(
        e['detail'] == dict(index=1, sigma=.6, r=None, commit=False, grad=False) for e in fwd[-2:])
checks['all_map_and_parameter_gradients_finite_positive'] = len(result['velocity_grad_norms']) == 8 and all(
    math.isfinite(v) and v > 0 for v in result['velocity_grad_norms'] + [
        result['student_target_grad_norm'], result['student_qkv_grad_norm'], result['student_clip_pre_norm']])
checks['fake_updates_and_cache_rebuilds'] = len(result['fake_warmups']) == 2 and all(
    x['cache_rebuilt'] and x['cache_read_unchanged'] and x['parameters_changed']
    and math.isfinite(x['loss']) and 0 < x['grad_norm'] < float('inf') for x in result['fake_warmups'])
checks['student_groups_changed'] = result['student_qkv_parameters_changed'] and result['student_target_parameters_changed']
checks['finite_dmd_direction_and_memory'] = math.isfinite(result['dmd_loss']) and result['direction_stats']['direction_rms'] > 0 and all(
    0 < v <= 44 for v in result['peak_allocated_gib_by_role'].values())
folder = OUT / 'cycle_01'
checks['paired_checkpoint_hashes'] = all(sha(folder / name) == expected for name, expected in result['checkpoint_files'].items())
qkv = torch.load(folder / 'student_qkv.pt', map_location='cpu', weights_only=True)
target = torch.load(folder / 'student_target.pt', map_location='cpu', weights_only=True)
fake = torch.load(folder / 'fake_qkv.pt', map_location='cpu', weights_only=True)
state = torch.load(folder / 'trainer_state.pt', map_location='cpu', weights_only=True)
checks['checkpoint_metadata_and_finite_tensors'] = (
    state['metadata'] == qkv['metadata'] == target['metadata'] == fake['metadata']
    and state['metadata']['parent_af2_step'] == 32
    and state['metadata']['source_manifest_sha256'] == sha(manifest_path)
    and target['velocity_convention'] == 'noise-clean' and target['gate'] == .25
    and qkv['rank'] == fake['rank'] == 8
    and qkv['block_indices'] == fake['block_indices'] == list(range(42, 50))
    and all(finite_tree(v) for v in (qkv, target, fake, state)))
checks['optimizer_and_rng'] = (
    len(state['student_optimizer']['state']) == 20 and len(state['fake_optimizer']['state']) == 16
    and all(float(x['step']) == 1 for x in state['student_optimizer']['state'].values())
    and all(float(x['step']) == 2 for x in state['fake_optimizer']['state'].values())
    and len(state['cuda_rng']) == 3
    and all(torch.is_tensor(state[k]) for k in ('cpu_rng', 'train_noise_rng', 'score_noise_rng')))
parent_q = torch.load(ROOT / manifest['sources']['af2_qkv']['path'], map_location='cpu', weights_only=True)
parent_t = torch.load(ROOT / manifest['sources']['af2_target']['path'], map_location='cpu', weights_only=True)
checks['saved_student_really_differs_from_parent'] = any(
    not torch.equal(x, y) for k in ('lora_A', 'lora_B') for x, y in zip(qkv[k], parent_q[k])) and any(
        not torch.equal(target['weights'][k], parent_t['weights'][k]) for k in target['weights'])
audit = dict(task=manifest['task'], pass_all=all(checks.values()), checks=checks,
    wall_seconds=budget['wall_seconds'], gpu_hours=budget['conservative_three_gpu_hours'],
    peak_allocated_gib_by_role=result['peak_allocated_gib_by_role'], checkpoint_files=result['checkpoint_files'],
    velocity_grad_norms=result['velocity_grad_norms'], direction_stats=result['direction_stats'],
    scope='One real-model DMD cycle engineering only. No VAE/video/quality claim; base and cache immutability rely on reviewed runtime assertions.')
(HERE / 'pilot_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
print(json.dumps(audit, indent=2))
assert audit['pass_all']
