"""Read-only CPU audit of completed DMD continuation, including noise replay."""
from pathlib import Path
import hashlib
import json
import math
import torch

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[2]
OUT = ROOT / 'H3-World/outputs/EXP-008_v3_dmd_train_v2'
PILOT = ROOT / 'H3-World/outputs/EXP-008_v3_dmd_pilot'

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(4 << 20), b''):
            h.update(b)
    return h.hexdigest()

def tensor_sha(x):
    return hashlib.sha256(x.cpu().numpy().tobytes()).hexdigest()

result = json.loads((OUT / 'result.json').read_text())
budget = json.loads((OUT / 'budget.json').read_text())
assert result['status'] in ('complete_8_pending_judge', 'budget_stopped_pending_judge') and 'end' in budget
last = result['last_complete_cycle']
assert 2 <= last <= 8
rows = result['cycles']
manifest_path = EXP / 'dmd_train/source_manifest_v2.json'
manifest = json.loads(manifest_path.read_text())
parent = torch.load(PILOT / 'cycle_01/trainer_state.pt', map_location='cpu', weights_only=True)
checks = {}
checks['sources_unchanged'] = result['source_manifest_sha256'] == sha(manifest_path) and all(
    sha(ROOT / e['path']) == e['sha256'] for e in manifest['sources'].values())
checks['complete_cycle_order_and_budget'] = (
    [x['cycle'] for x in rows] == list(range(2, last + 1))
    and (budget['forwards'], budget['backward'], budget['updates'], budget['vae']) == (1 + 14 * (last-1), 2 * (last-1), 2 * (last-1), 0)
    and budget['physical_gpu_ids'] == [0, 2, 5]
    and 0 < budget['wall_seconds'] <= 2700
    and abs(budget['conservative_three_gpu_hours'] - budget['wall_seconds'] * 3 / 3600) < 1e-9
    and budget['conservative_three_gpu_hours'] <= 2.25)
restored = result['restored_rng']
checks['actual_all_rng_restoration'] = (
    restored['cpu_sha256'] == tensor_sha(parent['cpu_rng'])
    and restored['cuda_sha256'] == [tensor_sha(x) for x in parent['cuda_rng']]
    and restored['train_noise_sha256'] == tensor_sha(parent['train_noise_rng'])
    and restored['score_noise_sha256'] == tensor_sha(parent['score_noise_rng']))
events = budget['events']
sigmas = json.loads((ROOT / 'H3-World/outputs/EXP-006_v3_fm8_full/chunk_0_12.json').read_text())['sigmas']
gen = torch.Generator(); gen.set_state(parent['train_noise_rng'])
score_gen = torch.Generator(); score_gen.set_state(parent['score_noise_rng'])
expected_rng = {}
for row in rows:
    cycle = row['cycle']; path = 'AD' if cycle % 2 == 0 else 'AA'
    subset = [x for x in events if x['detail'].get('cycle') == cycle]
    fwd = [x for x in subset if x['kind'] == 'forward']
    back = [x for x in subset if x['kind'] == 'backward']
    upd = [x for x in subset if x['kind'] == 'update']
    checks[f'cycle{cycle}_action_and_exact_calls'] = (
        row['status'] == 'complete' and row['path'] == path and row['action'] == path[-1]
        and row['forwards'] == 14 and len(fwd) == 14
        and [x['role'] for x in fwd] == ['student', 'fake'] + ['student']*8 + ['fake', 'fake', 'teacher', 'fake']
        and [x['role'] for x in back] == ['fake', 'student']
        and [x['role'] for x in upd] == ['fake', 'student']
        and all(x['detail']['path'] == path for x in fwd if x['detail']['index'] == 1))
    checks[f'cycle{cycle}_map_and_score_protocol'] = all(
        x['detail']['sigma'] == t and x['detail']['r'] == r and x['detail']['grad']
        for x, t, r in zip(fwd[2:10], sigmas[:-1], sigmas[1:])) and all(
            x['detail']['sigma'] == .6 and x['detail']['r'] is None and not x['detail']['grad'] for x in fwd[-2:])
    checks[f'cycle{cycle}_cache_and_base'] = all(row[k] for k in (
        'teacher_cache_reused_unchanged', 'student_cache_rebuilt_and_read_unchanged',
        'fake_cache_rebuilt_after_update_and_read_unchanged', 'base_weights_frozen'))
    checks[f'cycle{cycle}_gradients_and_updates'] = (
        len(row['velocity_grad_norms']) == 8
        and all(math.isfinite(v) and v > 0 for v in row['velocity_grad_norms'] + [row[k] for k in (
            'fake_grad_norm', 'student_target_grad_norm', 'student_qkv_grad_norm')])
        and all(row[k] for k in ('fake_changed', 'student_qkv_changed', 'student_target_changed'))
        and math.isfinite(row['fake_loss']) and math.isfinite(row['dmd_loss'])
        and all(0 < v <= 44 for v in row['peak_allocated_gib_by_role'].values()))
    noise = torch.randn((1, 24, 5, 30, 52), generator=gen, dtype=torch.float32)
    checks[f'cycle{cycle}_independent_noise_replay'] = tensor_sha(noise) == row['training_noise_sha256']
    for _ in range(2):
        torch.randn((1, 24, 5, 30, 52), generator=score_gen, dtype=torch.float32)
    expected_rng[cycle] = (gen.get_state().clone(), score_gen.get_state().clone())
receipts = result['checkpoint_paths']
checks['final_checkpoint_present'] = sum(x['cycle'] == last for x in receipts) == 1
for receipt in receipts:
    cycle = receipt['cycle']; folder = ROOT / receipt['path']
    checks[f'checkpoint{cycle}_hashes'] = all(sha(folder/name) == h for name, h in receipt['files'].items())
    states = {name: torch.load(folder/name, map_location='cpu', weights_only=True) for name in receipt['files']}
    state = states['trainer_state.pt']; metadata = state['metadata']
    checks[f'checkpoint{cycle}_metadata'] = all(x['metadata'] == metadata for x in states.values()) and (
        metadata['cycle'] == cycle and metadata['task'] == 'EXP-008/v2-DMD-TRAIN'
        and metadata['source_manifest_sha256'] == sha(manifest_path)
        and metadata['parent_pilot_result_sha256'] == manifest['sources']['pilot_result']['sha256'])
    checks[f'checkpoint{cycle}_optimizer_and_noise_rng'] = (
        len(state['student_optimizer']['state']) == 20 and len(state['fake_optimizer']['state']) == 16
        and all(float(v['step']) == cycle for v in state['student_optimizer']['state'].values())
        and all(float(v['step']) == cycle + 1 for v in state['fake_optimizer']['state'].values())
        and torch.equal(state['train_noise_rng'], expected_rng[cycle][0])
        and torch.equal(state['score_noise_rng'], expected_rng[cycle][1]) and len(state['cuda_rng']) == 3)
audit = dict(task=manifest['task'], pass_all=all(checks.values()), checks=checks,
    last_complete_cycle=last, forwards=budget['forwards'], backward=budget['backward'], updates=budget['updates'],
    wall_seconds=budget['wall_seconds'], gpu_hours=budget['conservative_three_gpu_hours'],
    peak_allocated_gib_by_role=result['peak_allocated_gib_by_role'], checkpoint_receipts=receipts,
    scope='Bounded training engineering only; video quality requires separate matched evaluation.')
(HERE / 'train_v2_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
print(json.dumps(audit, indent=2))
assert audit['pass_all']
