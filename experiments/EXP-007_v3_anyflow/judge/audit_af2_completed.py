"""Read-only CPU audit of a terminal AF2 run; no GPU or video claim."""
from pathlib import Path
import hashlib
import json
import math
import torch

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[2]
OUT = ROOT / 'H3-World/outputs/EXP-007_v3_anyflow_af2'


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    result = json.loads((OUT / 'result.json').read_text())
    budget = json.loads((OUT / 'budget.json').read_text())
    assert result['status'] in ('complete_32_pending_judge', 'budget_stopped_pending_judge')
    assert 'end' in budget, 'Training still owns an open budget'
    checks = {}
    last = result['last_complete_step']
    rows = result['updates']
    checks['complete_update_sequence'] = [x['step'] for x in rows] == list(range(2, last + 1))
    checks['terminal_status_matches_steps'] = (
        2 <= last <= 32 and (result['status'] == 'complete_32_pending_judge') == (last == 32))
    checks['exact_outer_calls'] = (
        budget['forwards'] == 17 * (last - 1) and budget['backward'] == 4 * (last - 1)
        and budget['updates'] == last - 1 and budget['vae'] == 0)
    checks['time_budget'] = 0 < budget['gpu_seconds'] <= 7200
    checks['per_step_actions_cache_and_counts'] = all(
        x['action'] == ('D' if x['step'] % 2 == 0 else 'A')
        and x['history_cache_read_unchanged'] and x['history_cache_rebuilt_this_step']
        and x['forwards'] == 17 * (x['step'] - 1)
        and x['backward'] == 4 * (x['step'] - 1)
        and x['updates'] == x['step'] - 1 for x in rows)
    checks['finite_positive_group_gradients'] = all(
        all(math.isfinite(x[k]) and x[k] > 0
            for k in ('target_grad_norm', 'qkv_grad_norm', 'grad_clip_pre_norm')) for x in rows)
    checks['sample_classes_and_losses'] = all(
        [s['sample_type'] for s in x['samples']] == ['diffusion', 'diffusion', 'endpoint', 'flow_map']
        and all(s['model_evaluations'] == 4 and 0 <= s['target_sigma'] <= s['sigma'] <= 1
                and all(math.isfinite(s[k]) for k in ('raw_loss', 'weight', 'weighted_loss',
                                                      'finite_difference_norm', 'adaptive_scale'))
                for s in x['samples']) for x in rows)
    checks['memory_cap'] = result['peak_allocated_gib'] <= 44 and all(
        x['peak_allocated_gib'] <= 44 for x in rows)
    manifest_path = EXP / 'source_manifest_v3_af2.json'
    manifest = json.loads(manifest_path.read_text())
    checks['manifest_and_sources_unchanged'] = (
        result['source_manifest_sha256'] == sha(manifest_path)
        and all(sha(ROOT / e['path']) == e['sha256'] for e in manifest['sources'].values()))
    checks['resume_step1_verified'] = (
        result['resume_verified']['step'] == 1
        and result['resume_verified']['optimizer_entries'] == 20
        and result['resume_verified']['logical_rng_restored']
        and result['resume_verified']['cpu_cuda_rng_restored'])
    receipts = result['checkpoint_paths']
    checks['unique_final_checkpoint'] = sum(x['step'] == last for x in receipts) == 1
    for receipt in receipts:
        step, folder = receipt['step'], ROOT / receipt['path']
        checks[f'step{step}_file_hashes'] = all(
            sha(folder / name) == expected for name, expected in receipt['files'].items())
        qkv = torch.load(folder / 'qkv.pt', map_location='cpu', weights_only=True)
        target = torch.load(folder / 'target_time.pt', map_location='cpu', weights_only=True)
        state = torch.load(folder / 'trainer_state.pt', map_location='cpu', weights_only=True)
        checks[f'step{step}_metadata'] = (
            state['metadata'] == qkv['metadata'] == target['metadata']
            and state['step'] == state['metadata']['step'] == step
            and state['metadata']['precision'] == 'h3_fp32'
            and state['metadata']['protocol'] ==
                'V3 strict causal/global/current-prefix/Single I0/12+5/sigma0 student KV')
        checks[f'step{step}_paired_weights'] = (
            state['qkv_sha256'] == receipt['files']['qkv.pt']
            and state['target_sha256'] == receipt['files']['target_time.pt']
            and target['velocity_convention'] == 'noise-clean'
            and target['gate'] == .25 and qkv['rank'] == 8
            and qkv['block_indices'] == list(range(42, 50)))
        checks[f'step{step}_optimizer_and_rng'] = (
            len(state['optimizer']['state']) == 20
            and all(float(v['step']) == step for v in state['optimizer']['state'].values())
            and all(torch.is_tensor(state[k]) for k in ('logical_rng_state', 'cpu_rng_state', 'cuda_rng_state')))
    audit = dict(task='EXP-007/v3-AF2', pass_all=all(checks.values()), checks=checks,
        last_complete_step=last, forwards=budget['forwards'], backward=budget['backward'],
        updates=budget['updates'], gpu_seconds=budget['gpu_seconds'],
        gpu_hours=budget['gpu_seconds'] / 3600, peak_allocated_gib=result['peak_allocated_gib'],
        checkpoint_receipts=receipts,
        scope='Terminal training engineering evidence only; video ability requires AF3 visual review',
        counting='Explicit outer forwards and backwards; internal gradient-checkpoint recomputation is not sampling NFE')
    (HERE / 'af2_completed_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps(audit, indent=2))
    assert audit['pass_all'], [k for k, v in checks.items() if not v]


if __name__ == '__main__':
    main()
