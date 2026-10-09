#!/usr/bin/env python3
"""Snapshot a running/completed AnyFlow pilot without declaring visual PASS.

Only complete video artifacts are included. Re-running updates the table and
contact sheets as the serial experiment queue produces more checkpoints.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path

import av
from PIL import Image, ImageDraw, ImageFont

from evaluate_action_control import evaluate as evaluate_flow
from evaluate_videos import evaluate as evaluate_video
from summarize_action_experiment import rgb_boundary


def audit_inputs(directory, teacher_directory):
    """Check saved conditioning tensors, not merely equal seed labels."""
    import torch
    torch.set_num_threads(4)
    current = torch.load(directory / 'conditioning.pt', map_location='cpu', weights_only=True)
    teacher = torch.load(teacher_directory / 'conditioning.pt', map_location='cpu', weights_only=True)
    checks = {}
    for key in ('initial_noise', 'audio_noise', 'prompt_embeds', 'anchor'):
        a, b = current[key], teacher[key]
        same_shape = a.shape == b.shape
        checks[key] = dict(shape=list(a.shape), dtype=str(a.dtype),
            exactly_equal=same_shape and a.dtype == b.dtype and torch.equal(a, b),
            max_abs_difference=float((a.float() - b.float()).abs().max()) if same_shape else None)
    checks['action_text_rows_equal'] = torch.equal(
        current['packed']['action_text_rows'], teacher['packed']['action_text_rows'])
    passed = all(checks[k]['exactly_equal'] for k in ('initial_noise', 'audio_noise', 'prompt_embeds', 'anchor'))
    passed = passed and checks['action_text_rows_equal']
    return dict(exactly_equal=passed, reference=str(teacher_directory.resolve()), checks=checks)


def summarize_video(directory, mode, *, label, action, history, step, nfe):
    runtime = json.loads((directory / f'{mode}.json').read_text())
    setup = json.loads((directory / 'setup.json').read_text())
    if runtime['status'] != 'complete':
        return None
    path = directory / f'{mode}.mp4'
    stats, flow = evaluate_video(path), evaluate_flow(path)
    if stats['frames'] != 39 or stats['fps'] != 24 or (stats['width'], stats['height']) != (832, 480):
        raise ValueError(f'Unexpected video protocol: {path}')
    return dict(label=label, action=action, history=history, optimizer_step=step,
        steps_per_chunk=nfe, path=str(path.resolve()), frames=stats['frames'],
        seconds=runtime['wall_including_shared_setup_seconds'],
        gpu_peak_MiB=runtime['memory_entire_run_peak_MiB'],
        vram_reserve_gib=setup.get('vram_reserve_gib'),
        cpu_kv_MiB=runtime.get('kv_cache_peak_MiB', 0),
        noisy_forwards=runtime['denoiser_forwards'],
        clean_commits=runtime.get('commit_forwards', 0),
        frame_gray_mad=stats['gray_pixel_difference_mean'],
        boundary_rgb_mad=rgb_boundary(path, [17, 34])['mean'],
        horizontal_flow=flow['horizontal_flow_px']['mean'])


def training_policy(runtime):
    """Read the generating checkpoint's run, including the historical default.

    Before the explicit flag was added, AnyFlow trained the target-time MLP.
    Keep those results distinct from the later frozen-time control.
    """
    checkpoint = runtime.get('anyflow_adapter')
    if not checkpoint:
        return 'not applicable'
    directory = Path(checkpoint).parent
    for path in (directory / 'training.json', directory.parent / 'training.json'):
        if path.exists():
            training = json.loads(path.read_text())
            flag = training.get('target_time_trainable',
                                training['config'].get('train_target_time', True))
            return 'trainable' if flag else 'frozen'
    raise ValueError(f'Missing training provenance for {checkpoint}')


def contacts(rows, path):
    indices = [0, 10, 20, 30, 38]
    image = Image.new('RGB', (320 * 5, len(rows) * 212), 'black')
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=13)
    for i, row in enumerate(rows):
        with av.open(row['path']) as c:
            frames = [f.to_image() for f in c.decode(video=0)]
        draw.text((4, i * 212 + 3), row['label'], fill='white', font=font)
        for j, frame in enumerate(indices):
            image.paste(frames[frame].resize((320, 185)), (j * 320, i * 212 + 27))
    image.save(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--pilot-root', type=Path, required=True)
    ap.add_argument('--teacher-root', type=Path, required=True)
    ap.add_argument('--smoke-root', type=Path)
    ap.add_argument('--comparison-root', type=Path, action='append', default=[],
                    help='Additional completed evaluations, e.g. uniform/frozen-time controls')
    args = ap.parse_args()
    report = args.pilot_root / 'report'
    report.mkdir(exist_ok=True)
    rows = []
    for action in 'AD':
        rows.append(summarize_video(args.teacher_root / f'action_{action}_teacher_39',
            'baseline', label=f'Original H3 / {action} / 30 full-sequence steps',
            action=action, history='bidirectional', step=None, nfe=None))
    for root in [args.pilot_root, *args.comparison_root]:
        for file in sorted((root / 'eval').rglob('evaluation.json')):
            data = json.loads(file.read_text())
            runtime = data['runtime']
            policy = training_policy(runtime)
            grid = runtime.get('video_sigma_grid', 'native')
            objective = data['objective']
            setup = json.loads((file.parent / 'setup.json').read_text())
            precision_profile = setup.get('precision', {}).get('profile', 'legacy')
            if objective == 'anyflow':
                objective += f' time={policy}'
            if precision_profile != 'legacy':
                objective += f' precision={precision_profile}'
            if setup.get('stage1_lora'):
                objective += ' scope=all_qkvo_ffn'
            training_config = setup.get('causal_adapter', {}).get('metadata', {}).get('config', {})
            history_gradient = training_config.get('history_gradient_mode', 'detached')
            if history_gradient != 'detached':
                objective += f' history_grad={history_gradient}'
            inference_shift = runtime.get('flow_shift', 2.22)
            training_shift = training_config.get('training_timestep_shift')
            training_shift = inference_shift if training_shift is None else training_shift
            validation_shift = training_config.get('validation_timestep_shift')
            validation_shift = inference_shift if validation_shift is None else validation_shift
            if training_shift != inference_shift:
                objective += f' train_shift={training_shift:g}'
            row = summarize_video(file.parent, 'cached',
                label=f"{objective} step{data['step']:02d} grid={grid} / {data['action']} / "
                      f"{data['steps_per_chunk']} steps/chunk / {data['history']}",
                action=data['action'], history=data['history'], step=data['step'], nfe=data['steps_per_chunk'])
            if row is not None:
                row.update(experiment=str(root.resolve()), precision_profile=precision_profile, target_time_policy=policy,
                           history_gradient_mode=history_gradient,
                           training_timestep_shift=training_shift,
                           validation_timestep_shift=validation_shift,
                           inference_flow_shift=inference_shift,
                           sigma_grid=grid, video_sigmas=runtime.get('video_sigmas'))
                rows.append(row)
    if args.smoke_root:
        file = args.smoke_root / 'eval4_verified/A/cached.json'
        if file.exists():
            rows.append(summarize_video(file.parent, 'cached',
                label='AnyFlow smoke step01 / A / 4 steps/chunk / generated',
                action='A', history='generated', step=1, nfe=4))
    rows = [r for r in rows if r is not None]
    for row in rows:
        if row['history'] != 'bidirectional':
            row['input_fairness'] = audit_inputs(Path(row['path']).parent,
                args.teacher_root / f"action_{row['action']}_teacher_39")
    checks = []
    for row in rows:
        if row['action'] != 'A':
            continue
        other = next((r for r in rows if r['action'] == 'D'
            and r.get('experiment') == row.get('experiment')
            and r['label'].replace(' / D / ', ' / A / ') == row['label']), None)
        if other:
            delta = row['horizontal_flow'] - other['horizontal_flow']
            checks.append(dict(method=row['label'].replace(' / A / ', ' / A,D / '),
                A=row['horizontal_flow'], D=other['horizontal_flow'], AD_separation=delta,
                numeric_gate=row['horizontal_flow'] > 0 and other['horizontal_flow'] < 0 and delta > 1.,
                visual_gate='NOT_AUTOMATICALLY_SCORED'))
    result = dict(snapshot_at=datetime.now().astimezone().isoformat(),
        videos=rows, paired_numeric_checks=checks, overall_gate='NOT_ACCEPTED',
        caveats=['MAD measures motion/activity, not video quality.',
                 'Flow signs/separation cannot establish intact geometry or absence of ghosting.',
                 'Runtime is one recorded run with CPU offload and shared hardware, not isolated speedup.',
                 'The archived original run does not record its offload reserve; GPU memory residency is not a matched comparison.',
                 'Initial image/seed/39f references are fixed; a separate seed and 124f/switching remain required.'])
    (report / 'metrics.json').write_text(json.dumps(result, indent=2) + '\n')
    lines = ['# Stage1 AnyFlow pilot snapshot', '', f"Snapshot: {result['snapshot_at']}", '',
        'Only completed videos are listed. Quality is not automatically accepted.', '',
        '| Method | E2E s | GPU MiB | Offload reserve GiB | CPU KV MiB | Noisy + commit | Gray MAD | Boundary RGB MAD | Horizontal flow |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['label']} | {r['seconds']:.1f} | {r['gpu_peak_MiB']:.1f} | "
            f"{r['vram_reserve_gib'] if r['vram_reserve_gib'] is not None else 'unrecorded'} | "
            f"{r['cpu_kv_MiB']:.1f} | {r['noisy_forwards']} + {r['clean_commits']} | "
            f"{r['frame_gray_mad']:.3f} | {r['boundary_rgb_mad']:.3f} | {r['horizontal_flow']:+.3f} |")
    lines += ['', '## A/D numeric checks', '']
    for check in checks:
        lines.append(f"- {check['method']}: A={check['A']:+.4f}, D={check['D']:+.4f}, "
            f"A-D={check['AD_separation']:.4f}; numeric gate={check['numeric_gate']}; visual gate is assessed separately.")
    if (report / 'VISUAL_REVIEW.md').exists():
        lines += ['', 'Recorded visual observations: [VISUAL_REVIEW.md](VISUAL_REVIEW.md). '
            'Only the videos explicitly covered there have been reviewed.']
    lines += ['', '## Input fairness against archived original H3', '',
        'Exact saved-tensor comparison of video/audio noise, prompt, initial image anchor and action row spans; '
        'this does not make the different offload configurations a fair speed/memory comparison.', '']
    for row in rows:
        if 'input_fairness' in row:
            lines.append(f"- {row['label']}: exactly equal = {row['input_fairness']['exactly_equal']}.")
    lines += ['', '## Fixed validation-noise raw residuals', '',
        'AnyFlow adaptive scaling can hide large endpoint/general-map residuals in weighted total. '
        'Compare these fixed-noise rows before/after training; FM and AnyFlow losses are different objectives.', '',
        '| Objective | Phase | Action | Sample type | sigma | target sigma | Raw loss | Scale | Weighted loss |',
        '|---|---|---|---|---:|---:|---:|---:|---:|']
    for objective in ('anyflow', 'fm'):
        file = args.pilot_root / f'train_{objective}' / 'training.json'
        if not file.exists():
            continue
        training = json.loads(file.read_text())
        for phase in ('before', 'after'):
            for case in training.get(f'validation_{phase}', []):
                for sample in case['samples']:
                    lines.append(f"| {objective} | {phase} | {case['action']} | {sample['sample_type']} | "
                        f"{sample['sigma']:.5f} | {sample['target_sigma']:.5f} | {sample['raw_loss']:.6f} | "
                        f"{sample['adaptive_scale']:.6f} | {sample['weighted_loss']:.6f} |")
    lines += ['', '## Interpretation limits', ''] + ['- ' + c for c in result['caveats']]
    (report / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    for action in 'AD':
        selected = [r for r in rows if r['action'] == action]
        if selected:
            contacts(selected, report / f'contact_{action}.jpg')
    print(json.dumps(dict(completed_videos=len(rows), paired_checks=checks,
                         report=str(report / 'REPORT.md')), indent=2))


if __name__ == '__main__':
    main()
