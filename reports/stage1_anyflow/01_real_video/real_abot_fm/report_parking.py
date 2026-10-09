"""Completed A/D pairs only: Original30 vs zero/trained real-data causal FM."""
import argparse
import csv
import json
from pathlib import Path
import sys

from PIL import Image, ImageDraw, ImageFont

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / 'runtime/code/causal'))
from report_stage1_anyflow import audit_inputs, summarize_video
from report_real import load_frames, write_video, video_metrics


def main(args):
    teacher = BASE.parents[2] / 'outputs/2026-10-02-03'
    checkpoints = [0] if args.step == 0 else [0, 48]
    entries = {}
    for action in 'AD':
        original = teacher / f'action_{action}_teacher_39'
        row = summarize_video(original, 'baseline', label='Original H3 30 full-seq', action=action,
                              history='full-sequence', step=None, nfe=30)
        assert row is not None
        row['steps_per_chunk'] = None
        row['generation_steps'] = '30 full-sequence'
        entries[action] = [(original, row)]
        for step in checkpoints:
            path = BASE / f'parking/step_{step:02d}/{args.steps}step' / action
            receipt = json.loads((path / 'evaluation.json').read_text())
            assert receipt['optimizer_step'] == step and receipt['steps_per_chunk'] == args.steps
            check = audit_inputs(path, original)
            assert check['exactly_equal']
            row = {k: v for k, v in receipt.items() if k != 'input_audit'}
            row['generation_steps'] = f'{args.steps}/chunk'
            row['exact_inputs_equal'] = check['exactly_equal']
            entries[action].append((path, row))
    dest = BASE / 'report' / f'parking_step{args.step:02d}_{args.steps}step'
    dest.mkdir(parents=True, exist_ok=False)
    metrics = [row for action in 'AD' for _, row in entries[action]]
    fonts = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 14)
    frames = {}
    for action in 'AD':
        for col, (_, row) in enumerate(entries[action]):
            current = load_frames(Path(row['path'])); frames[action, col] = current
            sheet = Image.new('RGB', (7 * 208, 6 * 138), 'white'); draw = ImageDraw.Draw(sheet)
            for i, im in enumerate(current):
                x, y = i % 7 * 208, i // 7 * 138
                sheet.paste(im.resize((208, 120)), (x, y + 18)); draw.text((x + 3, y + 2), f'frame {i}', fill='black')
            tag = 'original' if col == 0 else f'causal_step{checkpoints[col-1]:02d}'
            sheet.save(dest / f'{action}_{tag}_all39.jpg')
    assembled = []
    for i in range(39):
        canvas = Image.new('RGB', (416 * len(entries['A']), 604), 'white'); draw = ImageDraw.Draw(canvas)
        for a, action in enumerate('AD'):
            for col, (_, row) in enumerate(entries[action]):
                x, y = col * 416, a * 302
                title = 'Original H3' if col == 0 else f'Causal real-FM{row["optimizer_step"]}'
                draw.text((x + 5, y + 3), f'{action} | {title} | {row["generation_steps"]}', fill='black', font=fonts)
                draw.text((x + 5, y + 23), f'{row["noisy_forwards"]} noisy + {row["clean_commits"]} commits | {row["seconds"]:.1f}s recorded', fill='black', font=fonts)
                draw.text((x + 5, y + 43), f'frame {i}/38 | same image / actions / seed / noise', fill='black', font=fonts)
                canvas.paste(frames[action, col][i].resize((416, 240)), (x, y + 62))
        assembled.append(canvas)
    video = dest / 'original_causal_AD.mp4'; write_video(assembled, video)
    assert video_metrics(video)['frames'] == 39
    for action in 'AD':
        matched = Image.new('RGB', (6 * 277, len(entries[action]) * 180), 'white'); draw = ImageDraw.Draw(matched)
        for row, (_, metric) in enumerate(entries[action]):
            for col, frame in enumerate((0, 8, 16, 24, 30, 38)):
                x, y = col * 277, row * 180
                matched.paste(frames[action, row][frame].resize((277, 160)), (x, y + 20))
                label = 'Original30' if row == 0 else f'FM{metric["optimizer_step"]} {args.steps}/chunk'
                draw.text((x + 3, y + 3), f'{action} {label} frame{frame}', fill='black')
        matched.save(dest / f'{action}_matched.jpg')
    action_results = []
    for col in range(len(entries['A'])):
        a, d = entries['A'][col][1], entries['D'][col][1]
        separation = a['horizontal_flow'] - d['horizontal_flow']
        action_results.append(dict(method=a['label'], A=a['horizontal_flow'], D=d['horizontal_flow'],
                                   separation=separation,
                                   numeric_gate_only=a['horizontal_flow'] > 0 and d['horizontal_flow'] < 0 and separation > 1,
                                   visual_review='pending full-frame inspection'))
    report = dict(status='complete_artifacts', step=args.step, steps_per_chunk=args.steps,
                  rows=metrics, action_results=action_results,
                  precision_caveat='Original legacy released inference; causal0 and48 share h3_fp32. Pipeline reference, not isolated precision comparison.',
                  timing_caveat='Single runs on shared host, Original reserve unknown; no fair speedup or memory-reduction claim.',
                  metrics_caveat='MAD measures differences/activity, not video quality. Numeric gate alone cannot pass visual acceptance.')
    (dest / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n')
    fields = sorted(set().union(*(row.keys() for row in metrics)))
    with (dest / 'metrics.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(metrics)
    text = ['# 停车场已知动作正控：完整 A/D 对照', '',
            f'Causal 为 **{args.steps} steps/chunk**，三块共 {args.steps*3} noisy forwards + 3 clean commits；Original 为整段30次。', '',
            '[完整两行对比视频](original_causal_AD.mp4) · [完整指标CSV](metrics.csv) · [机器记录](metrics.json)', '',
            '| 方法 | flow(A) | flow(D) | A−D | 仅数值门槛 |', '|---|---:|---:|---:|---|']
    for row in action_results:
        text.append(f'| {row["method"]} | {row["A"]:+.6f} | {row["D"]:+.6f} | {row["separation"]:.6f} | {row["numeric_gate_only"]} |')
    text += ['', '最终画质/action验收仍须检查完整39帧，自动表格不判视觉PASS。Original旧legacy精度、causal0/48统一h3_fp32，且anchor/prefix协议不同，因此与Original是pipeline对比；训练前后causal才是匹配配置。', '',
             '时间是单次共享主机记录，Original offload reserve未知，不声称加速或显存优化。GT与generated-history不同场景的总flow不能代替这里的同图A/D。', '']
    (dest / 'README.md').write_text('\n'.join(text)); print(json.dumps(report, indent=2))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--step', type=int, choices=[0, 48], required=True)
    ap.add_argument('--steps', type=int, choices=[8, 30], required=True)
    main(ap.parse_args())
