#!/usr/bin/env python3
"""Original / step16 / step32 duration diagnostic grid, with complete footage.

Requires completed A/D evaluations. This is a diagnostic, not acceptance.
Reads recorded times; it does not claim an isolated-hardware speedup.
"""
import argparse
import hashlib
import json
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--pilot-root', type=Path, required=True)
    ap.add_argument('--teacher-root', type=Path, required=True)
    ap.add_argument('--previous-root', type=Path, required=True)
    ap.add_argument('--step', type=int, default=32)
    ap.add_argument('--evaluation-suffix', choices=['', '_native', '_uniform'], default='')
    ap.add_argument('--variant', default='Train shift2.22', help='Explicit precision/scope label for this diagnostic')
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    entries = []
    decoded = []
    for action in 'AD':
        sources = [(args.teacher_root / f'action_{action}_teacher_39', 'baseline', 'Original H3', 30, 0)]
        sources += [(args.previous_root / f'eval/anyflow/step_16/generated_8step_native/{action}',
                     'cached', f'{args.variant} step16 | 8/chunk', 24, 3),
                    (args.pilot_root / f'eval/anyflow/step_{args.step:02d}/generated_8step_native/{action}',
                     'cached', f'{args.variant} step{args.step} | 8/chunk', 24, 3)]
        for directory, mode, label, nfe, commits in sources:
            video = directory / f'{mode}.mp4'
            runtime = json.loads((directory / f'{mode}.json').read_text())
            if runtime['status'] != 'complete':
                raise ValueError(f'Incomplete run: {directory}')
            if runtime['denoiser_forwards'] != nfe or runtime.get('commit_forwards', 0) != commits:
                raise ValueError(f'Unexpected denoiser counts: {directory}')
            if video.resolve() == args.output.resolve():
                raise ValueError('Output must not replace input')
            with av.open(video) as container:
                stream = container.streams.video[0]
                if (stream.width, stream.height, stream.average_rate) != (832, 480, 24):
                    raise ValueError(f'Unexpected video format: {video}')
                frames = [(frame.time, frame.to_image()) for frame in container.decode(video=0)]
            if len(frames) != 39 or any(abs(t - i / 24) > 1e-6 for i, (t, _) in enumerate(frames)):
                raise ValueError(f'Unexpected frame count/timestamps: {video}')
            entries.append(dict(action=action, label=label, video=str(video.resolve()),
                sha256=hashlib.sha256(video.read_bytes()).hexdigest(),
                seconds=runtime['wall_including_shared_setup_seconds'], nfe=nfe, commits=commits))
            decoded.append([frame.resize((624, 360)) for _, frame in frames])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix('.tmp.mp4')
    font = ImageFont.load_default(size=18)
    with av.open(temporary, 'w', options={'movflags': '+faststart'}) as container:
        stream = container.add_stream('libx264', rate=24)
        stream.width, stream.height, stream.pix_fmt = 1872, 880, 'yuv420p'
        stream.options = {'crf': '18', 'preset': 'medium', 'threads': '4'}
        for index in range(39):
            canvas = Image.new('RGB', (1872, 880), 'black')
            draw = ImageDraw.Draw(canvas)
            draw.text((12, 8), 'Stage1 DURATION DIAGNOSTIC | A/D gate NOT PASSED | same image/action/seed/noise | single-run timings', font=font, fill='#ffcb66')
            for j, item in enumerate(entries):
                x, y = j % 3 * 624, 42 + j // 3 * 404
                draw.text((x + 8, y), f"{item['action']} | {item['label']}", fill='white', font=font)
                draw.text((x + 8, y + 22), f"{item['seconds']:.1f}s | noisy forwards {item['nfe']} + commits {item['commits']}", fill='white', font=font)
                canvas.paste(decoded[j][index], (x, y + 44))
            draw.text((12, 856), f'Frame {index:02d}/38 | {index/24:.3f}s | complete 39 frames / 24 fps | display scaled; source 832 x 480', fill='white', font=font)
            for packet in stream.encode(av.VideoFrame.from_ndarray(np.asarray(canvas), format='rgb24')):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    temporary.replace(args.output)
    with av.open(args.output) as container:
        count = sum(1 for _ in container.decode(video=0))
    if count != 39:
        raise RuntimeError('Output failed full decode verification')
    args.output.with_suffix('.json').write_text(json.dumps(dict(
        type='duration_learning_curve_diagnostic_not_final_demo', frames=count, fps=24, inputs=entries,
        output_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest()), indent=2) + '\n')
    print(f'Wrote {args.output} ({count} frames, H264/yuv420p/24fps)')


if __name__ == '__main__':
    main()
