"""CPU-only presentation from frozen videos; no inference or training.
Run from GWM: .venvs/h3world/bin/python submission/report/original_vs_causal/build_comparison.py
"""
from pathlib import Path
import hashlib
import json
import os
import shutil
import av
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCES = {
    'original': ROOT / 'H3-World/outputs/2026-10-01-21/action_A_teacher_latents/baseline.mp4',
    'v3_baseline': ROOT / 'submission/report/v3/v3_baseline/videos/V3_Baseline_AA_124.mp4',
    'fm8_swg': ROOT / 'submission/report/v3/fm2anyflow/fm8/videos/A_rollout_158.mp4',
}
VIDEO = HERE / 'videos/Original_vs_V3_SWG_FM-8NFE_A_common124.mp4'
FONT_PATH = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(4 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def read_video(path):
    frames = []
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        s.codec_context.thread_count = 2
        assert (s.width, s.height) == (832, 480)
        assert str(s.average_rate) == '24'
        for frame in c.decode(video=0):
            frames.append(frame.to_image().convert('RGB'))
    return frames


def main():
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:4])
    assert not VIDEO.exists(), 'Refuse to replace published comparison'
    frames = {key: read_video(path) for key, path in SOURCES.items()}
    assert [len(frames[k]) for k in SOURCES] == [124, 124, 158]
    font = ImageFont.truetype(FONT_PATH, 20)
    small = ImageFont.truetype(FONT_PATH, 16)
    boundaries = {39, 56, 73, 90, 107}
    with av.open(str(VIDEO), 'w') as out:
        stream = out.add_stream('libx264', rate=24, options={'crf': '18', 'preset': 'fast'})
        stream.width, stream.height, stream.pix_fmt = 1664, 576, 'yuv420p'
        stream.codec_context.thread_count = 4
        for i in range(124):
            im = Image.new('RGB', (1664, 576), '#142030')
            im.paste(frames['original'][i], (0, 64))
            im.paste(frames['fm8_swg'][i], (832, 64))
            d = ImageDraw.Draw(im)
            d.text((14, 6), 'Original H3-World | 30 NFE / full video', font=font, fill='white')
            d.text((846, 6), 'V3-SW-G + FM-8NFE | 8 NFE / chunk', font=font, fill='white')
            d.text((14, 35), 'Full-video bidirectional | released Action LoRA | no new training', font=small, fill='#bbccdd')
            d.text((846, 35), 'Strict causal + raw KV + Global RoPE | no new training', font=small, fill='#bbccdd')
            d.text((14, 550), f'A throughout | RGB {i:03d}/123 | 24 fps | original playback speed', font=small, fill='white')
            d.text((846, 550), 'Common first 124 of 158 frames; eviction is tested AFTER this excerpt', font=small, fill='#ffd080')
            if i in boundaries:
                d.rectangle((832, 64, 1663, 543), outline='#ffd080', width=3)
            for packet in stream.encode(av.VideoFrame.from_image(im)):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)
    with av.open(str(VIDEO)) as c:
        pts = [f.pts for f in c.decode(video=0)]
        assert len(pts) == 124 and all(a < b for a, b in zip(pts, pts[1:]))
        assert str(c.streams.video[0].average_rate) == '24'
    # Representative frames, not an exhaustive new quality assessment.
    selected = [38, 73, 83, 107, 123]
    sheet = Image.new('RGB', (1248, 44 + 268 * len(selected)), '#142030')
    d = ImageDraw.Draw(sheet)
    for col, title in enumerate(['Original H3 | 30 NFE/video', 'V3 Baseline | 30 NFE/chunk', 'V3 SW-G | FM-8NFE/chunk']):
        d.text((col * 416 + 8, 10), title, font=small, fill='white')
    for row, idx in enumerate(selected):
        y = 44 + row * 268
        for col, key in enumerate(SOURCES):
            sheet.paste(frames[key][idx].resize((416, 240), Image.Resampling.LANCZOS), (col * 416, y + 26))
            d.text((col * 416 + 8, y + 3), f'RGB {idx} | {idx / 24:.2f}s', font=small, fill='#ffd080')
    poster = HERE / 'comparison_keyframes.jpg'
    sheet.save(poster, quality=92)
    original_metrics = ROOT / 'H3-World/outputs/2026-10-01-21/action_A_teacher_latents/baseline.json'
    shutil.copyfile(original_metrics, HERE / 'evidence/original_timing.json')
    manifest = {
        'purpose': 'Existing-output comparison; CPU presentation only; zero GPU inference/training',
        'comparison_scope': 'A continuation, common RGB 0..123 at 24fps. Candidate source has158 frames; no padding, looping, interpolation or retiming. RGB124..157 and actual eviction are outside excerpt.',
        'protocol_scope': 'Same parking I0/seed13 and released weights. Different generation protocols and self-generated histories; not a single-factor or matched E2E benchmark. No claim of bitwise input-noise equality from video files.',
        'sources': {k: {'path': str(p.relative_to(ROOT)), 'sha256': sha(p), 'frames': len(frames[k])} for k, p in SOURCES.items()},
        'outputs': {
            'video': {'path': str(VIDEO.relative_to(HERE)), 'sha256': sha(VIDEO), 'frames': 124, 'fps': 24, 'size': [1664, 576], 'all_frames_decoded': True, 'pts_monotonic': True},
            'poster': {'path': poster.name, 'sha256': sha(poster), 'rgb_indices': selected},
        },
        'original_timing_snapshot': {'source': str(original_metrics.relative_to(ROOT)), 'sha256': sha(original_metrics), 'copy': 'evidence/original_timing.json'},
        'builder_sha256': sha(Path(__file__)),
    }
    (HERE / 'video_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest['outputs'], indent=2))


if __name__ == '__main__':
    main()
