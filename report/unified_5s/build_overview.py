"""CPU presentation for completed EXP-018 outputs. Input must contain audited rows.
Usage: .venvs/h3world/bin/python submission/report/unified_5s/build_overview.py INPUT.json
Input: {"rows": [{"id", "label", "video120", "wall_seconds", "peak_allocated_gib", "sampling_nfe_total", "quality"}]}
Paths are relative to GWM. Metrics must be measured, not copied from previous experiments.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import av
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(4 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def read(p):
    frames = []
    with av.open(str(p)) as c:
        s = c.streams.video[0]
        s.codec_context.thread_count = 2
        assert (s.width, s.height) == (832, 480), (p, s.width, s.height)
        assert str(s.average_rate) == '24'
        for f in c.decode(video=0):
            frames.append(f.to_image().convert('RGB'))
    assert len(frames) == 120, (p, len(frames))
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('input', type=Path)
    args = ap.parse_args()
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:4])
    data = json.loads(args.input.read_text())
    rows = data['rows']
    assert len(rows) == 7 and len({r['id'] for r in rows}) == 7
    sources = [ROOT / row['video120'] for row in rows]
    videos = [read(p) for p in sources]
    (HERE / 'videos').mkdir(exist_ok=True)
    output = HERE / 'videos/V0_V1_V2a_V2b_V3_V4_unified_5s.mp4'
    assert not output.exists(), 'Refuse to overwrite published overview'
    title = ImageFont.truetype(FONT, 19)
    small = ImageFont.truetype(FONT, 15)
    compact = ImageFont.truetype(FONT, 12)
    with av.open(str(output), 'w') as c:
        s = c.add_stream('libx264', rate=24, options={'crf': '18', 'preset': 'fast'})
        s.width, s.height, s.pix_fmt = 1664, 624, 'yuv420p'
        s.codec_context.thread_count = 4
        for frame_id in range(120):
            image = Image.new('RGB', (1664, 624), '#132030')
            d = ImageDraw.Draw(image)
            for i, row in enumerate(rows):
                x, y = (i % 4) * 416, (i // 4) * 312
                d.text((x + 8, y + 3), row['label'], font=title, fill='white')
                d.text((x + 8, y + 28), f"124f wall {'~' if row['id']=='v0' else ''}{row['wall_seconds']:.1f}s | GPU peak {row['peak_allocated_gib']:.2f} GiB", font=small, fill='#d1dfec')
                d.text((x + 8, y + 49), f"Sampling total {row['sampling_nfe_total']} NFE | {row['quality']}", font=compact, fill='#ffd080')
                image.paste(videos[i][frame_id].resize((416, 240), Image.Resampling.LANCZOS), (x, y + 72))
            x, y = 3 * 416, 312
            lines = ['EXP-018 | fixed parking / seed13', 'A throughout | native 832 x 480', 'Each method generated fresh 124f', 'Display first120 at24fps =5.000s', 'Timing is for full124f, single run', 'No new training | no borrowed first chunk', f'Display RGB {frame_id:03d}/119 | {frame_id/24:.2f}s', 'Protocols differ: compare full methods']
            for j, line in enumerate(lines):
                d.text((x + 8, y + 18 + j * 32), line, font=small, fill='white' if j != 4 else '#ffd080')
            for packet in s.encode(av.VideoFrame.from_image(image)):
                c.mux(packet)
        for packet in s.encode():
            c.mux(packet)
    with av.open(str(output)) as c:
        pts = [f.pts for f in c.decode(video=0)]
        assert len(pts) == 120 and all(a < b for a, b in zip(pts, pts[1:]))
        assert str(c.streams.video[0].average_rate) == '24'
    indices = [0, 24, 48, 72, 96, 119]
    sheet = Image.new('RGB', (1248, 154 * 7), '#132030')
    d = ImageDraw.Draw(sheet)
    for i, row in enumerate(rows):
        d.text((8, i * 154 + 4), row['label'] + ' | ' + row['quality'], font=small, fill='white')
        for j, idx in enumerate(indices):
            sheet.paste(videos[i][idx].resize((208, 120), Image.Resampling.LANCZOS), (j * 208, i * 154 + 30))
    poster = HERE / 'all_versions_keyframes.jpg'
    sheet.save(poster, quality=92)
    pairs = []
    for candidate in ('v3_30', 'v3_8'):
        pair_ids = [next(i for i,r in enumerate(rows) if r['id']==rid) for rid in ('v0',candidate)]
        pair_path = HERE / 'videos' / f'V0_vs_{candidate}_5s.mp4'
        assert not pair_path.exists()
        with av.open(str(pair_path), 'w') as c:
            s = c.add_stream('libx264', rate=24, options={'crf':'18','preset':'fast'})
            s.width, s.height, s.pix_fmt = 1664, 552, 'yuv420p'
            s.codec_context.thread_count = 4
            for frame_id in range(120):
                canvas = Image.new('RGB',(1664,552),'#132030')
                draw = ImageDraw.Draw(canvas)
                for col, idx in enumerate(pair_ids):
                    r = rows[idx]
                    x = col*832
                    draw.text((x+12,5),r['label'],font=title,fill='white')
                    draw.text((x+12,29),f"124f wall {'~' if r['id']=='v0' else ''}{r['wall_seconds']:.1f}s | GPU peak {r['peak_allocated_gib']:.2f} GiB | {r['quality']}",font=small,fill='#ffd080')
                    draw.text((x+12,50),'Same I0 / seed13 / A | display first120 at24fps = 5.000s',font=small,fill='white')
                    canvas.paste(videos[idx][frame_id],(x,72))
                for packet in s.encode(av.VideoFrame.from_image(canvas)):
                    c.mux(packet)
            for packet in s.encode():
                c.mux(packet)
        with av.open(str(pair_path)) as c:
            pts = [f.pts for f in c.decode(video=0)]
            assert len(pts)==120 and all(a<b for a,b in zip(pts,pts[1:]))
            assert str(c.streams.video[0].average_rate)=='24'
        pairs.append({'path':str(pair_path.relative_to(HERE)),'sha256':sha(pair_path),
            'source_ids':['v0',candidate],'frames':120,'fps':24,'seconds':5.0,'full_decode':True})
    manifest = {'input': str(args.input), 'input_sha256': sha(args.input), 'builder_sha256': sha(Path(__file__)), 'sources': [{**row, 'sha256': sha(p)} for row, p in zip(rows, sources)], 'overview': {'path': str(output.relative_to(HERE)), 'sha256': sha(output), 'frames': 120, 'fps': 24, 'seconds': 5.0, 'all_frames_decoded': True, 'pts_monotonic': True}, 'pairs':pairs, 'poster': {'path': poster.name, 'sha256': sha(poster), 'indices': indices}}
    (HERE / 'presentation_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(manifest['overview'], indent=2))


if __name__ == '__main__':
    main()
