"""CPU-only full-frame review sheets and matched three-column local videos.

Reads only completed cases. Never loads weights or changes experiment receipts.
"""
import argparse
import hashlib
import json
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont

BASE = Path(__file__).resolve().parent
ROLES = ['zero', 'fm_only', 'fm_action']
LABELS = ['Original weights / local N', 'FM-only / 4 updates', 'FM + action / 4 updates']
FONT = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 16)


def read(path):
    with av.open(str(path)) as container:
        return [frame.to_image().convert('RGB') for frame in container.decode(video=0)]


def main(args):
    source = json.loads((BASE/'review_step4/metrics.json').read_text())
    cases = {}
    for row in source['metrics']:
        cases.setdefault(row['case'], {})[row['role']] = row
    out = BASE/'review_step4'; assets = []
    for name, rows in sorted(cases.items()):
        if set(rows) != set(ROLES) or (args.case and name not in args.case):
            continue
        frames = {role: read(rows[role]['video']) for role in ROLES}
        lengths = {len(v) for v in frames.values()}; assert len(lengths) == 1
        count = lengths.pop(); start = 39 if name.startswith('parking_') else 81
        for role in ROLES:
            target = out/f'{name}_{role}_all_frames.jpg'
            if not target.exists():
                # Every frame appears once at half resolution; native details are separate.
                sheet = Image.new('RGB', (6*416, ((count+5)//6)*266), 'white')
                draw = ImageDraw.Draw(sheet)
                for i, frame in enumerate(frames[role]):
                    x, y = (i % 6)*416, (i//6)*266
                    sheet.paste(frame.resize((416, 240)), (x, y+26))
                    draw.text((x+4, y+3), f'{role} RGB{start+i}', font=FONT, fill='black')
                sheet.save(target, quality=94)
            assets.append(str(target))
        target = out/f'{name}_comparison.mp4'
        if not target.exists():
            with av.open(str(target), mode='w') as container:
                stream = container.add_stream('libx264', rate=24)
                stream.width, stream.height, stream.pix_fmt = 1248, 308, 'yuv420p'
                stream.options = {'crf': '18', 'preset': 'medium', 'threads': '2'}
                for i in range(count):
                    canvas = Image.new('RGB', (1248, 308), '#eeeeee'); draw = ImageDraw.Draw(canvas)
                    for col, role in enumerate(ROLES):
                        x = col*416
                        draw.text((x+6, 5), LABELS[col], font=FONT, fill='black')
                        canvas.paste(frames[role][i].resize((416, 240)), (x, 30))
                    label = f'{name} | RGB {start+i} | 30 steps | fixed reference history; local generation'
                    draw.text((5, 281), label, font=FONT, fill='black')
                    for packet in stream.encode(av.VideoFrame.from_ndarray(np.asarray(canvas), format='rgb24')):
                        container.mux(packet)
                for packet in stream.encode(): container.mux(packet)
        assert len(read(target)) == count
        assets.append(str(target))
    receipt = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in assets}
    (out/'review_assets.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps({'assets': len(assets), 'cases': len(assets)//4}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--case', action='append')
    main(parser.parse_args())
