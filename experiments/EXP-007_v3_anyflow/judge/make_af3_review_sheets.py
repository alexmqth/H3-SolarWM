"""CPU review sheets: every new frame, adjacent boundary, and native last frame."""
from pathlib import Path
import json
import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OUT = ROOT / 'H3-World/outputs/EXP-007_v3_anyflow_af3'
for branch in ('AA', 'AD'):
    for chunk, stop, start in [(2, 56, 39), (3, 73, 56)]:
        source = OUT / branch / f'published_{stop}.npy'
        if not source.exists():
            continue
        frames = np.load(source, mmap_mode='r')
        indices = list(range(start - 2, stop))
        sheet = Image.new('RGB', (4 * 416, 5 * 260), 'white')
        draw = ImageDraw.Draw(sheet)
        for j, i in enumerate(indices):
            x, y = j % 4 * 416, j // 4 * 260
            sheet.paste(Image.fromarray(frames[i]).resize((416, 240)), (x, y))
            draw.text((x + 4, y + 242), f'{branch} AF C{chunk} RGB{i}', fill='black')
        label = f'AF3_{branch}_C{chunk}'
        sheet.save(HERE / f'{label}.jpg', quality=92)
        Image.fromarray(frames[-1]).save(HERE / f'{label}_last.png')
        print(json.dumps({'sheet': label, 'all_new_frames': stop-start, 'boundary_context': 2}))
