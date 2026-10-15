from pathlib import Path
import json
import numpy as np
from PIL import Image, ImageDraw
ROOT = Path(__file__).resolve().parent
rows = json.loads((ROOT/'VIDEO_RESULTS.json').read_text())
indices = [38, 39, 42, 46, 50, 55]
thumb = (312, 180)
margin = 38
sheet = Image.new('RGB', (thumb[0]*len(indices), (thumb[1]+margin)*len(rows)), '#1e1e1e')
draw = ImageDraw.Draw(sheet)
for r, row in enumerate(rows):
    video = ROOT.parent.parent.parent.parent / row['video']
    frames = np.load(video.with_suffix('.rgb.npy'), mmap_mode='r')
    for c, frame_index in enumerate(indices):
        im = Image.fromarray(frames[frame_index]).resize(thumb)
        sheet.paste(im, (c*thumb[0], r*(thumb[1]+margin)+margin))
        draw.text((c*thumb[0]+5, r*(thumb[1]+margin)+7),
                  f"{row['clip_id'][0:8]} ckpt{row['checkpoint_updates']} f{frame_index}",
                  fill='white')
sheet.save(ROOT/'CONTACT_SHEET.png')
print(ROOT/'CONTACT_SHEET.png')
