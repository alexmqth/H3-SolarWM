"""CPU review sheets for every generated frame; no quality score is inferred."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import av
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('row')
    args = ap.parse_args()
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:4])
    metrics = ROOT / 'H3-World/outputs/EXP-018_unified_5s_benchmark' / args.row / 'metrics.json'
    row = json.loads(metrics.read_text())
    assert row['status'] == 'complete_pending_judge'
    source = ROOT / row['full_video']
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    assert digest == row['full_sha256']
    with av.open(str(source)) as c:
        c.streams.video[0].codec_context.thread_count = 2
        frames = [f.to_image().convert('RGB') for f in c.decode(video=0)]
    assert len(frames) == 124
    dest = HERE / 'visual_review' / args.row
    dest.mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 14)
    outputs = []
    for page, first in enumerate(range(0, 124, 64)):
        sheet = Image.new('RGB', (1664, 1168), '#132030')
        draw = ImageDraw.Draw(sheet)
        draw.text((8, 3), f'{args.row} | original 124 frames | page {page+1} | no interpolation', font=font, fill='white')
        for i in range(first, min(first + 64, 124)):
            x, y = ((i-first) % 8)*208, 24 + ((i-first)//8)*143
            draw.text((x+3, y), f'RGB {i:03d} / {i/24:.3f}s', font=font, fill='white')
            sheet.paste(frames[i].resize((208,120), Image.Resampling.LANCZOS), (x,y+20))
        path = dest / f'all_frames_{first:03d}_{min(first+63,123):03d}.jpg'
        sheet.save(path, quality=94)
        outputs.append(str(path.relative_to(HERE)))
    # Full-resolution boundary witnesses complement the small exhaustive sheets.
    boundaries = [38,39,55,56,72,73,89,90,106,107,119,123]
    for i in boundaries:
        frames[i].save(dest / f'rgb_{i:03d}.jpg', quality=95)
    (dest / 'manifest.json').write_text(json.dumps({'row':args.row,'source':row['full_video'],
        'source_sha256':digest,'frames':124,'all_frame_sheets':outputs,
        'full_resolution_indices':boundaries,'quality':'PENDING_HUMAN_OR_JUDGE_REVIEW'},indent=2)+'\n')
    print(json.dumps({'row':args.row,'sheets':outputs}))


if __name__ == '__main__':
    main()
