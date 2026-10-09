"""Extract source-only sheets for at most one train-only pure-label window/action."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

import av
from PIL import Image, ImageDraw, ImageFont

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE/'runtime/code/abot'))
import abot_action as A

audit = json.loads((BASE/'supervision_coverage.json').read_text())
prep = json.loads((BASE/'preparation.json').read_text())
out = BASE/'coverage_candidate_review'; out.mkdir(exist_ok=False)
font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 16)
records = []
for action in ('A', 'D'):
    available = sorted([r for r in audit['overlapping_candidates'] if r['action'] == action and r['window'] == 2],
                       key=lambda r: (r['sample_id'], r['src_start']))
    if not available:
        records.append(dict(action=action, status='no_label_candidate')); continue
    row = available[0]; sid = row['sample_id']; start = row['src_start']
    source = BASE.parents[2]/'data/abot_bridge/raw/data'/sid[:2]/sid/'video.mp4'
    original = next(r for r in prep['clips'] if r['sample_id'] == sid)
    assert original['split'] == 'train'
    digest = hashlib.sha256(source.read_bytes()).hexdigest(); assert digest == original['source_video_sha256']
    indices = [start+k for k in A.window_offsets(124)]
    wanted = set(indices); images = {}
    with av.open(str(source)) as container:
        stream = container.streams.video[0]; assert float(stream.average_rate) == 30
        for index, frame in enumerate(container.decode(video=0)):
            if index in wanted:
                image = frame.to_image().convert('RGB'); width = round(image.width*480/image.height/2)*2
                image = image.resize((width, 480), Image.Resampling.BICUBIC)
                images[index] = image.crop(((width-832)//2, 0, (width-832)//2+832, 480))
            if index >= indices[-1]: break
    assert len(images) == 124
    lo, hi = A.frame_spans(37)[24][0], A.frame_spans(37)[35][1]
    sheet = Image.new('RGB', (6*416, 7*266), 'white'); draw = ImageDraw.Draw(sheet)
    # Three context frames + all39 current source RGB, never model generation.
    selected = [0, 79, 80] + list(range(lo, hi)); assert len(selected) == 42
    for i, k in enumerate(selected):
        x, y = (i % 6)*416, (i//6)*266
        sheet.paste(images[indices[k]].resize((416, 240)), (x, y+26))
        label = f'GT {action} RGB{k} / src{indices[k]}'
        draw.text((x+3, y+4), label, font=font, fill='black')
    name = f'train_pure_{action}_source.jpg'; sheet.save(out/name, quality=94)
    details = Image.new('RGB', (3*420, 4*450), 'white'); draw = ImageDraw.Draw(details)
    for i, k in enumerate([79, 80, 81, 85, 89, 93, 97, 101, 105, 109, 114, 119]):
        x, y = (i % 3)*420, (i//3)*450
        details.paste(images[indices[k]].crop((170, 60, 590, 480)), (x, y+30))
        draw.text((x+4, y+4), f'GT {action} RGB{k} / src{indices[k]}', font=font, fill='black')
    detail_name = f'train_pure_{action}_details.jpg'; details.save(out/detail_name, quality=95)
    records.append(dict(**row, status='source_extracted_pending_review', sheet=name, details=detail_name,
                        source_video_sha256=digest, current_RGB_range=[lo, hi], current_source_indices=indices[lo:hi]))
result = dict(at=datetime.now().astimezone().isoformat(), status='pending_source_visual_review',
              GPU_calls=0, optimizer_updates=0, training_approved=False,
              selection='First valid history24 candidate per action, lexical episode then source start; train labels only.',
              records=records, warning='Observed source windows, not model output, not same-state A/D pair, not proof of action-effect delay.')
(out/'review.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps({'extracted': len(records), 'GPU_calls': 0}))
