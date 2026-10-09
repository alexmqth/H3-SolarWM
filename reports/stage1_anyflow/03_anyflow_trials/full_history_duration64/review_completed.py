"""Visual review artifacts for completed duration-only continuation videos."""
from pathlib import Path
import hashlib
import json
import av
from PIL import Image, ImageDraw, ImageFont

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
REPORT=OUT/'report'
REPORT.mkdir(exist_ok=True)
FONT=ImageFont.load_default(size=16)

def frames(path):
    with av.open(path) as c:
        s=c.streams.video[0]
        assert (s.width,s.height,s.average_rate)==(832,480,24)
        result=[f.to_image() for f in c.decode(video=0)]
    assert len(result)==39
    return result

artifacts=[]
for p in sorted((OUT/'eval').rglob('evaluation.json')):
    e=json.loads(p.read_text())
    if e['runtime']['status']!='complete':continue
    a,n,step=e['action'],e['steps_per_chunk'],e['step']
    current=Path(e['video']);decoded=frames(current)
    name=f'full_history_{step:02d}_{a}_{n}step'
    canvas=Image.new('RGB',(1664,1810),'black');draw=ImageDraw.Draw(canvas)
    draw.text((6,6),f'AnyFlow full-history train shift2.22 step{step} | {a} | {n} steps/chunk | 39 complete frames | not accepted',font=FONT,fill='white')
    for i,f in enumerate(decoded):
        x,y=i%5*332,34+i//5*220
        draw.text((x+4,y),f'frame {i:02d} / {i/24:.3f}s',font=FONT,fill='white')
        canvas.paste(f.resize((332,192)),(x,y+24))
    canvas.save(REPORT/f'{name}_all39.jpg',quality=95)
    sources=[('Original H3 30 full steps',ROOT/f'outputs/2026-10-02-03/action_{a}_teacher_39/baseline.mp4')]
    generated=ROOT/f'outputs/2026-10-08-08/stage1_anyflow39_full_history/eval/anyflow/step_16/generated_{n}step_native/{a}/cached.mp4'
    if (generated.parent/'evaluation.json').exists():
        sources.append((f'AnyFlow full-history step16 {n}/chunk',generated))
    sources.append((f'AnyFlow full-history step{step} {n}/chunk',current))
    canvas=Image.new('RGB',(1664,288*len(sources)),'black');draw=ImageDraw.Draw(canvas)
    for row,(label,video) in enumerate(sources):
        clip=frames(video)
        for col,index in enumerate((12,24,30,38)):
            x,y=col*416,row*288
            draw.text((x+4,y+4),f'{a} {label}',font=FONT,fill='white')
            draw.text((x+4,y+25),f'frame {index}',font=FONT,fill='white')
            canvas.paste(clip[index].resize((416,240)),(x,y+48))
    canvas.save(REPORT/f'{name}_comparison.jpg',quality=95)
    artifacts.append(dict(action=a,step=step,nfe=n,video=str(current),
                          sha256=hashlib.sha256(current.read_bytes()).hexdigest(),
                          frames=39,horizontal_flow=e['flow']['horizontal_flow_px']['mean']))
(REPORT/'completed_review_inputs.json').write_text(json.dumps(artifacts,indent=2)+'\n')
print(json.dumps(artifacts,indent=2))
