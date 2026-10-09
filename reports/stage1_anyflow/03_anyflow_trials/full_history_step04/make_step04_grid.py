"""Complete step04 screening video; optimizer budgets are explicitly different."""
from pathlib import Path
import json
import hashlib
import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
REPORT=OUT/'report'
FONT=ImageFont.load_default(size=17)
entries=[];clips=[]
old=json.loads((ROOT/'outputs/2026-10-08-06/stage1_anyflow39_fullscope/run.json').read_text())
new=json.loads((ROOT/'outputs/2026-10-08-09/stage1_full_history_step04_eval/run.json').read_text())
assert new['status']=='complete'
for action in 'AD':
    teacher=ROOT/f'outputs/2026-10-02-03/action_{action}_teacher_39'
    items=[('Original H3 30 full steps',teacher/'baseline.mp4',json.loads((teacher/'baseline.json').read_text()))]
    for label,state in [('Detached AnyFlow: 16 updates, 8/chunk',old),('Full-history AnyFlow: 4 updates, 8/chunk',new)]:
        row=next(r for r in state['evaluations'] if r['action']==action and r['steps_per_chunk']==8)
        assert row['input_fairness']['exactly_equal']
        items.append((label,Path(row['video']),row['runtime']))
    for label,path,runtime in items:
        assert runtime['status']=='complete'
        with av.open(path) as c:
            stream=c.streams.video[0]
            assert (stream.width,stream.height,stream.average_rate)==(832,480,24)
            decoded=[(f.time,f.to_image().resize((624,360))) for f in c.decode(video=0)]
        assert len(decoded)==39 and all(abs(t-i/24)<1e-6 for i,(t,_) in enumerate(decoded))
        clips.append([f for _,f in decoded]);entries.append(dict(action=action,label=label,video=str(path),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),seconds=runtime['wall_including_shared_setup_seconds'],
            noisy=runtime['denoiser_forwards'],commits=runtime.get('commit_forwards',0)))
video=REPORT/'full_history_step04_AD_screening.mp4'
with av.open(video,'w',options={'movflags':'+faststart'}) as c:
    stream=c.add_stream('libx264',rate=24);stream.width=1872;stream.height=880;stream.pix_fmt='yuv420p'
    stream.options={'crf':'18','preset':'medium','threads':'4'}
    for i in range(39):
        canvas=Image.new('RGB',(1872,880),'black');draw=ImageDraw.Draw(canvas)
        draw.text((8,8),'EARLY SCREENING | NOT PASSED | different optimizer budgets (16 vs 4); full-history16 pending',font=FONT,fill='#ffcb66')
        for j,e in enumerate(entries):
            x,y=j%3*624,42+j//3*404
            draw.text((x+6,y),f'{e["action"]} | {e["label"]}',font=FONT,fill='white')
            draw.text((x+6,y+22),f'{e["seconds"]:.1f}s | {e["noisy"]} noisy + {e["commits"]} commits',font=FONT,fill='white')
            canvas.paste(clips[j][i],(x,y+44))
        draw.text((8,856),f'Frame {i:02d}/38 | full 39f / 24fps | same image/action/seed/noise | concurrent offload timings; no speedup claim',font=FONT,fill='white')
        for packet in stream.encode(av.VideoFrame.from_ndarray(np.asarray(canvas),format='rgb24')):c.mux(packet)
    for packet in stream.encode():c.mux(packet)
with av.open(video) as c:
    decoded=list(c.decode(video=0));assert len(decoded)==39
    decoded[30].to_image().save(REPORT/'step04_preview.jpg',quality=95)
(REPORT/'step04_video.json').write_text(json.dumps(dict(scope='diagnostic_not_accepted_not_equal_updates',entries=entries,
    output=str(video),frames=39,sha256=hashlib.sha256(video.read_bytes()).hexdigest()),indent=2)+'\n')
print(video)
