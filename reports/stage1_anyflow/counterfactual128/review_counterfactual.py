"""Complete footage and static frames for the fixed-history action intervention."""
import hashlib
import json
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent
REPORT = OUT / 'report'
rows = json.loads((REPORT/'counterfactual_response.json').read_text())['rows']
font = ImageFont.load_default(size=17)
clips = []
for row in rows:
    with av.open(row['video']) as container:
        stream = container.streams.video[0]
        assert (stream.width,stream.height,stream.average_rate)==(832,480,24)
        clip = [frame.to_image() for frame in container.decode(video=0)]
    assert len(clip)==39
    clips.append(clip)
    if '_to_' not in row['name']:
        continue
    canvas=Image.new('RGB',(1664,1810),'black');draw=ImageDraw.Draw(canvas)
    draw.text((6,6),f"TEACHER-history counterfactual {row['name']} | step128 | 8 steps/chunk | only chunk1 attribution",font=font,fill='white')
    for i,frame in enumerate(clip):
        x,y=i%5*332,34+i//5*220
        draw.text((x+4,y),f'frame {i:02d} / {i/24:.3f}s',font=font,fill='white')
        canvas.paste(frame.resize((332,192)),(x,y+24))
    canvas.save(REPORT/f"{row['name']}_all39.jpg",quality=95)

# Four rows, one per history/current-action combination. Keep the target
# interval's interior and its final edge, with a pre-switch reference frame.
canvas=Image.new('RGB',(1664,1152),'black');draw=ImageDraw.Draw(canvas)
for i,(row,clip) in enumerate(zip(rows,clips)):
    for col,index in enumerate((12,20,26,33)):
        x,y=col*416,i*288
        draw.text((x+4,y+4),f"History {row['history_action']} / current {row['current_action']}",font=font,fill='white')
        draw.text((x+4,y+25),f'frame {index} | step128 | 8/chunk',font=font,fill='white')
        canvas.paste(clip[index].resize((416,240)),(x,y+48))
canvas.save(REPORT/'fixed_history_same_frames.jpg',quality=95)

path=REPORT/'fixed_history_action_intervention_39.mp4'
with av.open(path.with_suffix('.tmp.mp4'),'w',options={'movflags':'+faststart'}) as container:
    stream=container.add_stream('libx264',rate=24)
    stream.width,stream.height,stream.pix_fmt=1248,894,'yuv420p'
    stream.options={'crf':'18','preset':'medium','threads':'4'}
    for index in range(39):
        canvas=Image.new('RGB',(1248,894),'black');draw=ImageDraw.Draw(canvas)
        draw.text((10,7),'TEACHER HISTORY | FIXED PREFIX | action intervention at chunk1 | NOT free-running',font=font,fill='#ffcb66')
        draw.text((10,30),'Columns: current A / current D | Rows: past A / past D | step128, 8 steps/chunk',font=font,fill='white')
        for i,(row,clip) in enumerate(zip(rows,clips)):
            x,y=i%2*624,60+i//2*402
            draw.text((x+8,y),f"History {row['history_action']} | current {row['current_action']} | {row['name']}",font=font,fill='white')
            draw.text((x+8,y+21),f"RGB[17,34) flow {row['flow_chunk1']:+.3f} | full footage retained",font=font,fill='white')
            canvas.paste(clip[index].resize((624,360)),(x,y+42))
        draw.text((10,868),f'Frame {index:02d}/38 | only chunk1 has matched history/actions; chunk2 is not an isolated intervention',font=font,fill='#ffcb66')
        for packet in stream.encode(av.VideoFrame.from_ndarray(np.asarray(canvas),format='rgb24')):
            container.mux(packet)
    for packet in stream.encode():container.mux(packet)
path.with_suffix('.tmp.mp4').replace(path)
with av.open(path) as container:
    frames=list(container.decode(video=0))
assert len(frames)==39
frames[26].to_image().save(path.with_suffix('.frame26.jpg'))
path.with_suffix('.json').write_text(json.dumps(dict(frames=39,fps=24,
    codec='h264',pixel_format='yuv420p',inputs=rows,
    sha256=hashlib.sha256(path.read_bytes()).hexdigest()),indent=2)+'\n')
print(path)
