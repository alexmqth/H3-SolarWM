"""CPU-only saved-output verification and complete new-frame contact sheets."""
from pathlib import Path
import argparse, hashlib, json, math
import av
import numpy as np
from PIL import Image,ImageDraw

p=argparse.ArgumentParser()
p.add_argument('--rgb',type=Path,required=True)
p.add_argument('--video',type=Path,required=True)
p.add_argument('--prior',type=Path,required=True)
p.add_argument('--start',type=int,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
frames=np.load(a.rgb,mmap_mode='r'); prior=np.load(a.prior,mmap_mode='r')
assert frames.dtype==np.uint8 and frames.shape[1:]==(480,832,3)
assert len(prior)>=a.start and len(frames)>a.start
checks=dict(rgb_frames=len(frames),new_frames=len(frames)-a.start,
 prefix_unchanged=bool(np.array_equal(frames[:a.start],prior[:a.start])),
 RGB_sha256=hashlib.sha256(frames.tobytes()).hexdigest())
assert checks['prefix_unchanged']
with av.open(str(a.video)) as c:
 s=c.streams.video[0];rate=str(s.average_rate);pts=[f.pts for f in c.decode(video=0)]
 assert len(pts)==len(frames) and rate=='24'
 assert all(b>a for a,b in zip(pts,pts[1:]))
 checks.update(video_frames=len(pts),fps=rate,monotonic_pts=True)
indices=list(range(max(0,a.start-2),len(frames))); cols=4
sheet=Image.new('RGB',(416*cols,260*math.ceil(len(indices)/cols)),'white'); draw=ImageDraw.Draw(sheet)
for n,i in enumerate(indices):
 x=n%cols*416;y=n//cols*260
 sheet.paste(Image.fromarray(frames[i]).resize((416,240)),(x,y))
 draw.text((x+4,y+243),f'{a.output.name} RGB{i}',fill='black')
a.output.parent.mkdir(parents=True,exist_ok=True)
sheet.save(a.output.with_suffix('.jpg'),quality=93)
Image.fromarray(frames[-1]).save(a.output.with_name(a.output.name+'_last.png'))
a.output.with_suffix('.json').write_text(json.dumps(checks,indent=2)+'\n')
print(json.dumps(checks))
