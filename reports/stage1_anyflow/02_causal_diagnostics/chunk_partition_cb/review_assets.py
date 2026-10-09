"""CPU full-current-frame and original-resolution detail views for review."""
import argparse
from pathlib import Path
import av
from PIL import Image,ImageDraw

BASE=Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('case',choices=['C_A','C_D','B_first']);args=parser.parse_args()
folder=BASE/args.case
paths=sorted(folder.glob('*_current.mp4'))
assert paths,'Sampling/decode has not finished'
for path in paths:
    with av.open(str(path)) as c:frames=[Image.fromarray(f.to_ndarray(format='rgb24')) for f in c.decode(video=0)]
    picks=sorted(set([0,4,8,12,len(frames)-1]))
    output=Image.new('RGB',(416*len(picks),420),'white');draw=ImageDraw.Draw(output)
    for i,j in enumerate(picks):
        draw.text((416*i+4,4),f'{path.stem} currentRGB {j}',fill='black')
        output.paste(frames[j].crop((208,80,624,480)),(416*i,20))
    output.save(folder/f'{path.stem}_details.jpg',quality=94)
    if args.case!='B_first':
        with av.open(str(folder/'startup_reused.mp4')) as c:
            previous=[Image.fromarray(f.to_ndarray(format='rgb24')) for f in c.decode(video=0)]
        boundary=previous[-3:]+frames[:3]
        image=Image.new('RGB',(416*6,260),'white');draw=ImageDraw.Draw(image)
        for i,f in enumerate(boundary):
            draw.text((416*i+4,4),f'RGB {36+i}',fill='black');image.paste(f.resize((416,240)),(416*i,20))
        image.save(folder/f'{path.stem}_boundary.jpg',quality=94)
print(args.case,len(paths),'clips, full-resolution character details prepared')
