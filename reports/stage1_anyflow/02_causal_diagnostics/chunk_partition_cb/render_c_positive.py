"""Render the four complete56-frame N-history trajectories at native24fps."""
from pathlib import Path
import sys
import av
from PIL import Image,ImageDraw,ImageFont
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE/'runtime/code/causal'))
from benchmark import write_video

clips={}
for ref in 'AD':
    for action in 'AD':
        with av.open(str(BASE/f'C_{ref}/sigma_noised_{action}_rollout.mp4')) as c:
            clips[ref,action]=[Image.fromarray(f.to_ndarray(format='rgb24')) for f in c.decode(video=0)]
        assert len(clips[ref,action])==56
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17)
frames=[]
for j in range(56):
    image=Image.new('RGB',(1664,1040),'black');draw=ImageDraw.Draw(image)
    for row,ref in enumerate('AD'):
        for col,action in enumerate('AD'):
            x,y=col*832,row*520;image.paste(clips[ref,action][j],(x,y+40))
            current=ref if j<39 else action
            label=f'C: {ref} -> {action} | chunk {int(j>=39)} / RGB {j} | action {current} | 30 steps/chunk | N history'
            draw.text((x+8,y+10),label,font=font,fill='white')
    frames.append(image)
write_video(frames,BASE/'C_selfhistory_N_56.mp4')
