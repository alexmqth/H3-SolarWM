"""Native-size limb/motion details plus the shared history boundary, CPU only."""
from pathlib import Path
import json
import av
from PIL import Image,ImageDraw

BASE=Path(__file__).resolve().parent


def read(path):
    with av.open(str(path)) as c:return [Image.fromarray(f.to_ndarray(format='rgb24')) for f in c.decode(video=0)]


def main():
    for ref in 'AD':
        folder=BASE/f'window1_{ref}'
        r=json.loads((folder/'evaluation.json').read_text());assert r['status']=='complete_pending_visual_review'
        history=Image.open(folder/'history_last_visible.png').convert('RGB')
        for action in 'AD':
            frames=read(folder/f'{action}.mp4');assert len(frames)==42
            im=Image.new('RGB',(1664,1020),'white');draw=ImageDraw.Draw(im)
            for j,k in enumerate((0,13,27,41)):
                x=j%2*832;y=j//2*510;im.paste(frames[k],(x,y+30));draw.text((x+8,y+6),f'ref{ref} N current{action} RGB{39+k}',fill='black')
            im.save(folder/f'{action}_detail.jpg')
            old=read(BASE/f'source_coarse/coarse_{ref}/window1_{action}.mp4')
            im=Image.new('RGB',(1664,1020),'white');draw=ImageDraw.Draw(im)
            for j,(name,frame) in enumerate([('shared visible history RGB38',history),('shared visible history RGB38',history),('C first current RGB39',old[0]),('N first current RGB39',frames[0])]):
                x=j%2*832;y=j//2*510;im.paste(frame,(x,y+30));draw.text((x+8,y+6),f'ref{ref} current{action} | {name}',fill='black')
            im.save(folder/f'{action}_boundary.jpg')
    print('4 details +4 boundary sheets ready')


if __name__=='__main__':main()
