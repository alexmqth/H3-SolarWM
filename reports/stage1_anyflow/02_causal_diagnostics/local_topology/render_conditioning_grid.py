"""One diagnostic A/D grid for completed window12 conditioning controls."""
from pathlib import Path
import hashlib
import json
import av
from PIL import Image, ImageDraw, ImageFont

BASE = Path(__file__).resolve().parent
GROUPS = [('window12_calibration', 'Clean text time + dual image'),
          ('native_prefix12_calibration', 'Native text time + dual image'),
          ('native_single_anchor12_calibration', 'Native text time + original I0')]


def main():
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 17)
    small = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 14)
    cells = []
    for group, title in GROUPS:
        receipt = json.loads((BASE/group/'evaluation.json').read_text())
        assert receipt['status'] == 'complete_pending_visual_review'
        videos = {}
        for action in 'AD':
            with av.open(str(BASE/group/f'{action}.mp4')) as c:
                videos[action] = [f.to_image().resize((416, 240)) for f in c.decode(video=0)]
            assert len(videos[action]) == 39
        cells.append((title, receipt, videos))
    out = BASE/'conditioning_window12_AD.mp4'
    with av.open(str(out), 'w', options={'movflags': '+faststart'}) as c:
        s = c.add_stream('libx264', rate=24); s.width=1248; s.height=658
        s.pix_fmt='yuv420p'; s.options={'crf':'18', 'preset':'medium'}
        for frame in range(39):
            im=Image.new('RGB',(1248,658),'#12161d'); d=ImageDraw.Draw(im)
            d.text((12,8),'Original weights | 30 steps | 39 RGB / 12 latent | same image, prompt, seed13 and initial noise',font=font,fill='white')
            d.text((12,32),'Reference calibration only: all current-window actions known; NOT a 5-latent causal rollout. Zero training.',font=small,fill='#ffce73')
            for col,(title,receipt,videos) in enumerate(cells):
                x=col*416; d.text((x+8,62),title,font=font,fill='white')
                for row,action in enumerate('AD'):
                    r=next(r for r in receipt['records'] if r['action']==action)
                    flow=r['flow']['horizontal_flow_px']['mean']; y=88+row*280
                    d.text((x+8,y),f"{action} | flow {flow:+.3f} | sampling {r['sampling_seconds']:.1f}s",font=small,fill='white')
                    im.paste(videos[action][frame],(x,y+25))
            d.text((12,638),f'Frame {frame:02d}/38 | Fixed audio noise in all columns | Flow is a motion proxy; review appearance separately.',font=small,fill='#bac2cf')
            if frame in (0,12,24,38):im.save(BASE/f'conditioning_grid_frame{frame:02d}.jpg')
            for packet in s.encode(av.VideoFrame.from_image(im)):c.mux(packet)
        for packet in s.encode():c.mux(packet)
    with av.open(str(out)) as c:
        frames=list(c.decode(video=0));assert len(frames)==39
        assert c.streams.video[0].codec_context.name=='h264'
        assert c.streams.video[0].average_rate==24
        assert all(f.format.name=='yuv420p' for f in frames)
    receipt=dict(file=out.name,frames=39,fps=24,codec='h264',pix_fmt='yuv420p',
                 sha256=hashlib.sha256(out.read_bytes()).hexdigest(),
                 source_groups=[g for g,_ in GROUPS],stage1_acceptance=False,
                 scope='Controlled reference calibration; not persistent-KV or multi-chunk success')
    (BASE/'conditioning_grid_audit.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
