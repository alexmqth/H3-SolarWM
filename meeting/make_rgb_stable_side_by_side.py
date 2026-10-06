#!/usr/bin/env python3
"""Build honest original-vs-RGB-anchor visual-stability comparisons.

The original fixed-mix grid is retained as an action-geometry diagnostic. This
helper creates the presentation videos from the later RGB-consistent visual
adapter outputs, so the meeting demo does not show the known latent-anchor
failure as if it were the final visual result.
"""
from pathlib import Path
import argparse, json
import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ORIGINAL_TIMES = {"W": 441.5, "S": 444.8, "A": 454.2, "D": 450.4}

def get_font(size):
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()

def frame_iter(path):
    c=av.open(str(path)); stream=c.streams.video[0]
    try:
        for frame in c.decode(video=0):
            yield frame.to_image().convert("RGB")
    finally:
        c.close()

def causal_time(path):
    meta=json.loads((path.parent/'cached.json').read_text())
    return float(meta['wall_including_shared_setup_seconds'])

def build(action, original_root, causal_root, output):
    original=original_root/f'action_{action}_baseline30_124'/'baseline.mp4'
    causal=causal_root/action/'cached.mp4'
    with av.open(str(original)) as oc, av.open(str(causal)) as cc:
        osr, csr=oc.streams.video[0],cc.streams.video[0]
        if osr.width != csr.width or osr.height != csr.height:
            raise RuntimeError(f'resolution mismatch: {original} vs {causal}')
        output.parent.mkdir(parents=True,exist_ok=True)
        with av.open(str(output),'w',options={'movflags':'+faststart'}) as out:
            stream=out.add_stream('libx264',rate=osr.average_rate or 24)
            stream.width=osr.width+csr.width; stream.height=osr.height+48; stream.pix_fmt='yuv420p'
            stream.options={'crf':'18','preset':'medium'}
            title_font=get_font(18); info_font=get_font(13)
            causal_seconds=causal_time(causal)
            for left,right in zip(oc.decode(video=0),cc.decode(video=0)):
                li=left.to_image().convert('RGB'); ri=right.to_image().convert('RGB')
                canvas=Image.new('RGB',(li.width+ri.width,li.height+48),(0,0,0))
                canvas.paste(li,(0,48)); canvas.paste(ri,(li.width,48))
                draw=ImageDraw.Draw(canvas)
                draw.text((16,4),f'{action} | Original H3-World | 30 steps',fill='white',font=title_font)
                draw.text((li.width+16,4),f'{action} | causal RGB-anchor visual-adapted | 8 steps/chunk',fill='white',font=title_font)
                draw.text((16,29),f'recorded e2e: {ORIGINAL_TIMES[action]:.1f}s | 124 frames / 5.17s',fill=(220,220,220),font=info_font)
                draw.text((li.width+16,29),f'recorded e2e: {causal_seconds:.1f}s | 64 noisy forwards + 8 commits',fill=(220,220,220),font=info_font)
                vf=av.VideoFrame.from_ndarray(np.asarray(canvas),format='rgb24')
                for packet in stream.encode(vf): out.mux(packet)
            for packet in stream.encode(): out.mux(packet)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--original-root',type=Path,required=True)
    ap.add_argument('--causal-root',type=Path,required=True)
    ap.add_argument('--output-dir',type=Path,required=True)
    args=ap.parse_args()
    for action in 'WSAD':
        out=args.output_dir/f'h3world_rgb_stable_{action}_original_vs_causal_timed.mp4'
        build(action,args.original_root,args.causal_root,out)
        print(out)

if __name__=='__main__': main()
