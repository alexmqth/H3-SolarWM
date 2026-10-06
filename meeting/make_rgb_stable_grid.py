#!/usr/bin/env python3
"""Stack the four timed RGB-anchor side-by-side videos into one meeting grid."""
from pathlib import Path
import argparse
import av
import numpy as np
from PIL import Image

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input-dir',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); args=ap.parse_args()
    paths=[args.input_dir/f'h3world_rgb_stable_{a}_original_vs_causal_timed.mp4' for a in 'WSAD']
    containers=[av.open(str(p)) for p in paths]
    iterators=[c.decode(video=0) for c in containers]
    try:
        streams=[c.streams.video[0] for c in containers]
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with av.open(str(args.output),'w',options={'movflags':'+faststart'}) as out:
            stream=out.add_stream('libx264',rate=streams[0].average_rate or 24)
            stream.width=streams[0].width; stream.height=streams[0].height*4; stream.pix_fmt='yuv420p'; stream.options={'crf':'18','preset':'medium'}
            while True:
                frames=[]
                for it in iterators:
                    try: frames.append(next(it).to_image().convert('RGB'))
                    except (StopIteration, av.error.EOFError): frames=[]; break
                if not frames: break
                canvas=Image.new('RGB',(frames[0].width,sum(f.height for f in frames)),(0,0,0)); y=0
                for f in frames: canvas.paste(f,(0,y)); y+=f.height
                vf=av.VideoFrame.from_ndarray(np.asarray(canvas),format='rgb24')
                for packet in stream.encode(vf): out.mux(packet)
            for packet in stream.encode(): out.mux(packet)
    finally:
        for c in containers: c.close()
    print(args.output)
if __name__=='__main__': main()
