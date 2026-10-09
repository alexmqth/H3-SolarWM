"""CPU-only labelled V2b vs EXP-002 clips from existing real MP4 files."""
from pathlib import Path
import argparse
import hashlib
import json

import av
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = ROOT / "H3-World/outputs/EXP-002_native_cached"
V2B56 = ROOT / "H3-World/outputs/2026-10-09-22/chunk_partition_cb/C_A"
V2B73 = ROOT / "submission/experiments/EXP-001_v2b_124/artifacts/videos"


def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(4<<20),b""): h.update(b)
    return h.hexdigest()


def read(path):
    with av.open(str(path)) as c:
        stream=c.streams.video[0]
        frames=[f.to_image().convert("RGB") for f in c.decode(video=0)]
        assert stream.width==832 and stream.height==480
        assert str(stream.average_rate)=="24"
    return frames


def render(left,right,path,labels,action,boundary):
    assert len(left)==len(right) in (56,73)
    font=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",17)
    stream=av.open(str(path),"w")
    video=stream.add_stream("libx264",rate=24)
    video.width=1664; video.height=560; video.pix_fmt="yuv420p"
    video.options={"crf":"18","preset":"medium"}
    try:
        for i,(a,b) in enumerate(zip(left,right)):
            canvas=Image.new("RGB",(1664,560),"#10151d")
            draw=ImageDraw.Draw(canvas)
            canvas.paste(a,(0,40));canvas.paste(b,(832,40))
            draw.text((10,10),labels[0],font=font,fill="white")
            draw.text((842,10),labels[1],font=font,fill="white")
            stage="shared first12 A" if i<39 else f"current action {action}"
            draw.text((10,530),f"RGB {i:02d}  |  {stage}",font=font,fill="white")
            if i>=boundary:
                draw.text((842,530),f"generated history  |  boundary at RGB {boundary}",
                          font=font,fill="#ffcf80")
            frame=av.VideoFrame.from_image(canvas)
            for packet in video.encode(frame): stream.mux(packet)
        for packet in video.encode(): stream.mux(packet)
    finally: stream.close()
    assert len(read_comparison(path))==len(left)


def read_comparison(path):
    with av.open(str(path)) as c:
        assert c.streams.video[0].width==1664 and c.streams.video[0].height==560
        assert str(c.streams.video[0].average_rate)=="24"
        return [f.pts for f in c.decode(video=0)]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--path",choices=["AA","AD"],required=True)
    p.add_argument("--frames",type=int,choices=[56,73],required=True)
    args=p.parse_args()
    action=args.path[1]
    left=(V2B56/f"sigma_noised_{action}_rollout.mp4" if args.frames==56
          else V2B73/f"{args.path}_rollout_73.mp4")
    right=OUT/args.path/f"rollout_{args.frames}.mp4"
    assert left.exists() and right.exists()
    result=OUT/args.path/f"V2b_vs_cached_{args.path}_{args.frames}.mp4"
    assert not result.exists(), "never overwrite a comparison MP4"
    labels=("V2b Same-sigma | local bidir | no KV | 30-step",
            "EXP-002 | strict chunk causal | raw KV | 30-step")
    render(read(left),read(right),result,labels,action,39)
    manifest=dict(left=str(left),left_sha256=sha(left),right=str(right),
                  right_sha256=sha(right),output=str(result),output_sha256=sha(result),
                  frames=args.frames,fps=24,labels=labels,
                  note="Cross-protocol comparison: same first12 latent, I0, seed/noise; "
                       "V2b same-sigma recompute vs candidate sigma0 persistent KV, different attention topology. "
                       "After RGB39 generated histories differ; not a single-factor ablation.")
    (OUT/args.path/f"comparison_{args.frames}.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(result)


if __name__=="__main__": main()
