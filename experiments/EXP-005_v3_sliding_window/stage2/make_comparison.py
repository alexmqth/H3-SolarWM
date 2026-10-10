"""CPU-only labeled H.264 comparison of already generated equal-length videos."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess

import cv2
import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont


def probe(path: Path):
    cap=cv2.VideoCapture(str(path))
    if not cap.isOpened():raise ValueError(f"cannot open {path}")
    frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps=cap.get(cv2.CAP_PROP_FPS)
    width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return frames,fps,width,height


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("left","right","output","left-label","right-label"):
        p.add_argument("--"+name,required=True)
    p.add_argument("--note",default="RGB boundaries: 124 and 141 | 30 steps per new chunk")
    args=p.parse_args()
    left,right,out=Path(args.left),Path(args.right),Path(args.output)
    if out.exists():raise FileExistsError(out)
    a,b=probe(left),probe(right)
    if a!=b or a[1]!=24 or a[2:]!=(832,480):
        raise ValueError(f"videos must have identical 24fps 832x480 geometry: {a} vs {b}")
    count=a[0]
    header=Image.new("RGB",(1664,80),(24,29,38))
    draw=ImageDraw.Draw(header)
    font=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",25)
    small=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",17)
    draw.text((24,7),args.left_label,font=font,fill=(255,255,255))
    draw.text((856,7),args.right_label,font=font,fill=(255,255,255))
    draw.text((24,51),args.note,font=small,fill=(190,211,228))
    out.parent.mkdir(parents=True,exist_ok=True)
    banner=out.with_suffix(".header.png")
    header.save(banner)
    command=[imageio_ffmpeg.get_ffmpeg_exe(),"-hide_banner","-loglevel","error","-y",
             "-i",str(left),"-i",str(right),"-framerate","24","-loop","1","-i",str(banner),
             "-filter_complex","[0:v][1:v]hstack=inputs=2[body];[2:v][body]vstack=inputs=2[out]",
             "-map","[out]","-r","24","-frames:v",str(count),"-c:v","libx264",
             "-preset","medium","-crf","20","-pix_fmt","yuv420p",str(out)]
    subprocess.run(command,check=True)
    result=probe(out)
    if result!=(count,24.0,1664,560):
        raise RuntimeError(f"unexpected encoded geometry {result}")
    cap=cv2.VideoCapture(str(out));decoded=0
    while True:
        okay,frame=cap.read()
        if not okay:break
        if frame.shape[:2]!=(560,1664):raise RuntimeError("bad decoded frame geometry")
        decoded+=1
    cap.release()
    if decoded!=count:raise RuntimeError(f"only {decoded}/{count} frames decoded")
    print(f"H.264 comparison complete: {out}, {decoded} frames @ 24fps")


if __name__=="__main__":main()
