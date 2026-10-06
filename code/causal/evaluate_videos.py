#!/usr/bin/env python3
"""Decode every frame and report descriptive motion/edge statistics.

MAD is NOT a quality score: frozen or blurred videos can score lower.
No claimed optical-flow, perceptual or human-preference metric here.
"""
import argparse
import json
from pathlib import Path
import av
import numpy as np


def evaluate(path):
    histogram=np.zeros(256,dtype=np.int64)
    frame_mad=[]; edges=[]; previous=None; count=0
    with av.open(str(path)) as c:
        s=c.streams.video[0]
        info=dict(path=str(path),width=s.width,height=s.height,fps=float(s.average_rate),codec=s.codec_context.name)
        for frame in c.decode(video=0):
            gray=frame.to_ndarray(format='gray').astype(np.int16)
            edges.append(float((np.abs(np.diff(gray,axis=0)).mean()+np.abs(np.diff(gray,axis=1)).mean())/2))
            if previous is not None:
                diff=np.abs(gray-previous)
                histogram+=np.bincount(diff.ravel(),minlength=256)
                frame_mad.append(float(diff.mean()))
            count+=1
            previous=gray
    pixels=histogram.sum()
    if not pixels:
        raise ValueError(f'{path}: fewer than two frames')
    info.update(frames=count,duration_seconds=count/info['fps'],
                gray_pixel_difference_mean=float(histogram@np.arange(256)/pixels),
                gray_pixel_difference_p95=int(np.searchsorted(histogram.cumsum(),.95*pixels)),
                mean_spatial_edge_difference=float(np.mean(edges)),
                adjacent_frame_mad=frame_mad,
                caveat='Descriptive pixel statistics; lower MAD may mean blur or little motion, not better quality.')
    return info


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('videos',nargs='+',type=Path)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    result=[evaluate(p) for p in args.videos]
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps([{k:v for k,v in r.items() if k!='adjacent_frame_mad'} for r in result],indent=2))


if __name__=='__main__':
    main()
