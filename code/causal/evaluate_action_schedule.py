#!/usr/bin/env python3
"""Measure optical-flow response separately for chunk-action schedule segments."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import av
import cv2
import numpy as np


def read_gray(path: Path):
    with av.open(str(path)) as container:
        return [cv2.resize(cv2.cvtColor(f.to_ndarray(format='rgb24'), cv2.COLOR_RGB2GRAY),
                           (416, 240), interpolation=cv2.INTER_AREA)
                for f in container.decode(video=0)]


def segment_flow(path: Path, segments):
    frames = read_gray(path)
    flows = []
    for previous, current in zip(frames, frames[1:]):
        flow = cv2.calcOpticalFlowFarneback(
            previous, current, None, 0.5, 3, 21, 3, 5, 1.2, 0)
        h, w = current.shape
        flows.append(flow[12:h - 12, 21:w - 21])
    rows = []
    for label, start, stop in segments:
        # RGB frame f uses the flow from f-1 to f.  Keep the transition at the
        # left edge with its new segment, and exclude the terminal frame.
        part = flows[max(0, start - 1):min(len(flows), stop - 1)]
        if not part:
            raise ValueError(f'empty segment {label}: {start}:{stop}')
        values = np.concatenate([x.reshape(-1, 2) for x in part], axis=0)
        rows.append({
            'label': label, 'start_frame': start, 'stop_frame_exclusive': stop,
            'horizontal_flow_mean': float(values[:, 0].mean()),
            'horizontal_flow_median': float(np.median(values[:, 0])),
            'vertical_flow_mean': float(values[:, 1].mean()),
            'vertical_flow_median': float(np.median(values[:, 1])),
            'magnitude_mean': float(np.linalg.norm(values, axis=1).mean()),
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--segments', required=True,
                        help='semicolon-separated LABEL:START:STOP, e.g. W:0:51,A:51:85,D:85:124')
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('videos', nargs='+', metavar='NAME=VIDEO')
    args = parser.parse_args()
    segments = []
    for raw in args.segments.split(';'):
        label, start, stop = raw.split(':')
        segments.append((label, int(start), int(stop)))
    output = []
    for raw in args.videos:
        name, path = raw.split('=', 1)
        output.append({'name': name, 'path': path,
                       'segments': segment_flow(Path(path), segments)})
    payload = {'segments': segments, 'videos': output,
               'note': 'Farneback central-frame motion proxy; not strict action accuracy.'}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + '\n')
    print(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()
