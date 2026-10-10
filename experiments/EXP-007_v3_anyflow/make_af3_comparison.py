"""CPU-only matched 73-frame FM8 vs trained-AF continuation gallery."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import av
import imageio_ffmpeg
import numpy as np
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parents[3]
FM8=ROOT/"H3-World/outputs/EXP-006_v3_fm8_full"
AF3=ROOT/"H3-World/outputs/EXP-007_v3_anyflow_af3"
HERE=Path(__file__).resolve().parent
DEST=HERE/"artifacts/af3_videos"


def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(4<<20),b""):h.update(block)
    return h.hexdigest()


def check(path):
    with av.open(str(path)) as container:
        stream=container.streams.video[0]
        frames=list(container.decode(video=0))
        pts=[f.pts for f in frames]
        assert len(frames)==73 and len(set(pts))==73 and pts==sorted(pts)
        assert all(f.width==832 and f.height==480 for f in frames)
        assert stream.average_rate==24
    return frames


def render(path):
    left=FM8/path/"rollout_73.mp4"
    right=AF3/path/"rollout_73.mp4"
    f0,f1=check(left),check(right)
    # H.264 may encode an identical raw prefix slightly differently when the
    # future frames differ; compare the published raw RGB arrays exactly.
    raw_first=np.load(FM8/"published_39.npy")
    assert np.array_equal(raw_first,np.load(FM8/path/"published_73.npy")[:39])
    assert np.array_equal(raw_first,np.load(AF3/path/"published_73.npy")[:39])
    DEST.mkdir(parents=True,exist_ok=True)
    target=DEST/f"{path}_FM8_vs_AF8_continuation_73.mp4"
    ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
    proc=subprocess.Popen([ffmpeg,"-hide_banner","-loglevel","error","-y",
        "-f","rawvideo","-pixel_format","rgb24","-video_size","1248x424",
        "-framerate","24","-i","-","-an","-c:v","libx264","-crf","19",
        "-threads","4","-pix_fmt","yuv420p","-movflags","+faststart",str(target)],stdin=subprocess.PIPE)
    font=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",16)
    small=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",13)
    try:
        for i,(a,b) in enumerate(zip(f0,f1)):
            image=Image.new("RGB",(1248,424),"black")
            image.paste(Image.fromarray(a.to_ndarray(format="rgb24")).resize((624,360)),(0,40))
            image.paste(Image.fromarray(b.to_ndarray(format="rgb24")).resize((624,360)),(624,40))
            draw=ImageDraw.Draw(image)
            draw.text((8,10),"V3 ordinary FM8 | original weights",font=font,fill="white")
            draw.text((632,10),"V3 AnyFlow 8NFE | trained target/QKV",font=font,fill="white")
            draw.text((8,404),f"Shared FM8 C1; C2 matched history; C3 own history | {path} | frame {i}",font=small,fill="yellow")
            proc.stdin.write(image.tobytes())
        proc.stdin.close()
        assert proc.wait()==0
    finally:
        if proc.poll() is None:proc.kill()
    with av.open(str(target)) as container:
        output=list(container.decode(video=0))
        assert len(output)==73 and all(f.width==1248 and f.height==424 for f in output)
    return dict(path=str(target.relative_to(ROOT)),sha256=sha(target),frames=73,fps=24,
                source_FM8_sha256=sha(left),source_AF3_sha256=sha(right),
                note="C1 identical FM8 RGB; C2 same clean C1 but different weights/objective; C3 each own generated history")


def main():
    items=[render(path) for path in ("AA","AD")]
    (HERE/"af3_comparison_manifest.json").write_text(json.dumps(items,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(items,ensure_ascii=False))


if __name__=="__main__":main()
