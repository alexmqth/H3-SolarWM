"""CPU-only actual writer test using a real source clip; no generated output."""
from __future__ import annotations

import json
import numpy as np
from PIL import Image, ImageDraw

import fm_run as fm
import finalize_cpu as gallery
from common import HERE, ROOT, inputs, raw_source_rgb, sha, write


def main():
    _,_,_,train,_,roles=inputs()
    row=train[roles["train_constant"]["clip_id"]]
    source=raw_source_rgb(row)
    assert source.shape==(56,480,832,3)
    out=HERE/"media_smoke_source_only"
    if out.exists():raise FileExistsError(out)
    out.mkdir()
    single=out/"single_source_only.mp4"
    fm.save_video(source,single)
    a,fps=gallery.read_video(single)
    assert a.shape==source.shape and fps==24
    panels=[]
    for i in range(56):
        image=Image.new("RGB",(2496,540),(20,20,20))
        for j in range(3):image.paste(Image.fromarray(source[i]),(j*832,30))
        ImageDraw.Draw(image).text((8,518),f"CPU SOURCE CODEC TEST ONLY | frame {i}",fill="white")
        panels.append(np.asarray(image))
    paired=out/"paired_source_only.mp4"
    gallery.write_video(np.stack(panels),paired)
    count,fps=gallery.read_gallery(paired)
    assert count==56 and fps==24
    result=dict(status="PASS_CPU_REAL_SOURCE_CODEC_ONLY",gpu_calls=0,
                source_clip=row["clip_id"],
                single_video=str(single.relative_to(ROOT)),single_sha256=sha(single),
                paired_video=str(paired.relative_to(ROOT)),paired_sha256=sha(paired),
                single_rgb_recoverable=str(single.with_suffix(".rgb.npy").relative_to(ROOT)),
                single_media=json.loads(single.with_suffix(".media.json").read_text()),
                frames=56,fps=24,note="source video codec smoke only, not model inference")
    write(HERE/"MEDIA_SMOKE.json",result)
    print(json.dumps({k:v for k,v in result.items() if k!="single_media"},indent=2))


if __name__=="__main__":main()
