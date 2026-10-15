"""CPU exact identity→RGB→MP4 path using bfloat16 saved condition."""
from __future__ import annotations

import json
import torch

import fm_run as fm
import finalize_cpu as gallery
from common import HERE, ROOT, ENCODED, inputs, raw_source_rgb, sha, write


def main():
    _,_,_,train,_,roles=inputs()
    clip=roles["train_constant"]["clip_id"]
    f=torch.load(ENCODED/clip/"condition.pt",map_location="cpu",weights_only=True)
    record=json.loads((ENCODED/clip/"record.json").read_text())
    assert f["initial_noise"].dtype==torch.bfloat16
    data={"f":f,"record":record}
    frames=raw_source_rgb(train[clip])
    out=HERE/"media_smoke_identity"
    if out.exists():raise FileExistsError(out)
    out.mkdir()
    path=out/"source_identity_path_test.mp4"
    raw=fm.persist_rgb(frames,path)
    identity=fm.video_identity(data,"train_constant","eval_pre",
          "GT first39 displayed and committed")
    write(path.with_suffix(".identity.json"),identity)
    fm.save_video(frames,path)
    decoded,fps=gallery.read_video(path)
    assert decoded.shape==frames.shape and fps==24
    assert len(identity["initial_noise_sha256"])==64
    result=dict(status="PASS_CPU_BFLOAT16_IDENTITY_RGB_MP4",gpu_calls=0,
       clip_id=clip,initial_noise_dtype=str(f["initial_noise"].dtype),
       identity=identity,raw_path=str(raw.relative_to(ROOT)),raw_sha256=sha(raw),
       mp4_path=str(path.relative_to(ROOT)),mp4_sha256=sha(path),
       decoded_frames=56,fps=24,
       ffmpeg=json.loads(path.with_suffix(".media.json").read_text())["ffmpeg_path"],
       note="real source RGB codec test only, not model inference")
    write(HERE/"MEDIA_SMOKE_V2.json",result)
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()
