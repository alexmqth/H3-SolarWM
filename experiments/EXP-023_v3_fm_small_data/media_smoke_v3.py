"""CPU-only full media path and injected-failure recovery checks."""
from __future__ import annotations

import json
import numpy as np
from PIL import Image, ImageDraw
import torch
import imageio_ffmpeg

import fm_run as fm
import finalize_cpu as gallery
from common import HERE, ROOT, ENCODED, inputs, raw_source_rgb, sha, write


def fixture(clip,row):
    f=torch.load(ENCODED/clip/"condition.pt",map_location="cpu",weights_only=True)
    record=json.loads((ENCODED/clip/"record.json").read_text())
    assert f["initial_noise"].dtype==torch.bfloat16
    frames=raw_source_rgb(row)
    return {"f":f,"record":record},frames


def valid_case(out,name,data,frames,role,history):
    path=out/f"{name}.mp4"
    raw=fm.persist_rgb(frames,path)
    identity=fm.video_identity(data,role,"eval_pre",history)
    write(path.with_suffix(".identity.json"),identity)
    fm.save_video(frames,path)
    decoded,fps=gallery.read_video(path)
    assert decoded.shape==frames.shape and fps==24
    media=json.loads(path.with_suffix(".media.json").read_text())
    assert media["rgb_sha256"]==sha(raw) and media["mp4_sha256"]==sha(path)
    return dict(video=str(path.relative_to(ROOT)),video_sha256=sha(path),
                raw=str(raw.relative_to(ROOT)),raw_sha256=sha(raw),
                noise_sha256=identity["initial_noise_sha256"],
                ffmpeg_path=media["ffmpeg_path"],ffmpeg_sha256=media["ffmpeg_sha256"])


def main():
    _,_,_,train,dev,roles=inputs()
    gtclip=roles["train_constant"]["clip_id"]
    selfclip=roles["dev_switch"]["clip_id"]
    gtdata,gt=fixture(gtclip,train[gtclip])
    selfdata,selfrgb=fixture(selfclip,dev[selfclip])
    out=HERE/"media_smoke_v3_source_only"
    if out.exists():raise FileExistsError(out)
    out.mkdir()
    result={"status":"RUNNING","gpu_calls":0,"note":"source RGB codec test only, never generated model output"}
    result["GT_history_writer"]=valid_case(out,"gt_source_codec",gtdata,gt,
         "train_constant","GT first39 displayed and committed")
    result["self_history_writer"]=valid_case(out,"self_source_codec",selfdata,selfrgb,
         "dev_switch","self-generated C1 and C2")

    # Exercise the same ordered calls as evaluate(): persist first, then
    # metadata. A broken metadata function cannot destroy decoded RGB.
    metadata_path=out/"injected_identity_error.mp4"
    raw=fm.persist_rgb(gt,metadata_path);before=sha(raw)
    saved=fm.tensor_sha
    try:
        fm.tensor_sha=lambda _: (_ for _ in ()).throw(RuntimeError("INJECTED_IDENTITY_ERROR"))
        try: fm.video_identity(gtdata,"train_constant","eval_pre","GT first39 displayed and committed")
        except RuntimeError as error: assert str(error)=="INJECTED_IDENTITY_ERROR"
        else: raise AssertionError("identity injection not exercised")
    finally: fm.tensor_sha=saved
    assert raw.exists() and sha(raw)==before
    result["identity_failure_kept_rgb"]=True

    codec_path=out/"injected_codec_error.mp4"
    raw=fm.persist_rgb(gt,codec_path);before=sha(raw)
    saved_exe=imageio_ffmpeg.get_ffmpeg_exe
    try:
        imageio_ffmpeg.get_ffmpeg_exe=lambda: "/no/such/imageio-ffmpeg"
        try: fm.save_video(gt,codec_path)
        except AssertionError: pass
        else: raise AssertionError("codec injection not exercised")
    finally: imageio_ffmpeg.get_ffmpeg_exe=saved_exe
    assert raw.exists() and sha(raw)==before and not codec_path.exists()
    result["codec_failure_kept_rgb"]=True

    panels=[]
    for i in range(56):
        image=Image.new("RGB",(2496,540),(20,20,20))
        for j in range(3):image.paste(Image.fromarray(gt[i]),(j*832,30))
        ImageDraw.Draw(image).text((8,518),f"CPU SOURCE CODEC TEST ONLY | frame {i}",fill="white")
        panels.append(np.asarray(image))
    paired=out/"paired_source_codec.mp4"
    gallery.write_video(np.stack(panels),paired)
    frames,fps=gallery.read_gallery(paired)
    assert frames==56 and fps==24
    result["paired_writer"]=dict(video=str(paired.relative_to(ROOT)),sha256=sha(paired),
                                  frames=frames,fps=fps)
    result["status"]="PASS_CPU_GT_SELF_PAIRED_AND_FAULT_RECOVERY"
    write(HERE/"MEDIA_SMOKE_V3.json",result)
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()
