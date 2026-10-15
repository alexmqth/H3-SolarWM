"""CPU-only paired EXP-023 gallery and MP4 integrity check after final eval."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess

import av
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw

from common import HERE, ROOT, inputs, raw_source_rgb, sha, write


def read_video(path):
    with av.open(str(path)) as container:
        stream=container.streams.video[0]
        fps=float(stream.average_rate)
        frames=np.stack([frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)])
    assert frames.shape==(56,480,832,3), (path,frames.shape)
    assert abs(fps-24)<.001, (path,fps)
    return frames, fps


def write_video(frames,path):
    if path.exists(): raise FileExistsError(path)
    h,w=frames.shape[1:3]
    executable=Path(imageio_ffmpeg.get_ffmpeg_exe()).resolve()
    assert executable.is_file()
    temp=path.with_suffix(".partial.mp4")
    if temp.exists():raise FileExistsError(temp)
    cmd=[str(executable),"-v","error","-f","rawvideo","-pix_fmt","rgb24",
         "-s",f"{w}x{h}","-r","24","-i","pipe:0","-an","-c:v","libx264",
         "-threads","2","-crf","20","-pix_fmt","yuv420p",str(temp)]
    p=subprocess.Popen(cmd,stdin=subprocess.PIPE)
    p.communicate(np.ascontiguousarray(frames).tobytes())
    if p.returncode: raise RuntimeError(f"ffmpeg failure: {temp}")
    read_gallery(temp)
    temp.replace(path)


def action_for_frame(script,index):
    cursor=0
    for latent,action in enumerate(script):
        span=1 if latent%5==0 else 4
        if index<cursor+span:return latent,action
        cursor+=span
    raise AssertionError(index)


def main():
    before=HERE/"eval_before";after=HERE/"eval_after"
    b=json.loads((before/"results.json").read_text())
    a=json.loads((after/"results.json").read_text())
    assert len(a)==len(b)==6
    _,_,_,train,dev,roles=inputs()
    out=HERE/"paired_gallery"
    if out.exists():raise FileExistsError(out)
    out.mkdir()
    index_before={(x["role"],x["history"]):x for x in b}
    index_after={(x["role"],x["history"]):x for x in a}
    assert set(index_before)==set(index_after)
    records=[]
    for key,row0 in index_before.items():
        row1=index_after[key]
        assert row0["clip_id"]==row1["clip_id"]
        role,history=key;clip=row0["clip_id"]
        rgb=raw_source_rgb((train if role.startswith("train") else dev)[clip])
        x0,fps0=read_video(ROOT/row0["video"])
        x1,fps1=read_video(ROOT/row1["video"])
        assert fps0==fps1==24
        prefix_mad_0=prefix_mad_128=None
        if history.startswith("GT"):
            prefix_mad_0=float(np.mean(np.abs(x0[:39].astype(float)-rgb[:39].astype(float))))
            prefix_mad_128=float(np.mean(np.abs(x1[:39].astype(float)-rgb[:39].astype(float))))
            assert prefix_mad_0<10 and prefix_mad_128<10
        condition_path=HERE/"encoded"/clip/"condition.pt"
        import torch
        condition=torch.load(condition_path,map_location="cpu",weights_only=True)
        script=condition["script"]
        rows=[]
        for frame in range(56):
            canvas=Image.new("RGB",(3*832,540),(20,20,20));draw=ImageDraw.Draw(canvas)
            latent,action=action_for_frame(script,frame)
            for col,(title,arr) in enumerate((("SOURCE RGB / GT",rgb),
                 ("ZERO ADAPTER | FM-8NFE",x0),("FM-TRAIN-128UPDATES | FM-8NFE",x1))):
                canvas.paste(Image.fromarray(arr[frame]),(col*832,30))
                draw.text((col*832+10,8),title,fill="white")
            suffix=" | C1→C2 boundary" if frame==39 else ""
            draw.text((12,515),f"{role} | {history} | RGB {frame:02d}/55 | latent {latent:02d} | action: {action}{suffix}",fill="white")
            rows.append(np.asarray(canvas))
        stem=role+("_gt_history" if history.startswith("GT") else "_self_history")
        path=out/f"{stem}_source_vs_0_vs_128.mp4"
        write_video(np.stack(rows),path)
        verified,fps=read_gallery(path)
        sheet=Image.new("RGB",(3*832,6*270),(20,20,20))
        draw=ImageDraw.Draw(sheet)
        for n,frame in enumerate((0,18,38,39,43,55)):
            sample=Image.fromarray(rows[frame]).crop((0,0,3*832,510)).resize((3*832,255))
            sheet.paste(sample,(0,n*270));draw.text((10,n*270+256),f"RGB {frame}",fill="white")
        sheet_path=out/f"{stem}_contact.jpg";sheet.save(sheet_path,quality=88)
        records.append(dict(role=role,history=history,clip_id=clip,
             video=str(path.relative_to(ROOT)),sha256=sha(path),frames=verified,fps=fps,
             contact=str(sheet_path.relative_to(ROOT)),
             gt_prefix_codec_mad_0=prefix_mad_0,gt_prefix_codec_mad_128=prefix_mad_128,
             before=row0,after=row1,
             protocol_note="matched source/actions/seed/NFE; self-history trajectories differ by checkpoint"))
        print("PAIRED",stem,flush=True)
    write(HERE/"FINAL_VIDEO_AUDIT.json",dict(status="PASS_6_PAIRED_PLAYABLE",
          gallery=records,original_video_count=12,paired_video_count=6))


def read_gallery(path):
    with av.open(str(path)) as c:
        stream=c.streams.video[0]
        fps=float(stream.average_rate)
        count=0
        for frame in c.decode(video=0):
            assert frame.width==3*832 and frame.height==540
            count+=1
    assert count==56 and abs(fps-24)<.001
    return count,fps


if __name__=="__main__":main()
