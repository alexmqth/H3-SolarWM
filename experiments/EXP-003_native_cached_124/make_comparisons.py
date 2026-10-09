"""CPU-only real-video comparisons for EXP-003; never runs H3 inference."""
from pathlib import Path
import hashlib
import json

import av
from PIL import Image, ImageDraw, ImageFont

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
OUT=ROOT/"H3-World/outputs/EXP-003_native_cached_124"
ORIGINAL=ROOT/"H3-World/outputs/2026-10-01-21/action_A_teacher_latents/baseline.mp4"
V2B=ROOT/"submission/experiments/EXP-001_v2b_124/artifacts/videos/AA_rollout_124.mp4"
AD73=ROOT/"submission/experiments/EXP-002_native_cached/artifacts/videos/V2b_vs_cached_AD_73.mp4"


def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(4<<20),b""):h.update(b)
    return h.hexdigest()


def read(path):
    with av.open(str(path)) as c:
        s=c.streams.video[0]
        assert (s.width,s.height)==(832,480) and str(s.average_rate)=="24"
        return [f.to_image().convert("RGB") for f in c.decode(video=0)]


def make(left,right,target,left_title,right_title,left_timing,right_timing):
    assert not target.exists(),f"refuse overwrite: {target}"
    a,b=read(left),read(right)
    assert len(a)==len(b)==124
    font=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",17)
    small=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",14)
    with av.open(str(target),"w") as c:
        stream=c.add_stream("libx264",rate=24,options={"crf":"20","preset":"fast"})
        stream.width=1664;stream.height=570;stream.pix_fmt="yuv420p"
        for i,(x,y) in enumerate(zip(a,b)):
            im=Image.new("RGB",(1664,570),"#17202a")
            im.paste(x,(0,57));im.paste(y,(832,57))
            draw=ImageDraw.Draw(im)
            draw.text((15,10),left_title,font=font,fill="white")
            draw.text((847,10),right_title,font=font,fill="white")
            draw.text((15,37),left_timing,font=small,fill="#d2dce5")
            draw.text((847,37),right_timing,font=small,fill="#d2dce5")
            draw.text((15,543),f"A | RGB {i:03d}/123 | 24 fps | same I0 / seed 13 / noise",font=small,fill="white")
            draw.text((847,543),"cross-protocol capability comparison; not a single-factor ablation",font=small,fill="white")
            if i in (39,56,73,90,107):
                draw.line((832,57,832,537),fill="#fac858",width=4)
            frame=av.VideoFrame.from_image(im)
            for packet in stream.encode(frame):c.mux(packet)
        for packet in stream.encode():c.mux(packet)
    with av.open(str(target)) as c:
        s=c.streams.video[0]
        frames=[f.pts for f in c.decode(video=0)]
        assert len(frames)==124 and str(s.average_rate)=="24"
        assert all(x<y for x,y in zip(frames,frames[1:]))
    return dict(path=str(target),sha256=sha(target),left=str(left),left_sha256=sha(left),
                right=str(right),right_sha256=sha(right),fps=24,frames=124,
                left_timing=left_timing,right_timing=right_timing)


def main():
    candidate=OUT/"AA/rollout_124.mp4"
    assert candidate.exists() and ORIGINAL.exists() and V2B.exists()
    budget=json.loads((OUT/"budget.json").read_text())
    aa=next(x for x in budget["runs"] if x["path"]=="AA")
    candidate_wall=aa["end"]-aa["start"]
    original_meta=json.loads((ROOT/"H3-World/outputs/2026-10-01-21/action_A_teacher_latents/baseline.json").read_text())
    original_e2e=original_meta["wall_including_shared_setup_seconds"]
    v2b_metrics=json.loads((ROOT/"submission/experiments/EXP-001_v2b_124/metrics.json").read_text())
    # The published manifest is the source of the old three-process increment.
    v2b_comparison=json.loads((ROOT/"submission/experiments/EXP-001_v2b_124/artifacts/comparison_manifest.json").read_text())
    v2b_increment=v2b_comparison["videos"]["A"]["v2b_73_to_124_incremental_wall_seconds"]
    target=HERE/"artifacts/videos";target.mkdir(parents=True,exist_ok=True)
    result={"comparison_type":"cross-protocol capability, not single-factor KV speedup",
            "timing_scope":"candidate increment includes loading, cache IO, sampling, VAE, encoding and human review waits; not full from-zero E2E",
            "videos":{}}
    right=f"73->124f increment {candidate_wall:.1f}s incl review; earlier 73f reused"
    result["videos"]["Original_vs_candidate_AA_124"]=make(ORIGINAL,candidate,
        target/"Original_vs_candidate_AA_124.mp4",
        "Original H3 | A | 30 full-video steps",
        "EXP-003 strict causal raw KV | AA | 30 steps/chunk",
        f"full E2E {original_e2e:.1f}s",right)
    result["videos"]["V2b_vs_candidate_AA_124"]=make(V2B,candidate,
        target/"V2b_vs_candidate_AA_124.mp4",
        "V2b Same-sigma local bidir | AA | 30 steps/chunk",
        "EXP-003 strict causal raw KV | AA | 30 steps/chunk",
        f"73->124f increment {v2b_increment:.1f}s; prior 73f reused",right)
    assert AD73.exists()
    ad_target=target/"V2b_vs_candidate_AD_73.mp4"
    assert not ad_target.exists()
    ad_target.write_bytes(AD73.read_bytes())
    result["videos"]["V2b_vs_candidate_AD_73"]={
        "path":str(ad_target),"sha256":sha(ad_target),"source":str(AD73),
        "source_sha256":sha(AD73),"frames":73,"fps":24,
        "note":"copied accepted EXP-002 common-range comparison; Original A->D matched source absent"}
    (HERE/"artifacts/comparison_manifest.json").write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps(result,indent=2,ensure_ascii=False))


if __name__=="__main__":main()
