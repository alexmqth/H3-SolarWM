"""Report completed real-video controls; never fill missing models as results."""
import argparse
import csv
import json
from pathlib import Path
import sys
import av
import numpy as np
from PIL import Image,ImageDraw,ImageFont

BASE=Path(__file__).resolve().parent
sys.path[:0]=[str(BASE/'runtime/code/causal')]
from evaluate_videos import evaluate as video_metrics
from evaluate_action_control import evaluate as flow_metrics
from summarize_action_experiment import rgb_boundary
from evaluation_queue import CLIPS

def load_frames(path):
    with av.open(str(path)) as c:frames=[x.to_image().convert('RGB') for x in c.decode(video=0)]
    assert len(frames)==39
    return frames

def write_video(frames,path):
    tmp=path.with_suffix('.tmp.mp4')
    with av.open(str(tmp),'w',options={'movflags':'+faststart'}) as c:
        s=c.add_stream('libx264',rate=24);s.width,s.height=frames[0].size;s.pix_fmt='yuv420p'
        s.options={'crf':'18','preset':'medium'}
        for im in frames:
            for p in s.encode(av.VideoFrame.from_ndarray(np.asarray(im),format='rgb24')):c.mux(p)
        for p in s.encode():c.mux(p)
    tmp.replace(path)

def main(args):
    dest=BASE/'report'/args.label;dest.mkdir(parents=True,exist_ok=False)
    rows=[];grids=[]
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
    for clip in CLIPS:
        orig=BASE/'eval/original/generated_30'/clip
        if not (orig/'evaluation.json').exists():continue
        original=json.loads((orig/'evaluation.json').read_text())
        if original['status']!='complete':continue
        entries=[('Real ABot GT',Path(original['source']['video']),None)]
        entries.append(('Original 30 full-seq',orig/'video.mp4',original))
        for step in (0,48):
            p=BASE/f'eval/step_{step:02d}/{args.history}_{args.steps}'/clip
            if not (p/'evaluation.json').exists():continue
            m=json.loads((p/'evaluation.json').read_text())
            if m['status']!='complete':continue
            assert m['encoded_sha256']==original['encoded_sha256'] and m['input_fingerprints']==original['input_fingerprints']
            entries.append((f'Causal FM{step} {args.steps}/chunk {args.history}',p/'video.mp4',m))
        clips=[]
        for name,video,m in entries:
            fs=load_frames(video);clips.append(fs)
            stats=video_metrics(video);flow=flow_metrics(video);boundary=rgb_boundary(video,[17,34])
            rows.append(dict(clip=clip,method=name,video=str(video),history=args.history if name.startswith('Causal') else None,
                steps=m['steps'] if m else None,denoiser_forwards=m['denoiser_forwards'] if m else None,
                clean_commits=m['commit_forwards'] if m else None,
                inference_after_load_seconds=m['inference_after_load_seconds'] if m else None,
                end_to_end_seconds=m['end_to_end_seconds'] if m else None,
                GPU_peak_MiB=m['GPU_peak_MiB'] if m else None,CPU_KV_peak_MiB=m['cpu_kv_peak_MiB'] if m else None,
                first_chunk_seconds=m['chunk_seconds'][0] if m and m['chunk_seconds'] else None,
                frame_MAD=stats['gray_pixel_difference_mean'],boundary_MAD=boundary['mean'],
                horizontal_flow_mean=flow['horizontal_flow_px']['mean'],
                all_frames_decode=True,input_fairness=True if m else None))
            sheet=Image.new('RGB',(7*208,6*138),'white');draw=ImageDraw.Draw(sheet)
            for i,im in enumerate(fs):
                x=i%7*208;y=i//7*138;sheet.paste(im.resize((208,120)),(x,y+18));draw.text((x+3,y+2),f'frame{i}',fill='black')
            sheet.save(dest/(clip+'_'+name.replace(' ','_').replace('/','_')+'_all39.jpg'))
        assembled=[]
        for frame in range(39):
            canvas=Image.new('RGB',(len(entries)*416,302),(250,250,250));draw=ImageDraw.Draw(canvas)
            for col,(name,_,m) in enumerate(entries):
                x=col*416;canvas.paste(clips[col][frame].resize((416,240)),(x,62))
                draw.text((x+5,3),name,fill='black',font=font)
                label='Recorded RGB; combined keys/camera' if m is None else f"{m['denoiser_forwards']} noisy forwards; {m['inference_after_load_seconds']:.1f}s after load"
                draw.text((x+5,23),label,fill='black',font=font)
                draw.text((x+5,43),f'frame {frame}/38 | same image, prompt/actions, noise',fill='black',font=font)
            assembled.append(canvas)
        path=dest/(clip+'_comparison.mp4');write_video(assembled,path)
        assert video_metrics(path)['frames']==39
        # Matched temporal selections for every available column, no missing
        # model placeholders and no promotion to the meeting deliverable.
        sheet=Image.new('RGB',(6*277,len(entries)*180),'white');draw=ImageDraw.Draw(sheet)
        for row,(name,_,_) in enumerate(entries):
            for col,frame in enumerate((0,8,16,24,30,38)):
                x=col*277;y=row*180;sheet.paste(clips[row][frame].resize((277,160)),(x,y+20))
                draw.text((x+3,y+3),f'{name} f{frame}',fill='black')
        sheet.save(dest/(clip+'_matched.jpg'))
        grids.append(dict(clip=clip,video=path.name,columns=[x[0] for x in entries]))
    report=dict(scope='held-out real combined actions; natural clips do not prove A/D counterfactual control',
        history=args.history,steps=args.steps,rows=rows,grids=grids,
        visual_review='pending manual full-frame review',timing='single shared-host cached-conditioning runs, no speedup claim')
    (dest/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    if rows:
        with (dest/'metrics.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    lines=['# 真实ABot验证片段：已完成结果','',f'History treatment: **{args.history}**. Causal **{args.steps} steps/chunk**. Original30为整段30次前向。','',
        '仅列出真实完成且输入一致的结果；缺失checkpoint不填充成模型输出。GT-history是oracle历史诊断，拼接的预测chunk不是自由rollout。两条真实动作含联合按键/镜头操作，片段间光流差不作为A/D控制恢复证据。','',
        '时间来自单次共享主机、预缓存conditioning，不能宣称加速；MAD表示运动/帧差，不是画质。自然户外场景的绝对MAD不能套用停车场阈值。视频画质还需人工检查全部39帧，当前报告生成器不自动判PASS。','']
    for grid in grids:lines+=['- ['+grid['clip']+']('+grid['video']+') — '+' / '.join(grid['columns'])]
    lines+=['','[逐条指标](metrics.csv) · [完整机器记录](metrics.json)','']
    (dest/'README.md').write_text('\n'.join(lines));print(json.dumps(dict(output=str(dest),rows=len(rows),grids=grids)))

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--label',required=True)
    ap.add_argument('--history',choices=['gt','generated'],required=True);ap.add_argument('--steps',choices=[8,30],type=int,default=30)
    main(ap.parse_args())
