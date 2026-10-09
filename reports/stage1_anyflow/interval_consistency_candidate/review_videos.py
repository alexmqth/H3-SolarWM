"""Review complete fork evaluations; retain all39 frames and original timings.

Can make contact sheets while the queue runs. Comparison grids require both
A/D at the selected NFE for every included variant. No visual PASS is inferred.
"""
import argparse
import hashlib
import json
from pathlib import Path

import av
import numpy as np
from PIL import Image,ImageDraw,ImageFont

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
SOURCE=ROOT/'outputs/2026-10-08-13/stage1_parallel_resume68_to128'
FONT=ImageFont.load_default(size=15)


def read_clip(path):
    with av.open(path) as container:
        stream=container.streams.video[0]
        if (stream.width,stream.height,stream.average_rate)!=(832,480,24):
            raise ValueError(f'Unexpected source format: {path}')
        frames=[(frame.time,frame.to_image()) for frame in container.decode(video=0)]
    if len(frames)!=39 or any(abs(t-i/24)>1e-6 for i,(t,_) in enumerate(frames)):
        raise ValueError(f'Incomplete footage/timestamps: {path}')
    return [frame for _,frame in frames]


def source(action,nfe,variant):
    if variant=='original':
        directory=ROOT/f'outputs/2026-10-02-03/action_{action}_teacher_39'
        mode,label,noisy,commits='baseline','Original H3 | 30 full-sequence steps',30,0
    elif variant=='initial128':
        directory=SOURCE/f'eval/anyflow/step_128/generated_{nfe}step_native/{action}'
        mode,label,noisy,commits='cached',f'AnyFlow128 | {nfe} steps/chunk',3*nfe,3
    else:
        directory=OUT/variant/f'eval/{nfe}step/{action}'
        if not (directory/'evaluation.json').exists():
            raise FileNotFoundError(f'Completed evaluation required: {directory}')
        mode='cached';label=f'136 {variant} | {nfe} steps/chunk';noisy,commits=3*nfe,3
    runtime=json.loads((directory/f'{mode}.json').read_text())
    if (runtime['status']!='complete' or runtime['denoiser_forwards']!=noisy
            or runtime.get('commit_forwards',0)!=commits):
        raise ValueError('Incomplete run or incorrect forward counts')
    if variant!='original' and runtime['sampler']!='anyflow_finite_map':
        raise ValueError('Only normal finite-map sampling belongs in this comparison')
    path=directory/f'{mode}.mp4'
    item=dict(action=action,variant=variant,label=label,path=str(path),
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),noisy_forwards=noisy,
        clean_commits=commits,seconds=runtime['wall_including_shared_setup_seconds'])
    return item,read_clip(path)


def contacts(report):
    receipts=[]
    for variant in ('control','auxiliary'):
        for file in sorted((OUT/variant/'eval').rglob('evaluation.json')):
            evaluation=json.loads(file.read_text())
            action,nfe=evaluation['action'],evaluation['steps_per_chunk']
            item,frames=source(action,nfe,variant)
            name=f'{variant}136_{action}_{nfe}step'
            canvas=Image.new('RGB',(1664,1810),'black');draw=ImageDraw.Draw(canvas)
            draw.text((6,6),f"{item['label']} | {action} | complete 39-frame diagnostic | not accepted",font=FONT,fill='white')
            for i,frame in enumerate(frames):
                x,y=i%5*332,34+i//5*220
                draw.text((x+4,y),f'frame {i:02d} | {i/24:.3f}s',font=FONT,fill='white')
                canvas.paste(frame.resize((332,192)),(x,y+24))
            canvas.save(report/f'{name}_all39.jpg',quality=95)
            clips=[source(action,nfe,v) for v in ('original','initial128',variant)]
            canvas=Image.new('RGB',(1664,288*len(clips)),'black');draw=ImageDraw.Draw(canvas)
            for row,(info,clip) in enumerate(clips):
                for col,index in enumerate((12,24,30,38)):
                    x,y=col*416,row*288
                    draw.text((x+4,y+4),f"{action} | {info['label']}",font=FONT,fill='white')
                    draw.text((x+4,y+25),f'frame {index}',font=FONT,fill='white')
                    canvas.paste(clip[index].resize((416,240)),(x,y+48))
            canvas.save(report/f'{name}_comparison.jpg',quality=95)
            receipts.append(item)
    (report/'review_inputs.json').write_text(json.dumps(receipts,indent=2)+'\n')
    return receipts


def grid(report,nfe,variants):
    columns=['original','initial128',*variants];entries=[];clips=[]
    for action in ('A','D'):
        for variant in columns:
            item,clip=source(action,nfe,variant);entries.append(item);clips.append(clip)
    width,height=416*len(columns),674
    output=report/f'original128_{"_".join(variants)}136_{nfe}step_AD.mp4'
    temporary=output.with_suffix('.tmp.mp4')
    preview=None
    with av.open(temporary,'w',options={'movflags':'+faststart'}) as container:
        stream=container.add_stream('libx264',rate=24)
        stream.width,stream.height,stream.pix_fmt=width,height,'yuv420p'
        stream.options={'crf':'18','preset':'medium','threads':'4'}
        for index in range(39):
            canvas=Image.new('RGB',(width,height),'black');draw=ImageDraw.Draw(canvas)
            draw.text((8,8),'Stage1 objective diagnostic | same image/action/seed/noise | complete footage',font=FONT,fill='#ffcb66')
            draw.text((8,28),'128 initializer vs eight-update forks | single-run timings; no speedup or quality claim',font=FONT,fill='white')
            for j,item in enumerate(entries):
                x,y=j%len(columns)*416,56+j//len(columns)*294
                draw.text((x+6,y),f"{item['action']} | {item['label']}",font=FONT,fill='white')
                draw.text((x+6,y+20),f"{item['seconds']:.1f}s | noisy {item['noisy_forwards']} + clean {item['clean_commits']}",font=FONT,fill='white')
                canvas.paste(clips[j][index].resize((416,240)),(x,y+46))
            draw.text((8,650),f'Frame {index:02d}/38 | {index/24:.3f}s | 39 frames / 24fps | sources 832x480',font=FONT,fill='white')
            if index==30:preview=canvas.copy()
            for packet in stream.encode(av.VideoFrame.from_ndarray(np.asarray(canvas),format='rgb24')):container.mux(packet)
        for packet in stream.encode():container.mux(packet)
    with av.open(temporary) as container:
        stream=container.streams.video[0]
        assert stream.codec_context.name=='h264' and stream.pix_fmt=='yuv420p'
        decoded=list(container.decode(video=0));assert len(decoded)==39
        assert all(abs(frame.time-i/24)<1e-6 for i,frame in enumerate(decoded))
    temporary.replace(output);preview.save(output.with_suffix('.jpg'),quality=95)
    output.with_suffix('.json').write_text(json.dumps(dict(inputs=entries,frames=39,fps=24,
        output_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),acceptance='NOT_REVIEWED'),indent=2)+'\n')
    print(output)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--steps-per-chunk',type=int,choices=(4,8),default=8)
    ap.add_argument('--variants',nargs='+',choices=('control','auxiliary'),default=['control','auxiliary'])
    ap.add_argument('--contacts-only',action='store_true')
    args=ap.parse_args();report=OUT/'report';report.mkdir(exist_ok=True)
    receipts=contacts(report)
    print(f'Contact sheets for {len(receipts)} completed evaluations')
    if not args.contacts_only:grid(report,args.steps_per_chunk,args.variants)
