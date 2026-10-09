"""CPU-only matched-codec metrics and honest partial/full step4 review assets."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import av
import numpy as np
from PIL import Image,ImageDraw,ImageFont

BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE/'runtime/code'))
from causal.evaluate_action_control import evaluate as flow_metrics


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(path):
    with av.open(str(path)) as c:
        s=c.streams.video[0];frames=[f.to_image().convert('RGB') for f in c.decode(video=0)]
        assert s.codec_context.name=='h264' and s.codec_context.pix_fmt=='yuv420p' and float(s.average_rate)==24
    return frames


def main(args):
    out=BASE/'review_step4';out.mkdir(exist_ok=True)
    spec=json.loads((BASE/'evaluation_protocol.json').read_text());old=Path(spec['parking_root'])
    videos={};records={};running=[]
    for arm in ['fm_only','fm_action']:
        for suite in ['parking_D','parking_A','gt']:
            path=BASE/f'eval_{arm}_{suite}/evaluation.json'
            if not path.exists():running.append(dict(arm=arm,suite=suite,status='not_started'));continue
            r=json.loads(path.read_text())
            if r['status']!='complete_pending_visual_review':running.append(dict(arm=arm,suite=suite,status=r['status']))
            for row in r['records']:
                assert sha(row['current_video'])==row['video_sha256']
                name=row['name'];videos.setdefault(name,{})['zero']=Path(row['zero_video'])
                videos[name][arm]=Path(row['current_video']);records[(name,arm)]=row
    if running and not args.partial:raise ValueError('Preselected evaluations are incomplete')
    prep=json.loads((BASE/'preparation.json').read_text());gt={}
    for row in prep['clips']:
        if row['split']=='validation':gt[row['clip_id']]=row
    metrics=[];review_assets=[];font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
    for name,paths in sorted(videos.items()):
        park=name.startswith('parking_');start=39 if park else 81
        if park:
            ref=name.split('_')[1][-1]
            prior=Image.open(old/f'window1_{ref}/history_last_visible.png').convert('RGB')
            definition='MP4 first current vs identical lossless decoded reference history lastRGB38'
        else:
            with np.load(gt[name]['path'],allow_pickle=False) as d:prior=Image.fromarray(d['rgb'][80])
            definition='MP4 first current vs identical raw observed GT RGB80'
        previous=np.asarray(prior.convert('L'),dtype=np.float32);decoded={}
        for role,path in paths.items():
            frames=read(path);decoded[role]=frames
            gray=np.stack([np.asarray(f.convert('L'),dtype=np.float32) for f in frames])
            flow=flow_metrics(path) if park else None
            metrics.append(dict(case=name,role=role,frames=len(frames),video=str(path),sha256=sha(path),
                frame_MAD_MP4=float(np.abs(np.diff(gray,axis=0)).mean()),boundary_MAD_MP4=float(np.abs(gray[0]-previous).mean()),
                boundary_definition=definition,horizontal_flow_mean=None if flow is None else flow['horizontal_flow_px']['mean'],
                flow=flow,sampling_seconds=records[(name,role)]['sampling_seconds'] if role!='zero' else None))
        if set(decoded)=={'zero','fm_only','fm_action'}:
            assert len({len(v) for v in decoded.values()})==1
            roles=['zero','fm_only','fm_action']
            # Full RGB positions are deliberately the same across methods.
            chosen=[0,20,23,26,29,41] if park else [0,7,15,23,31,38]
            sheet=Image.new('RGB',(3*420,6*450),'white');draw=ImageDraw.Draw(sheet)
            for col,role in enumerate(roles):
                for row,k in enumerate(chosen):
                    frame=decoded[role][k];x=col*420;y=row*450
                    sheet.paste(frame.crop((170,60,590,480)),(x,y+30))
                    draw.text((x+5,y+5),f'{role} RGB{start+k}',font=font,fill='black')
            target=out/f'{name}_details.jpg';sheet.save(target,quality=95);review_assets.append(str(target))
            boundary=Image.new('RGB',(4*416,270),'white');draw=ImageDraw.Draw(boundary)
            for col,(label,frame) in enumerate([('Known history',prior)]+[(role,decoded[role][0]) for role in roles]):
                boundary.paste(frame.resize((416,240)),(col*416,30));draw.text((col*416+5,5),label,font=font,fill='black')
            target=out/f'{name}_boundary.jpg';boundary.save(target,quality=94);review_assets.append(str(target))
    result=dict(at=datetime.now().astimezone().isoformat(),status='partial' if running else 'all_videos_complete_pending_visual_gate',
                metrics=metrics,pending=running,review_assets=review_assets,action_gate_accepted=False,
                metric_policy='All0/4 frame and boundaryMAD recomputed from H264 decoded MP4 at full832x480. Farneback same416x240 center80% method. No quality score invented.')
    (out/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# E2 step4 objective comparison\n\n',f"Status: **{result['status']}**. Visual and action acceptance are not automatic.\n\n",'| Case | Method | Flow | MP4 frameMAD | MP4 boundaryMAD |\n|---|---|---:|---:|---:|\n']
    for r in metrics:
        flow='—' if r['horizontal_flow_mean'] is None else f"{r['horizontal_flow_mean']:+.6f}"
        lines.append(f"| {r['case']} | {r['role']} | {flow} | {r['frame_MAD_MP4']:.4f} | {r['boundary_MAD_MP4']:.4f} |\n")
    lines.append('\nRaw transition-model receipts use pre-encode frames; this table recomputes all comparisons from the same codec to avoid mixing definitions. Parking reference histories are Original-generated, not realGT; natural joint-control cases are realGT but have no pure-lateral flow sign gate.\n')
    if running:lines.append('\nPending: '+', '.join(x['arm']+'/'+x['suite'] for x in running)+'. No final efficacy conclusion.\n')
    (out/'METRICS.md').write_text(''.join(lines))
    print(json.dumps(dict(status=result['status'],rows=len(metrics),assets=len(review_assets),pending=running)))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--partial',action='store_true');main(ap.parse_args())
