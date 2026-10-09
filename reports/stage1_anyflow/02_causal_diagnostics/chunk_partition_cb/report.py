"""CPU aggregate and fully decode completed C/B outputs; never starts training."""
import csv
import hashlib
import json
from pathlib import Path
import sys

BASE=Path(__file__).resolve().parent
sys.path[:0]=[str(BASE/'runtime/code/causal')]


def main():
    import av
    from PIL import Image,ImageDraw,ImageFont
    from benchmark import write_video
    from evaluate_action_control import evaluate
    def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    def frames(p):
        with av.open(str(p)) as c:return [Image.fromarray(f.to_ndarray(format='rgb24')) for f in c.decode(video=0)]
    reports={case:json.loads((BASE/case/'evaluation.json').read_text()) for case in ('C_A','C_D','B_first')}
    assert all(d['status']=='complete_pending_visual_review' for d in reports.values())
    launch=json.loads((BASE/'launch.json').read_text())
    for name,h in launch['sources_sha256'].items():assert sha(BASE/name)==h,name
    metrics=[]
    for case,d in reports.items():
        assert d['history_and_noise_unchanged'] and d['frozen_parameter_versions_unchanged']
        for row in d['records']:
            metrics.append(dict(case=case,history=d['history_kind'],mode=row['mode'],action=row['action'],
                current_RGB=row['flow']['frames'],flow_mean_x=row['flow']['horizontal_flow_px']['mean'],
                current_MAD=row['frame_gray_MAD'],boundary_MAD=row.get('boundary_gray_MAD'),
                sample_seconds=row['sampling_seconds'],diagnostic_seconds=row['diagnostic_probe_seconds'],
                noisy_forwards=row['noisy_forwards'],GPU_peak_MiB=row['GPU_peak_MiB'],CPU_KV_MiB=d['CPU_KV_MiB'],
                history_latents_MiB=d['history_latents_MiB'],
                previous5_revision_MAD=row.get('previous5_RGB_revision_MAD_not_applied')))
    with (BASE/'metrics.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(metrics[0]));w.writeheader();w.writerows(metrics)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17)
    contexts={case:frames(BASE/case/'comparison_context.mp4') for case in ('C_A','C_D')}
    assert all(len(v)==25 for v in contexts.values())
    combined=[]
    for j in range(25):
        image=Image.new('RGB',(1664,560),'black');draw=ImageDraw.Draw(image)
        for ci,case in enumerate(('C_A','C_D')):
            image.paste(contexts[case][j].resize((832,520)),(ci*832,40))
            draw.text((ci*832+8,10),f'{case}: own {case[-1]} startup; left CLEAN / right NOISED; top A / bottom D',font=font,fill='white')
        combined.append(image)
    write_video(combined,BASE/'C_second_chunk_summary.mp4')
    # Common first22 RGB only: the C12 source can see all first39 RGB-time
    # actions/latents within its current window, so not equal latency/context.
    b={a:frames(BASE/'B_first'/f'clean_{a}_current.mp4') for a in 'AD'}
    c={a:frames(BASE/'source_coarse/coarse_A'/f'window0_{a}.mp4')[:22] for a in 'AD'}
    combined=[]
    for j in range(22):
        image=Image.new('RGB',(1664,1040),'black');draw=ImageDraw.Draw(image)
        for ai,a in enumerate('AD'):
            for ci,(clip,label) in enumerate(((b,'B: first7 / 22RGB'),(c,'C: first12 / 39RGB, display first22'))):
                image.paste(clip[a][j],(ci*832,ai*520+40))
                draw.text((ci*832+8,ai*520+10),f'{label} | action {a} | 30 steps | RGB{j}',font=font,fill='white')
        combined.append(image)
    write_video(combined,BASE/'B7_vs_C12_first22.mp4')
    startup_comparison={}
    for a in 'AD':
        path=BASE/f'C12_{a}_first22_reference.mp4';write_video(c[a],path)
        startup_comparison[a]=evaluate(path)
    videos=[]
    paths=sorted(set(BASE.glob('*/*.mp4')) | set(BASE.glob('*.mp4')))
    for p in paths:
        with av.open(str(p)) as container:
            stream=container.streams.video[0]
            count=sum(1 for _ in container.decode(video=0))
            row=dict(path=str(p.relative_to(BASE)),sha256=sha(p),frames=count,
                width=stream.width,height=stream.height,fps=float(stream.average_rate),
                codec=stream.codec_context.name,pixel_format=stream.codec_context.format.name)
            assert row['codec']=='h264' and row['pixel_format']=='yuv420p' and row['fps']==24
            videos.append(row)
    pairs=[]
    for case,d in reports.items():
        for mode in sorted(set(r['mode'] for r in d['records'])):
            a=next(r for r in metrics if r['case']==case and r['mode']==mode and r['action']=='A')
            z=next(r for r in metrics if r['case']==case and r['mode']==mode and r['action']=='D')
            pairs.append(dict(case=case,mode=mode,A=a['flow_mean_x'],D=z['flow_mean_x'],
                separation=a['flow_mean_x']-z['flow_mean_x'],
                direction_proxy_pass=a['flow_mean_x']>0 and z['flow_mean_x']<0))
    summary=dict(status='metrics_complete_visual_review_required',metrics=metrics,action_pairs=pairs,
        videos=videos,original_C_first22=startup_comparison,
        noisy_forwards=sum(d['denoiser_forwards'] for d in reports.values()),
        diagnostic_forwards=sum(d['diagnostic_forwards'] for d in reports.values()),
        sources={case:sha(BASE/case/'evaluation.json') for case in reports},
        caveats=['Optical flow is a proxy, no automated video quality acceptance.',
                 'C videos reuse already generated own first39 RGB; no GT or Original-history reset.',
                 'T2 recomputes history; CPU hiddenKV=0; not a persistent-KV performance experiment.',
                 'Different first-window widths see different amounts of current context.',
                 'Single seed/scene, not independent generalization or speedup evidence.'])
    (BASE/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(dict(pairs=pairs,videos=len(videos),noisy_forwards=summary['noisy_forwards']),indent=2))


if __name__=='__main__':main()
