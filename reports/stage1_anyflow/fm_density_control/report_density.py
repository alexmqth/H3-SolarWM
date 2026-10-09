"""Matched, complete density-control reports; never infer a visual PASS."""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from statistics import mean
import sys

BASE=Path(__file__).resolve().parent
CONTROL=BASE.parents[1]/'2026-10-08-18/stage1_real_abot_fm'
NOISE=BASE.parents[1]/'2026-10-08-22/stage1_fm_noise_audit'
CLIPS=('118eb5d8b75e1b8ac23a4e9ae77af9a9_A_1140','dfec8ed3237860eba14d67c089ecd041_D_1750')


def require(value,message):
    if not value:raise ValueError(message)


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def key(row):return row['clip'],row['chunk'],row['sigma']


def completed(path):
    data=read(path);require(data['status']=='complete',f'Incomplete receipt: {path}');return data


def emit(dest,report,lines):
    dest.mkdir(parents=True,exist_ok=False)
    (dest/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    rows=report.get('rows',[])
    if rows:
        with (dest/'metrics.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=sorted(set().union(*(r.keys() for r in rows))))
            writer.writeheader();writer.writerows(rows)
    (dest/'README.md').write_text('\n'.join(lines)+'\n')


def noise_rows(probes):
    require(len(probes)==3,'Need FM0 and both FM48 policies')
    indexed=[]
    for data in probes:
        require(data['status']=='complete' and len(data['records'])==54,'Need complete54 states')
        require(all(c['read_only'] for c in data['cache_audits']),'Cache mutation')
        table={key(r):r for r in data['records']};require(len(table)==54,'Duplicate noise state');indexed.append(table)
    require(all(set(t)==set(indexed[0]) for t in indexed),'Noise state keys differ')
    fields=('source_sha256','noise_sha256','state_sha256','history_sha256','clean_target_sha256',
            'prompt_sha256','anchor_sha256','audio_sha256','gaussian_weight','native8')
    rows=[]
    for state in sorted(indexed[0]):
        triplet=[t[state] for t in indexed];ref=triplet[0]
        for row in triplet[1:]:
            require(all(row[k]==ref[k] for k in fields),f'Noise inputs differ: {state}')
            require(row['roles']['original']==ref['roles']['original'],f'Original full output changed: {state}')
        result=dict(clip=state[0],chunk=state[1],sigma=state[2],native8=ref['native8'])
        roles=[('original',ref['roles']['original'])]+[(name,row['roles']['causal']) for name,row in zip(('fm0','shift12','shift2_22'),triplet)]
        for name,values in roles:
            for metric in ('raw_velocity_mse','clean_endpoint_mse','normalized_velocity_mse'):
                result[f'{name}_{metric}']=values[metric]
        rows.append(result)
    return rows


def report_noise(dest):
    paths=[NOISE/'step_00/probe.json',NOISE/'step_48/probe.json',BASE/'noise/step_48/probe.json']
    rows=noise_rows([completed(p) for p in paths]);groups={}
    for sigma in sorted({r['sigma'] for r in rows},reverse=True):
        selected=[r for r in rows if r['sigma']==sigma]
        groups[str(sigma)]={k:mean(r[k] for r in selected) for k in rows[0] if k.endswith('_mse')}
    lines=['# 固定weight函数，只改sampling density：54点噪声对照','',
        '同一真实GT状态/自然联合动作，Original完整输出hash逐点一致。所有causal checkpoint各自重建只读GT KV。','',
        '| sigma | Original | FM0 | FM48 shift12 | FM48 shift2.22 |','|---:|---:|---:|---:|---:|']
    for sigma,g in groups.items():
        lines.append('| '+f'{float(sigma):.6f}'+' | '+' | '.join(f"{g[n+'_raw_velocity_mse']:.6f}" for n in ('original','fm0','shift12','shift2_22'))+' |')
    lines+=['','表中为velocity MSE，固定noise−GT目标；每sigma为6状态等权均值。单次endpoint MSE也保存，但它等于sigma²×velocity MSE，不是完整采样终点或独立画质证据。',
        '这不是A/D干预，也不能由MSE判断动作或视觉PASS。[逐状态CSV](metrics.csv) · [完整记录](metrics.json)']
    emit(dest,dict(status='complete_matched_statistics',rows=rows,by_sigma=groups,
        receipts={str(p):sha(p) for p in paths},teacher_full_outputs_identical=True,
        visual_acceptance='not measured',new_optimizer_steps=0),lines)


def validate_geometry_source(before,after):
    audit=read(BASE/'geometry_source_adaptation.json')
    require(before['source_sha256']['probe_real_geometry.py']==audit['reference_sha256'],'Unexpected old probe source')
    require(after['source_sha256']['probe_real_geometry.py']==audit['candidate_sha256'],'Unexpected candidate source')
    require(sha(BASE/'probe_real_geometry.py')==audit['candidate_sha256'],'Candidate script changed')
    source=(BASE/'probe_real_geometry.py').read_text()
    for original,replacement in reversed(list(audit['replacements'].items())):
        require(source.count(replacement)==1,'Path adaptation no longer reversible');source=source.replace(replacement,original)
    require(source==(CONTROL/'probe_real_geometry.py').read_text(),'Changes beyond audited paths/provenance')
    require(sha(CONTROL/'probe_real_geometry.py')==audit['reference_sha256'],'Frozen original script changed')
    for field in ('geometry_primitives.py','prepare_counterfactual.py'):
        require(before['source_sha256'][field]==after['source_sha256'][field]==sha(CONTROL/field),'Condition/model helper changed')


def geometry_rows(before,after):
    sys.path.insert(0,str(CONTROL))
    from compare_geometry import validate
    old,new=validate(before),validate(after)
    require(set(old)==set(new),'Geometry state keys differ')
    for field in ('history_source','state_construction','teacher_backend'):
        require(before[field]==after[field],f'Geometry protocol differs: {field}')
    validate_geometry_source(before,after)
    fields=('state_sha256','history_sha256','pair_sha256','source_sha256','endpoint_source','endpoint_sha256')
    rows=[]
    for state in sorted(old):
        x,y=old[state],new[state]
        require(all(x[k]==y[k] for k in fields),f'Geometry inputs differ: {state}')
        pairs=[(x['absolute'][a],y['absolute'][a]) for a in 'AD']+[(x['action_delta'],y['action_delta'])]
        pairs+=list(zip(x['action_delta']['per_latent_frame'],y['action_delta']['per_latent_frame']))
        for a,b in pairs:
            require(all(math.isclose(a[k],b[k],rel_tol=1e-9,abs_tol=1e-12) for k in ('teacher_rms','teacher_norm')),
                f'Teacher scalar changed: {state}')
        row=dict(clip=state[0],chunk=state[1],sigma=state[2])
        for name,r in [('shift12',x),('shift2_22',y)]:
            row[f'{name}_whole_cosine']=mean(r['absolute'][a]['cosine'] for a in 'AD')
            for metric in ('cosine','norm_ratio','relative_delta_error'):
                row[f'{name}_delta_{metric}']=r['action_delta'][metric]
        rows.append(row)
    return rows


def report_geometry(dest,history):
    paths=[base/f'geometry/step_48_{history}/probe.json' for base in (CONTROL,BASE)]
    rows=geometry_rows(*(completed(p) for p in paths))
    groups={}
    for chunk in (None,0,1,2):
        selected=[r for r in rows if chunk is None or r['chunk']==chunk]
        groups['all' if chunk is None else str(chunk)]={k:mean(r[k] for r in selected) for k in rows[0] if k not in ('clip','chunk','sigma')}
    lines=['# 相同48更新，采样density对动作差分的影响','',f'固定状态来源：{history}。18点当前chunk A/D反事实；各模型独立构建历史KV，A/D内部只读。','',
        '| chunk | whole cos shift12 / 2.22 | A/D delta cos shift12 / 2.22 | delta norm ratio shift12 / 2.22 |','|---|---:|---:|---:|']
    for group,g in groups.items():
        cells=[' / '.join(f'{g[p+"_"+m]:.6f}' for p in ('shift12','shift2_22')) for m in ('whole_cosine','delta_cosine','delta_norm_ratio')]
        lines.append('| '+group+' | '+' | '.join(cells)+' |')
    lines+=['','模型计算源相同，诊断入口只作经逐字反向核验的输出/checkpoint路径与来源记录变动。逐state/action/endpoint哈希和teacher范数一致；旧收据没有完整anchor或teacher输出hash，不能把标量核验称为完整tensor逐bit证明。',
        'Original仍双向重算历史，student使用已commit的causal KV；插值状态不是实际solver中间状态。不能由cosine单独判定视频方向或画质。','',
        '[逐状态CSV](metrics.csv) · [完整记录](metrics.json)']
    emit(dest,dict(status='complete_matched_statistics',history=history,rows=rows,by_chunk=groups,
        receipts={str(p):sha(p) for p in paths},source_adaptation_audited=True,
        visual_acceptance='pending videos/manual review',teacher_full_tensor_hashes_available=False),lines)


def video_helpers():
    sys.path.insert(0,str(CONTROL))
    import report_real
    from PIL import Image,ImageDraw,ImageFont
    return report_real,Image,ImageDraw,ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)


def assemble(dest,collections):
    """Render only complete39-frame groups; return artifact names for review."""
    helper,Image,Draw,font=video_helpers();artifacts=[]
    for tag,entries in collections.items():
        frames=[helper.load_frames(Path(row['video'])) for row in entries]
        for j,(row,fs) in enumerate(zip(entries,frames)):
            sheet=Image.new('RGB',(7*208,6*138),'white');draw=Draw.Draw(sheet)
            for i,im in enumerate(fs):
                x=i%7*208;y=i//7*138;sheet.paste(im.resize((208,120)),(x,y+18));draw.text((x+3,y+2),f'frame{i}',fill='black')
            sheet.save(dest/f'{tag}_{j}_all39.jpg')
        assembled=[]
        for i in range(39):
            canvas=Image.new('RGB',(416*len(entries),302),'white');draw=Draw.Draw(canvas)
            for j,row in enumerate(entries):
                x=j*416;draw.text((x+4,3),row['method'],fill='black',font=font)
                timing='Recorded GT' if row['seconds'] is None else f"{row['noisy_forwards']} noisy + {row['clean_commits']} commits | {row['seconds']:.1f}s recorded"
                draw.text((x+4,23),timing,fill='black',font=font)
                draw.text((x+4,43),f'frame{i}/38 | matched initial/actions/noise',fill='black',font=font)
                canvas.paste(frames[j][i].resize((416,240)),(x,62))
            assembled.append(canvas)
        video=dest/f'{tag}_comparison.mp4';helper.write_video(assembled,video)
        require(helper.video_metrics(video)['frames']==39,'Output video is incomplete')
        artifacts.append(video.name)
        sheet=Image.new('RGB',(6*277,len(entries)*180),'white');draw=Draw.Draw(sheet)
        for j,row in enumerate(entries):
            for col,i in enumerate((0,8,16,24,30,38)):
                x=col*277;y=j*180;sheet.paste(frames[j][i].resize((277,160)),(x,y+20));draw.text((x+3,y+3),f"{row['method']} f{i}",fill='black')
        sheet.save(dest/f'{tag}_matched.jpg')
    return artifacts


def report_natural(dest,history,steps):
    helper,_,_,_=video_helpers();rows=[];collections={}
    for clip in CLIPS:
        original_path=CONTROL/'eval/original/generated_30'/clip
        original=completed(original_path/'evaluation.json')
        entries=[('Real ABot GT',Path(original['source']['video']),None),('Original30 full-seq',original_path/'video.mp4',original)]
        for label,root in [('shift12',CONTROL),('shift2.22',BASE)]:
            path=root/f'eval/step_48/{history}_{steps}'/clip;r=completed(path/'evaluation.json')
            require(r['optimizer_step']==48 and r['steps']==steps and r['history_source']==history,'Wrong natural video protocol')
            require(r['encoded_sha256']==original['encoded_sha256'] and r['input_fingerprints']==original['input_fingerprints'],'Video inputs differ')
            entries.append((f'FM48 {label} {steps}/chunk',path/'video.mp4',r))
        group=[]
        for label,path,r in entries:
            metrics=helper.video_metrics(path);require(metrics['frames']==39,'Incomplete natural video')
            row=dict(clip=clip,method=label,video=str(path),history=history,
                seconds=r['inference_after_load_seconds'] if r else None,
                end_to_end_seconds=r['end_to_end_seconds'] if r else None,
                timing_scope='after load; conditioning was cached' if r else 'recorded GT',
                noisy_forwards=r['denoiser_forwards'] if r else None,clean_commits=r['commit_forwards'] if r else None,
                gpu_peak_MiB=r['GPU_peak_MiB'] if r else None,cpu_kv_MiB=r['cpu_kv_peak_MiB'] if r else None,
                first_chunk_seconds=r['chunk_seconds'][0] if r and r['chunk_seconds'] else None,
                chunk_seconds_json=json.dumps(r['chunk_seconds']) if r else None,
                frame_gray_MAD=metrics['gray_pixel_difference_mean'],boundary_rgb_MAD=helper.rgb_boundary(path,[17,34])['mean'],
                horizontal_flow=helper.flow_metrics(path)['horizontal_flow_px']['mean'])
            group.append(row);rows.append(row)
        collections[clip]=group
    report_video(dest,rows,collections,scope=f'natural {history} {steps}/chunk; joint keys/camera, not pure A/D',
        note='两自然场景来自held-out真实ABot。GT-history为oracle历史，不能当free rollout；自然片段之间flow差不能当A/D控制恢复。')


def report_parking(dest,steps):
    sys.path.insert(0,str(CONTROL/'runtime/code/causal'))
    from report_stage1_anyflow import audit_inputs,summarize_video
    teacher=CONTROL.parents[2]/'outputs/2026-10-02-03';rows=[];collections={}
    for action in 'AD':
        original=teacher/f'action_{action}_teacher_39'
        reference=summarize_video(original,'baseline',label='Original30 full-seq',action=action,history='full-sequence',step=None,nfe=30)
        require(reference is not None,'Missing Original video');entries=[('Original30 full-seq',reference)]
        for label,root in [('shift12',CONTROL),('shift2.22',BASE)]:
            path=root/f'parking/step_48/{steps}step'/action;r=read(path/'evaluation.json')
            require(r['optimizer_step']==48 and r['steps_per_chunk']==steps and r['frames']==39,'Wrong parking protocol')
            require(audit_inputs(path,original)['exactly_equal'],'Parking input mismatch')
            entries.append((f'FM48 {label} {steps}/chunk',r))
        group=[]
        for label,r in entries:
            row=dict(action=action,method=label,video=r['path'],seconds=r['seconds'],
                timing_scope='benchmark wall including shared setup; see original runtime',
                noisy_forwards=r['noisy_forwards'],clean_commits=r['clean_commits'],gpu_peak_MiB=r['gpu_peak_MiB'],cpu_kv_MiB=r['cpu_kv_MiB'],
                frame_gray_MAD=r['frame_gray_mad'],boundary_rgb_MAD=r['boundary_rgb_mad'],horizontal_flow=r['horizontal_flow'])
            group.append(row);rows.append(row)
        collections[action]=group
    responses=[]
    for a,d in zip(collections['A'],collections['D']):
        separation=a['horizontal_flow']-d['horizontal_flow']
        responses.append(dict(method=a['method'],A=a['horizontal_flow'],D=d['horizontal_flow'],separation=separation,
            numeric_gate_only=a['horizontal_flow']>0 and d['horizontal_flow']<0 and separation>1))
    report_video(dest,rows,collections,scope=f'parking current A/D, {steps}/chunk',
        note='Original是旧legacy精度/原始条件，两个FM48采用同h3_fp32 causal/RGB协议；与Original为pipeline参照，两种density才是受控训练比较。',responses=responses)


def report_video(dest,rows,collections,*,scope,note,responses=None):
    # Completion is written only after every source and rendered MP4 decodes.
    dest.mkdir(parents=True,exist_ok=False)
    artifacts=assemble(dest,collections)
    report=dict(status='complete_artifacts',scope=scope,rows=rows,artifacts=artifacts,
        action_responses=responses,visual_review='pending full39-frame manual inspection',
        timing='single shared-host runs, recorded scope differs for natural/parking, no fair speedup claim',
        caveat=note,stage1_accepted=False)
    (dest/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    with (dest/'metrics.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=sorted(set().union(*(r.keys() for r in rows))));w.writeheader();w.writerows(rows)
    lines=['# FM48固定权重的density对照：完整39帧','',scope,'',note,'',
        '时间为各入口记录的单次共享主机耗时，不能跨自然/停车场入口混用或宣称公平加速。GPU peak为入口记录的allocated峰值；没有独立分解weights/activations，CPU KV另列，不能将offload显存差归因为算法收益。MAD是帧差/活动量，不是画质；需检查全部39帧，尤其18–38帧重影和人物结构。','']
    lines+=['- ['+name+']('+name+')' for name in artifacts]
    if responses:
        lines+=['','| 方法 | A flow | D flow | A−D | 仅数值门槛 |','|---|---:|---:|---:|---|']
        for r in responses:lines.append(f"| {r['method']} | {r['A']:+.6f} | {r['D']:+.6f} | {r['separation']:.6f} | {r['numeric_gate_only']} |")
    lines+=['','[逐条指标](metrics.csv) · [完整记录](metrics.json)','',
        '**自动产出不代表画质PASS；本报告不会替换meeting视频。**']
    (dest/'README.md').write_text('\n'.join(lines)+'\n')


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--kind',choices=['noise','geometry','natural','parking'],required=True)
    ap.add_argument('--history',choices=['gt','generated','step00_generated']);ap.add_argument('--steps',type=int,choices=[8,30]);args=ap.parse_args()
    if args.kind=='geometry' and args.history not in ('gt','step00_generated'):ap.error('Geometry needs fixed GT or step00-generated history')
    if args.kind=='natural' and (args.history not in ('gt','generated') or args.steps is None):ap.error('Natural needs history and steps')
    if args.kind=='parking' and args.steps is None:ap.error('Parking needs steps')
    label='_'.join(str(x) for x in (args.kind,args.history,args.steps) if x is not None)
    dest=BASE/'report'/label
    if args.kind=='noise':report_noise(dest)
    elif args.kind=='geometry':report_geometry(dest,args.history)
    elif args.kind=='natural':report_natural(dest,args.history,args.steps)
    else:report_parking(dest,args.steps)
    print(json.dumps(dict(status='complete',output=str(dest))))


if __name__=='__main__':main()
