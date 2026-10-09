"""Compare the two completed FM48 training receipts; no model calls."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean


def main(args):
    paths=[args.control,args.candidate]
    runs=[json.loads(p.read_text()) for p in paths]
    old,new=runs
    assert all(d['status']=='complete' and len(d['updates'])==48 for d in runs)
    assert old['validation_before']==new['validation_before']
    invariant=('objective','precision_profile','adapter_scope','bank_rank','bank_alpha',
        'history_gradient_mode','anchor_mode','chunk_frames','history_chunks',
        'action_prefix_mode','action_feedback','flow_shift','validation_timestep_shift',
        'logical_batch','lr','seed','validation_seed','real_manifest_sha256')
    assert all(old['config'][k]==new['config'][k] for k in invariant)
    assert old['config']['objective']=='fm'
    assert old['config']['training_timestep_shift']==12 and new['config']['training_timestep_shift']==2.22
    assert old['config'].get('training_weight_shift',12)==new['config']['training_weight_shift']==12
    rows=[]
    for i,before in enumerate(new['validation_before']):
        controls=[d['validation_after'][i] for d in runs]
        assert all((x['action'],x['chunk'])==(before['action'],before['chunk']) for x in controls)
        for j,initial in enumerate(before['samples']):
            pair=[x['samples'][j] for x in controls]
            assert all(all(s[k]==initial[k] for k in ('sigma','target_sigma','weight','sample_type')) for s in pair)
            assert all(math.isfinite(s['raw_loss']) for s in [initial,*pair])
            sigma=initial['sigma'];band='low' if sigma<=.2407808991 else ('mid' if sigma<=.6894410401 else 'high')
            rows.append(dict(clip=before['action'],chunk=before['chunk'],sigma=sigma,noise_band=band,
                initial_raw=initial['raw_loss'],shift12_raw=pair[0]['raw_loss'],shift2_22_raw=pair[1]['raw_loss'],
                candidate_minus_control=pair[1]['raw_loss']-pair[0]['raw_loss']))
    assert len(rows)==16
    summaries={}
    for band in ('low','mid','high','all'):
        group=[r for r in rows if band=='all' or r['noise_band']==band]
        if not group:
            summaries[band]=dict(count=0,measured=False);continue
        values={k:mean(r[k] for r in group) for k in ('initial_raw','shift12_raw','shift2_22_raw')}
        summaries[band]=dict(count=len(group),measured=True,**values,
            candidate_vs_control_percent=100*(values['shift2_22_raw']/values['shift12_raw']-1),
            improved_vs_initial=sum(r['shift2_22_raw']<r['initial_raw'] for r in group),
            improved_vs_control=sum(r['shift2_22_raw']<r['shift12_raw'] for r in group))
    runtime=[dict(method=label,visible_devices=d['visible_devices'],vram_reserve_gib=d['vram_reserve_gib'],
        wall_seconds=d['wall_seconds'],allocated_peak_MiB=d['gpu_allocated_peak_MiB'])
        for label,d in zip(('shift12','shift2.22'),runs)]
    output=dict(status='complete_receipt_comparison',sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        rows=rows,by_noise=summaries,runtime=runtime,
        scope='fixed four validation clips x four sigma values, all final chunk2 (two latent frames); training before/after raw FM residuals',
        limitation='No low-sigma validation samples. Actual old validation noise tensor hashes unavailable; matching source/seed/time policy and identical initial validation receipts, not a full tensor identity proof.',
        quality_gate='NOT_EVALUATED: no action/visual acceptance from training loss',
        speedup_claim=False,new_optimizer_updates=0)
    args.out.mkdir(parents=True,exist_ok=False)
    (args.out/'metrics.json').write_text(json.dumps(output,indent=2)+'\n')
    with (args.out/'metrics.csv').open('w') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['# FM密度对照：48次训练完成，效果验收待评测','',
        '从同一初始化完成两个48-update分支，复用已完成shift12对照。新分支只改变数学协议中的训练sigma采样shift为2.22，Gaussian权重函数保持shift12；采样变化会改变实际scalar weight和gradient mass，没有importance correction。','',
        '| 验证noise段 | 数量 | 初始化raw MSE | FM48 shift12 | FM48 shift2.22 | 新分支相对旧分支 |',
        '|---|---:|---:|---:|---:|---:|']
    for band,s in summaries.items():
        if not s['measured']:lines.append(f'| {band} | 0 | 未测 | 未测 | 未测 | 不推断 |');continue
        lines.append(f"| {band} | {s['count']} | {s['initial_raw']:.6f} | {s['shift12_raw']:.6f} | {s['shift2_22_raw']:.6f} | {s['candidate_vs_control_percent']:+.2f}% |")
    lines+=['','负百分比表示raw residual更小。中段有所改善，高段均值略差，不能称为全面改善；all均值由本组12个mid/4个high的固定数量决定，不是自然噪声分布下的期望性能。',
        '完整16项初始validation记录两分支相同，clip/chunk/sigma/weight及验证策略匹配。全部为最后chunk2（两latent帧），不能代替所有chunk的检查。历史control未保存validation完整noise tensor哈希，证据限于源码、种子与记录匹配。这里是GT clean-history局部FM拟合，不是当前A/D反事实或视频验收。','',
        '| 运行 | 记录训练wall(s) | allocated peak(MiB) | GPU | reserve(GiB) |',
        '|---|---:|---:|---|---:|']
    for r in runtime:lines.append(f"| {r['method']} | {r['wall_seconds']:.2f} | {r['allocated_peak_MiB']:.2f} | {r['visible_devices']} | {r['vram_reserve_gib']:.1f} |")
    lines+=['','两次在不同GPU和共享负载下运行，allocated峰值也不同；不将训练wall差归因于采样分布或宣称算法加速。训练峰值不是推理峰值，GPU权重/激活未独立分解。',
        '最终checkpoint/source/课程检查通过；另有step48 Adam/完整noise RNG审计。预算已封顶，没有追加更新。后续54点noise扫描、GT/generated各18点A/D及10条39帧视频由原队列继续执行；完整结果和全帧评审之前不通过B/C/D质量门槛。','',
        '[逐项CSV](metrics.csv) · [完整比较](metrics.json)']
    (args.out/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summaries,indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--control',type=Path,required=True);ap.add_argument('--candidate',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    main(ap.parse_args())
