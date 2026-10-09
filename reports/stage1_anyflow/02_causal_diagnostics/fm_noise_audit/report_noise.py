"""Match complete fixed-noise FM0/FM48 probes, separating density from weight."""
import csv
from collections import Counter,defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import statistics

BASE=Path(__file__).resolve().parent
BRIDGE=BASE.parents[1]/'2026-10-08-18/stage1_real_abot_fm'
TRAINING=BRIDGE/'train_48/training.json'
if not TRAINING.exists():
    # The submission ships the exact completed receipt, without model caches.
    TRAINING=BASE.parent/'real_abot_fm/training_snapshot.json'


def bin_name(s):return 'low' if s<=.2407808991 else ('mid' if s<=.6894410401 else 'high')
def mean(xs):return statistics.mean(xs)


def coverage():
    train=json.loads(TRAINING.read_text())
    assert train['status']=='complete' and len(train['updates'])==48
    rows=[dict(step=u['step'],clip=u['action'],chunk=u['chunk'],sigma=s['sigma'],noise_bin=bin_name(s['sigma']),
        weight=s['weight'],raw_loss=s['raw_loss'],weighted_loss=s['weighted_loss']) for u in train['updates'] for s in u['samples']]
    assert len(rows)==192
    groups={};total_weight=sum(x['weight'] for x in rows);total_loss=sum(x['weighted_loss'] for x in rows)
    for name in ('low','mid','high'):
        items=[x for x in rows if x['noise_bin']==name]
        groups[name]=dict(samples=len(items),sample_share=len(items)/192,weight_mass_share=sum(x['weight'] for x in items)/total_weight,
            weighted_loss_share=sum(x['weighted_loss'] for x in items)/total_loss,
            raw_loss_mean=mean(x['raw_loss'] for x in items),per_chunk=dict(Counter(x['chunk'] for x in items)),
            distinct_clip_chunks=len({(x['clip'],x['chunk']) for x in items}))
    result=dict(source_sha256=hashlib.sha256(TRAINING.read_bytes()).hexdigest(),
        summary=groups,rows=rows,limitation='Training model/data/noise change between rows; loss mass is not gradient mass or validation quality.')
    (BASE/'training_coverage.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    training=coverage()
    before=json.loads((BASE/'step_00/probe.json').read_text());after=json.loads((BASE/'step_48/probe.json').read_text())
    assert before['status']==after['status']=='complete'
    assert before['sigmas']==after['sigmas'] and len(before['records'])==len(after['records'])==54
    assert all(x['read_only'] for p in (before,after) for x in p['cache_audits'])
    key=lambda x:(x['clip'],x['chunk'],x['sigma'])
    old={key(x):x for x in before['records']};rows=[]
    fields=['source_sha256','noise_sha256','state_sha256','history_sha256','clean_target_sha256',
        'prompt_sha256','anchor_sha256','audio_sha256','gaussian_weight','native8']
    for y in after['records']:
        x=old[key(y)];assert all(x[f]==y[f] for f in fields)
        # Duplicate teacher computation across independent checkpoints/GPUs
        # provides a genuine full-output identity control.
        assert x['roles']['original']==y['roles']['original']
        row=dict(clip=y['clip'],chunk=y['chunk'],sigma=y['sigma'],noise_bin=bin_name(y['sigma']),native8=y['native8'])
        for role,source in [('original',x['roles']['original']),('causal0',x['roles']['causal']),('causal48',y['roles']['causal'])]:
            row.update({f'{role}_{m}':source[m] for m in ('raw_velocity_mse','clean_endpoint_mse','normalized_velocity_mse')})
        row.update(causal0_vs_original_cos=x['student_teacher']['cosine'],causal48_vs_original_cos=y['student_teacher']['cosine'],
            gaussian_weight12=y['gaussian_weight']['12.0'],gaussian_weight2_22=y['gaussian_weight']['2.22'])
        rows.append(row)
    rows.sort(key=key)
    with (BASE/'metrics.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    metrics=[k for k in rows[0] if k not in ('clip','chunk','sigma','noise_bin','native8')]
    predicates=[('all_grid_points',lambda x:True),('native8_only',lambda x:x['native8'])]
    predicates += [(f'bin_{b}',lambda x,b=b:x['noise_bin']==b) for b in ('low','mid','high')]
    predicates += [(f'sigma_{s:.6f}',lambda x,s=s:x['sigma']==s) for s in sorted(before['sigmas'],reverse=True)]
    predicates += [(f'chunk_{k}',lambda x,k=k:x['chunk']==k) for k in (0,1,2)]
    groups={}
    for name,predicate in predicates:
        selected=[x for x in rows if predicate(x)]
        groups[name]=dict(count=len(selected),**{m:mean(x[m] for x in selected) for m in metrics},
            improved_cases=sum(x['causal48_raw_velocity_mse']<x['causal0_raw_velocity_mse'] for x in selected))
    result=dict(status='complete_matched_noise_sweep',at=datetime.now().astimezone().isoformat(),groups=groups,
        input_hashes_matched=True,original_teacher_full_output_hashes_match=True,
        training_coverage=training['summary'],new_optimizer_steps=0,video_quality='not measured by this probe',
        probe_sha256={str(s):hashlib.sha256((BASE/f'step_{s:02d}/probe.json').read_bytes()).hexdigest() for s in (0,48)})
    (BASE/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# FM48：固定GT状态的分噪声velocity/endpoint误差','',
        '2个held-out自然场景×3个chunk×9个sigma，54个逐一匹配状态。8个点来自实际8-step shift2.22网格，另补30-step末端sigma0.071108。不是新增训练、A/D反事实或视频质量评测。', '',
        '所有输入/noise/history/anchor/prompt/source hash一致；两独立GPU上Original teacher完整输出hash逐点一致。每个checkpoint用自己权重构造GT clean history KV，预测不写cache。Original依然双向重算历史，所有分支统一SDPA和匹配条件。', '',
        '| sigma | Original raw MSE | causal0 raw MSE | causal48 raw MSE | Original endpoint MSE | causal0 endpoint MSE | causal48 endpoint MSE |',
        '|---:|---:|---:|---:|---:|---:|---:|']
    for sigma in sorted(before['sigmas'],reverse=True):
        g=groups[f'sigma_{sigma:.6f}']
        vals=[g[f'{role}_{metric}'] for metric in ('raw_velocity_mse','clean_endpoint_mse') for role in ('original','causal0','causal48')]
        lines.append('| '+f'{sigma:.6f}'+' | '+' | '.join(f'{v:.6f}' for v in vals)+' |')
    lines+=['','每个sigma均是6状态等权平均；不是按训练分布或推理轨迹出现概率加权。', '',
        'Velocity目标为真实GT的noise−clean。Endpoint是单次预测构造的 z_t−sigma*v，不是完整8/30步生成结果；其MSE在同一插值状态上等于sigma²×velocity MSE。两列帮助区分误差尺度，不是两个独立验证集。GT conditional flow本身可多模态，Original对某个单样本GT的误差也不为0，必须与Original对照，不能将所有低sigma误差归因于causal训练。', '',
        '| 训练noise段 | 样本数 | 样本比例 | 权重总量比例 | 加权loss比例 |',
        '|---|---:|---:|---:|---:|']
    for name,x in training['summary'].items():
        lines.append(f"| {name} | {x['samples']} | {x['sample_share']:.2%} | {x['weight_mass_share']:.2%} | {x['weighted_loss_share']:.2%} |")
    lines+=['','这些训练loss来自不同更新、数据、噪声，不能画成固定输入学习曲线；加权loss比例不等于参数梯度比例。Shift同时影响sigma采样density和Gaussian normalization，表中单独保存shift12/2.22的固定sigma weight以供后续受控设计。当前未改sampling/weight。', '',
        '[逐状态CSV](metrics.csv) · [完整统计](analysis.json) · [训练覆盖](training_coverage.json)','']
    (BASE/'RESULTS.md').write_text('\n'.join(lines))
    print(json.dumps(dict(status=result['status'],groups=groups),indent=2))


if __name__=='__main__':main()
