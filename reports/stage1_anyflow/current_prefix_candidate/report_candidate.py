"""Summarize complete paired geometry; first-chunk identity is not improvement."""
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import statistics

BASE=Path(__file__).resolve().parent


def mean(xs):return statistics.mean(xs)


def main():
    probe=json.loads((BASE/'probe.json').read_text())
    assert probe['status']=='complete' and len(probe['records'])==18
    assert len(probe['cache_audits'])==12
    assert all(x['read_only'] and x['before_sha256']==x['after_sha256'] for x in probe['cache_audits'])
    assert all(x['baseline_vs_previous_delta_cos_absdiff']<1e-8 for x in probe['records'])
    assert all(x.get('candidate_first_chunk_original_identity') for x in probe['records'] if x['chunk']==0)
    rows=[]
    for x in probe['records']:
        for role,y in x['roles'].items():
            d=y['action_delta']
            rows.append(dict(clip=x['clip'],chunk=x['chunk'],sigma=x['sigma'],role=role,
                whole_velocity_cosine=mean(v['cosine'] for v in y['absolute'].values()),
                delta_cosine=d['cosine'],delta_norm_ratio=d['norm_ratio'],
                delta_relative_error=d['relative_delta_error'],
                response_relative_velocity=d['student_response_relative_to_velocity']))
    with (BASE/'metrics.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    groups={}
    for name,predicate in [('all',lambda x:True),('first_chunk_identity',lambda x:x['chunk']==0),
                            ('later_chunks',lambda x:x['chunk']>0),
                            *[(f'chunk_{k}',lambda x,k=k:x['chunk']==k) for k in (1,2)],
                            *[(f'sigma_{s:.6f}',lambda x,s=s:x['sigma']==s) for s in sorted({x['sigma'] for x in rows})]]:
        groups[name]={}
        for role in ('baseline','candidate'):
            selected=[x for x in rows if predicate(x) and x['role']==role]
            groups[name][role]=dict(count=len(selected),**{k:mean(x[k] for x in selected)
                for k in ('whole_velocity_cosine','delta_cosine','delta_norm_ratio','delta_relative_error')})
    later=[x for x in probe['records'] if x['chunk']>0]
    improving=sum(x['roles']['candidate']['action_delta']['cosine']>x['roles']['baseline']['action_delta']['cosine'] for x in later)
    result=dict(status='complete_frozen_routing_probe',timestamp=datetime.now().astimezone().isoformat(),
        groups=groups,later_cosine_improved_cases=improving,later_case_count=len(later),
        source_sha256=hashlib.sha256((BASE/'probe.json').read_bytes()).hexdigest(),
        optimizer_steps=0,video_acceptance='not measured',
        limitation='Identical raw history, state and conditions; cached causal ancestors and bidirectional teacher ancestors differ. First chunk is identity control.')
    (BASE/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# Current-prefix候选：真实33B同状态路由诊断','',
        '冻结Original H3+released action LoRA，零训练更新；2个scene×3chunks×3sigmas。每个状态只替换当前chunk A/D。', '',
        '**首块候选=Original是身份正控，不能当作修复收益。后续chunk单独列出。没有生成候选视频，也没有action/画质PASS。**','',
        '| 分组 | 点数 | 整体cos baseline→candidate | A/D delta cos baseline→candidate | delta相对误差 baseline→candidate |',
        '|---|---:|---:|---:|---:|']
    for name,data in groups.items():
        a,b=data['baseline'],data['candidate']
        lines.append(f"| {name} | {a['count']} | {a['whole_velocity_cosine']:.6f} → {b['whole_velocity_cosine']:.6f} | {a['delta_cosine']:.6f} → {b['delta_cosine']:.6f} | {a['delta_relative_error']:.6f} → {b['delta_relative_error']:.6f} |")
    lines+=['',f'后续12点中{improving}点delta cosine增加；这不是光流方向正确率。', '',
        '控制：baseline重跑与旧probe逐状态delta cosine差<1e−8；input/history/pair匹配；当前anchor及teacher完整输出hash已保存。两种causal路由各自重建KV，A/D内部共享只读cache。chunk1分别以A/D重建历史KV一致；当前预测不修改KV/参数；重复RMSE=0。', '',
        '解释边界：候选的own直接绑定和current-video公共prefix两项改变在先前首块2×2实验中已分开；本次测试它们组合到persistent cache的行为。Teacher仍双向重算历史，不能把候选当作Original完整等价模型。状态是固定endpoint加噪，不是在线solver采样到的中间状态，也不是候选自生成分布。', '',
        f"真实执行：{probe['model_forwards']} noisy/reference forwards、{probe['clean_forwards']} clean-history forwards；{probe['wall_seconds']:.2f}s、allocated peak{probe['gpu_allocated_peak_MiB']:.2f}MiB，单次共享主机不作speedup声明。",'',
        '[逐状态指标](metrics.csv) · [完整probe](probe.json) · [CPU协议](README.md)','']
    (BASE/'RESULTS.md').write_text('\n'.join(lines))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
