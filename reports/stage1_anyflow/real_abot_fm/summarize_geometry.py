"""Summarize completed real-scene geometry probes without a video PASS claim."""
import argparse
import csv
import json
from pathlib import Path
from statistics import mean

def main(args):
    data=json.loads(args.input.read_text())
    if data['status']!='complete' or len(data['records'])!=18:raise ValueError('Requires all18 real-scene probe cases')
    assert len(data['cache_audits'])==6 and all(x['read_only'] for x in data['cache_audits'])
    assert all(x.get('rebuilt_D_history_equals_A',True) for x in data['cache_audits'])
    rows=[]
    for r in data['records']:
        delta=r['action_delta']
        rows.append(dict(clip=r['clip'],chunk=r['chunk'],sigma=r['sigma'],
            velocity_cosine=mean(r['absolute'][a]['cosine'] for a in 'AD'),
            delta_cosine=delta['cosine'],delta_norm_ratio=delta['norm_ratio'],
            student_response_relative_to_velocity=delta['student_response_relative_to_velocity'],
            teacher_response_relative_to_velocity=delta['teacher_response_relative_to_velocity'],
            current_sha256=r['state_sha256'],history_sha256=r['history_sha256']))
    result=dict(status='complete',step=data['step'],history_source=data['history_source'],
        records=len(rows),mean_velocity_cosine=mean(r['velocity_cosine'] for r in rows),
        mean_delta_cosine=mean(r['delta_cosine'] for r in rows),
        min_delta_cosine=min(r['delta_cosine'] for r in rows),max_delta_cosine=max(r['delta_cosine'] for r in rows),
        min_delta_norm_ratio=min(r['delta_norm_ratio'] for r in rows),max_delta_norm_ratio=max(r['delta_norm_ratio'] for r in rows),
        by_chunk={str(c):dict(mean_velocity_cosine=mean(r['velocity_cosine'] for r in rows if r['chunk']==c),
            mean_delta_cosine=mean(r['delta_cosine'] for r in rows if r['chunk']==c)) for c in (0,1,2)},
        by_sigma={str(s):dict(mean_delta_cosine=mean(r['delta_cosine'] for r in rows if r['sigma']==s)) for s in sorted({r['sigma'] for r in rows})},
        cache_readonly=True,current_future_action_does_not_change_history_KV=True,
        repeat_errors=[{k:r[k] for k in ('student_repeat_rmse','teacher_repeat_rmse')} for r in data['records'] if 'student_repeat_rmse' in r],
        execution={k:data[k] for k in ('model_forwards','clean_forwards','wall_seconds','gpu_allocated_peak_MiB')})
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    with (args.out/'metrics.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    lines=['# 真实验证场景：同状态当前chunk A/D动作差分','',
        f"Checkpoint: **FM step {data['step']}**. History/state source: **{data['history_source']}**. 2 held-out scenes × 3 chunks × 3 sigmas = 18 cases.",'',
        f"整体velocity平均cosine **{result['mean_velocity_cosine']:.6f}**；A/D差分平均cosine **{result['mean_delta_cosine']:.6f}**，范围 {result['min_delta_cosine']:.6f}–{result['max_delta_cosine']:.6f}。Student/teacher动作差分范数比 {result['min_delta_norm_ratio']:.3f}–{result['max_delta_norm_ratio']:.3f}。",'',
        '| chunk | 整体velocity平均cos | A/D差分平均cos |','|---|---:|---:|']
    for chunk,value in result['by_chunk'].items():lines.append(f"| {chunk} | {value['mean_velocity_cosine']:.6f} | {value['mean_delta_cosine']:.6f} |")
    lines+=['','## 控制与解释边界','',
        '- 每个状态下历史与noisy current固定，仅改当前chunk A/D；过去/未来动作embedding复用，A/D paired layout完全一致。原始自然动作句长度可与替换句不同，因此全部历史KV按此次layout重建，没有导入别的布局/模型的KV。',
        '- 每个chunk的历史用自己对应的RGB dual anchor在sigma=0提交。真实33B上检查chunk1的A/D历史重建KV哈希相同；每次当前干预期间KV内容/commit及参数version不变。',
        '- Student是部署cached路径；Original权重teacher在相同条件下双向重算历史，使用原始directed action predicate的SDPA，与student统一后端。两者历史内部依赖图不相同。',
        '- 状态为固定endpoint的显式加噪插值，不声称是实际solver中间状态。FM瞬时速度对比，不含AnyFlow finite-map语义。',
        '- 低cosine是动作条件场差分不一致的诊断，不自动等同于视频方向错误或画质失败。自然片段含联合动作/镜头变化；需另看同首帧纯A/D生成、完整视频及边界情况。',
        '- step00是零更新causal baseline；不能把它当48更新训练效果。GT和固定step00 generated state用于前后匹配；step48 own generated state若后续提供，单独作为自生成分布诊断，不冒充相同输入对照。','',
        '## 执行','', '```json',json.dumps(result['execution'],indent=2),'```','']
    (args.out/'RESULTS.md').write_text('\n'.join(lines));print(json.dumps(result,indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--input',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    main(ap.parse_args())
