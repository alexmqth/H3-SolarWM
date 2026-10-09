"""Summarize the completed read-only six-state first-chunk route experiment."""
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean

BASE = Path(__file__).resolve().parent
LABELS = {
    'own_common': 'Own action + video-updated prefix (Original)',
    'own_static': 'Own action + static prefix',
    'causal_common': 'Past/own actions + video-updated prefix',
    'causal_static': 'Past/own actions + static prefix (current causal)',
}


def main():
    data = json.loads((BASE / 'probe.json').read_text())
    assert data['status'] == 'complete' and len(data['records']) == 6
    assert data['model_forwards'] == 56 and data['parameter_versions_unchanged']
    repeats = [r for r in data['records'] if 'original_identity_rmse' in r]
    assert len(repeats) == 2
    assert all(all(v == 0 for v in r['original_identity_rmse'].values()) for r in repeats)
    rows = []
    for record in data['records']:
        for role, value in record['roles'].items():
            delta = value['action_delta']
            rows.append(dict(clip=record['clip'], sigma=record['sigma'], role=role,
                whole_velocity_cosine=mean(v['cosine'] for v in value['absolute'].values()),
                delta_cosine=delta['cosine'], delta_norm_ratio=delta['norm_ratio'],
                delta_relative_error=delta['relative_delta_error'],
                student_delta_relative_to_velocity=delta['student_response_relative_to_velocity'],
                teacher_delta_relative_to_velocity=delta['teacher_response_relative_to_velocity'],
                state_sha256=record['state_sha256'], anchor_sha256=record['anchor_sha256'],
                pair_sha256=record['pair_sha256']))
    columns = ('whole_velocity_cosine', 'delta_cosine', 'delta_norm_ratio', 'delta_relative_error',
               'student_delta_relative_to_velocity', 'teacher_delta_relative_to_velocity')
    def summary(selected):
        return dict(count=len(selected), **{f'mean_{k}': mean(r[k] for r in selected) for k in columns},
                    min_delta_cosine=min(r['delta_cosine'] for r in selected),
                    max_delta_cosine=max(r['delta_cosine'] for r in selected))
    result = dict(status='complete', scope='Six fixed generated endpoint/noise states, first chunk without history',
        by_role={role: summary([r for r in rows if r['role'] == role]) for role in LABELS},
        by_sigma={str(s): {role: summary([r for r in rows if r['role'] == role and r['sigma'] == s])
                          for role in LABELS} for s in sorted({r['sigma'] for r in rows})},
        independent_original_identity_rmse=[r['original_identity_rmse'] for r in repeats],
        cached_vs_full_causal=[dict(clip=r['clip'], sigma=r['sigma'], **r['cached_vs_full_causal']) for r in repeats],
        model_forwards=data['model_forwards'], wall_seconds=data['wall_seconds'],
        gpu_allocated_peak_MiB=data['gpu_allocated_peak_MiB'], optimizer_steps=0,
        probe_sha256=hashlib.sha256((BASE / 'probe.json').read_bytes()).hexdigest())
    (BASE / 'analysis.json').write_text(json.dumps(result, indent=2) + '\n')
    with (BASE / 'metrics.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    selected_roles = [r for r in LABELS if r != 'own_common']
    figure, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), layout='constrained')
    for ax, clip in zip(axes, dict.fromkeys(r['clip'] for r in rows)):
        sigmas = sorted({r['sigma'] for r in rows}, reverse=True)
        values = np.array([[next(r['delta_cosine'] for r in rows if r['clip'] == clip and r['role'] == role and r['sigma'] == sigma)
                            for sigma in sigmas] for role in selected_roles])
        plot = ax.imshow(values, vmin=-1, vmax=1, cmap='coolwarm')
        ax.set_xticks(range(3), [f'{s:.3f}' for s in sigmas]); ax.set_xlabel('sigma (fixed generated-state interpolant)')
        ax.set_yticks(range(3), ['Own action\nstatic prefix', 'Past/own action\nvideo-updated prefix', 'Past/own action\nstatic prefix'])
        ax.set_title('Forest/buildings' if clip.startswith('118') else 'Medieval village')
        for i in range(3):
            for j in range(3): ax.text(j, i, f'{values[i, j]:+.3f}', ha='center', va='center', color='black')
    figure.colorbar(plot, ax=axes, shrink=.8, label='cos(student A-D delta, Original A-D delta)')
    figure.suptitle('First chunk: same weights/state, two attention-edge factors\nOriginal = own action + video-updated prefix (identity checked); no training/video PASS')
    figure.savefig(BASE / 'route_geometry.png', dpi=160); plt.close(figure)
    lines = ['# 首块的两类 attention 边：固定权重与状态的 2×2 消融', '',
        '本实验隔离“当前动作行已经对齐，但差分方向仍失配”的结构因素。两个验证场景的固定step00生成endpoint，各取高/中/低三个sigma，只比较首个5-latent块，因此没有历史KV。所有分支用相同Original权重、发布action LoRA、token范围、anchor、时间、A/D pair和SDPA调用方式。', '',
        '| 视频读取动作的规则 | 通用prefix能否读取当前视频 | 整体velocity cos | A/D delta cos | delta cos范围 |',
        '|---|---|---:|---:|---:|']
    for role in LABELS:
        r = result['by_role'][role]
        own = '仅自己绑定动作' if role.startswith('own_') else '过去及自己绑定动作'
        common = '允许' if role.endswith('_common') else '禁止'
        lines.append(f"| {own} | {common} | {r['mean_whole_velocity_cosine']:.6f} | {r['mean_delta_cosine']:.6f} | {r['min_delta_cosine']:.6f}～{r['max_delta_cosine']:.6f} |")
    lines += ['', '![分状态动作差分](route_geometry.png)', '',
        '当前动作自身的action→video反馈在四个分支全部保留。这里的“通用prefix”是非action的图像/场景文本/音频等行，不能混同于当前action反馈开关。', '',
        '“仅自己动作＋允许prefix读取视频”的首块mask与Original逐元素相同；两个场景中sigma的独立Original重放A/D共四次RMSE均为0。因此该行的cos=1是身份正控，不是新模型取得恢复。两个单边改动和现行causal分支的效应见完整逐状态CSV；不能把余弦差当成可相加的归因比例。', '',
        'CPU以真实layout核查四种mask只改变两类声明的边，且attention图的50层可达性中未来chunk的action内容不能流入当前video。这不是多chunk泄漏验收，text-refiner等其它路径的语义仍沿用已核查的输入协议。', '',
        '另在两个中sigma状态，把现行causal的完整prefix SDPA与部署split-query cached路径对照；空cache保持零commit/零字节，结果单列在analysis.json，避免混淆mask差异和执行形状的数值影响。', '',
        '**限制：** 这不是新训练、不是完整rollout，也没有证明单独修改某条边就能修好39/124帧视频。历史KV交互未在本实验中测试，首块以外不能直接推广。两项路由一起恢复到Original的首块身份正控，也不能当作完整causal方案。普通FM48训练与其评测协议保持不变。', '',
        '[逐状态指标](metrics.csv) · [全部原始前向记录](probe.json) · [执行与identity控制](analysis.json) · [CPU mask检查](preflight.json)', '',
        f"执行：{data['model_forwards']}次真实H3预测/重复前向，0 optimizer updates，{data['wall_seconds']:.2f}s，GPU allocated peak {data['gpu_allocated_peak_MiB']:.2f}MiB。这是诊断成本，不是视频推理benchmark。", '']
    (BASE / 'RESULTS.md').write_text('\n'.join(lines))
    print(json.dumps(result['by_role'], indent=2))


if __name__ == '__main__':
    main()
