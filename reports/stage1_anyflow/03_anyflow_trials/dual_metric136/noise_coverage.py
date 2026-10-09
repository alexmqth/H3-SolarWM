"""Read-only audit: realized time coverage and effective scalar loss weights.

These weights are not measured gradient contributions. No new training occurs.
"""
from pathlib import Path
import json
import statistics

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
SOURCE=ROOT/'outputs/2026-10-08-13/stage1_parallel_resume68_to128/train_128/training.json'
BOUNDS=(0.,.24078089904785155,.4252873229980469,.6894410400390625,1.)
train=json.loads(SOURCE.read_text())
assert train['status']=='complete' and len(train['updates'])==128
records=[]
for kind in ('diffusion','endpoint','flow_map'):
    samples=[s for u in train['updates'] for s in u['samples'] if s['sample_type']==kind]
    for lo,hi in zip(BOUNDS,BOUNDS[1:]):
        selected=[s for s in samples if lo<s['sigma']<=hi or (lo==0 and s['sigma']==0)]
        def mean(key):return statistics.mean(s[key] for s in selected) if selected else None
        records.append(dict(sample_type=kind,lower_exclusive=lo,upper_inclusive=hi,
            count=len(selected),total=len(samples),expected_shift12_probability=
                hi/(12.-11.*hi)-lo/(12.-11.*lo),
            mean_raw_residual_loss=mean('raw_loss'),mean_gaussian_weight=mean('weight'),
            mean_adaptive_scale=mean('adaptive_scale'),mean_weighted_loss=mean('weighted_loss'),
            sum_effective_scalar_weight=sum(s['weight']*s['adaptive_scale']/4. for s in selected),
            mean_model_evaluations=mean('model_evaluations')))
low_cases={}
for kind in ('diffusion','endpoint','flow_map'):
    low_cases[kind]=[dict(step=u['step'],action=u['action'],chunk=u['chunk'],
        sigma=s['sigma'],target_sigma=s['target_sigma']) for u in train['updates']
        for s in u['samples'] if s['sample_type']==kind and s['sigma']<=BOUNDS[1]]
data=dict(source=str(SOURCE),records=records,low_noise_cases=low_cases,
    sampling='u~Uniform(0,1), sigma=k*u/(1+(k-1)*u); P(sigma<=s)=s/(k-(k-1)*s)',
    low_noise_probability={str(k):BOUNDS[1]/(k-(k-1)*BOUNDS[1]) for k in (12.,2.22)},
    caveats=[
        'Small per-bin counts and changing checkpoints/states; mean training residual is not a matched validation error.',
        'Scalar weight is not gradient norm or direction and cannot establish a causal explanation of ghosting.',
        'The current training-timestep-shift controls BOTH time density and Gaussian weighting.',
        'Changing that flag alone is not a pure sampling-density ablation.',
        'Historical shift2.22 training did not pass the video/action gate; lowering shift is not a proven fix.',
    ])
(OUT/'noise_coverage.json').write_text(json.dumps(data,indent=2)+'\n')
lines=['# 低噪声覆盖与损失权重：只读检查','',
    '当前AnyFlow v1.5的未shift时间t来自Uniform(0,1)，不是log-normal。shift12下P(sigma≤0.240781)≈2.57%，128个endpoint样本的期望约3.3个，实际2个符合稀疏采样预期，不能凭此断言实现bug。','',
    '| type | sigma范围 | 样本数 | raw residual均值 | Gaussian weight均值 | adaptive scale均值 |','|---|---|---:|---:|---:|---:|']
for r in records:
    def fmt(key):return f"{r[key]:.6g}" if r[key] is not None else '—'
    lines.append(f"| {r['sample_type']} | ({r['lower_exclusive']:.3f},{r['upper_inclusive']:.3f}] | {r['count']}/{r['total']} | {fmt('mean_raw_residual_loss')} | {fmt('mean_gaussian_weight')} | {fmt('mean_adaptive_scale')} |")
lines+=['',
    '这些training样本跨checkpoint/action/chunk且各段数量不均，不是匹配的validation分段误差；用于解释监督覆盖，局部误差以固定输入双指标探针为准。adaptive和Gaussian标量也不是实际梯度贡献，不据此推断唯一根因。','',
    '低噪声endpoint的两个实际样本只有step52的D/chunk1（sigma0.124565）和step121的A/chunk0（sigma0.088974）。A/chunk2在128次更新内三种原AnyFlow样本都没有sigma≤0.240781的样本。这是更具体的状态覆盖缺口，但不证明它单独导致后段崩溃；完整逐样本记录保存在noise_coverage.json。','',
    '**下一项受控实验要解耦两个变量**：当前`training_timestep_shift`同时用于采样density和Gaussian loss weighting。直接从12换2.22会同时改变二者。若136仍失败，应先固定Gaussian权重、原AnyFlow目标和模型，只改变时间采样覆盖，再单独决定是否改变低噪声监督目标；不要把二者与新LoRA容量一起改。','',
    '已有shift2.22的64次训练并未通过画质/动作验收，因此“降低shift必然修复”没有依据。当前一致性辅助也只覆盖固定teacher/noise上的6种局部状态，不代表随机低噪声分布已经覆盖。', '',
    '此文仅准备后续归因，不启动136以上训练。先完成当前两组正常4/8步视频与双局部指标；成功须体现完整画面和动作响应。']
(OUT/'NOISE_COVERAGE.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(data['low_noise_probability']))
