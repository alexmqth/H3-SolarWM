"""Archive Experiment A without claiming causal-FM or video acceptance."""
import csv
import json
from pathlib import Path
from statistics import mean

OUT=Path(__file__).resolve().parent
receipts={a:json.loads((OUT/f'field_{a}.json').read_text()) for a in 'AD'}
assert all(d['status']=='complete' and len(d['records'])==6 for d in receipts.values())
rows=[]
for a,d in receipts.items():
    for r in d['records']:
        for role,v in r['comparisons'].items():
            rows.append(dict(history=a,chunk=r['chunk'],sigma=r['sigma'],role=role,
                velocity_cosine_A=v['absolute_velocity']['A']['cosine'],
                velocity_cosine_D=v['absolute_velocity']['D']['cosine'],
                delta_cosine=v['action_delta']['cosine'],norm_ratio=v['action_delta']['norm_ratio'],
                semantics=v['semantics']))
summary={}
for role in receipts['A']['roles']:
    selected=[r for r in rows if r['role']==role]
    summary[role]=dict(mean_velocity_cosine=mean([r['velocity_cosine_A'] for r in selected]+[r['velocity_cosine_D'] for r in selected]),
        mean_delta_cosine=mean(r['delta_cosine'] for r in selected),
        min_delta_cosine=min(r['delta_cosine'] for r in selected),max_delta_cosine=max(r['delta_cosine'] for r in selected),
        mean_delta_norm_ratio=mean(r['norm_ratio'] for r in selected))
backend=[r['backend_control']['delta']['cosine'] for d in receipts.values() for r in d['records'] if 'backend_control' in r]
initial_zero=all(r['initializer_diagonal_preservation']['velocity_rmse']==[0.,0.] for d in receipts.values() for r in d['records'])
result=dict(status='complete',summary=summary,records=rows,initializer_diagonal_exact=initial_zero,
    Original_SDPA_vs_native_Flex_delta_cosines=backend,
    execution={a:{k:d[k] for k in ('model_forwards','wall_seconds','gpu_allocated_peak_MiB')} for a,d in receipts.items()})
(OUT/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
with (OUT/'metrics.csv').open('w') as file:
    w=csv.DictWriter(file,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
lines=['# 实验A：因果化前后的整体速度与动作差分','',
    '**12个固定状态的完整对照已完成：未进行AnyFlow更新、未加载我们新增适配的Original H3，仅换causal路由，整体velocity平均余弦仍为0.9963，但动作差分平均余弦降到0.0602。动作几何的明显失配在AnyFlow训练之前就已出现。**','',
    '这支持先做真实视频causal FM桥接；不支持继续用增加AnyFlow updates作为首要修复。当前仍没有通过视频验收。','',
    '| 角色 | 整体velocity平均cos | A/D差分平均cos | 差分cos范围 |',
    '|---|---:|---:|---:|']
labels={'original':'Original权重＋双向', 'causal_original':'同一Original权重＋causal（无我们新增适配）',
    'causal_initializer':'此前训练的共同初始化（已有visual/action适配，bank零输出）',
    'causal_initializer_diagonal':'共同初始化＋未训练AnyFlow r=t',
    'causal_fm32':'匹配的Causal FM32', 'causal_anyflow32':'匹配的Causal AnyFlow32 r=t',
    'causal_anyflow128':'AnyFlow128 r=t（额外更新，仅补充）'}
for role,label in labels.items():
    m=summary[role];lines.append(f"| {label} | {m['mean_velocity_cosine']:.6f} | {m['mean_delta_cosine']:.6f} | {m['min_delta_cosine']:.6f}–{m['max_delta_cosine']:.6f} |")
lines += ['', '## 控制了哪些变量','',
    '- 固定AnyFlow128保存的A/D generated endpoints及其历史，显式加噪；后两个chunk×三个sigma×两条history。不是实际solver中间状态。',
    '- 所有角色输入相同长度的完整prefix＋clean history＋当前noisy chunk；global positions、dual RGB anchors、prefix时间和当前A/D spans一致。只更改当前动作，过去/未来动作不变。',
    '- 原始与causal均使用同一个SDPA后端；CPU逐元素验证Original predicate等于原生directed mask。causal保持chunk内双向、跨chunk只向过去、past/current action visibility与own-frame feedback。',
    '- 因果部分重算完整祖先和历史action-prefix状态。这是输入范围受控的field诊断，不冒充部署时persistent-KV等价推理。当前anchor也作用于重算的历史，因此不是完整rollout质量验收。',
    '- 原始H3-World action LoRA始终保留。第一组causal_original关闭我们新增的visual/bank/time/residual，与Original角色具有完全相同权重。',
    '- FM32/AnyFlow32共同配置仅objective/out_dir/resume来源不同，旧审计已匹配初始化、训练动作/chunk/sigma/噪声序列和32更新。两者训练数据仍是teacher生成片段，不能算实验B的真实视频基线；等更新数不等训练算力。',
    '- 多加入共同初始化，区分此前visual/action适配的影响；AnyFlow128不与FM32冒充同预算对照。',
    '', '## 对角条件与计算后端控制','',
    f'- 12个状态、两种动作下，共同初始化的AnyFlow r=t与不启用AnyFlow的FM输出逐元素相同：**{initial_zero}**。这只证明本初始化对角条件保持，不代表off-diagonal或训练后场正确。',
    '- Original SDPA与原生FlexAttention的中sigma动作差分cos分别为：'+', '.join(f'{x:.6f}' for x in backend)+'。BF16下小幅动作差分对后端误差敏感，重复同后端输出为0误差不能当作所有精度误差都为0。',
    '- 主表全部使用统一SDPA后端，避免把这项后端差异直接算进因果化对照。结果仍限定于本组生成状态及实现。',
    '- r<t有限区间输出的全部数据保留在原始receipt和CSV，仅作描述，不能与瞬时teacher作同语义优劣结论。',
    '', '## 接下来','',
    '按目标附件继续实验B：用真实ABot视频、有效动作标注和GT历史训练普通causal FM。当前已选择6个episode、16 train＋8 validation个39帧片段；按episode隔离，真实VAE latent正在编码。B的训练与画质/动作评测尚未完成。',
    '本次证据说明AnyFlow不是失配开始出现的唯一环节；不证明普通FM能自动修复，更不证明Stage2已具备可靠前置checkpoint。C/D仍需依次验证。','',
    '## 执行记录','', '```json',json.dumps(result['execution'],indent=2),'```','',
    '两条history共224次诊断前向，无optimizer。共享主机上的wall time只作执行记录，不是推理加速评测。GPU5/6进程已正常退出。','']
(OUT/'RESULTS.md').write_text('\n'.join(lines))
print(json.dumps(summary,indent=2))
