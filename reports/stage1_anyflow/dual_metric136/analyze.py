"""Summarize both local metric families without inferring visual acceptance."""
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import statistics

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
PREVIOUS=ROOT/'outputs/2026-10-08-15/stage1_finite_interval_probe128'
REFINED=ROOT/'outputs/2026-10-08-15/stage1_finite_interval_refinement128'
FORK=ROOT/'outputs/2026-10-08-15/stage1_interval_consistency_candidate'
VARIANTS=('initial128','control136','auxiliary136')
rows=[];states={};conditioning={};input_hashes={};checks=[]
for variant in VARIANTS:
    states[variant]={}
    for action in ('A','D'):
        file=OUT/variant/f'probe_{action}.json'
        if not file.exists():
            states[variant][action]='not_started';continue
        result=json.loads(file.read_text());states[variant][action]=result['status']
        if result['status']=='complete':
            assert len(result['records'])==9 and result['model_forwards']==159
            assert result['clean_commit_forwards']==3 and result['parameter_versions_unchanged']
            assert all(x['unchanged'] for x in result['cache_checks'])
        keys=('clean_sha256','noise_sha256','action_sha256','anchor_sha256','prompt_sha256','audio_sha256')
        if not all(k in result for k in keys):continue
        current={k:result[k] for k in keys}
        if action in conditioning:assert conditioning[action]==current
        else:conditioning[action]=current
        for r in result['records']:
            key=(action,r['chunk'],r['native_step'])
            if key in input_hashes:assert input_hashes[key]==r['initial_sha256']
            else:input_hashes[key]=r['initial_sha256']
            target=Path(r['target_file'])
            assert hashlib.sha256(target.read_bytes()).hexdigest()==r['target_sha256']
            finest=str(r['subdivisions'][-1])
            if variant=='initial128':
                old=json.loads(((REFINED if r['native_step']==7 else PREVIOUS)/f'probe_{action}.json').read_text())
                previous=next(x for x in old['records'] if (x['chunk'],x['native_step'])==(r['chunk'],r['native_step']))
                assert r['initial_sha256']==previous['initial_sha256']
                assert r['direct_endpoint_sha256']==previous['direct_endpoint_sha256']
                assert r['reference_endpoint_sha256'][finest]==previous['reference_endpoint_sha256'][finest]
                checks.append(dict(action=action,chunk=r['chunk'],step=r['native_step'],repeats_previous_exactly=True))
            rows.append(dict(variant=variant,action=action,chunk=r['chunk'],noise_region=r['noise_region'],
                sigma=r['sigma'],target_sigma=r['target_sigma'],reference_substeps=int(finest),
                self_consistency_velocity_relative_rmse=r['velocity_relative_rmse'],
                self_consistency_endpoint_rmse=r['finite_vs_diagonal_endpoint_rmse'][finest],
                reference_refinement_rmse=r['diagonal_refinement_endpoint_rmse'],
                finite_rmse_to_teacher_at_r=r['endpoint_rmse_to_teacher_interpolant_at_r']['finite'],
                diagonal_rmse_to_teacher_at_r=r['endpoint_rmse_to_teacher_interpolant_at_r']['diagonal'+finest],
                finite_t_to_zero_rmse_to_teacher_clean=r['finite_t_to_zero_rmse_to_teacher_clean'],
                frozen128_training_target_velocity_rmse=r.get('finite_velocity_rmse_to_frozen128_training_target'),
                seconds=r['seconds']))

complete=all(states[v][a]=='complete' for v in VARIANTS for a in ('A','D'))
aggregates=[]
for variant in VARIANTS:
    for region in ('high','middle','low'):
        group=[r for r in rows if r['variant']==variant and r['noise_region']==region]
        if not group:continue
        metrics=('self_consistency_velocity_relative_rmse','self_consistency_endpoint_rmse',
            'reference_refinement_rmse','finite_rmse_to_teacher_at_r','diagonal_rmse_to_teacher_at_r',
            'finite_t_to_zero_rmse_to_teacher_clean')
        aggregates.append(dict(variant=variant,noise_region=region,count=len(group),
            **{k:statistics.mean(r[k] for r in group) for k in metrics}))

coverage={}
bounds=[0.,.24078089904785155,.4252873229980469,.6894410400390625,1.]
for variant in VARIANTS:
    train=(ROOT/'outputs/2026-10-08-13/stage1_parallel_resume68_to128/train_128' if variant=='initial128'
           else FORK/variant.removesuffix('136')/'train_136')/'training.json'
    if not train.exists():continue
    result=json.loads(train.read_text());coverage[variant]={}
    for kind in ('diffusion','endpoint','flow_map'):
        samples=[s for u in result['updates'] for s in u['samples'] if s['sample_type']==kind]
        coverage[variant][kind]=dict(total=len(samples),bins=[dict(lower=lo,upper=hi,
            count=sum((lo<s['sigma']<=hi) or (lo==0 and s['sigma']==0) for s in samples))
            for lo,hi in zip(bounds,bounds[1:])])
    auxiliary=[u['interval_consistency'] for u in result['updates'] if u.get('interval_consistency')]
    coverage[variant]['additional_frozen_target_samples']=dict(total=len(auxiliary),
        note='Fixed teacher/noise state per action/chunk at sigma .240780899 ->0; not randomized low-noise coverage')

data=dict(updated_at=datetime.now().astimezone().isoformat(),complete=complete,states=states,
    rows=rows,aggregates=aggregates,training_time_coverage=coverage,
    initial128_repetition_audit=checks,conditioning_consistent=True,
    visual_acceptance='Separate full39-frame4/8 video review required',
    limitations=[
      'One teacher clip/noise per action; clean-history interpolation states are not generated-history states.',
      'Current diagonal numerical trajectory and frozen128 auxiliary target are different references; neither is ground truth.',
      'At target r>0 the pseudo-target is (1-r)*teacher_clean+r*same_noise. Comparing to clean would be invalid.',
      'Direct t->0 at high/middle noise is a diagnostic long map, not the actual native8-step inference interval.',
      'Three noise anchors, not full-bin expectations; training bin counts are reported separately.',
      'Do not aggregate different map widths as a calibrated video quality score.',
      'Pseudo-GT is an Original H3 generated clip, not a real ground-truth video.',
    ])
(OUT/'analysis.json').write_text(json.dumps(data,indent=2)+'\n')
if rows:
    with (OUT/'metrics.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
lines=['# 128/136双局部指标：'+('诊断完成' if complete else '运行中，非最终结论'),'',
    '同clean history/teacher-noise插值状态，A/D×3chunks。高/中/低是三个固定噪声锚点，不是整个噪声段的期望值。正常finite-map4/8完整视频另行验收。','',
    '| checkpoint | 噪声锚点 | cases /6 | finite/当前diagonal相对速度RMSE | finite/diagonal端点RMSE | finite到teacher(r) RMSE | diagonal到teacher(r) RMSE | finite(t→0)到clean RMSE |',
    '|---|---|---:|---:|---:|---:|---:|---:|']
for r in aggregates:
    lines.append(f"| {r['variant']} | {r['noise_region']} | {r['count']} | {r['self_consistency_velocity_relative_rmse']:.2%} | {r['self_consistency_endpoint_rmse']:.6f} | {r['finite_rmse_to_teacher_at_r']:.6f} | {r['diagonal_rmse_to_teacher_at_r']:.6f} | {r['finite_t_to_zero_rmse_to_teacher_clean']:.6f} |")
lines+=['','每个case原值见metrics.csv/analysis.json。末区间额外记录到冻结128训练target的距离，不能替代当前模型self-consistency。高/中区间用4/8细分，末区间用8/16，参考细分误差单列。','',
    'r>0时teacher(r)=(1-r)×clean+r×同noise；不能把带噪端点直接和clean比较。另列的t→0是独立端点诊断；高/中t→0不是实际8步推理的单步。Original teacher仅伪GT，内部误差不能冒充视频质量。','',
    '比较同噪声锚点、同action/chunk的128/control136/auxiliary136，以分辨更自洽是否伴随伪目标距离恶化。不得将跨区间宽度的数值混成统一画质分数。','',
    '训练覆盖按(0,.240781]、(.240781,.425287]、(.425287,.689441]、(.689441,1]和diffusion/endpoint/map分开记录在analysis.json。固定8个辅助低噪声参考不等于随机低噪声训练分布已补齐。']
if complete:
    lines+=['','## 本轮解释','',
        '低噪声当前自一致性相对速度RMSE：128为18.12%，control136为18.13%，auxiliary136为17.90%；auxiliary只比128下降约0.22个百分点。端点self-consistency RMSE为0.054262→0.053334。','',
        'finite到Original teacher clean伪GT的平均RMSE为0.058802→0.058319，约下降0.82%；6个case中5个略降，D/chunk1略升。**平均值没有支持“一致性改善是以finite伪目标正确性变差为代价”**，但改善很小，不能据此接受视频。','',
        '另一个容易混淆的数：到冻结128训练参考的velocity RMSE为0.225360→0.210929（约下降6.4%），明显大于当前模型自一致性的相对改善。同时当前diagonal16到teacher的误差从0.064893升至0.067320。说明frozen-reference fit与current self-consistency不是同一个测量，diagonal也不是GT。','',
        '真实正常finite-map4/8生成视频必须独立验收。当前8步辅助A/D仍重影且分离度0.635174，没有恢复原128的0.759660；不能把局部数值的小幅改善当作画质修复。完整4/8视频见[136结果](../interval_consistency_candidate/FINAL_RESULTS.md)。','',
        '所有54个case已完成，6个进程各159 noisy+3 clean；全部参数version及KV内容/commit数量不变，128重复输入/finite/diagonal输出hash与先前结果一致。共享主机只读探针不是速度benchmark。','',
        '![按噪声锚点分别比较两类指标](local_metric_tradeoff.png)']
(OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(dict(complete=complete,states=states,rows=len(rows)),ensure_ascii=False))
