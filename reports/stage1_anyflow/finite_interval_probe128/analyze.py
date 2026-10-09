"""Summarize completed read-only probes and actual training time coverage."""
from pathlib import Path
import hashlib
import json
import torch

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
REFINE=OUT.parent/'stage1_finite_interval_refinement128'
TRAIN=ROOT/'outputs/2026-10-08-13/stage1_parallel_resume68_to128/train_128/training.json'
torch.set_num_threads(4)
analysis=dict(scope='Frozen128 clean-history numerical diagnostic; not video acceptance',
              actions={},training_time_coverage={})
for action in ('A','D'):
    coarse=json.loads((OUT/f'probe_{action}.json').read_text())
    fine=json.loads((REFINE/f'probe_{action}.json').read_text())
    assert coarse['status']==fine['status']=='complete'
    assert len(coarse['records'])==9 and len(fine['records'])==3
    for result in (coarse,fine):
        assert result['parameter_versions_unchanged']
        assert len(result['cache_checks'])==3 and all(x['unchanged'] for x in result['cache_checks'])
    for key in ('checkpoint','clean_sha256','noise_sha256','action_sha256','anchor_sha256','prompt_sha256','audio_sha256'):
        assert coarse[key]==fine[key],key
    clean=torch.load(ROOT/f'outputs/2026-10-02-03/action_{action}_teacher_39/baseline_latents.pt',
                     map_location='cpu',weights_only=True).float()
    low=[]
    for f in fine['records']:
        c=next(x for x in coarse['records'] if x['chunk']==f['chunk'] and x['native_step']==7)
        # Repeated8 references and direct predictions are byte-for-byte identical.
        assert c['initial_sha256']==f['initial_sha256']
        assert c['direct_endpoint_sha256']==f['direct_endpoint_sha256']
        assert c['reference_endpoint_sha256']['8']==f['reference_endpoint_sha256']['8']
        path=Path(f['target_file']);assert hashlib.sha256(path.read_bytes()).hexdigest()==f['target_sha256']
        data=torch.load(path,map_location='cpu',weights_only=True)['tensors']
        gt=clean[:,:,f['chunk']*5:(f['chunk']+1)*5]
        direct=data['initial']-f['sigma']*data['finite_velocity']
        local_rmse=dict(finite=float((direct-gt).square().mean().sqrt()),
                        diagonal16=float((data['reference_endpoints'][16]-gt).square().mean().sqrt()))
        low.append(dict(chunk=f['chunk'],velocity_relative_rmse16=f['velocity_relative_rmse'],
            finite_vs_reference_rmse={**c['finite_vs_diagonal_endpoint_rmse'],**f['finite_vs_diagonal_endpoint_rmse']},
            refinement_4_to8=c['diagonal_refinement_endpoint_rmse'],
            refinement_8_to16=f['diagonal_refinement_endpoint_rmse'],
            refinement_ratio=f['diagonal_refinement_endpoint_rmse']/c['diagonal_refinement_endpoint_rmse'],
            gap_over_refinement16=f['endpoint_error_over_refinement'],
            same_input_direct_reference8_hashes=True,
            endpoint_rmse_to_original_teacher_pseudo_gt=local_rmse))
    analysis['actions'][action]=dict(coarse=coarse['records'],low_interval=low,
        coarse_wall_s=coarse['wall_seconds'],refinement_wall_s=fine['wall_seconds'],
        coarse_peak_MiB=coarse['gpu_allocated_peak_MiB'],refinement_peak_MiB=fine['gpu_allocated_peak_MiB'])
train=json.loads(TRAIN.read_text())
for kind in ('diffusion','endpoint','flow_map'):
    samples=[s for u in train['updates'] for s in u['samples'] if s['sample_type']==kind]
    analysis['training_time_coverage'][kind]=dict(total=len(samples),
        sigma_le_last8=sum(s['sigma']<=.24078089904785155 for s in samples),
        sigma_le_last4=sum(s['sigma']<=.4252873229980469 for s in samples))
analysis['limitations']=[
    'Interpolation states use one Original pseudo-GT clip and its saved noise per action; not free-running states.',
    'Same-model diagonal16 trajectories are numerical self-consistency references, not ground truth.',
    '4/8/16 refinement is evidence of a persistent mismatch, not a rigorous integration error bound.',
    'The normal finite map is closer to the Original pseudo-GT endpoint than diagonal16 in these local cases.',
    'Only real normal finite-map4/8 generated-history videos can establish benefit from an auxiliary objective.',
]
(OUT/'analysis.json').write_text(json.dumps(analysis,indent=2)+'\n')
rows=['# AnyFlow128有限区间诊断完成','',
    'A/D各9个4/8细分case、各3个末区间8/16细分case全部完成；参数version与真实CPU KV内容hash不变。',
    '重复计算的输入、正常finite端点、8细分端点hash在两轮中逐项相同。没有optimizer更新。','',
    '| action | chunk | 末区间相对速度偏差（16参考） | finite/16端点RMSE | 8→16参考变化 | 差距/参考变化 |',
    '|---|---:|---:|---:|---:|---:|']
for action,content in analysis['actions'].items():
    for r in content['low_interval']:
        rows.append(f"| {action} | {r['chunk']} | {r['velocity_relative_rmse16']:.2%} | {r['finite_vs_reference_rmse']['16']:.6f} | {r['refinement_8_to16']:.6f} | {r['gap_over_refinement16']:.2f} |")
rows += ['',
    '低噪声最后一区间的差异显著大于更细参考之间的变化，且跨A/D及三个chunk存在。8→16参考变化约为4→8的一半；有限映射对更细参考的差异并未缩小。它支持局部有限映射与模型自身对角轨迹不一致，不能证明模型数学实现错误或完整画质修复。', '',
    '**反向证据必须保留**：六个局部case中，正常finite端点都比diagonal16更接近Original教师伪标签的latent。不能把diagonal16称为更准确的GT。之前r=t真实生成的重影较少，但动作分离也较低，辅助监督是否有用仍未知。', '',
    '训练日志实际512个逻辑样本中，256 diffusion/128 endpoint/128 general map。sigma≤0.240781分别只有6/2/3个；sigma≤0.425287分别14/5/8个。低噪声覆盖稀少是有限预算训练的可检验因素，不是已证实的唯一根因。', '',
    '下一步限定为同128初始化/Adam/RNG、相同8次额外更新的两组：原AnyFlow loss控制，与额外0.25权重的末区间一致性辅助。架构、原loss、anchor、推理4/8网格不变；辅助参考是冻结128的16细分结果，清楚标注为官方loss以外的实验。正常finite-map完整A/D视频决定去留，不用内部loss替代画质/动作验收。', '',
    '首块/clean-history的局部问题与generated-history累积应分别判断；这项数值诊断不检验Stage2的效果。']
(OUT/'RESULTS.md').write_text('\n'.join(rows)+'\n')
print(json.dumps(analysis['training_time_coverage']))
