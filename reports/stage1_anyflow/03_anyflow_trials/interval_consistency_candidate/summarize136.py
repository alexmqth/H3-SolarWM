"""Consolidate completed finite-budget forks; visual review is an explicit input."""
from datetime import datetime
import json
from pathlib import Path

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
REPORT=OUT/'report'
SOURCE=ROOT/'outputs/2026-10-08-13/stage1_parallel_resume68_to128'
reviews=json.loads((REPORT/'visual_review.json').read_text())
assert reviews['scope']=='Static all39-frame and matched-frame review, not real-time playback'
results=[];training={}
for variant in ('control','auxiliary'):
    run=json.loads((OUT/f'run_{variant}.json').read_text())
    assert run['status']=='complete' and len(run['evaluations'])==4
    train=json.loads((OUT/variant/'train_136/training.json').read_text())
    assert train['status']=='complete' and len(train['updates'])==136
    updates=[u for u in train['updates'] if u['step']>128]
    assert len(updates)==8 and train['frozen_visual_unchanged'] and not train['target_time_parameters_changed']
    training[variant]=dict(new_updates=8,wall_seconds=train['wall_seconds'],
        update_seconds=sum(u['seconds'] for u in updates),gpu_allocated_peak_MiB=train['gpu_allocated_peak_MiB'],
        current_forwards=sum(u['model_evaluations'] for u in updates),
        physical_clean_forwards=sum(u['clean_commit_forwards'] for u in updates),
        differentiable_history_forwards=sum(u['differentiable_history_forwards'] for u in updates),
        counting_note='Excludes validation and activation-checkpoint backward recomputation')
    for e in run['evaluations']:
        key=f"{variant}_{e['steps_per_chunk']}_{e['action']}"
        assert key in reviews['clips']
        assert e['input_fairness']['exactly_equal'] and e['runtime']['sampler']=='anyflow_finite_map'
        results.append(dict(**e,visual_review=reviews['clips'][key]))
audit=json.loads((OUT/'sample_audit.json').read_text());assert audit['complete_pair']
separation=[]
for nfe in (8,4):
    for variant in ('control','auxiliary'):
        rows=[x for x in results if x['steps_per_chunk']==nfe and x['variant']==variant]
        a=next(r for r in rows if r['action']=='A')['flow']['horizontal_flow_px']['mean']
        d=next(r for r in rows if r['action']=='D')['flow']['horizontal_flow_px']['mean']
        separation.append(dict(variant=variant,steps_per_chunk=nfe,A=a,D=d,separation=a-d,
            numeric_action_gate=a>0 and d<0 and a-d>1.,
            visuals_accepted=all(r['visual_review']['accepted'] for r in rows)))
data=dict(completed_at=datetime.now().astimezone().isoformat(),training=training,
    action_summary=separation,evaluations=results,review_scope=reviews['scope'],
    maximum_optimizer_step=136,stage2_started=False,
    limitations='Single seed, shared-host single timings, pseudo-GT local targets. No official-scale reproduction.')
(REPORT/'final_results.json').write_text(json.dumps(data,indent=2)+'\n')
lines=['# 136次有限预算对照：完整结果','',
    '[8步/chunk完整四列A/D视频](original128_control_auxiliary136_8step_AD.mp4) · [4步/chunk完整四列A/D视频](original128_control_auxiliary136_4step_AD.mp4)','',
    '每个视频包含Original H3 / AnyFlow128 / 原loss136 / +一致性136，保留全部39帧，不剪除18–38帧失败段。两组从同128权重/Adam/RNG各新增8次更新，架构、数据、原loss和推理设置固定。辅助项每更新增加一次finite forward及其history反传，因此等更新不等算力。','',
    '| checkpoint | steps/chunk | A flow | D flow | A−D | 数值动作门槛 | 静态视觉检查 |',
    '|---|---:|---:|---:|---:|---|---|']
for r in separation:
    lines.append(f"| {r['variant']}136 | {r['steps_per_chunk']} | {r['A']:+.6f} | {r['D']:+.6f} | {r['separation']:.6f} | {'PASS' if r['numeric_action_gate'] else 'FAIL'} | {'可接受' if r['visuals_accepted'] else '未通过'} |")
lines+=['','128参考：8步+0.040414/−0.719245，分离度0.759660；4步−0.751431/−0.885575，分离度0.134144。Original30步+1.181253/−0.842124，分离度2.023377。Farneback是运动响应proxy，不是严格的动作准确率。','',
    '| variant | action | steps/chunk | E2E s | allocated peak MiB | CPU KV MiB | noisy+clean | gray MAD | boundary RGB MAD |',
    '|---|---|---:|---:|---:|---:|---|---:|---:|']
for e in results:
    r=e['runtime']
    lines.append(f"| {e['variant']} | {e['action']} | {e['steps_per_chunk']} | {r['wall_including_shared_setup_seconds']:.2f} | {r['memory_entire_run_peak_MiB']:.2f} | {r['kv_cache_peak_MiB']:.2f} | {r['denoiser_forwards']}+{r['commit_forwards']} | {e['video_stats']['gray_pixel_difference_mean']:.4f} | {e['boundary']['mean']:.4f} |")
lines+=['','两边输入与Original逐张量一致、832×480/39f/24fps/seed13。每条8步为24 noisy+3 commits，4步为12+3；不是用8/4次forward生成全39帧。MAD仅活动量，雾化可能降低帧差；不作画质分数。无warmup重复计时和权重/激活精确拆分，不能据单次共享主机时间声称speedup。','',
    '视觉检查方式：所有0–38帧contact sheets和对应12/24/30/38帧静态查看，非实时播放。逐clip观察：','']
for e in results:
    lines.append(f"- {e['variant']} {e['action']} {e['steps_per_chunk']}步：{e['visual_review']['notes']}")
lines+=['','训练预算已到136停止，无新Stage2训练。双局部指标的[独立报告](../dual_metric136/RESULTS.md)同时保留自一致性和teacher伪目标距离；[噪声覆盖检查](../dual_metric136/NOISE_COVERAGE.md)说明后续如何控制采样density与loss权重，不能用局部residual替代以上实际视频。']
(REPORT/'FINAL_RESULTS.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(dict(action_summary=separation,training=training),indent=2))
