"""Build the E2 decision report only after complete video and static review."""
from datetime import datetime
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
metrics = json.loads((BASE/'review_step4/metrics.json').read_text())
review = json.loads((BASE/'review_step4/visual_review.json').read_text())
fit = json.loads((BASE/'heldout_fit/summary.json').read_text())
assert not metrics['pending'] and len(metrics['metrics']) == 18
assert review['status'] == 'complete_static_review' and len(review['cases']) == 6
assert not review['stage1_accepted']
roles = ['zero', 'fm_only', 'fm_action']
rows = metrics['metrics']
parking = []
for history in ('A', 'D'):
    for role in roles:
        a = next(r for r in rows if r['case'] == f'parking_history{history}_currentA' and r['role'] == role)
        d = next(r for r in rows if r['case'] == f'parking_history{history}_currentD' and r['role'] == role)
        parking.append(dict(history=history, role=role, flow_A=a['horizontal_flow_mean'],
                            flow_D=d['horizontal_flow_mean'], separation=a['horizontal_flow_mean']-d['horizontal_flow_mean']))
costs = []
for arm in roles[1:]:
    for suite in ('parking_A', 'parking_D', 'gt'):
        receipt = json.loads((BASE/f'eval_{arm}_{suite}/evaluation.json').read_text())
        assert receipt['status'] == 'complete_pending_visual_review'
        costs.append(dict(arm=arm, suite=suite, wall_seconds=receipt['wall_seconds'],
                          sampling_seconds=[r['sampling_seconds'] for r in receipt['records']],
                          GPU_peak_MiB=receipt['GPU_peak_MiB'], CPU_hidden_KV_MiB=0,
                          noisy_forwards=receipt['sampling_forwards'], identity_forwards=receipt['diagnostic_forwards']))
decision = dict(at=datetime.now().astimezone().isoformat(), status='complete4_no_go_for_promotion',
                stage1_accepted=False, AnyFlow_or_Stage2_started=False,
                automatic_training_extension=False, parking=parking, costs=costs,
                diagnosis='No consistent added action/structure benefit from four-update ranking loss over matched FM-only. Limits do not establish impossibility of consequence supervision or a unique cause of artifacts.',
                next='Audit reliable action-outcome supervision and effect timing before another controlled E2 run; do not tune this validation set or extend the current run automatically.')
(BASE/'decision.json').write_text(json.dumps(decision, indent=2)+'\n')
lines = ['# E2 真实动作后果监督：4-update受控结果\n\n',
         '**本轮不通过局部动作＋画面联合验收；不扩至16更新，不进入AnyFlow/Stage2。** 两臂完成固定4更新及全部预登记评测。停车场两份历史均保留A正/D负，但A分支重影仍在，FM+action没有一致优于FM-only。四步小试不能证明动作后果监督不可行，也没有识别人物重影的唯一成因。\n\n',
         '## 可播放三列对比\n\n',
         '列顺序：**Original权重下的局部N / FM-only4 / FM+action4**。左列也是局部T2/N推理，不能标成Original全长双向生成。全部30steps/current window，832×480源片；展示缩为每列416×240，24fps。三列视频不含GT上下文、不补帧、不平滑。\n\n']
for history in ('A', 'D'):
    for action in ('A', 'D'):
        name = f'parking_history{history}_current{action}'
        lines.append(f'- [固定{history}历史 → 当前{action}](review_step4/{name}_comparison.mp4)：42当前RGB，全球索引39–80；历史来自Original生成。\n')
for name, label in [('118eb5d8b75e1b8ac23a4e9ae77af9a9_D_1015', '真实GT-history：W+A+J及观测F'),
                    ('dfec8ed3237860eba14d67c089ecd041_A_1080', '真实GT-history：D+L及观测F')]:
    lines.append(f'- [{label}](review_step4/{name}_comparison.mp4)：39当前RGB，索引81–119；文件名动作是旧首窗标签，以此处当前条件为准。\n')
lines += ['\n这是固定reference/GT history上的局部测试，**不是124f自由生成**。两份GT例子含相机联合控制，F由当前观察速度派生；它们是oracle条件结构检查，不证明实时交互时可取得该速度。停车场纯A/D没有F。\n\n',
          '## 同状态动作响应\n\n',
          '| 固定历史 | 方法 | A flow | D flow | A−D |\n|---|---|---:|---:|---:|\n']
for r in parking:
    lines.append(f"| {r['history']} | {r['role']} | {r['flow_A']:+.6f} | {r['flow_D']:+.6f} | {r['separation']:.6f} |\n")
lines += ['\nFarneback沿用416×240中心80%的相邻帧平均水平flow；单位px/frame。两个history分别评价，不与旧full-horizon teacher的2.023直接比较。符号与separation不是画质分数。\n\n',
          'D-history的分离度从1.287926变为FM-only1.366635、FM+action1.274145；A-history则由3.192228降到两臂约3.05。动作项没有一致增益。人物方面，D-history/currentA约RGB55–68仍多重手臂/躯干残影；A-history/currentA也仍有肢体残影。当前D的两组结构相对完整。\n\n',
          '## 真实GT局部结构与完整帧检查\n\n',
          '| GT场景 | 方法 | Frame MAD | 首边界MAD |\n|---|---|---:|---:|\n']
for r in rows:
    if not r['case'].startswith('parking_'):
        lines.append(f"| {r['case'][:6]} | {r['role']} | {r['frame_MAD_MP4']:.4f} | {r['boundary_MAD_MP4']:.4f} |\n")
lines += ['\n所有MAD均从同编码H264 MP4重算。Frame MAD是活动量；GT首边界为当前第一帧与同一rawGT RGB80的灰度差。旧运行收据中的未压缩/decoded-GT MAD定义不同，不混算。没有报告FVD、LPIPS、GT PSNR或warmup均值。\n\n']
for r in review['cases']:
    lines.append(f"- **{r['name']}**：{r['observation']}\n")
lines += ['\n评审覆盖全部738个当前RGB（三种参数bank、六个局部case），包括全帧sheet、预先固定位置的原尺寸人物crop及历史边界；属于静态逐帧检查，不声称实时播放评审。[完整记录](review_step4/visual_review.json) · [全指标](review_step4/METRICS.md)。\n\n',
          '## 正确动作绝对拟合与错误动作排序\n\n',
          '| 参数bank | 正确动作FM均值 | 交换动作FM均值 | gap | 正确排序状态 |\n|---|---:|---:|---:|---:|\n']
for r in fit['summary']:
    if r['sigma'] is None:
        lines.append(f"| {r['role']} | {r['positive_FM']:.8f} | {r['negative_FM']:.8f} | {r['gap']:+.8f} | {r['correct_order']}/{r['n']} |\n")
lines += ['\n正例均值改善不足0.03%，FM+action不比FM-only更好。两验证状态×四sigma不是八个独立视频样本；没有用此验证结果调lambda/margin/noise分布。[分sigma完整表](heldout_fit/RESULTS.md)。错误动作没有反事实真实视频，排序不能当方向真值。\n\n',
          '## 计算与可复核性\n\n',
          '| 方法 | 评测组（每组两条） | 整组wall s | 每条采样 s | 峰值allocated GiB | CPU hidden KV | noisy/identity forwards |\n|---|---|---:|---|---:|---:|---:|\n']
for r in costs:
    times = '/'.join(f'{x:.2f}' for x in r['sampling_seconds'])
    lines.append(f"| {r['arm']} | {r['suite']} | {r['wall_seconds']:.2f} | {times} | {r['GPU_peak_MiB']/1024:.2f} | 0 | {r['noisy_forwards']}/{r['identity_forwards']} |\n")
lines += ['\n这是并行共享主机上的单次局部作业成本，整组含加载/两视频/解码/诊断；不是整视频warmup后延迟，也不提供speedup。T2逐sigma重算可见hidden，CPU hiddenKV为0但仍存放raw history/noise。权重、临时KV、激活的峰值分解没有独立测量，不能相减推算。\n\n',
          '两臂训练各4更新、logical batch2、released action LoRA tail8 QKV/out共10,092,544参数，LR2e-5；只有objective不同。训练wall284.76/416.62秒，峰值allocated均27.91GiB。初始参数bank完全一致、更新有限非零、梯度replay误差0。六组视频合计360采样＋12identity forward，held-out另48forward。Original→step0输出identity均0，停车场旧N场重放均0；冻结backbone参数版本保持，307文件runtime核验。\n\n',
          '## 决策与下一步\n\n',
          '本轮停止在4更新。它支持“当前小样本动作排序项尚未显示额外收益”，不支持“所有动作监督都无效”，也不支持用Stage2替代局部能力问题。\n\n',
          '[监督覆盖审计](SUPERVISION_COVERAGE.md)发现本次8个microbatch无纯A/D，且3个混有相机控制。CPU扫描训练原片后找到A/D各一段更纯候选，原片人物完整，但A存在待核查的输入到运动时序；仅完成源数据静态评审，没有训练准入、VAE编码或新optimizer。先核对可靠动作后果与时序、同状态配对来源，再冻结一个受控E2修订；不在当前验证集上扫loss系数或平移标签。\n\n',
          '阶段顺序保持：可信causal30 → AnyFlow4/8 → on-policy Stage2。局部生成仍未联合通过，会议主demo不以本轮短片替换。\n']
(BASE/'FINAL_RESULTS.md').write_text(''.join(lines))
print(json.dumps({'status': decision['status'], 'stage1_accepted': False, 'video_jobs': len(costs)}))
