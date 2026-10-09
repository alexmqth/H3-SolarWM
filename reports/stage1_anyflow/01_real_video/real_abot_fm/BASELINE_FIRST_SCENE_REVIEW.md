# 第一真实验证场景：Original、未训练causal与两种历史

这份记录针对零更新baseline，不能当作正在训练的48-update模型效果。片段为独立验证episode `118eb5d8b75e1b8ac23a4e9ae77af9a9_A_1140`，39帧、24fps、832×480；包含forward＋strafe left＋camera pan left。两条causal均使用普通FM 30steps/chunk、5+5+2 latent chunks、CPU raw KV、RGB dual anchor，没有AnyFlow、action residual或旧visual adapter。

| 方法 | noisy / clean前向 | 模型加载后推理含解码/编码 (s) | GPU peak MiB | CPU KV MiB | frame MAD | 指定17/34帧的RGB边界差均值 | 水平光流均值 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 真实RGB | — | — | — | — | 27.104 | 27.079 | +3.192 |
| Original30，整段 | 30 / 0 | 227.67 | 25928.91 | 0 | 19.052 | 18.754 | +2.289 |
| Causal0，GT history | 90 / 3 | 533.97 | 26002.10 | 6484.13 | 17.143 | 47.278 | +0.886 |
| Causal0，generated history | 90 / 3 | 576.63 | 26002.32 | 6484.13 | 9.845 | 17.100 | +0.467 |

这都是单次共享主机运行、已缓存conditioning，未作warmup多次均值。30/chunk=90 noisy forward，不能与Original30按相同步数口径宣称加速。first_chunk_seconds表示latent chunk生成/commit用时，不等于最终RGB首帧的实测显示延迟。

## 全39帧静态检查

已检查Original及两条causal的全部39帧contact sheet，并对照0/8/16/24/30/38帧。人物都保留到最后，未见旧8步诊断中的严重分解/全面雾化。未训练causal首chunk已改变场景/相机轨迹；generated-history分支后段运动偏弱、建筑细节漂移，不能称为动作正确或长期稳定。

GT-history分支在17/34附近有明显位置/场景跳变。它每个chunk接受真实历史，最终视频拼接各自的预测chunk；上一预测终点与下一真实历史并非同一状态。因此这条oracle-history视频的边界尖峰，不直接等同于自主rollout崩溃，也不能用它的边界指标代替generated-history指标。H3 temporal VAE还包含跨chunk重叠解码；局部latent能力与拼接视频连续性分开报告。

这里的“无严重分解”只是39帧、30steps/chunk和这个场景的静态观察，不外推8步、124帧或20秒；未进行人工实时播放。

## 动作解读

自然片段有联合按键和镜头转动，水平flow同时受它们影响。Causal的总flow比Original弱，是待解释的响应差异，不能当作纯A正确率或A-D separation。严格纯A/D同首帧的Original参照已另起队列，后续与训练完成模型比较。

真实GT状态上的当前chunk反事实也单独完成18点。零更新causal的整体velocity平均cos0.988193，A/D差分平均cos0.017516；无历史chunk0平均delta cos0.034156。历史KV对当前A/D重建相同、读期间不变，重复前向误差0。这说明差分失配已存在于GT历史/无历史条件，不能全部归因为generated-history漂移；低cosine本身仍不自动判定视频动作错误。

原始receipt：`eval/original/generated_30/`、`eval/step_00/gt_30/`、`eval/step_00/generated_30/`。
完整并排视频/contact sheets：`report/baseline_partial_gt30/`、`report/baseline_partial_generated30/`。
