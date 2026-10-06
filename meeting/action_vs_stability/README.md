# 同时展示动作响应与视觉稳定性的取舍

这个目录并列保留两套已有 causal checkpoint。它们是两个不同的协议，不是同一模型的随机波动，也不是单变量 ablation。

推荐先播放 `original_action_stronger_visual_stable_AD_124.mp4`（A/D 两行三列），再按需播放 `original_action_stronger_visual_stable_grid_124.mp4`（W/S/A/D 四行三列）。每行依次为原始 H3、旧 fixed-mix causal、当前 RGB visual causal。全部是 124 帧 / 24 fps / 5.167 秒，保留完整后段，不裁掉漂移。

| 项目 | 原始 H3 | 动作响应较强的旧 causal | 当前视觉稳定 causal |
|---|---|---|---|
| 模型/adapter | released H3-World | fixed_mix_0.5 action residual | tail16 RGB visual QKV + action residual |
| visual tail16 adapter | 无额外 causal adapter | 无 | 有 |
| anchor | 固定首帧 | latent dual | RGB decode/re-encode dual |
| action prefix / feedback | 原始 directed routing | own / false | causal / true |
| history | 双向完整 horizon | generated + CPU raw KV | generated + CPU raw KV |
| noisy calls | 30 full-horizon | 64 chunk + 8 clean commits | 64 chunk + 8 clean commits |
| flow(A) | +1.0767 | +0.1427 | -0.7841 |
| flow(D) | -1.6019 | -0.3105 | -1.0075 |
| A-D separation | 2.6786 | 0.4532 | 0.2233 |
| A/D 方向符号 | 正确 | 正确，但幅度弱 | A 的符号错误 |
| 本组 124 帧视觉观察 | 参考 | 后段 tearing / ghosting / 场景漂移 | 结构较完整，但模糊、动作同向 |

旧版的“动作较好”是相对于当前稳定版而言：它保持 A 正、D 负，并非恢复了原始 H3 的完整控制力。旧版 separation 约为原始 H3 的 16.9%，稳定版约为 8.3%；光流是在 416×240 下中央裁剪区域计算的平均横向位移；这不是 action accuracy。W/S 的水平光流无法严格判定前后移动，不作方向验收。旧片的撕裂与漂移也可能影响光流，因此“较强动作响应”必须连同视频一起解读，不能仅凭正确符号认定角色控制完全正确。

## 保留的 checkpoint 和出处

- 旧 action residual：[legacy_fixed_mix](../../checkpoints/legacy_fixed_mix/)。
- 当前 RGB adapter：[visual_rgb_tail16](../../checkpoints/visual_rgb_tail16/)。
- 旧视频与指标：[`../diagnostics/legacy_fixed_mix/`](../diagnostics/legacy_fixed_mix/)。
- 稳定视频与指标：[`../annotated/`](../annotated/)、[`../source_metrics/rgb_visual/`](../source_metrics/rgb_visual/)。
- 新三列视频的原始输入和 label：`grid_spec.json`、`AD_spec.json`。

在源项目根目录重建：

```bash
python submission/meeting/render_comparison.py \
  --spec submission/meeting/action_vs_stability/grid_spec.json \
  --source-root . \
  --output submission/meeting/action_vs_stability/original_action_stronger_visual_stable_grid_124.mp4
```

原始输入来自 `H3-World/outputs/2026-10-01-20/`、`2026-10-03-02/final_fixed_mix124_8step/`、`2026-10-06-09/visual_online_rgb_tail16_endpoint_ad2/eval124/`。重新渲染只加标签、并排和编码，不改变播放速度或插值。
