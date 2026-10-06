# 主展示选片审计：为什么旧视频会崩

## 事件

会议包第一版把 `H3-World/outputs/2026-10-03-02/final_fixed_mix124_8step/` 的 W/S/A/D grid 作为主展示。它覆盖四个动作，且 MP4 能被 PyAV 完整解码，因此当时被错误地当成了“正式结果”。完整解码只检查了容器、H.264、YUV420P、帧数和 fps，不能证明长时视觉稳定性。

## 旧协议

| 项目 | 旧 fixed-mix 主视频 |
|---|---|
| visual causal adapter | 无（`causal_adapter=null`） |
| action adapter | `fixed_mix_0.5/action_adapter.pt` |
| anchor | `dynamic_last_frame_dual` |
| anchor protocol | `global_retimed_latent_dual_v1` |
| history | generated |
| action prefix | `own` |
| action feedback | false |
| solver | 8 steps/chunk，5 latent frames/chunk，5-chunk history |

这套协议不是“只差一个开关”：它没有后续的 visual tail16 QKV adaptation，且 latent-only inference anchor 与后续训练的 RGB prefix semantics 不一致。generated-history rollout 会把这种误差逐 chunk 累积成 temporal-VAE ghosting、人物透明和场景 tearing。

## 当前替换的主协议

| 项目 | RGB visual-main |
|---|---|
| visual causal adapter | `visual_online_rgb_tail16_endpoint_ad2/causal_adapter.pt` |
| action adapter | 同一 run 的 `action_adapter.pt` |
| anchor | `dynamic_last_frame_rgb_dual` |
| anchor protocol | `global_retimed_rgb_prefix_last_image_dual_v2` |
| history | generated |
| action prefix | `causal` |
| action feedback | true |
| solver | 8 steps/chunk，5 latent frames/chunk，5-chunk history |

当前协议在 124 帧末尾保留了更完整的人物和车库结构，故被选作会议主视觉 demo。它仍有 blur/ghosting，且 A/D flow gate 失败；它不能被写成“动作完整保真”。

## 责任边界

这是一次选片和标注错误：我把覆盖四个方向的旧实验误提升为主视觉结果，并把“可播放”当成了“视觉合格”。修正后，旧文件和旧指标仍保留在本目录供审计，新的主入口统一使用 `meeting/annotated/h3world_rgb_stable_*` 和 `meeting/source_metrics/rgb_visual/`。

由于当前修复同时改变了 visual adapter、anchor、routing 和 feedback，不能从现有材料断言某个单项配置独自造成全部改善。可以确认的是：旧视频的问题不是 MP4 播放器或编码器导致，RGB-consistent protocol + visual adaptation 的组合显著降低了后段结构分解；action geometry 仍需独立解决。
