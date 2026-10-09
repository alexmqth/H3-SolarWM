# 真实验证场景：同状态当前chunk A/D动作差分

Checkpoint: **FM step 0**. History/state source: **gt**. 2 held-out scenes × 3 chunks × 3 sigmas = 18 cases.

整体velocity平均cosine **0.988193**；A/D差分平均cosine **0.017516**，范围 -0.018145–0.073592。Student/teacher动作差分范数比 0.315–1.364。

| chunk | 整体velocity平均cos | A/D差分平均cos |
|---|---:|---:|
| 0 | 0.988610 | 0.034156 |
| 1 | 0.988204 | 0.013026 |
| 2 | 0.987764 | 0.005365 |

## 控制与解释边界

- 每个状态下历史与noisy current固定，仅改当前chunk A/D；过去/未来动作embedding复用，A/D paired layout完全一致。原始自然动作句长度可与替换句不同，因此全部历史KV按此次layout重建，没有导入别的布局/模型的KV。
- 每个chunk的历史用自己对应的RGB dual anchor在sigma=0提交。真实33B上检查chunk1的A/D历史重建KV哈希相同；每次当前干预期间KV内容/commit及参数version不变。
- Student是部署cached路径；Original权重teacher在相同条件下双向重算历史，使用原始directed action predicate的SDPA，与student统一后端。两者历史内部依赖图不相同。
- 状态为固定endpoint的显式加噪插值，不声称是实际solver中间状态。FM瞬时速度对比，不含AnyFlow finite-map语义。
- 低cosine是动作条件场差分不一致的诊断，不自动等同于视频方向错误或画质失败。自然片段含联合动作/镜头变化；需另看同首帧纯A/D生成、完整视频及边界情况。
- step00是零更新causal baseline；不能把它当48更新训练效果。GT和固定step00 generated state用于前后匹配；step48 own generated state若后续提供，单独作为自生成分布诊断，不冒充相同输入对照。

## 执行

```json
{
  "model_forwards": 84,
  "clean_forwards": 8,
  "wall_seconds": 784.7798011139967,
  "gpu_allocated_peak_MiB": 16160.4521484375
}
```

GT与step00 generated两套诊断改变了历史、当前endpoint和相应RGB anchor，不是只替换history的单变量消融。每个A/D pair内部保持相同state/history；训练前后才应复用同一套固定states比较。
