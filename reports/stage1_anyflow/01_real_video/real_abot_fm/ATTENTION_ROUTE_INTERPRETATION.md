# “动作行对齐且 mask 符合设计”并不表示保留了原始信息流

固定 history/current noisy state 的 A/D 反事实已证实：当前动作确实能够干预输出，但 student 与 Original 的动作差分方向弱相关。下面将“边存在”和“整体动作表示相同”分开。

使用两个真实验证场景、三个 chunk 的实际 packed layout，在 CPU 调用部署 `ChunkAttention.attend`，拦截其两个 SDPA boolean masks；与统一 SDPA Original directed action predicate 比较。Q/K/V 为零占位，仅检查可见性，**不是新的真实 H3 前向，也不测边的贡献大小**。所有六个 layout 的结论一致。

| Query 读取 Key | Original directed H3 | 当前 causal + feedback | 含义 |
|---|---|---|---|
| 当前视频 latent → 自己绑定的 action rows | 允许 | 允许 | 当前动作的直接入口存在 |
| 当前 action rows → 自己绑定的视频 latent | 允许 | 允许 | 当前块动作反馈入口存在 |
| 当前视频 latent → 更早帧的 action rows | 直接禁止 | 允许 | `action_prefix_mode=causal` 额外开放了过去动作的直接入口 |
| 通用 prefix（图像/场景文字/音频行）→ 当前视频 | 允许 | 禁止 | causal prefix 的状态更新依赖图发生变化，首块也如此 |
| 过去 action rows → 自己绑定的历史视频 | 允许 | 当前 prefix 前向禁止 | teacher 重算的历史动作表示与 cached student 的当前 prefix 不同 |
| 当前视频 → 历史视频 K/V | 允许 | 允许 | 历史仍可用，但 teacher 重算历史状态，student 复用 clean KV |

因此，首块尚无历史 KV 时也出现低 A/D-delta cosine，并不自相矛盾：首块的通用 prefix 回读视频、视频直接读取过去 action 的规则就已经不同。历史块出现后，还增加了历史 action 是否重新接收对应视频状态、历史 K/V 是否重算的差别。

这些变化符合现有 causal 实现，但不能称为 Original attention 仅删掉未来块后的完全等价结构。原始 H3 的逐 latent 直接绑定让其它帧获取某动作时须经过其绑定视频状态；`causal` action visibility 开放了额外直接路径。另一方面，完全照抄 Original 的通用 prefix 双向反馈可能把未来视频信息带回过去块，不能未经泄漏检查就直接恢复。

**这份检查没有证明上述哪条变化造成了实际失败。** 旧 `own/causal/all` 和 feedback 诊断有各自的权重、状态和条件，不能把旧差分幅度直接拼成本次真实数据的单边因果消融。现阶段保持已启动 FM48 的协议不变；先收齐同状态训练前后 geometry 与完整视频，再决定是否需要一条改变单一依赖边、并保留无未来泄漏约束的消融。

对应源码：

- `H3-World/code/causal/h3_cached.py`：`ChunkAttention.masks`、`ChunkAttention.attend`。
- `H3-World/DiffSynth-Studio-h3-v2/diffsynth/models/minimax_h3_dit.py`：`_build_action_block_masks` 的 directed predicate。
- 本目录 `geometry_primitives.py`：统一 SDPA Original / full-prefix causal predicates；它明确不将完整历史重算冒充 persistent-KV。
- [本次六布局 CPU 原始记录](attention_route_inventory.json)、[检查脚本](inspect_attention_routes.py)。真实 33B 的时间索引、0/25/49层实际 mask、未来内容负对照和 KV 哈希检查在此前 generated_action_geometry128 报告中另行保留。
