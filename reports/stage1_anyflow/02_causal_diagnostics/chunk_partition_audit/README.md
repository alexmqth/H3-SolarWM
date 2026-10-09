# Chunk partition 审计：配对已实现，VAE 相位值得对照，但不是已证实根因

2026-10-09。结论基于本项目实际 DiffSynth/H3 实现，而非只根据长度公式推断：**没有发现旧版把 Action–Video pair 按物理 packed token 数切坏；发现不同推理协议确实改变了原始信息流，且 VAE 的 prefix 输出需要区分“能解码”与“不会被未来 latent 改写”。建议的 `[2,5,…]` 不能直接运行：当前 temporal decoder 对仅2 latent 返回 `None`。**

本次只运行CPU结构审计和既有测试，没有加载33B/VAE权重、启动生成或训练、修改推理默认值。研究冻结保持。新证据为 [脚本](audit.py)、[receipt](receipt.json)、[17项CPU测试](cpu_tests.log)。真实VAE像素依赖沿用此前的 [preparation.json](../local_topology/inputs/preparation.json)，没有把本次shape stub当作新的画质实测。

## 1. Action–Video pair 和时间网格：建议正确，但已在代码中实现

- `code/abot/abot_action.py::frame_spans` 使用 `(1,4,4,4,4)` 分配RGB动作；`bin_to_latent`据此聚合按键及连续控制。
- DiffSynth `MiniMaxH3Unit_PackedSequenceBuilder`用同一非均匀网格构建video时间坐标，以及逐latent完整`action_text_rows`。Action span与video的RoPE时间有固定偏移，不要求两者时间坐标数值相等。
- `patchify_video`的patch为`(1,2,2)`；832×480对应latent空间30×52，每个latent frame有390个DiT tokens。本次对完整37帧patchify核对了全部空间行。
- `h3_cached.slice_packed`按`video_start + latent_index*frame_rows`切video，并保留完整prefix。不是按整个packed序列每隔固定token数切分，也不会把一个action句子拆成两半。
- 后来的`local_topology.visible_inputs`在refiner前物理删除未来action/video，保持已知span和全局位置。CPU审计对旧5、旧12和候选A/B/C的所有显式区间验证了完整span、390空间行及global RoPE复制关系。这只验证**切片工具**，不代表生产rollout已支持非均匀chunk。

连续按住A可以跨chunk；一个chunk也可以含多个不同action。要求是已知当前chunk的动作、保留每个latent的逻辑配对，而非把整段相同按键强行放进一个chunk。当前chunk内部允许双向传播，因此其全部动作应在该chunk采样前确定；这不等于逐RGB帧零延迟交互。

## 2. 两种不同的VAE边界，不能只选一种叫“原生”

实际encoder将RGB分成17帧片段，每段产生5 latent，不足17帧先复制末帧补齐；最后删除3 latent (`token_drop=3`)。因此124RGB先补到136RGB，得到40 latent，再截成37。`37=2+7×5`这个等式**不表示encoder有一个特殊的“先编码2 latent”启动块**。

`5+17m ↔ 2+5m`是pipeline的整视频长度约定；`17RGB ↔ 5latent`则是encoder分组主体。首帧图像条件C0另外存在，不能自动把前2个待生成video latent当作已经给定的C0。

| 分块 | 37 latent的长度列表 | 累计RGB边界（跨度分配） | 审计判断 |
|---|---|---|---|
| 旧uniform5 | `5,5,5,5,5,5,5,2` | 17,34,51,68,85,102,119,124 | 前7边界一直是latent模5=0，相位没有逐块漂移；对齐encoder主体，但prefix解码需要尾部padding |
| 旧uniform12 | `12,12,12,1` | 39,81,120,124 | 相位变化；历史E1只评前三窗至120RGB，不能把最后1 latent当作已评 |
| 候选A | `2,5,5,5,5,5,5,5` | 5,22,39,56,73,90,107,124 | 模5=2；首2latent的当前decoder控制流不支持 |
| 候选B | `7,5,5,5,5,5,5` | 22,39,56,73,90,107,124 | 模5=2；尚无该分块的真实生成评测 |
| 候选C | `12,5,5,5,5,5` | 39,56,73,90,107,124 | 模5=2；首12窗口有正控，后续5窗口仍未验证 |

这些是全局时间分配和累计prefix的RGB长度，**不是证明每块可独立解码或每个latent只依赖所分配的RGB**。旧uniform5也保持固定相位，因此“随便每5个切导致相位不断变化”不准确。

## 3. Decoder的5＋2规则是本次关键核查点

`minimax_h3_video_vae.py`的实际配置：`tokens_chunk_size=5`、`token_overlap=2`、`frame_pre_padding=3`、`frame_overlap=5`。第i次神经解码读取`z[5i:5i+7]`，并与上一段融合5RGB。ViT decoder在这个窗口内没有temporal causal mask。

CPU调用了实际`decode_temporal`、frame plan、padding、streaming和blend函数，仅将昂贵的`tiled_decode`替换为正确输出形状的零张量：

| 可用prefix latent | 实际调度 | 输出RGB数量 | 含义 |
|---:|---|---:|---|
| 2 | `num_chunks=0`，无decoder调用 | 无，返回`None` | `decode_video`随后访问`recon.float()`会失败；A方案不能照搬 |
| 5 | 复制末latent补2，解码7 | 17 | 旧版可以运行，不是非法长度；末端用了人工补齐 |
| 7 | 无padding，解码7 | 22 | 有完整第一个7-latent窗口，但末5RGB仍是overlap输出 |
| 10 | 尾部补2，两次解码7 | 34 | 同旧版第二边界 |
| 12 | 无padding，两次解码7 | 39 | 最后5RGB仍会参与下一窗口融合 |
| 37 | 无padding，七次解码7 | 124 | 完整视频结束时输出最后5RGB |

1-latent temporal路径还会触发frame-plan错误；这与独立`process_image=True`图像解码不是一条路径。本次没有修这个边界条件，也没有声称所有H3实现都具有同样限制。

**prefix decode不使用尚未生成的latent，并不保证与未来完整decode的同一RGB前缀一致。** 即使是7/12/17…，当前末5RGB也可能在之后被重算或混合。由窗口依赖可推导，7latent可先稳定提交前17RGB、12latent可先提交前34RGB，末5RGB等待后续或在视频结束时flush；这是待真实VAE逐像素验证的输出协议，尚未实现成新的流式播放器，也不是模型生成质量结论。若要严格符合已显示画面不可更改，应明确选择延迟提交，或沿用prefix-only且冻结已显示RGB，并报告边缘差异。

此前真实VAE审计已经证实：改变latent5及以后，前17RGB仍发生变化，A/D两参考的mean absolute差为0.00727/0.00705（0–1RGB）。反转RGB17/34之后的像素，过去5/10latent的最大变化为0（两真实ABot片段）。它支持encoder前缀因果性，同时否定“完整decode自动就是在线因果输出”。后来的局部T2视频已经改用prefix-only追加；旧benchmark在整段rollout结束后full decode，不能将两者混称为已验证的在线显示。

## 4. Pair对齐不等于保留Original信息流

Original directed single-egress只允许video帧k直接读取自己的action span k，action还可读取自己的video；动作可经video-video边间接传播。

实际persistent-KV有多个版本，不能说“所有mask都正确保留Original”：

- 早期`action_feedback=False`删除了action读取所属video的反馈；后来`True`只恢复当前chunk自己的对应video。
- `action_prefix_mode='own'`保留直接own-action规则；`'causal'`允许video帧k直接读取所有action≤k，**改变了Original的single-egress拓扑**，即使索引完全正确。CPU实际predicate确认video5在own下只读action5，在causal下直接读action0–5。
- 基础cached实现的静态prefix不读video；历史hidden KV在clean commit后固定。这也与Original双向重算不同。T1/T2研究就是为检查这些变化；T2每sigma重算可见窗口，不应宣称其复用了persistent hidden KV。
- Legacy cached保留所有action行，需靠隔离refiner和mask控制可见性。未来action文本**长度**还能改变native packed RoPE原点；`position_contract.py`已有独立修复及测试，但不是benchmark自动开启的默认值。后来的固定布局/物理删除协议规避了此问题；固定等长A/D的无泄漏证据不能推广到任意变长未来动作。
- cached模式保留全长audio noise作为固定prefix；它不来自未来真实音频，但本项目是silent-video诊断，不是已验收的AV同步流式模型。Original联合audio/video去噪与cached固定audio也是既有协议差异。

本次原有`test_local_topology.py`、`test_h3_cached.py`、`test_action_position_contract.py`共17项通过，覆盖小H3的相关mask、未来干预、KV重放和位置契约。它证明这些明确条件下的行为，不证明预训练33B在新拓扑下仍有正确动作语义。

## 5. 为什么不能把失败统一归因于5-latent边界

已有 [首窗口对照](../coarse_window12/FIRST_WINDOW_RESULTS.md)在同full37布局/noise下：首5latent的A/D flow约−1.169/−1.174；首12latent约+0.863/−1.174。更长首窗口改善了正控，但同时增加了可见上下文与控制延迟，不能将收益唯一归因于VAE相位。

已有 [解码前同状态动作探针](../generated_action_geometry128/INTERPRETATION.md)中，AnyFlow128的student/teacher动作差分平均余弦0.0382；检查未发现该固定A/D布局的action错位、内容泄漏或KV污染。这一差异发生在VAE decode之前，说明decoder输出伪影不足以解释全部问题。该探针也不能唯一归因于权重：teacher重算history与student固定KV仍是结构差异。

## 6. 若恢复研究，先把生成分块与显示提交拆开

1. **仅VAE对照**：用保存的同一完整latent，比较5/7/10/12等prefix与完整decode；干预未来，定位可安全提交的RGB。保持DiT、权重、动作不动，单独测padding/overlap，先验证上述延迟提交推导。
2. **显式区间接口**：统一记录`(start,stop)`和cache的全局frame范围，再接A/B/C。目前`chunk_forward`、T1的历史索引、T2等都使用`index*chunk_frames`；直接修改循环长度会错配全局action/RoPE。cache eviction、训练窗口、replay、动作schedule与边界指标需共用同一partition。不是简单改CLI里的chunk size。
3. **有限Original30步正控**：若前两项通过，优先考虑已有首12正控的C，再研究B；A需先决定首2的解码策略，且更短首窗没有动作收益证据。保留旧uniform5作控制；同一全局layout、首图、noise及明确固定history来源，对每个边界做同状态当前A/D分叉。不同partition的history长度和响应延迟不同，不能称为只改变“相位”的严格单变量试验。
4. **多个后续chunk同时验动作＋人物结构**，不能只凭首chunk好看通过。先做局部充分采样正控，再决定generated-history、AnyFlow或Stage2。当前不启动这些实验。

## CPU复现

在已经配置好的H3环境中，从项目根目录运行（依赖已有patched DiffSynth，不下载权重）：

```bash
CUDA_VISIBLE_DEVICES='' python submission/reports/stage1_anyflow/02_causal_diagnostics/chunk_partition_audit/audit.py --runtime H3-World --output /tmp/h3_chunk_audit.json
CUDA_VISIBLE_DEVICES='' PYTHONPATH="$PWD/H3-World/DiffSynth-Studio-h3-v2" python -m pytest submission/tests/test_local_topology.py submission/tests/test_h3_cached.py submission/tests/test_action_position_contract.py -q
```

CPU shape审计不生成视频，不修复2-latent decoder，不替代真实权重的动作和视觉评测。原视频、checkpoint和冻结实验收据保持原样。
