# Action Routing / Persistent KV：代码审计与四路对照计划

**2026-10-10更新：已按P0优先完成固定状态实验，[完整结果、失败轮次与源码](execution_20261010/README.md)。6状态×A/D的受控数值KV/velocity严格等价通过；公共prefix、video因果化、提交时间及past-action直读分别测量。未训练或生成新视频。下文保留2026-10-09的审计与预先计划，不将其历史“未执行”状态当作当前状态。**

2026-10-09。**代码和已有证据审计完成；下文新实验均未执行。** 本轮未加载模型、运行神经网络前向、生成视频或训练，也未重跑神经网络测试。AnyFlow/DMD保持暂停。审计起点：`1a0ad4df2cbb42f0ea3a6123219289e944ba7a10`；[源码指纹](source_manifest.json)。旧实验源码、数值和视频不变。

**结论：`causal` action prefix确实改变了Original Single-Egress；原型同时改变了action/video反馈、公共prefix grounding及时间条件。无历史KV的旧实验也出现动作差分失配，因此不能全部归因于KV。尚无足够干净的多chunk四路对照来确定哪项主导实际动作失控。**

## 源码事实

本文“Q(X)读取K/V(Y)”表示信息从Y流向X。i/j是latent时间索引，c(i)才是chunk编号。

Original实际predicate位于工作树 `H3-World/DiffSynth-Studio-h3-v2/diffsynth/models/minimax_h3_dit.py:242–250`，对应包内[H3 action patch](../../../../code/diffsynth_h3_action.patch)。审计按predicate而非函数顶部的旧概括注释判断：

- V_i query只能直接读A_i；A_i query在video keys中只能读V_i。
- Action可以读自身span、公共条件，不能读其他action。
- 公共语义、I0、audio等非action prefix不直接读取action，但能读取video。
- Video/video双向。历史动作可先进入对应V_j，再经以V_j为必经点的间接路径传播。

### 基础缓存路径不是Original信息流的原样保留

[h3_cached.py](../../../../code/causal/h3_cached.py) 的 `ChunkAttention.masks()`（258–288）和 `attend()`（291–357）：

| 读取关系 | Original | `own`基础缓存 | `causal`基础缓存 |
|---|---|---|---|
| V_i读action | 仅A_i | 仅A_i | A_j，j≤i |
| A_i读V_i | 允许 | feedback开启时，仅当前块恢复 | 同左 |
| 旧A_j重新读历史V_j | 重算时允许 | 不允许 | 不允许 |
| 公共prefix读video | 允许 | 禁止 | 禁止 |
| 当前video读历史video | 双向重算后的状态 | 已提交KV | 已提交KV |
| 历史video读当前video | 允许 | 不更新历史 | 不更新历史 |

`causal`使用`ann <= current_frame`，**包括同一chunk内较早latent的actions，不只是更早chunk**。首块没有历史KV时也会偏离Single-Egress。当前API/benchmark默认是`own`，旧实验曾显式设置`causal`，不能按今天的默认值推断所有旧实验。

`action_feedback`默认False；True才恢复当前action query读自身video。仅将`causal`改成`own`并不恢复完整Original图。

**缓存没有action tokens。** Keys的实际拼接是`[fresh prefix, cached historical video, current video]`（299–306），commit只存`k[p:], v[p:], rope[p:]`（355–356）。这些是历史video的逐层pre-RoPE K/V和位置。额外直接读过去action来自**fresh action prefix**，而非缓存里保存的action rows。

禁止直接读past actions后，结构上仍有`A_j → 历史V_j hidden/KV → 当前V_i`。这是间接路径存在的判断，不是动作语义正确的证明；旧“修改历史value能改变输出”也只说明历史被使用。

### 现有recompute接口不能直接用于这次严格对照

`recompute_forward()`（463–504）接收`action_prefix_mode`、`action_cond`、`action_adapter`，但函数体**未使用/传递它们**，也没有`action_feedback`参数。`benchmark.py:549–575`向两入口传参不代表实际生效一致。

它调用vendor `_build_causal_block_masks()`：`static_to_static | video_to_static | causal_video_to_video`先排除全部prefix-query→video-key，然后才与action限制相交。故**连A_i读自身V_i的反馈也被删除**；这不是只切video时间边。

`chunk_forward()`与`recompute_forward()`还都硬编码`fixed_prefix_timesteps=True`。原生model_fn默认False，text/action time为`1−sigma`；True固定为1。因此不能直接运行`cached vs recompute + causal + feedback`后将差额全部解释成KV冻结。这是当前源码确认的接口/对照局限，不使旧own/无feedback/固定时间协议下的特定测试失效。

### T1、T2、最近C12→5的区别

[local_topology.py](../../../../code/causal/local_topology.py)：

- T1 `GroundedPrefixAttention`要求own+feedback，公共prefix能读可见历史/当前video，旧action能读对应cached video；历史video KV不可变。它与基础缓存路径不是同一个图。
- T2 `OriginalWindowAttention.build_mask()`（152–162）使用Original directed predicate，**可见历史和当前video在一次调用内仍双向交互**。固定的是raw历史latent，hidden历史每次重算。
- 最新[C/B interval_forward](../chunk_partition_cb/interval_forward.py)使用T2、native time、Single I0。它在生成窗口间不读未知未来块，但不是“严格chunk-causal video attention＋历史重算”，不能充当本次第2路。
- 缓存`start=index*chunk_frames`、T1历史`entry.index*fixed_chunk_frames`没有通用非均匀分区语义。新诊断须显式对应`[0,12)`和`[12,17)`，不能套`index*5`；C/B显式interval接口目前只有T2无缓存路径。

## KV冻结：执行等价和条件冻结是两件事

对于同一个确定性causal计算图，若历史latent、全祖先、时间、条件、位置、窗口和精度都一致，缓存可以与重算等价。**“使用KV就必然破坏动作”没有依据。**

Original双向模型允许`当前action → 当前video → 历史hidden → 当前输出`，严格chunk-causal图已切断这个回路；不能全记到KV账上。

另外clean commit在sigma0发生，当时native text/action time=1、I0 time=1。当前sigma>0时text/action time=`1−sigma`、I0 time=`max(1−sigma,.999)`。用当前prefix条件重算历史可能不同于commit时历史；同raw history不代表同hidden state。因此必须分开：

1. 按原commit的完整条件重建各层KV，验证执行等价。
2. 按当前native prefix时间重算历史，比较frozen clean KV，测提交上下文冻结的影响。

最新C+N每步又将历史输入改成`(1−sigma)*history+sigma*history_noise`、历史video time=sigma。**N重算与sigma0 clean persistent KV不具有相同历史输入，不能作纯缓存对照。** 四路主体先统一clean history，N作为独立后续因素。

公共prefix不能遗漏：如果所有历史和当前video共享一个读取全video的可变公共prefix，即使V/V mask三角化，仍可有`当前V → 公共P → 历史V`的多层绕路。要严格因果，必须定义公共prefix政策；本次最小审计用static common prefix并单列其代价，不新建chunk-scoped prefix架构。

## 已有证据及归因边界

| 证据 | 已有结果 | 能说明什么 / 不能说明什么 |
|---|---|---|
| [首块2×2](../first_chunk_routes/INTERPRETATION.md) | 原始own＋公共prefix读video：delta cos=1；仅屏蔽公共反馈：−0.061621；仅增加past-action直读：0.240543；两者均改：−0.019933 | 无历史KV也失配。cos=1是身份正控；采用旧原型时间/anchor，并非完整原生推理或视频修复 |
| [完整历史重算field](../field_factorization/RESULTS.md) | 同Original权重12状态：整体cos=0.996253，delta cos=0.060153 | 无persistent缓存也失配；同时改action直读、公共prefix和video topology，且双方fixed-prefix time，不能单归video causality |
| [own＋current-prefix候选](../current_prefix_candidate/RESULTS.md) | 首块6点cos=1；后续12点delta cos 0.028777→0.008548；相对误差1.195180→1.443233 | 首块恢复未延续到历史条件；两因素同时改，各自重建cache，teacher双向重算。没有候选视频 |
| [AnyFlow128固定状态](../generated_action_geometry128/INTERPRETATION.md) | 后续12点delta cos平均0.0382；实际mask/行配对、重复RMSE0和cache hash均检查 | 未发现该协议的错位/分支污染；不等于语义控制正确，权重/条件/拓扑/cache仍混杂 |
| [时间/Single I0校准](../local_topology/CONDITIONING_RESULTS.md) | 12latent整窗口30step的A−D：0.021764→0.798524→2.227735；最终A=+1.250421，D=−0.977313 | 原型时间和anchor修改能显著抑制实际控制，不可将旧失败全部归因mask/KV；不是多chunk通过 |
| [C12→5自身历史](../chunk_partition_cb/README.md) | N的A-history当前A/D=+2.146440/−1.751855；D-history=+2.607594/−1.414301；clean D-history当前A=−0.176696失败 | 局部双向重算＋N有第二块动作/结构正控；不是严格causal或persistent KV成功，只有一场景/seed |

Cosine是动作velocity差分指标，不是视频质量。旧BF16实验同Original的SDPA/Flex差分cos约0.84–0.87，提示小差分对后端/精度敏感；新主对照统一backend和precision。各实验数字不能跨协议相加计算失败归因比例。

**Replay=0的覆盖范围有限：** [test_h3_cached.py](../../../../tests/test_h3_cached.py)的真tiny-H3 cached/recompute测试用own、默认无feedback、fixed-prefix time，容差`atol=2e−6,rtol=2e−5`；[T1 replay测试](../../../../tests/test_local_topology.py)是按相同clean commit顺序重建缓存；训练tail replay=0是同一捕获图；C/B first12 replay=0是旧solver状态重放。均不能直接推导33B native＋feedback下第2↔3路动作等价。本轮仅读这些测试和报告，没有重跑。

## 最小计划：先固定状态，不训练（未执行）

先准备隔离的审计入口，补足上述参数和mask对齐；不修改生产默认。只用Original H3＋released action LoRA，关闭新增visual/action/prefix adapters和AnyFlow，统一SDPA与h3_fp32政策。

**输入银行：** 复用[C12→5已归档状态](../../../../experiments/11_causal_12_then5_selfhistory/README.md)，两份自身first12 A/D history、当前`[12,17)`；每份history预先固定一个endpoint及noise，两个action和全部角色共用同一个z_sigma。三sigma取`.939540/.689441/.240781`，共6状态。插值得到的noisy state明确标注为endpoint/noise诊断状态，不能称第二块真实在线solver状态。首12旧solver states另作空history身份检查。

固定Single I0原位置、same RGB/prompt/seed13、full37 global RoPE/noise。A/D等长布局；future action/video在refiner前删除，位置origin预先固定。`position_contract.py`是opt-in，不假定benchmark自动启用。history raw latent保持clean、其video time=1；当前/live text使用native时间。Audio共用已有固定noise/time0，四路一致，但不同于完整Original联合audio/video去噪。

第1路仅保留相同可见输入范围，记为**Original-visible**，不声称是含未来video的完整124f Original；已保存的完整视频另列为动作正控。

| 角色 | Action规则 | Video / 历史规则 | 目的 |
|---|---|---|---|
| R1 | Original own，action读自身video | 同可见范围，Original双向重算 | 同状态动作参考 |
| R2 | own＋自身feedback | 严格chunk-causal，重算全历史祖先，live prefix按当前native时间 | causal图影响 |
| R3 | own＋当前feedback，历史反馈在clean commit内构造 | 同chunk-causal，persistent clean video KV | 历史提交上下文冻结影响 |
| R4 | V_i可直接读A_j(j≤i)，其他边同R3 | 同chunk-causal＋persistent KV | past-action直接入口影响 |

必须补两个小控制，避免四路表掩盖混杂：

- **R1-P：** 在R1仅屏蔽非action公共prefix读video，保留action自身feedback及双向V/V。R2/R3/R4统一这套static公共prefix。R1→R1-P测公共grounding；R1-P→R2才测video三角化。R2的“Original Single-Egress”指action直接边保留，不是Original所有公共prefix边都保留。R2历史action可读其历史video，R3历史表示在各自commit时形成。
- **R4-live / R4-rebuilt：** 先固定R3同一cache，仅切当前前向own→causal；再按causal规则重建历史cache。前者测即时直读效应，后者测完整规则含历史累积效应，不能混成一个数字。

共6角色×6状态×A/D=**72次当前诊断前向**；prefill、重复、泄漏/间接路径探针另记，72不是所有模型调用的总预算。不需要新自由rollout。

验收顺序：

1. 核对完整latent空间行、双向own-action边及第0/25/49层实际mask，检查完整多层依赖。未来action内容、span长度、video扰动分别作为负对照。
2. 同commit完整条件、同全祖先重建，报逐层K/V、当前velocity和A/D delta误差；再比较按当前native条件重算历史。时间/I0差额单列，不为提高cosine静默改成fixed-prefix time。若同图重建先失败，修执行/索引/位置后再讨论模型能力。
3. 间接历史动作：固定cache后只改过去action文本，基础own图预期不变；固定raw history，仅在prefill改历史action并重建KV，再恢复当前prefix，检查输出是否变化。前者验证无直接旁路，后者验证历史video表示的间接传递；非零不等于方向正确。
4. A/D期间不得commit或改history/cache/模型；保存全部hash。权重、路由或条件改变后重建各自cache。不得把删掉祖先的滑窗重算差异误记成cache损失。
5. 每状态报告delta cosine、范数比、相对L2、delta RMS/整体velocity RMS，另报A/D单个v、layer KV、repeat/future perturbation误差。Teacher delta接近数值噪声时标记，不按不稳定cosine排名。

决策：R3↔R4-live测额外直读；R1-P→R2测video因果化；R2尚可信而R3失配且同commit重建通过，才支持提交上下文冻结造成额外损失。多个因素同时有影响则报告交互，不强行选一个主因。R1→R1-P已失配时必须承认公共prefix因素。

只在机制对照可信后，考虑预定D-history上R1与一个候选的A/D **4条30step局部视频（120采样forward，另计诊断/解码）**。看完整当前17RGB、人物结构、光流方向、boundary/frame MAD；不能以cosine改善颁发画质PASS。无自动124f扩展、架构扫参、AnyFlow或DMD。

当前下一任务以此机制对照为先，暂不继续此前C第三/第四块扩展建议；已有C+N局部正结果保留并与strict causal/KV结论分开。
