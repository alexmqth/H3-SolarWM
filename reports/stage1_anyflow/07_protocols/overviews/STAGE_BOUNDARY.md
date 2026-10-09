# Stage1能力与Stage2历史漂移：分开验收

2026-10-08，回应“后段崩溃是否因为缺少Stage2”。

**Generated-history分布偏移是合理且已有支持的解释，但不是目前全部失败的已证实原因。** SolarWM README的Training progression明确说明：Stage1用干净历史条件下的teacher forcing和AnyFlow；Stage2通过SGF在student自己的autoregressive rollout上训练，使用frozen teacher和trainable critic做DMD。官方H3的`sgf_rollout.py`实现detached raw-KV rollout及gradient replay；`stage2.py`负责生成状态和双向score输入。SGF是训练/梯度组织方法，DMD是分布匹配目标，不是仅把teacher replay换个名字。

## 最新128对照补充

128的generated/teacher-history完整对照已完成：分离度0.759660/1.333539，首块latent逐元素相同；teacher明显减轻后段重影，但有边界重置。固定同history改变当前动作的实验也完成：两种历史下A−D为0.314431/0.426827，相对响应方向一致。两组证据支持“历史分布有影响、动作并未完全失效”，尚不能证明Original动作保真或free-running质量通过。原始H3同历史反事实参考缺失，不能单凭切换后的绝对flow符号判定惯性/控制失效。详见[history128](../../03_anyflow_trials/history128/FINAL_RESULTS.md)和[counterfactual128](../../02_causal_diagnostics/counterfactual128/FINAL_RESULTS.md)。以下step16作为历史证据保留，已不再是唯一诊断。

## 同权重有限步/对角条件的新证据

128、generated history、相同8步网格，只将模型的目标时间r设为当前t，对角速度条件明显减轻后段人像/场景叠影，但A/D分离度0.662低于正常finite-map的0.760。实际time pairs/conditioning/权重来源核查通过。因此当前视觉退化也与有限步预测的使用有关，不能完全归因于Stage2缺失；同一seed消融不能证明数学实现有bug，也不排除与历史分布的交互。下一步先定位有限区间预测误差。见[完整结果](../../03_anyflow_trials/diagonal_inference128/FINAL_RESULTS.md)。

## 已有的可区分证据

同一shift12 AnyFlow **step16**、39帧、8 steps/chunk，conditioning与noise相同，仅改变commit和下一块RGB anchor的历史来源：

| history | A flow | D flow | A−D |
|---|---:|---:|---:|
| generated | -1.312275 | -1.492164 | 0.179888 |
| teacher/oracle | -0.211312 | -0.875811 | 0.664499 |

Teacher history明显缓解后续状态偏离，但A仍方向错误，块边界也有重置。第一块5个latent在两个history条件下逐元素一致，而A的第一个RGB区间[0,17) flow为−1.207126，Original对应+0.083398。该区间尚无前一生成chunk的误差累积，因此不能把这部分偏差归因于长时历史累积。RGB指标受temporal VAE影响，不等于独立latent动作准确率。

这是step16证据，不能冒充step64/96/128的teacher-history结果；新checkpoint需要新的对应诊断。Oracle history本身含动作之后的世界状态，后续正flow也可能来自历史信息，不能单独证明当前action binding。

完整视频、逐帧复核与latent对照见[历史诊断报告](../../03_anyflow_trials/shift12_history_diagnostic/FINAL_RESULTS.md)。

## 调整后的阶段判断

1. **进入Stage2的Stage1前提**：首块及受控干净历史下，少步生成应能保持人物/场景、对A/D产生与Original相符的不同响应；在同一history上切换当前action作counterfactual诊断，避免oracle历史代替动作输入。保留AnyFlow对匹配FM的实际少步比较。数值测试和loss下降不能代替这些效果证据。
2. **Stage2所针对的问题**：上述局部能力成立，但free-running后续chunk持续漂移。此时应进入SGF/DMD检验generated-history适应，不要求Stage1先消除全部长时漂移。
3. **项目最终交付门槛保持**：39f自生成A>0、D<0、A−D>1并且画面完整；再验证独立seed、动作切换、124f与公平效率表。它是最终效果门槛，不能全部当作进入Stage2之前必须完成的条件。

128更新及4/8步视频现已全部完成，当前仍未证明首块/clean-history动作与画质条件已充分满足。下一项是同权重、同clean history的有限区间预测诊断，不自动继续加训练次数，也不立即启动Stage2。若clean与generated都失败，优先检查Stage1/条件路径；若局部生成与动作响应合理而generated后续仍退化，应有针对性地进入Stage2，不把free-running最终gate提前变成Stage1必须独自解决的问题。

以上是本项目在有限算力下用于归因和选择下一实验的工程标准，并非论文规定的硬性门槛。首块失败排除了“所有偏差都来自历史累积”的解释，但不证明Stage2无法改善首块：分布匹配也可能改善少步生成本身和动作响应。是否改善必须用同一个AnyFlow初始化的受控Stage2对照验证。

已有Stage2-lite基于旧FM原型，不是当前AnyFlow checkpoint的匹配对照，不能用旧结果证明新Stage2必然无效或必然有效。DMD同样不保证修复动作语义、条件路由错误或全部视觉问题。

## 对当前疑问的直接回答（2026-10-08 15:32）

**是，缺少针对当前AnyFlow初始化的Stage2自生成轨迹训练，很可能是后段退化的原因之一；已有同权重历史干预支持这一判断。** 这不是已证实的唯一原因，也不是Stage2效果的实测承诺。Stage1训练的历史来自干净teacher片段，部署时历史来自自身输出；轻微人物重影或几何偏差一旦进入后续anchor/KV，就会改变下一块的条件分布。缓存正确只表示历史计算被正确复用，不表示历史内容正确。SGF组织student自身rollout及梯度replay，DMD用frozen teacher与学习student分布的critic提供分布匹配信号，针对的正是这种训练/推理差异。

当前最直接的对照是[Original / generated / teacher history视频](../../03_anyflow_trials/history128/original_generated_clean_history_AD.mp4)。分离度0.760→1.334和末段人物/车库改善支持历史来源的影响；oracle边界重置及动作历史泄漏限制仍然保留。另一个[同权重finite / diagonal视频](../../03_anyflow_trials/diagonal_inference128/original_finite_diagonal_AD.mp4)说明有限步条件的使用也影响退化，可能和历史分布相互作用。

有限区间只读探针已通过CPU解析场/随机小H3检查，随后在重新确认空闲的GPU5/6上启动A/D。它固定teacher history与teacher/noise插值状态，覆盖三个chunk、高/中/低三个8-step区间，比较一次finite map与4/8细分diagonal积分。细分轨迹不是GT，4/8差异仅用于识别数值参考本身的不确定性；该探针本身不证明Stage1画质成功，也不能检验Stage2疗效。状态和限制见[probe报告](../../03_anyflow_trials/finite_interval_probe128/README.md)。


## 16:04实验推进

有限区间与8/16细分诊断现已全部完成，见[结果](../../03_anyflow_trials/finite_interval_probe128/RESULTS.md)。低噪声末区间内部差异持续，但diagonal16到教师伪GT的latent误差反而更大，不能作为正确答案。开始[同初始化、各8更新的辅助loss对照](../../03_anyflow_trials/interval_consistency_candidate/README.md)，由正常finite-map视频决定是否有收益；未开始Stage2。Stage1无需独自解决长时历史漂移的原则不变。

## 2026-10-08 17:01：进入Stage2前，保留可核查的局部条件

当前136原loss控制组4/8步均未通过，辅助组尚在训练。最多3张GPU、136上限不变。正常finite-map完整39帧（尤其18–38帧）与双局部指标共同决定本轮结果。当前模型的finite/diagonal一致性、到Original伪GT的距离以及对冻结128训练参考的fit分别记录；任何一项下降都不单独构成Stage1成功。

准备Stage2时先明确三条证据：clean-history局部画面可信；同状态A/D动作响应能与Original同状态参照比较（该Original参照仍缺）；有限区间生成基本可信。Stage1不必先独自消除generated-history的全部长期漂移。局部条件成立后，才用同一AnyFlow初始化比较有无Stage2对自生成历史退化的影响，不能用旧FM Stage2-lite替代该对照。

若136继续失败，首先检验低噪声覆盖/监督。现有training-timestep-shift同时改采样density和Gaussian weight；下一项应解耦它们、每次只改一个因素，而不是直接降shift后把效果归因于noise density。已有shift2.22训练的负结果也必须保留。
