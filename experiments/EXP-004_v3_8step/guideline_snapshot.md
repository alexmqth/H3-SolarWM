
# H3-World 项目协作与实验管理规则

更新日期：2026-10-10（Asia/Hong_Kong）

## 1. 项目目标与基本原则

本项目目标是将原始双向 H3-World 转化为具有动作控制能力、视觉稳定性和高效推理能力的自回归视频世界模型，并进一步探索 AnyFlow 少步生成与 On-policy DMD。

研究路线：

**Original H3-World → Causalization → Visual & Action Recovery → Efficient Causal Generation → AnyFlow → On-policy DMD**

项目遵循以下原则：

1. **主线优先**：每项实验必须服务于明确的研究问题，不能因为出现新想法就无限扩展实验。
2. **单变量优先**：机制实验尽量只改变一个因素；多因素变化必须明确记录，不能作单因素归因。
3. **证据优先**：区分代码实现成功、训练执行成功、指标改善和实际视频效果通过。
4. **负结果保留**：失败实验同样具有研究价值，但应归档，不反复开启已排除的方向。
5. **阶段性收敛**：每轮研究必须有明确交付物、验收标准、资源预算和停止条件。
6. **版本与任务分离**：EXP 任务完成不代表产生新模型版本；只有具有明确能力定位和可复现证据的结果，才可以升级为正式版本。
7. **禁止无授权扩展**：不自行增加训练步数、实验分支、GPU任务或大规模参数搜索。

## 2. 协作角色

项目设置两个角色：

- **Judge**：研究负责人，负责研究路线、实验设计、结果审计、验收和任务发布。本会话承担该职责。
- **Exp Worker**：实验执行者，负责代码实现、受控实验、证据收集和结果汇报。执行实验的 Codex 会话承担该职责。

### Judge 职责

1. 根据项目总体目标确定当前最重要的研究问题。
2. 阅读任务报告，检查代码、实验协议、结果和失败原因。
3. 区分实现正确性、模型能力和研究结论，不将局部指标改善直接视为模型成功。
4. 决定接受、返工、停止或归档当前任务。
5. 维护主线版本、研究分支和正式验收结果。
6. 在发布下一项任务前评估：该实验是否会改变接下来的研究决策。
7. 控制研究范围，避免连续开展收益不明确的消融。
8. 负责发布 `next_plan.md` 并维护正式进展记录。

### Exp Worker 职责

1. 开始前阅读 `guideline.md`、`progress.md` 和 `next_plan.md`。
2. 核实当前任务编号、版本、验收标准和资源限制。
3. 只执行当前获批任务及必要的实现修复、验证。
4. 完整记录实际运行的命令、环境、随机种子、模型、GPU、输出路径及结果。
5. 主动报告异常、失败、混杂因素和无法支持的结论。
6. 实验执行完毕后提交 `report.md`，等待 Judge 验收。
7. 不自行修改正式研究目标、扩大实验预算或宣布项目进入下一阶段。
8. 可以在报告中提出新假设，但不得未经授权直接启动衍生实验。

**Worker 的首要任务是完成已经定义的研究问题，而不是尽可能多地运行实验。**

## 3. 项目文件职责

项目根目录的文件：

| 文件             | 职责                                                   | 维护者     |
| ---------------- | ------------------------------------------------------ | ---------- |
| `guideline.md` | 项目协作制度、研究版本、实验管理和验收规则             | Judge      |
| `progress.md`  | 当前项目状态、主线版本、关键结果、未解决问题和任务索引 | Judge      |
| `next_plan.md` | 当前唯一有效的任务书、预算、验收标准和 Judge 评价      | Judge      |
| `report.md`    | 当前任务的真实执行过程、结果、证据和建议               | Exp Worker |
| `archive.md`   | 已结束任务的完整历史记录与验收结论                     | Judge      |
| `README.md`    | 项目整体介绍、架构、依赖和运行说明                     | 按需维护   |

### 文件使用要求

- `progress.md` 是当前正式研究状态的唯一摘要来源。
- `next_plan.md` 是当前可执行任务的唯一授权来源。
- `report.md` 是本轮 worker 实际执行情况的证据入口。
- `archive.md` 保留完整历史，不承担当前状态摘要职责。
- 具体实验的原始配置、日志、指标和视频必须保留在对应实验目录。
- `submission/report/` 是汇报展示目录，不应替代原始证据和实验日志。

不允许将 worker 的未验收结论直接写入 `progress.md`。

## 4. 正式研究主线与版本管理

主线版本按照研究能力与技术路线组织，而不是简单按实验执行日期命名。

### V0：Original H3-World

**Bidirectional Baseline**

- 原始 H3-World 和 released action LoRA。
- Original Single-Egress Action Routing。
- 完整视频联合双向去噪。
- 原生 Single I0、时间与位置条件。

职责：作为视觉质量、动作方向和原始计算行为的主要参考。

### V1：Native Chunk-Causal H3-World

**Initial Causalization**

- 视频 latent 按时间分块。
- 引入 chunk-causal attention 和 persistent video KV。
- 实现 generated-history autoregressive rollout。
- 包含相关早期 latent-anchor 探索，但必须区分具体配置。

主要研究问题：原始 H3 能否直接转换为高效的分块因果生成模型？

状态：工程可行性已有验证，视觉稳定和动作控制未同时通过。

### V2a：RGB-Anchor Causal H3-World

**Visual Consistency Branch**

- 从 V1 探索 RGB-consistent image conditioning。
- 使用原生 H3 image branch 生成 dynamic RGB anchor。
- 相关正式候选包含 visual adaptation 和 endpoint supervision。
- 保留 persistent KV 的因果生成路线。

主要目标：减少跨块视觉漂移和人物分解。

状态：已有124帧视觉结构基本稳定的结果，但动作方向未恢复。

### V2b：Same-σ History H3-World

**Action-Preserving Local Autoregressive Branch**

- 重新采用 Original H3 + released action LoRA。
- Single I0 与原生条件协议。
- 采用12-latent初始窗口与后续5-latent窗口。
- 每个去噪步骤使用 Same-σ History。
- 历史与当前视频在可见窗口内联合双向重算。
- 不使用 persistent hidden KV。

主要目标：在自生成历史续写中同时保留人物结构和动作响应。

状态：EXP-001已验收单停车场seed13持续A/D的124帧可行性；动作切换仍有限制，10/20秒长期稳定和高效缓存尚未通过。

**版本关系说明：**

V2a 和 V2b 是并行研究路线，不是直接的 checkpoint 继承关系。

V2a 证明部分视觉连续性修复可行；V2b 为恢复局部动作能力提供正结果。

V2b 不等于已经完成 strict chunk-causal attention，也不能因为短时成功就宣称124帧长视频通过。

### V3：Efficient Causal H3-World

**Feasibility Version — EXP-002/003 accepted**

目标是在同一个模型中实现：

- Strict chunk-causal attention。
- 可复用的 persistent KV。
- 正确的 action-conditioned response。
- 可接受的多窗口视觉稳定性。
- 能够支持后续少步生成训练。

2026-10-10已验收同一native Single I0/current-prefix/strict-KV配置，AA/AD124帧可行性成立，零新增训练。保留AA短暂明显人体形变后恢复、边界跳变和单scene/seed限制；画质/严格连续性PARTIAL，完整从零E2E、公平Original speedup与长期泛化未验证。正式入口：submission/mainline/V3_efficient_causal/README.md。

### 版本升级规则

只有同时满足以下条件，Judge 才可以创建或升级正式版本：

1. 说明 Parent Version 和核心改动。
2. 给出准确的模型 checkpoint、代码 revision 和推理协议。
3. 有可追溯的代表性输出。
4. 有对应的视觉、动作和系统指标。
5. 明确解决了什么问题，以及仍未解决什么。
6. 不存在将多个不同实验配置混合成单一版本的情况。

不能因为训练成功、loss下降、运行通过或单个 cosine 改善，就自动升级版本。

## 5. 研究分支管理

非主线实验统一归入以下分支。

### Branch A：Causal Mechanism Diagnostics

包括：

- Action Routing / Single-Egress。
- Public Prefix Feedback。
- Video Attention Topology。
- Recompute vs Persistent KV。
- Clean vs Same-σ History。
- Native timestep / RoPE。
- Chunk size 与 VAE decoder boundary。
- Single I0 / Latent Anchor / RGB Anchor。

目标是解释能力退化的机制。

机制定位与模型修复是不同任务，不能把审计完成写成动作或画质已经恢复。

### Branch B：Causal Adaptation & Action Recovery

包括：

- Ordinary Flow Matching。
- Causal QKV LoRA。
- Teacher Forcing / Scheduled Sampling。
- Generated-history adaptation。
- Action velocity / counterfactual alignment。
- Endpoint、boundary、action ranking 等监督方法。
- Real ABot 数据训练。

目标是恢复严格因果模型的控制能力和视觉质量。

所有训练实验需要有预定义的预算、基线、验证集及停止条件。

### Branch C：AnyFlow & On-policy DMD

包括：

- AnyFlow objective 与 target-time network。
- 4/8-step generation。
- Finite-map / r=t consistency。
- Teacher-generated 与 Real ABot 数据实验。
- Stage2-lite DMD、critic 和自生成历史训练。

早期实现验证和短预算训练统一标记为 preliminary exploration。

不得将 Stage2-lite 的工程运行成功写成完整 SolarWM-style DMD 已通过，也不得将训练步数直接当作模型能力证据。

**主线优先级始终高于分支数量。**

## 6. 任务管理与实验预算

任务编号使用：

`EXP-001`、`EXP-002`、`EXP-003`……

返工保留 `task_id`，递增 `plan_version`。

每个 `next_plan.md` 必须明确：

| 字段                 | 必填内容                              |
| -------------------- | ------------------------------------- |
| Task ID / Version    | 唯一任务编号及版本                    |
| Research Track       | Mainline / Branch A / B / C           |
| Parent Version       | 当前基于哪个正式版本或 checkpoint     |
| Research Question    | 本轮唯一核心研究问题                  |
| Hypothesis           | 待检验的假设，而非预设结论            |
| Baseline             | 与什么配置比较                        |
| Controlled Variables | 必须保持不变的条件                    |
| Changed Variables    | 本轮允许修改的条件                    |
| Inputs               | 数据、checkpoint、history、动作、噪声 |
| Execution Plan       | 具体步骤和命令                        |
| Resource Budget      | GPU、训练更新次数、推理数量及资源上限 |
| Evaluation           | 数值指标、视频检查和验收标准          |
| Stop Conditions      | 提前终止或停止扩展的条件              |
| Deliverables         | 代码、报告、指标、视频、日志          |
| Out of Scope         | 明确不做的工作                        |

### 任务设计原则

- 尽量每轮只回答一个核心问题。
- 优先复用已保存的 latents、checkpoints、noisy states 和视频。
- 机制诊断优先使用固定状态 forward，必要时才进行完整视频推理。
- 只有固定状态结果具有解释价值且能够影响决策时，才扩展到30-step视频生成。
- 不允许因为某个局部指标改善而自动扩大训练或生成更长视频。
- 如一次实验包含多个主要变量，必须将其标记为联合方案评估，而不是严格消融。
- 新想法记录为候选任务，不自动进入运行队列。

### 实验停止规则

以下情况应暂停并请求 Judge 决策：

1. 关键验收条件已经失败，继续增加预算无法回答新问题。
2. 发现协议不一致、数据泄漏、错误 checkpoint 或缓存不等价。
3. 需要修改主线架构、输入条件或训练目标。
4. 原定实验预算用尽。
5. 训练不稳定、OOM或出现关键数值异常。
6. 已经获得充分证据支持或否定当前假设。

Judge 应明确选择：`continue`、`revise`、`accept`、`stop` 或 `archive`，避免任务无期限运行。

## 7. H3-World 实验协议

所有关键实验必须完整记录以下条件。

### 模型与计算

- Original H3 checkpoint 及其 hash/revision。
- Released action LoRA 和新增 adapter。
- Attention mask 与 action routing。
- Action feedback 是否开启。
- Public Prefix Feedback 是否保留。
- Video attention 是 full bidirectional、local bidirectional 还是 strict chunk-causal。
- Transformer 精度、attention backend 和数值计算设置。

### 输入与时间条件

- Single I0 / RGB Dual Anchor / Latent Dual Anchor。
- Native timestep、history timestep 和 current timestep。
- Clean history / Same-σ history。
- History 来源：GT、teacher-generated 或 student-generated。
- History 是否为真实 free-running rollout。
- Global RoPE、action-span 与 video-latent 对齐。
- 是否存在未来动作或视频信息泄漏。

### 生成与缓存

- RGB frames、latent frames 和 chunk partition。
- Sampling steps、sigma schedule、flow shift。
- History window size。
- Recompute / Persistent KV。
- KV prefill、commit、freeze 和 reuse 协议。
- Decoder prefix、overlap 和已显示帧的保留方式。

**特别约束：**

1. Persistent KV 中缓存的是 Transformer 历史 K/V，不应简单描述为“缓存了 clean latent”。
2. KV Reuse Correctness 与 History Commit Semantics 必须分别验证。
3. Same-σ history 的 T2 重算不能描述为 persistent-KV 版本。
4. 逐块生成不等于 strict chunk-causal attention；窗口之间不使用未来视频，并不代表窗口内部和历史之间没有双向交互。
5. 若改变 native time、anchor 或 attention topology，必须将其明确列为实验变量。

## 8. 模型能力验收标准

所有正式生成实验分别评价三类能力。

### A. Action Fidelity

核心检查：

- 固定起始状态下的 A/D counterfactual response。
- A、D 是否产生符合场景约定的方向。
- 是否支持动作切换。
- 是否存在动作响应消失、混淆或明显延迟。

停车场场景可采用预先固定的水平 optical-flow proxy；但 flow 只能作为辅助指标，不能代替视觉和动作语义判断。

Velocity cosine、norm ratio、relative L2 等只能用于内部机制诊断，不能直接代表视频动作正确。

### B. Visual Stability

检查：

- 人物是否保持单一完整的身体结构。
- 是否发生 ghosting、透明化、肢体分解或人物消失。
- 场景是否发生不合理跳变。
- Chunk boundary 是否明显断裂。
- 已显示 RGB frames 是否被无声明地重新修改。
- 长时间 rollout 是否出现累积漂移。

必须检查完整视频帧序列和必要的原分辨率细节，不能只依据 MP4 能解码或首末帧判断。

### C. Efficiency

检查：

- 每个 chunk 的 sampling steps。
- 总 denoiser forward 数量。
- 历史是否每步重算。
- Persistent KV memory。
- GPU allocated/peak memory。
- Sampling latency 与 end-to-end latency。
- VAE decode/re-encode、CPU offload、prefill 和 cache commit 的额外成本。

不同 sampling steps、GPU占用、offload方式或硬件条件下的耗时不得直接宣称公平 speedup。

### 验收状态

能力状态使用：

- `PASS`：达到预先声明的验收条件。
- `PARTIAL`：部分能力得到验证，但尚不满足完整要求。
- `FAIL`：结果完成但未通过能力验收。
- `NOT_TESTED`：尚未进行相应测试。
- `INVALID`：协议错误、数据泄漏或实验不可比较。

执行状态与能力状态必须分别记录。

**任务 completed 不等于模型 PASS。**

## 9. 实验报告与证据管理

每个实验使用独立目录，建议：

`submission/experiments/EXP-XXX_<short_name>/`

每个正式实验目录至少包含：

- `README.md`：研究问题、实验配置、结果与结论。
- `config.yaml` 或等价冻结配置。
- `metrics.json`：机器可读指标。
- `artifacts/`：必要视频、图像和诊断输出。
- `logs/`：执行日志。
- `MANIFEST.md`：checkpoint、代码 revision、原始输出及依赖路径。

大型 checkpoint 可保持原位置，不重复复制。

报告必须严格区分：

1. 历史已知证据。
2. 本轮实际执行。
3. 本轮观察结果。
4. 可以支持的结论。
5. 尚未验证的假设。
6. 下一步建议。

不得将之前的结果复制成当前实验的新发现。

### 对比视频规则

正式主线版本需要尽可能提供：

- Original H3 vs Current Version。
- Parent Version vs Current Version。
- 必要的 A/D 或 W/S/A/D 对照。

视频采用横向并排形式，默认左边为 baseline，右边为当前方案。

要求：

- 标明 action、帧数、sampling steps、history 与 anchor。
- 优先匹配相同初始图像、动作、噪声和生成范围。
- 不同 checkpoint、采样步数、历史协议或 topology 的比较，必须说明混杂因素。
- V2a 和 V2b 属于平行能力对照，不能宣称为严格的单变量消融。
- 没有匹配视频时标记缺失，不擅自启动 GPU 推理补齐。

不要求每个小型机制诊断都生成对比视频，避免无意义的渲染和汇报膨胀。

## 10. Submission 与 Report 组织

`submission/` 按研究主线、研究分支和汇报材料组织。

```text
submission/
├── mainline/
│   ├── V0_original_bidirectional/
│   ├── V1_native_chunk_causal/
│   ├── V2a_rgb_anchor_causal/
│   ├── V2b_same_sigma_local_bidir/
│   └── V3_efficient_causal/
├── branches/
│   ├── A_causal_diagnostics/
│   ├── B_causal_adaptation/
│   └── C_anyflow_dmd/
├── experiments/
│   └── EXP-XXX_.../
├── archive/
└── report/
    ├── README.md
    ├── roadmap.md
    ├── versions/
    ├── comparisons/
    └── research_branches/
```

其中：

- `mainline/` 记录每个版本的定义、证据和研究结论。
- `branches/` 归档解释机制或尝试新训练方法的实验。
- `experiments/` 保存任务级原始产物。
- `archive/` 保留旧组织方式和重要历史。
- `report/` 只保留用于汇报的精简版本、核心代码、视频和说明。

不得因目录重组而丢失原始日志、破坏 checkpoint 引用或覆盖已有产物。

只有 Judge 正式接受的结果才能升级到 `mainline/` 的已验收结论。

每次正式版本更新后，应同步检查 `submission/report/` 中对应版本的说明和代表性视频是否需要更新。

## 11. Judge 验收和任务交接

Judge 每轮按以下顺序工作：

1. 阅读 `guideline.md`、`progress.md`、`next_plan.md`、`report.md`。
2. 核对 `task_id`、`plan_version` 和实际执行状态。
3. 检查配置、运行日志、checkpoint、指标、视频和异常。
4. 判断任务是否按要求执行。
5. 独立判断模型能力是否通过验收。
6. 将评价写入 `next_plan.md` 的 Judge 验收栏。
7. 将已结束任务和完整报告归档至 `archive.md`。
8. 在 `progress.md` 中更新正式结论。
9. 必要时更新主线版本与汇报目录。
10. 决定下一项任务或明确暂停实验。

Judge 验收状态：

- `pending`
- `accepted`
- `needs_revision`
- `cancelled`

### Worker 状态

Worker 使用：

- `not_started`
- `running`
- `completed`
- `blocked`
- `failed`

Worker 将 `report.md` 标记为 `completed` 仅表示本轮执行结束。

任务被 Judge 正式接受后，才能成为项目已验收记录。

### 交接顺序

Judge 发布 `next_plan.md`
→ Worker 执行并维护 `report.md`
→ Judge 审计与验收
→ 完整记录归档
→ 更新 `progress.md`
→ 必要时更新版本与 report
→ 发布下一轮任务。

上一轮未验收或仍在运行时，不得覆盖其任务书及报告。

## 12. Progress 精简规则

`progress.md` 应帮助新会话在几分钟内理解项目。

建议固定以下结构：

1. **Project Objective**：最终目标。
2. **Current Research Stage**：当前主要研究问题。
3. **Mainline Status**：V0、V1、V2a、V2b、V3 的状态。
4. **Best Available Evidence**：当前最可信的动作、视觉、效率结果。
5. **Known Limitations**：尚未解决的问题。
6. **Active Task**：唯一当前执行任务。
7. **Next Decision**：下一项需要 Judge 决定的研究方向。
8. **Completed Task Index**：已验收任务的简短索引。

历史细节、逐step日志、失败尝试和完整实验报告保留在 `archive.md` 与实验目录中。

不要因为每天完成了大量实验，就在 `progress.md` 中不断增加长篇流水账。

一个实验如果没有改变当前正式结论，只需索引归档，不必扩大主线叙述。

## 13. GPU、代码与仓库安全规则

- 默认使用项目已有环境与依赖，不擅自修改系统级配置。
- 启动 GPU 前检查实时使用情况和用户当前授权。
- 不终止、抢占或修改其他用户进程。
- 不依据历史 GPU 编号推断当前资源可用。
- 禁止未经授权地启动批量训练或长期运行队列。
- 所有训练任务设置资源预算、日志和停止机制。
- 修改代码后运行与修改内容相关的测试。
- 对数值等价、cache correctness 或 gradient correctness 的测试，必须注明使用的精度与后端。
- 代码、配置和结果应能追溯到 Git commit/hash。
- 不覆盖正式 baseline checkpoint、历史实验或原始数据。
- 对已有仓库结构进行大规模移动前，先建立 manifest，并检查 Git 状态及引用路径。
- 提交与推送 GitHub 遵循当前任务授权；未经要求不进行大规模历史改写或 force push。
- 不把真实 checkpoint、大型原始数据、密钥或敏感配置意外纳入 Git。

## 14. 研究决策与防止实验失控

在发布下一轮实验前，Judge 必须回答：

**Q1. 本轮实验究竟要验证什么？**

必须能够用一句话清晰描述。

**Q2. 已有实验是否回答过相同问题？**

如已有充分证据，不重复运行。

**Q3. 如果结果为正或负，我们下一步分别做什么？**

如果两种结果都不会影响决策，应考虑取消该实验。

**Q4. 是否有更便宜的验证方式？**

优先使用固定状态、CPU测试、已有 latent 和短窗口，不直接启动长视频或训练。

**Q5. 本轮成功后能交付什么？**

必须明确对应的代码、指标、视频、能力证明或研究结论。

**Q6. 什么时候停止？**

任务必须有明确的 GPU、时间、更新次数或实验数量预算，以及提前终止条件。

### 当前项目的优先级原则

整体研究优先顺序：

1. 明确可用的 causal action/visual baseline。
2. 统一视觉稳定、动作响应和高效因果推理。
3. 在可信 causal baseline 上研究 AnyFlow 少步生成。
4. 通过 On-policy DMD 改善自生成历史下的长期质量。

机制诊断是服务于这些目标的辅助工作，不应无限推迟模型能力建设。

不要求完全解释每一个历史失败实验后才能推进研究，但必须保证下一阶段的训练目标建立在可信的实现与数据协议之上。

---

**最终原则：每项实验都应当让研究路线更清晰，而不是让仓库里多出一个无法解释的 checkpoint。**

Judge 负责控制研究方向和证据标准；Worker 负责高质量执行、记录和复现。整个项目始终围绕模型能力的可验证提升推进。


## 15. 用户补充：研究ROI与审核尺度（2026-10-10）

用户要求Judge把握大局，不为很小的指标收益反复增加实验，不把审核设成脱离研究目的的机械关卡。

1. 优先完成能改变研究决策的能力交付和直观对比。工程、协议、数据真实性的检查要充分；达到足够证据后停止重复核查。
2. 单一辅助proxy轻微波动或反号，先结合完整画面和动作语义判断。没有严重结构崩坏或明显控制失效时，不因此自动否定所有其他有价值的验证；保留缺点、范围与不确定性。
3. 若原任务门槛过严或范围收益不足，Judge应显式修订plan_version、冻结旧协议和结果，并缩小到最有决策价值的交付。不能事后改写旧FAIL为PASS，也不要求每条研究分支全部通过才交付有效结果。
4. 优先复用已有输出；不为占满GPU而增加场景、seed、消融或训练。停止条件聚焦协议无效、资源超限、严重质量失败和继续实验已无法改变决策。
5. 每轮结束以“当前可用到什么程度、主要瓶颈是什么、下一步哪项工作收益最大”组织结论；不无限追求局部指标完美。


## 16. 持续Judge目标与GitHub交付（2026-10-10）

用户已授权Judge持续监督Worker报告、验收后发布下一任务，直到V3版本真正完成，并要求每次Worker完成任务后更新submission、提交到GitHub且注意仓库合并。

- 先核对报告及实际进程。正在执行的任务继续按当前任务书推进；完成、失败或阻塞均须按真实证据处理，不能只依据文件状态猜测运行是否结束。
- 每轮形成可审计结果后，把冻结任务书、完整Worker报告、Judge评价和原始证据索引存入对应submission实验目录，更新必要的mainline/report导航与能力边界。研究失败结果也应按实际范围交付，未验收结果明确pending/needs_revision。
- Judge统一负责本轮Git发布：检查工作区和待提交内容，fetch远端；有分叉时保留双方有效改动并完成必要验证。禁止force push或改写历史，不覆盖其他工作。只提交本轮应交付材料，不带基础权重、大型原始tensor、临时缓存或敏感配置。
- 提交后推送正常分支并核对远端commit一致，记录SHA和验证范围；远端发生竞态则再次fetch、合并、验证后重试，不能凭本地提交声称已同步GitHub。
- 完成一个EXP或V2b长视频不等于V3完成。V3须同一明确checkpoint/推理协议同时具备strict chunk-causal、实际persistent KV复用、可辨动作响应、可用多窗口视觉，以及可追溯效率测量和代表视频。未证实的范围写明限制，不能通过改名或缩小目标宣布完成。


## 17. 用户补充：当前为可行性验证阶段（2026-10-10）

当前模型未经大量针对性训练，Judge以可行性与下一步研究价值判断结果，不要求成熟产品画质。

1. 动作响应可辨、人物/场景基本可用、严格因果与真实KV复用等核心机制成立时，可以接受有明确限制的原型证据。模糊、动作不自然、局部肢体细节、普通边界跳变不单独否定整条路线；如实说明程度与范围。
2. 对持续没有有效信号、关键结构明显不成立或进一步投入难以改变决策的方向，在有限预算后停止并归档。无需解释完所有失败机制，也不自动追加大规模训练来挽救每个方向。
3. 负结果后的选择可以是一次有依据的有限适配，也可以直接换路线；Judge必须结合已有证据和ROI决定，不预设“失败就继续加预算”。
4. 严格区分实现正确、原型可行、成熟质量三个层次；当前优先前两层。正式V3仍须同一配置具备strict chunk-causal、真实persistent KV、可辨动作、可用多窗口结构与可信效率记录；质量结论标明可行性范围，不额外暗加高保真或大规模泛化门槛。
5. 用户后续要求优先于旧文档中过严的画质或诊断门槛。更新验收解释时保留旧结果与协议，不追溯伪造PASS。
