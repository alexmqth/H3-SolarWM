# 报告常用缩写与计数

## 统一命名：训练 updates 与推理 NFE（2026-10-12）

训练状态、推理配置和实验阶段分别命名。以后不单独用 `step` 表示一个模型或实验配置。

| 旧写法 | 真正含义 | 当前规范写法 |
| --- | --- | --- |
| AF2 | EXP-007 的第二个训练阶段；阶段编号，不表示2次采样 | EXP-007 / AF2 训练阶段 |
| step32 / step_32 | 学生累计完成32次 optimizer updates | **AF-train-32updates** |
| AF8 / AF4 | AnyFlow 推理时，每个生成 chunk 用8 / 4次去噪网络求值 | **AF-8NFE / AF-4NFE** |
| FM8 / FM4 / FM30 | 普通 Flow Matching 推理时，每块用8 / 4 / 30次去噪网络求值 | **FM-8NFE / FM-4NFE / FM-30NFE** |
| 8 NFE | 8次函数求值（Number of Function Evaluations） | 写明 **8 NFE/chunk** 及方法 |

**完整示例：`AF-train-32updates + AF-8NFE` = 累计训练更新32次的 AnyFlow 学生，每个待生成 chunk 使用8次采样网络求值。**

EXP-007 的 AF1 完成1次更新，AF2新增31次，合计32次；AF3是匹配NFE的视频评估阶段。因此 `AF2`、`AF3` 都不能读作 `AF-2NFE`、`AF-3NFE`。同一个 `AF-train-32updates` checkpoint 分别用于 EXP-007/012 的 `AF-8NFE` 和 EXP-009 的 `AF-4NFE`，没有为这两个采样配置分别重训。详见[训练审核](../experiments/EXP-007_v3_anyflow/judge/AF2_REVIEW.md)。

### 计数与后续报告规则

- `optimizer_updates_total`：累计实际 optimizer 更新次数；`optimizer_updates_this_stage`：本阶段新增次数。前向、反向、梯度累积 microsteps 不等于 optimizer updates。
- `sampling_steps_per_chunk`：采样器的迭代步数；`sampling_nfe_per_chunk`：每块采样实际网络求值次数，独立记录。当前上述FM/AF采样配置每步一次网络求值，所以二者相等；多阶段求解器、多次模型求值等其他协议须按实测填写，不一般化为 step = NFE。
- clean KV prefill/commit、训练与反向重算、text/image encode、VAE decode 单列；它们不计入这里的采样NFE。NFE不代表完整E2E耗时。
- 新报告、任务书、视频标题使用规范名称，并分别列训练checkpoint/累计updates、方法、NFE/chunk、首窗来源与生成范围。例如73帧AF续写借用FM首39帧，`AF-8NFE`只描述AF生成的后续块。
- `AF-train-32updates`描述训练状态，不能单独唯一定位权重；复现同时记录EXP编号、checkpoint路径和SHA。

### 历史文件兼容

现有 `fm8/`、`fm4_vs_af4/`、含 `FM8`/`AF8` 的视频文件、`step_32/` checkpoint目录，以及冻结配置中的 `step`/`checkpoint_step` 等字段保留历史身份。它们是上述规范名称的旧别名。正文、链接显示文字和视频导航使用新名；原始日志、冻结任务书、源码、权重、视频内已有字幕和SHA不改写。这里的旧 `checkpoint_step=32`是训练累计updates，采样次数须另读采样配置。

## 其他缩写

| 名称 | 全称 / 中文解释 | 本项目怎么读 |
| --- | --- | --- |
| FM | Flow Matching，流匹配 | 基础模型的速度场建模；当前 FM 减步评估使用原权重，不新增训练 |
| FM-30NFE / FM-8NFE / FM-4NFE | 每块30 / 8 / 4次普通FM采样网络求值 | 当前采样器对应30 / 8 / 4个采样步；不表示训练更新次数 |
| AF / AnyFlow | AnyFlow 少步有限映射方法；AF为本项目简称 | 新增目标时间条件并训练 finite map；与直接减少FM采样步数分开 |
| AF-8NFE / AF-4NFE | AnyFlow每块8 / 4次采样网络求值 | `AF-train-32updates` 单独表示学生累计32次训练更新 |
| NFE | Number of Function Evaluations，函数求值次数 | 本实验计每块采样的去噪网络求值；不含单独KV提交或VAE等调用 |
| SW / G / L | Sliding Window / Global / Local，滑动窗口 / 全局 / 局部 | SW-G和SW-L保持同窗口，仅位置协议不同 |
| RoPE / MM-RoPE | Rotary Position Embedding / Multi-Modal RoPE，旋转 / 多模态旋转位置编码 | 本项目Global/Local主要讨论视频时间坐标；空间、prefix处理必须单独说明 |
| PRoPE | 相机/相对位姿相关的位置编码 | SolarWM参考涉及该协议；当前H3实验路径没有SolarWM camera PRoPE |
| KV / raw KV | Attention Key / Value，注意力键和值；raw为未施加视频RoPE的缓存表示 | 缓存由各层历史隐藏状态计算；不是保存clean latent，也不是可以任意无损重算的位置元数据 |
| Persistent KV / clean commit | 持久KV / 在干净端点提交缓存 | 一个块生成后按约定条件计算并保存历史KV，后续块复用 |
| prefix / current-prefix | 非action的公共前缀条件 / 当前块前缀反馈 | 含原生条件token；位置、可见性和反馈协议不能随意改变 |
| I0 / Single I0 | 初始图像 / 单初始图像条件 | 使用一张起始图作为原生视觉条件 |
| chunk / C1、C2… | 视频块 / 第1、第2…块 | V3首块12 latent对应39 RGB帧；后续每块5 latent追加17 RGB帧 |
| latent / RGB | 模型潜空间表示 / 解码后图像像素 | latent数与视频帧数不同 |
| AA / AD | 先A后继续A / 先A后切换D | 动作分支标签；实际切换块以各实验协议为准。A/D为相反横向控制，画面方向受场景影响 |
| LoRA / QKV | Low-Rank Adaptation，低秩适配 / Query、Key、Value投影 | released Action LoRA与新增学生adapter分开记录 |
| target time / finite map | 目标时间 / 有限时间间隔映射 | AnyFlow学习从当前噪声时间到指定目标时间的映射 |
| teacher / student / target | 教师 / 学生 / 监督目标 | 教师提供参考，学生被训练；target不自动等于正确标签 |
| GT / teacher forcing | Ground Truth，真实记录 / 使用给定真实或教师历史条件训练 | 给定历史局部续写与自身历史自由生成须分开 |
| on-policy | 使用当前学生自己生成的样本 | EXP-008只覆盖固定C1后的当前块，不代表完整长程自生成训练 |
| SGF | Self-Gradient Forcing，自梯度强制 | SolarWM Stage2的方法；本项目归为V4研究路线 |
| DMD / critic / fake score | Distribution Matching Distillation，分布匹配蒸馏 / 可训练评分模型 / 学生分布的score估计 | 冻结teacher与可训练fake角色协同更新student；工程跑通不代表视频通过 |
| VAE / PSNR | Variational Autoencoder，变分自编码器 / Peak Signal-to-Noise Ratio，峰值信噪比 | VAE做视频编码/解码；PSNR用于重建比较，不等于生成质量 |
| E2E | End-to-End，端到端 | 包含完整流程开销；sampling秒数不能直接当完整速度收益 |
| PASS / PARTIAL / FAIL / NOT_TESTED | 通过 / 部分成立 / 失败 / 未测试 | 工程、动作、画质与效率分别验收 |

数字例子：**V3-FM-8NFE 的“3”是研究版本，“8”是每个视频块的采样网络求值次数。AF-train-32updates + AF-8NFE 则是“训练32次更新的学生，用8次求值生成一个待生成块”。**
