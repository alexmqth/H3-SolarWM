# 报告常用缩写与计数

| 名称 | 全称 / 中文解释 | 本项目怎么读 |
| --- | --- | --- |
| FM | Flow Matching，流匹配 | 基础模型的速度场建模；当前 FM 减步评估使用原权重，不新增训练 |
| FM30 / FM8 / FM4 | 每块30 / 8 / 4步普通FM采样 | 数字是生成采样步数，不是帧数、训练步数或版本号 |
| AF / AnyFlow | AnyFlow 少步有限映射方法；AF为本项目简称 | 新增目标时间条件并训练 finite map；与直接减少FM采样步数分开 |
| AF8 / AF4 | AnyFlow每块8 / 4次网络求值 | `step32` 则表示学生累计32次训练更新 |
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

数字例子：**V3-FM8 的“3”是研究版本，“8”是每个视频块的普通采样步数。AF step32 / 8NFE 则是“训练32次更新的学生，用8次求值生成一块”。**
