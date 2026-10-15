# EXP-022/v1 — V3普通FM真实训练入口与2-update profile

2026-10-15 HKT，Judge批准立即执行。Research Track：Branch B；Parent：冻结V3 + EXP-019固定动作布局；承接EXP-021已验收CPU清单。执行者：既有Exp Worker会话01a0ece6-63b6-7f42-8c39-cd3ec39a9c80。

## 核心问题与授权

在不改变V3条件/attention协议的情况下，真实33B能否完成普通FM的forward/backward/update，成本多少？这是工程和预算测试，不是训练效果评估。**本任务同时授权CPU实现与下述有限GPU profile。CPU及来源自检通过后Worker直接开始GPU，不需再等待额外Judge marker。** 已完成EXP-021不重复运行。

## 冻结变量与允许修改

Original H3 + released Action LoRA冻结；Single I0、native time、Global RoPE、12+5分块、current non-action prefix/own-action/action feedback、clean raw KV、EXP-019精度与布局不变。普通FM目标为epsilon-clean，z_sigma=(1-sigma)clean+sigma*epsilon。不含AnyFlow target time、DMD或额外损失。

在本EXP新建可微wrapper，复用现有模块，不能删改冻结推理入口的no-grad guard来冒充完成。新adapter采用EXP-021候选tail8 QKV rank8、零增量初始化；实际层名/可训练参数量必须核对。本配置仅用于测训练通路，不是正式容量选择。只允许新adapter进入optimizer，AdamW lr=1e-4，明确其他参数、seed13、batch1、无梯度累积；训练状态与CUDA/CPU/RNG、optimizer需可恢复。允许activation checkpoint/CPU offload，但保持数值/attention语义，记录具体策略。

## 执行顺序

### A. CPU实现和数据物化

1. 复用EXP-021冻结DATA_MANIFEST，物化剩余28train+8dev的56帧RGB；已有4条复用。保持源索引、分辨率832×480、24fps和17动作。完整解码并检查来源/帧数/黑帧/损坏、动作时间网格，保存精简contact sheet和质量排除原因，不换窗口挑结果。此阶段不做VAE/text编码，不下载新数据。
2. 实现独立可微wrapper与profile runner，核对当前块物理未来截断、prefix位置、cache read、clean commit无梯度。profile使用**b784d995已物化的原56帧样本**及EXP-016已有I0/prefix12/full17 latents、EXP-019 E fixture；不等待新36条全部物化，不重新编码。
3. 冻结源码与输入SHA、配置、启动命令、计数器、watchdog和空闲卡检查；先CPU自检，再进入B。冻结之前必要的CPU修复可直接进行。

### B. 有限GPU profile（直接授权）

单张空闲L40，优先GPU0，若被占则选择另一张实际空闲卡，启动前再次检查，不抢占。保持EXP-019 precision/backend。

1. checkpoint0：零adapter条件下，用同一GT C1、C2初噪、sigma=0.5，比较冻结V3与新wrapper的velocity；比较50层clean KV及位置。检查未来动作改变不影响C1可见输入。记录max/relative误差、dtype/backend，不能靠放宽容差或改变条件掩盖差异。若同backend应逐值一致；非零差异先停止训练并报告，不能擅改协议。
2. 固定C2，sigma=0.5与0.75各进行1次真实FM forward/backward/optimizer update，共最多2updates。每次更新前按当前adapter重建GT C1 clean KV，避免更新后复用旧参数缓存。cache与GT历史不接收梯度，冻结base/released LoRA无梯度；新adapter梯度finite且至少有有效非零更新。记每次loss、梯度范数、参数变化与各阶段耗时。
3. 保存checkpoint0和训练状态，第二次后保存并重载，固定状态检查重载输出一致，检查optimizer/RNG。重载探针不做额外optimizer update。输出不是视频，本阶段不渲染生成质量对比。

## 硬预算和停止条件

- CPU实现/物化首轮≤2小时，≤8线程，host RAM≤128GiB，新磁盘≤12GiB。实物可压缩/复用，原始大tensor不提交Git。
- GPU最多1卡，累计进程wall≤3600 GPU秒、peak allocated≤44GiB；所有失败加载/调用计入。最多20次顶层DiT forward（包括commit/基准/比较/训练/重载），最多2次backward、2次optimizer update；0text encoder、0VAE encode/decode、0新视频生成采样。backward内activation-checkpoint层重算单列次数/时间，不伪称免费，也不与顶层forward重复计数；所有成本均受3600秒限制。
- GPU阶段用独立watchdog，预算在每次调用前记账；显存allocated/reserved、host RSS、CPU KV bytes、加载/commit/forward/backward/update/保存/重载耗时全部记录。
- OOM、NaN/Inf、错误梯度、前向/KV不等价、来源不符或超预算：停止受影响GPU流程，保留日志；不自动重试、不增加卡数/rank/更新。CPU数据物化可独立完成。若CPU无法实现协议正确的可微入口，交具体阻塞与最小修复建议，不能换成旧协议训练冒充完成。
- 项目资源总规则仍是夜间≤8空闲卡，09:00后≤3卡；本任务本身≤1GPU。没有为了占满卡追加任务的授权。

## 验收、交付与后续

提交run/config/source manifest、数据物化清单与质控、CPU测试、GPU原始日志/账本、可训练参数清单、完整2update或失败报告、checkpoint本地路径/SHA、峰值内存和吞吐。PASS仅表示真实训练通路可用，不能称动作或画质改善。基于实测估计后续32条小数据拟合的预算，并提出匹配checkpoint0/训练后、NFE/历史条件/动作切换的最小评测；此处不授权扩大训练。

写本EXP的WORKER_REPORT.md及根report.md；开始GPU和完成/失败均给Judge发消息。SolarWM标准推理另案，不混用相机/人物控制接口。Judge负责审核和Git push，Worker不操作Git或冻结旧实验。
