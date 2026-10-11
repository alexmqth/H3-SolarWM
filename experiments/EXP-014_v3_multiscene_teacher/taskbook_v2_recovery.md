# EXP-014/v2 — 补齐两次中断的AD分支

2026-10-11，Judge。**只授权CPU恢复入口准备；两次恢复GPU运行仍需新的Judge marker。** 原v1代码、配置、日志、账本和已生成C1/AA/完整scene保持冻结。

## 原因与范围

s1_7199292c、s3_b784d995进程分别在AD完成17/19步后消失；预记账分别为78/80 sampling、各1commit/2decode，无AD endpoint/video和stage_stop。不能宣称这两路生成质量失败。可见cgroup OOM计数均0，内核日志不可读，实际退出原因尚未证实；Worker应保存原exec退出信息，缺失则如实记录未知。

s0已完整有限可行、quality PARTIAL；s2完整协议/结构通过，但AA/AD同向，D切换PARTIAL，不能当正确AD动作监督。不得为s2改动作/seed、追加C3或重跑。

## 唯一研究问题 / 控制变量

补齐预定四图中缺失的两条AD，判断该固定teacher集合的实际控制与结构覆盖。仅重做未产出视频的AD，从各自已有冻结C1、自己的已保存sigma0 cache和原C2初始噪声开始。权重、动作条件、Single I0、native时间/位置、precision/attention、sampling30、解码和旧RGB保持v1不变。不重新生成C1、AA，不重新commit，不重做编码，不换图/seed。

## CPU准备与审核

新增独立recovery runner/config/source manifest和每scene恢复账本，不修改v1冻结文件或覆盖原AD半成品。核对实际C1/AA/fixture/cache摘要、全部50层index0、缓存容量、旧RGB、原AD动作span和原始噪声/native30网格。恢复结果写独立目录，明确关联原attempt，训练target索引也另存；保留原始未关闭账本。

提交CPU预检与可运行命令。Judge核对后签发只覆盖s1/s3的恢复marker。一次恢复只跑一个scene，两scene顺序执行，避免重复三进程并发这一未排除因素；不会声称串行已证明解决中断原因。

## 修订预算 / Stop

新增最多**60 sampling forward、2decode、0commit/0encode/0train**，每scene只允许一次30forward/1decode恢复、最多600GPU秒，两次共1200GPU秒。全任务预记账denoiser上限从364调整为**402**（原342已启动/预记账+60），包含两次中断中预记账但未确认完成的forward。全任务decode仍最多12，教师GPU秒累计仍≤5400、编码≤600，不重置失败成本。旧两中断按原exec可证耗时计费；缺终点时采用Judge保守上界，每scene≤515.555秒，原账本gpu_seconds=0不代表免费。

沿用磁盘≥60GiB、allocated≤44GiB；优先空闲GPU0，09:00后项目≤3卡；08:40后不启动新恢复，09:00停止本任务入口。保留进程PID、执行退出状态、开始/结束时间和调用前账本。若再次无解释中断或模型/缓存协议失败，停止恢复并按未完成归档，不自动第二次重试。

## 交付与判断

两条AD端点/完整56帧/新增17帧、完整动作与结构观察、与已有AA成对比较、旧39RGB不变、实际KV来源/数据不变性、总成本和中断说明。逐scene明确可用/切换PARTIAL/NOT_EVALUATED，不能把缺失或弱响应teacher当正确监督。此补齐任务不授权任何训练；是否开展独立多场景student仍由完整teacher证据与ROI决定。
