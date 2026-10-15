# EXP-026/v3：四卡 SP4/DP1 真实加载验证

2026-10-16 HKT。用户已批准开始四卡方案。Judge授权原Exp Worker立即实现，并在下述CPU/资源检查通过后自动执行一次真实GPU加载；无需等用户或Judge再次发marker。已有v1/v2和正式V3证据冻结。

## 唯一问题与范围

同一公开SolarWM-H3 Stage2-fix 33B base + released rank384 EMA，能否在四张L40上，通过CPU加载与物理四卡冻结参数分片，完成真实模型包装和权重恢复？这是加载可行性实验，0去噪/codec forward、0更新、0视频，不把成功当生成能力通过。真实数据访问缺失不阻塞纯加载；不构造假数据绕过reader。

Parent：官方checkout ce1da4e7705391eda8eeda6016c0fd3f614b975e及memory_load_v2独立patch。固定158帧Stage2允许SP4且world可被4整除；Judge已验证world4/SP4/sample_count1通用与H3配置合同。v2 patch的world8/shard8硬编码是本轮要调整的范围。

## 允许改动及CPU门控

全部新代码放memory_load_v3/，在独立临时或实验源码副本应用patch；不修改官方冻结checkout或memory_load_v2。

1. 改为world_size4、SP4、DP1、sample_count1、balance_by_dataset=false；model CPU deferred load开启，显式frozen_base_shard_size=4。保持原native时间、相机/MM-RoPE/PRoPE、dtype、adapter、attention、KV/采样配置；配置变化仅资源/样本数/路径。
2. 复用现有frozen_base_groups，以torchrun LOCAL_WORLD_SIZE=4建立物理四rank shard及replica1。不是让默认DP1分片。四个可见设备映射物理0/1/6/7，记录GPU UUID。
3. CPU测试覆盖官方validate_h3_config、拒绝whole-model .to(cuda)、base CPU/dtype、LoRA ignored参数位置/调用顺序及EMA小模型拷贝；现有生产分组代码四进程Gloo检查一次即可。已经验证的无关分支不重复扩展。
4. 制作只负责真实加载的入口：初始化四rank NCCL→CPU strict-load真实33B→注入官方LoRA→分片→恢复本地真实Stage2-fix EMA→核对descriptor/形状/有限参数及受控数值样本→记录资源并释放退出。必须在reader/sampler/codec构造前结束。正式checkpoint加载receipt必须明确step3900、来源与SHA既有证据；若权重loader需data contract，只能验证checkpoint自身与配置的结构一致性，明确缺失encoder/silence数据验证，不能伪装完整输入合同通过。
5. CPU冻结config/patch/源码SHA、命令、预计count、watchdog与资源检查；通过则直接GPU执行，不因仅欠真实样本而等待。

## 资源预算与启动条件

- CPU实现窗口45分钟，最多8 CPU worker、CPU小模型检查聚合RSS≤32GiB、新文件≤2GiB；不下载新模型/数据、不复制基础权重。
- **GPU授权一次加载尝试，最多4张L40×600秒=2400 GPU秒。** CUDA/NCCL初始化、加载、包装、EMA恢复、核对、释放全部计入；0 DiT/denoise/codec forward、0backward/updates。
- 优先CUDA_VISIBLE_DEVICES=0,1,6,7；启动前再次确认4卡均无其他compute process且占用<1GiB。不使用2–5，不修改或终止他人进程。
- 初始MemAvailable≥400GiB，运行时MemAvailable<128GiB或本任务聚合RSS>400GiB即停止本任务；记录主机可用内存与各rank RSS采样峰，监测swap并避免依赖swap容纳模型。400GiB是本轮保守启动条件，不是实测峰。
- 每rank CUDA allocated>40GiB、OOM、NCCL/FSDP错误、NaN、checkpoint/shape/dtype不符立即停止全部本任务rank。外层独立watchdog在580秒TERM，600秒内清理本任务进程组；不杀其他任务。
- 夜间最多项目8空闲卡、09:00后最多3卡；本四卡任务不得跨09:00，最晚08:48启动，否则报告等待时段。现在约01:04 HKT。
- 仅一次GPU尝试，失败不自动重试/换配置/改精度/缩模型；保存失败调用和资源，不把失败预算抹掉。无GPU常驻队列。

## 验收与交付

CPU结果、可复现patch/config及原始diff、load-only入口、冻结manifest；GPU_LAUNCH（设备/UUID/PID/命令/开始结束/退出）、各rank日志、全任务预算账本、base strict-load与EMA恢复receipt、逐rank与合计RSS/GPU allocated/reserved采样峰及阶段耗时。说明FSDP的shard4/replica1和SP4/DP1；权重恢复后校验不能全量gather33B回单卡。若加载成功，只接受加载/参数分片可行性；attention运行、KV、真实reader和视频仍NOT_TESTED。

Worker实现开始、GPU启动、完成/失败均主动消息Judge；更新根report.md并归档WORKER_REPORT.md。Judge负责审核/Git，不自行运行研究GPU。CPU及资源门控通过后自动连续执行本GPU阶段，任务结束立即释放4卡。官方真实视频待数据权限和本结果后另定预算，不在本v3授权内。
