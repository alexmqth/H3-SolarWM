# EXP-026/v4：隔离环境修复后一次四卡加载验证

2026-10-16 HKT。用户已批准四卡可行性工作，Judge基于v3 ABI失败授权必要修复与一次新尝试。v3一次失败已收口，不能覆盖或清除105.265GPU秒开销。原Worker立即开工，CPU依赖与既有资源门控通过后自动执行，不需再次请求用户/marker。

## 问题与固定条件

只修复flash-attn原生扩展与当前torch2.6.0+cu124/Python3.10/C++ ABI不匹配。保留官方FlashAttention/flex语义，不用SDPA/eager回退伪装修复，不升级torch/diffusers/transformers，不改相机/time/RoPE/KV/采样。复用已审核v3四卡SP4/DP1/shard4/单样本load-only路径；原SolarWM、既有虚拟环境及v1–v3冻结证据保持不变。

## 阶段A：CPU环境修复（立即执行）

1. 记录Python/torch/CUDA build/C++ ABI、编译器/系统库及失败binary SHA。使用独立包overlay或独立venv引用原依赖，避免复制全部大型torch/模型，不改共享环境。新实现放memory_load_v4/。
2. 优先选Dao-AILab/flash-attention官方发布的匹配Python/torch/CUDA/C++ ABI wheel，保留版本、URL、SHA和安装命令，安装不自动升级依赖。找不到匹配wheel时允许一次官方源码限定编译（使用现有CUDA toolkit，MAX_JOBS≤4），不下载/安装新的系统CUDA或系统库。不要无限尝试多个版本。
3. CPU实际import torch、flash_attn及其原生extension，调用与本次报错对应的diffusers backend requirement检查（保持CUDA未初始化）；确认导入路径指向新overlay、torch仍原版本、没有fallback警告。检查模型backend配置仍flash/flex。入口/各rank进程必须使用同一新环境路径。若某检查确实必须初始化CUDA，先在CPU报告，不自行做额外GPU kernel smoke。
4. 沿用v3配置与来源SHA，只新增环境/entry路径和必要日志字段。补记max_memory_reserved；不重复所有已通过CPU/Gloo组测试，仅核对必要变化。冻结ENV_PREFLIGHT.json、源码/patch/config/launch环境manifest。

CPU总窗口45分钟（其中若需编译≤30分钟），并发≤8、编译MAX_JOBS≤4、聚合RSS≤32GiB；新下载≤2GiB、新磁盘≤8GiB。允许下载必要官方flash-attn包，不下载数据/模型。超限、找不到匹配包或必须升级torch/改backend时立即报告，不启动GPU。

## 阶段B：一次真实四卡load-only，A通过自动执行

沿用v3：优先物理0/1/6/7、world4/SP4/DP1、物理shard4、真实base与Stage2-fix EMA step3900。新gpu_attempt_001在v4下，独立attempt lock和日志，不用v3旧锁或覆盖旧结果。

最多一次，4卡×600秒=2400GPU秒；watchdog580秒TERM/600秒清理。0去噪/codec forward、0backward/updates、0视频。任务前查四卡无其他compute进程且<1GiB；初始MemAvailable≥400GiB，运行≥128GiB、本任务聚合RSS≤400GiB、每rank allocated≤40GiB。只清理本任务进程组。09:00前释放，最晚08:48开始。资源变动不抢他人卡，不自动轮询队列。

所有加载/EMA校验沿用v3，真实reader仍未验证；不得编造encoder/silence合同。阶段日志准确区分strict base完成、LoRA注入、真实FSDP包装、EMA恢复、数值样本和释放。失败不自动第二次重试或更换配置，原始计时/计数保留。

## 交付与验收

ENV_PREFLIGHT、依赖安装/编译日志及URL/SHA、隔离路径、代码/config来源、真实GPU命令/PID/UUID/退出与各rank日志、预算总账、GPU allocated/reserved峰及采样物理显存/CPU RSS峰、base/EMA receipt、WORKER_REPORT。v3与v4费用分别列并累计。加载PASS只代表加载/参数分片可行，不代表attention kernel forward、KV、视频或动作能力通过。

Worker开始、依赖修复、GPU启动、完成/失败主动消息Judge；不操作Git。该轮必要依赖修复在范围内，不把环境错误变成新研究分支。
