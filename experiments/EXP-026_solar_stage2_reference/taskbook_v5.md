# EXP-026 / plan v5：用户目录CUDA12.4工具链与四卡加载

2026-10-16 HKT。用户已明确批准开始匹配独立工具链修复。Judge授权原Exp Worker立即执行；CPU门控通过后自动进入一次四卡load-only，不等待额外marker。plan v5是任务修订编号，不是模型V5。

## 目标与范围

保留torch2.6.0+cu124/Python3.10及官方FlashAttention/flex协议，用与torch匹配的独立CUDA12.4开发工具链构建可导入的flash-attn，再回答真实33B+released Stage2-fix EMA是否可在四张L40上完成加载/分片。此前v3实际NCCL/shard4已到，FlashAttention ABI导致FSDP前失败；v4仅CPU诊断，累计旧GPU费用105.26510238647461秒，不得清零。

Original H3/SolarWM权重、rank384 LoRA、BF16/FP32单元、camera/time/RoPE/KV/NFE不变。禁止替换为SDPA/eager、升级torch/diffusers/transformers、修改CUDA driver或全局系统CUDA。真实相机数据仍缺失，本轮没有视频生成/训练授权。

## A：独立工具链与构建

1. 新代码/报告放memory_load_v5/；工具链、构建缓存和overlay放/home/qma/work/GWM/.toolchains/exp026_v5/等专用用户目录，记录精确路径。不能覆盖原venv、v4失败overlay、官方checkout或系统文件。
2. 从NVIDIA官方发布源取得CUDA12.4的最小开发组件（nvcc、ptxas、headers、cudart开发库及实际必要组件），允许官方Python/Conda模块包或仅解包官方installer到用户目录。不得使用sudo或安装driver/system-wide包；不下载不需要的cuDNN/训练数据/模型。使用任务进程局部CUDA_HOME/PATH/库路径，不能改zshrc/bashrc。
3. 实际验证nvcc版本、headers/runtime库来源和torch的_check_cuda_version，并记录torch CXX11 ABI=False。FlashAttention使用官方v2.8.3源码固定tag/commit；强制从源构建，不能再次取同一已失败wheel。匹配当前torch的C++ ABI，记录编译命令中的ABI标志；禁止用跳过ABI/CUDA版本检查代替修复。
4. 在隔离overlay以--no-deps安装构建结果，torch等现有依赖版本不变。限制编译目标到当前L40兼容的官方支持架构集合以控制成本，保留其构建配置，不能裁剪H3实际需要的kernel/功能。不复制基础权重或整个大型venv。
5. CPU真实import torch/flash_attn/flash_attn_2_cuda，执行Diffusers原Flash backend requirement检查；CUDA保持未初始化，必须没有导入错误/回退。检查新so实际路径与SHA、torch版本/ABI、工具链版本，冻结ENV_PREFLIGHT和BUILD_MANIFEST。对端到端GPU函数仍不得宣称测试通过。

CPU/网络预算：总准备窗口90分钟，实际编译累计≤60分钟；MAX_JOBS≤4、nvcc内部线程≤1（可配置时）、同一时刻最多8个CPU执行worker，任务聚合RSS≤48GiB。新下载累计≤8GiB、新磁盘≤20GiB，至少保留200GiB可用盘。最多一个固定源码版本，允许在同一构建树内修复工具路径/缺必要header等普通构建问题并继续，累计不超过两次build命令和上述总预算；不得无限换包/版本或增加预算。网络单次无进展15分钟报告。下载前核对来源/大小，记录bytes/SHA。超过预算或必须改变torch/backend才可继续则立即报告Judge，不自动扩大。

## B：门控通过后自动执行一次真实四卡load-only

复用v3已审计的SP4/DP1/world4/物理shard4/单样本config与源码patch；只改隔离环境和必要日志/输出路径，不重新研究并行策略。不重复无关CPU/Gloo测试。新load-only入口及watchdog必须从v5授权读预算、独立attempt lock，不复用v3锁或覆盖旧证据。

- 优先物理GPU0/1/6/7，启动前再次确认无其他compute进程、每卡占用<1GiB，记录UUID。GPU2–5不动。
- **最多一次4卡×600秒=2400GPU秒**；独立watchdog580秒TERM、600秒内只清理本任务进程组。初始化/加载/FSDP/EMA/核对/释放全部计时。
- 初始MemAvailable≥400GiB；运行时MemAvailable<128GiB、任务聚合RSS>400GiB、每rank allocated>40GiB、OOM、collective错误、NaN、wrong shape/dtype/weights均停止。记录max_memory_allocated和max_memory_reserved，CPU RSS/物理显存按采样峰注明。不得依赖swap放模型。
- 保留base strict-load receipt、LoRA624 tensors/2,075,394,048参数、实际FSDP shard4/replica1、Stage2-fix EMA step3900加载及稀疏数值对照，不能全量gather33B到单GPU。各rank须加载相同环境/patch。
- 0denoise/codec/model forward、0backward/updates、0视频。结束点在reader/sampler前；真实数据未读、encoder/silence合同未验收必须明示。缺官方样本不阻塞本纯加载。
- 最晚08:48 HKT启动，09:00前释放；夜间全项目≤8空闲卡/09:00后≤3卡。资源临时不足即报告，不抢卡、不创建无人值守轮询队列。
- GPU失败不自动重试或修改配置，本轮一次预算用尽后提交结果。CPU阶段通过可直接启动本B，无需再请用户批准。

## 交付与Judge决策

工具链与source来源/版本/URL/SHA/容量、完整安装构建命令日志、ENV_PREFLIGHT、实际so/依赖解析来源、代码/config manifest；GPU命令/PID/UUID/阶段日志/退出及总账、逐rank峰显存与主机RSS、strict base/EMA receipt、WORKER_REPORT及根report.md。旧105.265GPU秒与本轮分别列并累加。

修复/加载成功只验收环境与加载可行性；FlashAttention真实kernel、KV、相机reader/视频仍未测。若本轮仍受阻，明确停在何处及最小外部变化，不为1个依赖无限扩展研究。Worker在开工、工具链就绪、编译开始/完成、GPU启动/结束时消息Judge；Judge审核/Git，Worker不操作Git。
