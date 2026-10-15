# EXP-027/v2 补充01：修复新审计入口的commit索引并一次恢复

2026-10-16 HKT。Judge已核对真实traceback及冻结coverage.fm_step：C2训练前先以index0完成GT C1 clean commit，新runner把所有审计调用都限制为index1而误拒绝。旧EXP-025允许0/1；故障属于新观测断言，不是训练forward/backward协议失败。用户既有ABot任务授权覆盖此必要修复，无需再次询问。

## 保留与复用

原run_exp027.py、CPU_PREFLIGHT、GPU_LEDGER/GPU_RUN、checkpoint000、PROBE000、ZERO_ADAPTER_REGRESSION与失败报告全部冻结。首次账本369.2553288709605秒、51DiT尝试（50完成+1进入后失败）、0backward/updates/decode；GPU0已释放。保留失败调用，不能清零累计费用。零adapter三路逐值相等/完整KV摘要一致，初始40probes已完成。

## 唯一修复

在独立recovery_01目录创建入口/证据，检查production审计helper的phase映射：commit→index0；forward/backward→index1；own-action/action_feedback保持原检查；未知phase与错index必须拒绝。CPU测试实际helper三种合法路径及非法phase/index/动作模式/反馈，检查实际入口调用该helper。冻结源码和模型数学计算不改。

CPU验证恢复入口路径、导入的旧模块/正式源、输出目录、输入与既有checkpoint0/PROBE0 SHA、噪声/optimizer保存恢复。使用已保存0update状态，恢复optimizer为空以及CPU/CUDA/training-noise RNG，训练顺序仍从第1条开始；若重置optimizer须等同原初始AdamW设置。不能沿用失败尝试已消费一次训练噪声之后的RNG。checkpoint0/初始probe只引用或等SHA复制，不能重复执行零回归/PROBE0/数据编码。

## CPU门控通过后自动一次恢复

Judge授权原Worker CPU上述检查通过后直接恢复train→probe32→train→probe64→必要四视频→条件可选两视频，不需额外marker。首次真实梯度门控仍需通过方可optimizer.step；其调用计入原64updates，不另做GPU profile。

**恢复尝试最多5500单卡GPU进程秒，且含旧尝试累计总≤6000秒；新增286顶层DiT、64backward/64updates、最多6decode/0encode。累计DiT上限337（原336+一次失败commit）。** 使用单张启动时空闲L40优先GPU0；新外层watchdog5480TERM并5500前清理本任务组，若累计剩余更少取较小值。failed/CPU准备后GPU加载/保存/释放均入账。四个必要视频预留800秒；其余资源阈值与样本/噪声/精度/优化器/验收标准完全沿用v2。

这是一轮最后的最小恢复；不自动再重试或扩大GPU预算。CPU失败先报告，不为赶时间跳过检查。完成后提交两次尝试分别及合并账本、首步梯度、更新/probes/视频与报告。失败更正不得覆盖旧证据。
