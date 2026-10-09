# Stage1 参数策略复核与修正（2026-10-08）

**官方 MiniMax-H3 Stage1 冻结目标时间 MLP，仅优化 LoRA。本地已完成的16-update pilot额外训练了该 MLP，是一个变体，不能写成官方训练参数策略的复现。** 这不推翻已有 loss/gradient 公式对照，但修正了“time MLP必须实际更新才算AnyFlow实现通过”的错误验收条件。

## 官方源码证据

固定版本：`ce1da4e7705391eda8eeda6016c0fd3f614b975e`。

- [`runtime.py` 的 Stage1 初始化](https://github.com/Junchao-cs/SolarWM/blob/ce1da4e7705391eda8eeda6016c0fd3f614b975e/src/solarwm/backends/minimax_h3/runtime.py#L493)：先 `enable_anyflow` 克隆目标时间 MLP，随后对整个 transformer 执行 `requires_grad_(False)`，再注入 LoRA。
- 同一文件的 optimizer 使用 `self.lora.parameters`。
- [`lora.py` 的目标模块与断言](https://github.com/Junchao-cs/SolarWM/blob/ce1da4e7705391eda8eeda6016c0fd3f614b975e/src/solarwm/backends/minimax_h3/lora.py#L211)：目标是50个主块和2个refiner块的Q/K/V/out/FFN，并断言 **all and only H3 LoRA parameters must be trainable**；不包含时间MLP。

冻结时间编码不妨碍AnyFlow学习。固定的 `(t,r)` 编码向模型提供起点和终点，LoRA学习在这些时间条件下的有限步映射。模块有独立参数不代表这些参数必须进入优化器。

机器可读证据和源码哈希见 [official_parameter_policy_audit.json](reports/stage1_anyflow/03_anyflow_trials/early_pilot/official_parameter_policy_audit.json)。未为这次核查下载官方4.15GB权重；源码已足以确认参数策略。

## 本地影响与实验边界

旧pilot优化了3,440,640个tail16 QKV参数和15,835,008个目标时间MLP参数。它已完成真实33B训练和视频评测，但4-step画面严重退化，8-step动作gate仍失败。保持这些记录，不把它们重标为冻结时间MLP的结果。

已在独立实验副本中增加 `--train-target-time / --no-train-target-time`，默认冻结。小型实际H3的两步CPU训练验证了：

| 分支 | QKV更新 | 时间MLP更新 | 初始时间MLP权重 |
|---|---|---|---|
| 默认冻结 | 是 | 否，严格不变 | 两分支完全相同 |
| 显式 `--train-target-time` | 是 | 是 | 两分支完全相同 |

证据：[parameter_policy_cpu_validation.json](reports/stage1_anyflow/03_anyflow_trials/early_pilot/parameter_policy_cpu_validation.json)。该CPU检查仅验证参数策略；随后真实33B冻结分支16次更新与6条39帧视频均已完成，三组采样动作gate全部失败，见[冻结分支结果](reports/stage1_anyflow/03_anyflow_trials/frozen_time_snapshot/README.md)。

真实33B GPU2分支现已完成前4次更新：逐张量核对step00/04，16个QKV块全部变化，目标时间MLP权重严格不变；32个Adam参数状态与adapter SHA256校验通过。见[parameter_policy_gpu_step04.json](reports/stage1_anyflow/03_anyflow_trials/early_pilot/parameter_policy_gpu_step04.json)。这确认冻结策略在真实训练中生效，后续16-update评测已完成，质量未通过。

## GPU 2对照进度

GPU任务串行执行，等待中的控制进程不加载模型：

1. 原pilot已完成FM16、step00/04/08/12学习曲线和clean-history诊断。
2. 同一旧AnyFlow step16与step00的uniform4 A/D已完成；分离度-0.4424/-0.5705且后段雾化，未通过。
3. 同初始化、同数据/噪声/anchor/chunk/routing/shift/loss、同16次更新，只冻结时间MLP的33B训练已完成16次更新；native4/native8/uniform4的A-D分别0.0031/0.2293/-0.3223，均未通过。

源运行目录：

```text
H3-World/outputs/2026-10-08-02/stage1_anyflow39_pilot/
H3-World/outputs/2026-10-08-03/stage1_anyflow39_uniform/
H3-World/outputs/2026-10-08-04/stage1_anyflow39_frozen_time/
```

两项新实验使用各自 `runtime/` 的代码副本，记录源码哈希和共享依赖哈希。原pilot已完成并退出后，冻结策略、明确的采样网格选项及恢复CLI已合入主源码/提交包；36项源测试通过。原trainer/benchmark保存在pilot的`original_source/`。变更前后的补丁仍以 [freeze_target_time.patch](reports/stage1_anyflow/03_anyflow_trials/early_pilot/freeze_target_time.patch)、[benchmark_uniform.patch](reports/stage1_anyflow/03_anyflow_trials/early_pilot/benchmark_uniform.patch)保存，供审计；旧结果保持原参数策略标签。两个排队实验的隔离副本未随主源码合入而改变。

冻结分支保存 `trainer_state.pt`（Adam、logical/global RNG、更新数及adapter SHA256），并支持 `--resume-from <checkpoint目录>`；`--steps` 是目标**总更新数**。小型实际H3的CPU验证中，冻结时间AnyFlow、可训练时间AnyFlow、FM三种分支的连续4步与2步后恢复到4步，adapter/Adam/RNG及loss/gradient记录逐项完全相等；错误LR、错配权重和缺失optimizer状态会被拒绝。见 [resume_validation.json](reports/stage1_anyflow/03_anyflow_trials/early_pilot/resume_validation.json)。此外，真实33B从GPU2的step08迁到GPU0时已逐项核验adapter、Adam与logical/CPU/CUDA RNG一致，并实际完成后8次更新，见[迁移审计](reports/stage1_anyflow/05_runtime/gpu0_migration/gpu0_restore_audit.json)。这证明状态恢复一致，不是与未中断CUDA整条训练轨迹的对照。旧pilot没有保存optimizer/RNG，不能把旧权重热启动称为精确续训。安装文档中的DiffSynth依赖后，可从submission运行 `python reports/stage1_anyflow/03_anyflow_trials/early_pilot/verify_resume.py --work-dir /tmp/h3_resume_check_<新目录>` 复核CPU续训；工具拒绝覆盖既有目录。

参数策略差异是已确认事实，但上述对照已表明单独冻结目标时间MLP没有修复本轮短片。冻结时间MLP后仍只训练tail16 QKV rank8，和官方全块QKVO/FFN rank384有很大容量差异；其他已列出的数据、音频、两流/缓存训练差异也仍存在。Stage1质量验收标准不降低，Stage2保持暂缓。

后续[同协议训练量对照](reports/stage1_anyflow/03_anyflow_trials/duration32_snapshot/README.md)已完成32次更新，A-D从16次的0.2293降为0.1424，方向和画面未修复。64次分支在第33次更新前暂缓，完整checkpoint/Adam/RNG保留。独立FP32数值诊断完成后，当前优先进行[原生FP32训练候选](reports/stage1_anyflow/04_numerical_checks/native_fp32_candidate/README.md)：仍冻结time-MLP、只训练tail16 QKV，但保留官方FP32输入/输出/时间边界及有限差分输入。精度策略改变不是legacy checkpoint的精确续训；质量未通过前不会扩展Stage2。
