# Stage1 全部Q/K/V/out/FFN的rank8候选

状态：CPU预检查完成；native-FP32 tail16已完整评测且失败，真实GPU0单次更新已完成，正在从自身step01恢复至16次。初始A/D固定噪声验证的8个样本记录逐项与旧native初始化相同，见gpu_initial_validation_snapshot.json。只使用GPU0、reserve6；父队列失败则停止，父模型任一设置通过短片动作数值gate则暂停本候选，先等待视觉审查。

目前小规模AnyFlow训练只优化最后16块的QKV（3,440,640参数）。官方H3 Stage1覆盖50个主块和2个token refiner的Q/K/V/out/FFN，共312个线性投影。新候选保持原生FP32数值策略、相同visual/action初始函数、数据/seed/shift2.22/RGB dual/chunk5/history5/action routing与feedback，只改变训练参数化：

- 原有visual QKV和action adapter冻结保留。
- 全50+2块上增加零初始化LoRA，rank=alpha=8；DiffSynth虽将QKV融合为一层，仍给Q/K/V各自独立A/B因子。
- 208个实际wrapper对应312个逻辑线性投影，43,237,376参数，正好是官方rank384参数量的1/48。
- 新bank的alpha/rank=1；旧tail16 adapter的scale=1/8。因而这不是“只多几个块、其他优化参数化严格相同”的消融，而是更接近官方覆盖与缩放的受控Stage1候选。初始输出保持相同，不直接把任何潜在收益归因于某一个投影。
- 冻结time-MLP；loss仍为TF-AnyFlow v1.5；旧clean CPU raw KV仍每次更新后重建。没有新增anchor、action residual、attention拓扑或Stage2目标。

权重+梯度+Adam的FP32存储预算约659.75MiB，不包含激活、33B base和旧冻结adapter。GPU能否fit必须通过1次真实更新测量，不能仅凭参数内存估算承诺。

CPU检查：38项集成和6项官方loss/gradient oracle通过（后者需PYTHONPATH指向SolarWM/src；最初完整测试未设置该引用路径）。新增测试使用真实小H3、FP32/BF16、实际CPU offload wrapper和非零旧visual adapter，验证新bank初始化输出逐bit相同、RNG不变、各组梯度有限非零、更新后base/visual/time不变、KV只读及保存回载相同。连续4次与2+恢复到4次的adapter/Adam/RNG/loss/gradient逐bit一致；旧native/tail checkpoint通过新trainer恢复，也与原连续训练逐bit一致。scope/rank/alpha改变不能精确恢复。

有限GPU队列：相同native初始化→1次真实更新→核对初始A/D固定noise验证与旧native初始化完整记录相同→恢复自身Adam/RNG到16次→39f A/D4和8 steps/chunk。每4步保存。大bank另存stage1_lora.pt，推理必须同时加载匹配的冻结visual、action、AnyFlow time和bank checkpoint，不能漏掉bank后误评旧模型。

本实验继续使用冻结的隔离runtime；2026-10-08 07:50时段，可选全覆盖LoRA及FM/AnyFlow评测入口已合入主代码和submission，完整测试67 passed。fullscope_runtime.patch是相对准备时主代码的候选补丁；run.json为队列状态。已有真实单次更新完成证据，尚无画质收益结论，不替换会议视频，不扩124帧，不启动Stage2。

真实单次GPU更新及续训审计见[GPU_RESULTS.md](GPU_RESULTS.md)。
