# EXP-007 AF0：V3-AF CPU预审

2026-10-11，Judge实施与核查。**CPU_PASS，未加载33B、0GPU。** 结果见[cpu_preflight.json](cpu_preflight.json)，代码[source manifest](cpu_source_manifest.json)。后续必须有真实模型初始化/训练可运行性验证，不将小模型结果当生成能力。

## 实际验证

真实tiny H3（两层、随机权重）走当前model_fn、native packed、current_prefix及raw KV，而非替换模型输出的假函数。首12+续5区间新student入口在未安装target模块时与EXP-005普通FM逐值一致。克隆target-time MLP不改变RNG；r=t初始化输出maxabs0。实际model.forward收到的时序行检查通过：外部sigma=.6映射内部video/text=.4，audio=0；target=.2仅令当前video目标时间=.8，prefix目标行保持其当前时间，future action物理裁剪至17。

Finite-map执行3次detached场估计和1次有梯度prediction；target-time与QKV都有非零、finite梯度。activation-checkpoint反向后历史entry/storage/version及K/V数值均不变，历史张量无梯度。参数更新后重建首窗KV确实变化，证明正式训练不得跨optimizer step复用旧权重cache。解析常场map符号与线性时间场的raw1000导数单位抵消检查通过。

## 与旧训练协议的差异

旧chunk_forward硬编码均匀分块并使用fixed_prefix_timesteps=True；V3要求首12后5与False。新interval_student保留EXP-005所有条件/缓存校验，独立开启allow_grad_read、checkpoint与target_timestep_video，限制Global且不允许带梯度commit。Baseline文件不改。

外部`timestep_video=1000*sigma`进入冻结model_fn后转换为内部原生`1-sigma`；因此旧conditioner的native_time='1-sigma'标签与V3外部入口本身不冲突，不能重复变号。旧checkpoint依然不可直接作为新V3训练产物，因为其prefix/anchor/分块/历史协议可能不同。

目标时间MLP混合应用于原H3 unique time pairs；prefix的target坐标仍等于current坐标。训练后的MLP可能改变包括prefix在内的embedding数值，这是新student权重的一部分，不意味着prefix timestep被固定或attention路由改变。commit使用sigma=r=0，必须由当前student自身构建。

## 后续建议

新student从原H3+released Action LoRA创建，target-time gate0.25、最后8层rank8 QKV LoRA为初始有限容量，base/released权重冻结。先同源实际C1/C2做初始化对角回归，再最多一个logical batch4的完整optimizer update，核查显存、梯度、cache和checkpoint，随后根据实际成本放行最多32 updates的有限试验。训练只使用冻结V3 generated clean latents，分别标明训练端点与未训练的噪声/闭环条件，不能称独立泛化。正式GPU预算由Judge下一版任务书发布。
