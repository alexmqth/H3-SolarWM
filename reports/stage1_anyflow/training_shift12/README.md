# Stage1训练时间分布对照：train shift12 / validation与inference2.22

最新：本组已完成，见[FINAL_RESULTS.md](FINAL_RESULTS.md)及完整诊断视频。以下保留启动和执行过程记录。

官方H3 Stage1使用train.video_timestep_shift=12。现有固定序列16次更新中，A的8个endpoint样本都低于sigma0.9（其他样本仍有高噪声监督），D仅1/8达到0.9；同一基础随机样本改shift12时分别为2/8与3/8。覆盖不足是候选原因，尚未证明导致坏视频。

保持full-history、同原始visual/action初始化、全50+2 QKVO/FFN rank8、FP32、LR3e-5、logical batch4、16updates、RGB dual、chunk5/history5、action routing/feedback、原image/prompt/noise不变。只改训练time-pair采样及与其对应的Gaussian权重shift。validation_shift与inference_flow_shift仍为2.22，保持公共held-out raw residual与视频采样协议。两者不能再用同一个参数强制绑定。

CPU17项相关检查通过；默认新入口与旧full4的权重/Adam/RNG/sample loss/gradient逐bit相同。shift12连续4次vs自身step2恢复到4次逐bit相同，改变train shift不能当作exact resume。可选参数经GPU初始/采样审计后已合入主源码和submission，完整CPU检查84 passed；实时GPU训练仍用冻结的隔离runtime。

GPU1先fresh16，再A/D39f的4/8 steps/chunk。运行时审计step00四套adapter、三种RNG与公共初始验证sample；训练sigma/r必须符合单独冻结的shift12预定序列；实际历史前向次数必须chunk×4。GPU独立反传存在微小重复性误差，不能将微小loss差解释为效果。必须查看人物/场景与A>0、D<0、A-D>1。Stage1未验收，Stage2暂缓。

CPU恢复审计的历史receipt中，`shift12_exact_cpu_resume.validation_before=true`是未比较占位，不是一致性断言：恢复前是step2，连续训练前是step0。真实比较通过的范围为最终adapter/Adam/RNG、逐update loss/gradient与validation_after；默认新旧路径的validation_before另有实际相等检查。详见cpu_audit_scope_correction.json，冻结原receipt保持不变。

## 16次训练完成（2026-10-08T10:21:24.044459+08:00）

全部初始化/采样审计通过，公共验证A endpoint下降而D上升；[完整公共验证表](GPU_RESULTS.md)。视频评测与同checkpoint的clean-history定位开始，尚无通过结果。

## cached.json字段更正

冻结runtime的anyflow_training_shift字段旧实现误取flow_shift，因此本组cached.json写成2.22；实际训练为12，验证与推理为2.22。以checkpoint/setup的config.training_timestep_shift及training.json的time_sampling为准。本报告均按这些实际配置汇总。主代码已修正，旧runtime/原始结果不改写，见[更正记录](../metadata_shift_correction.json)。
