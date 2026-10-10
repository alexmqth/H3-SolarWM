# EXP-008 / v1 — V3 causal DMD 单cycle可行性验证草案

2026-10-11，Judge。**当前仅CPU准备，没有GPU授权。** AF3视频审核后发布正式任务书和独立授权marker；不凭本草案启动。

## 研究问题

Track C，Parent为EXP-007 AF2 step32，并以AF3匹配8NFE续写检查其基本可用性。验证当前33B、冻结V3 causal/KV协议下，一条真实8-map生成链能否承受完整DMD反向及一次student更新，测量显存与训练成本，为是否继续有限cycles提供依据。工程结果不能代表画质改善或完整SolarWM复现。

## 三角色与控制条件

- Teacher：Original H3 + released Action LoRA，冻结V3 strict causal入口，无target-time、无新增训练。该teacher不是原始双向H3 teacher。
- Fake score：独立模型和新last8 rank8 QKV，普通velocity/FM模型，无target-time；2次FM warmup只消费student的detach生成端点。
- Student：AF2 step32配套target-time gate .25和last8 rank8 QKV；保留全部8个finite maps之间的计算图。
- 各角色独立权重对象、设备和自身clean C1 raw KV。fake每次更新后重建自身KV；student更新后的旧KV失效，不能供未来训练或视频直接复用。
- Original base及released action LoRA冻结。Single I0、native timestep、own-action/action feedback、current non-action prefix、Global RoPE、12后5分块、max_history5、h3_fp32及现有attention backend保持。

## 数据与训练配置

共同clean C1来自EXP-006新FM8 first12。当前块为AA C2，使用独立training noise seed170008，score noise seed170108；不直接训练用于seed13视频比较的噪声。仅固定C1下的current-block on-policy训练，不宣称完整多块self-history on-policy。

Student沿用native 8-step shift2.22。`x_r=x_t+(r-t)*v(x_t,t,r)`。Teacher/fake评分使用相同detach+renoise端点、sigma .6；方向为`fake_x0-real_x0=sigma*(v_teacher-v_fake)`，使用已CPU验证的detach normalization。student/fake各自新AdamW，lr1e-4、betas(.9,.95)、wd .01、clip1；不复用AF2优化器作为DMD既有训练状态。

## 一个cycle与预算

1. Teacher/fake/student各prefill C1：3forward。
2. Student连续8map生成C2，保留梯度：8forward。
3. Fake进行2次普通FM更新，每次更新后重建C1：4forward、2backward、2update。
4. Teacher/fake对相同renoise端点评分：2forward。
5. DMD沿原8map反向并更新student一次：1backward、1update。

总计**17forward、3backward、3update、0VAE，最多1cycle**。三个真实空闲GPU建议0/2/5，GPU3/4他人进程不动。保守计3×全过程wall time，包含加载、等待、保存和失败；**≤30分钟wall、≤1.5GPU小时、每卡allocated≤44GiB、绝对截止09:00 HKT**。内部checkpoint重算计入时间与反向，不计为额外采样NFE。09:00后项目最多3卡。

## 验收与停止

必须保存每次forward/backward/update角色账本、完整source/input/parent checkpoint清单、实际导入DMD源码、独立KV来源/只读/重建检查、8个map velocity有效梯度、student target/QKV及fake参数变化、base冻结、梯度角色隔离、三卡峰值与总GPU时间。保存student/fake配套weights/optimizer/RNG，不复制base或逐步大KV。

任何OOM、nonfinite、cache或角色协议错误、预算耗尽立即停止并报Judge；不得detach map链或共享可变角色权重来降低成本，不自动重跑。一次工程成功后只提交报告，不自动追加训练。普通缺陷按可行性尺度评价。

## 后续决策

若完整8-map反向成立且成本合理，再单独预算有限追加cycles和匹配AF3/FM8的视频。无可辨动作/结构收益则收口，不扫学习率、critic比例或任意增加steps。该草案不授权后续cycles，也不升级正式V3 Baseline。
