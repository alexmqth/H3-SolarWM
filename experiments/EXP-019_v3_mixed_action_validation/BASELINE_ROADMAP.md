# 后续基线路线与决策门

用户于2026-10-15批准开始。EXP-019执行第一门；后续GPU任务由Judge根据本门结果签发具体编号/预算，避免把多周建议直接当无上限计算授权。

## 主线与现有基础

主线为H3-World已发布Action LoRA的人物/键盘接口。V0含作者已训练权重；下一训练学习的是V3因果协议适配，不从头重训角色控制。SolarWM Stage2-fix为独立相机控制参考，不在相同语言Action prompt上冒充同任务比较。

## Gate A：实际动作接口与短续写（EXP-019）

EXP-017 canonical输出/KV回归→真实head/独立动作编码→四scene GT-history FM30/FM8。先固定问题与样本，不筛seed。变长动作语义位置和未来隔离失败时先修复接口。此门不证明闭环、反转或广泛泛化。

## Gate B：固定诊断集与Original/冻结V3

目标6–8个不同源episodes；现有4 train和2 validation源已观察，只能算训练/回归，不能包装为盲测。保留两固定种子，预先定义持续/反转/真实混合动作；按episode隔离新增测试源。先少量5秒样本验证输入，再分批扩大。GT真实脚本与反事实动作分别记账。已生成相同配置可复用；不重复VAE检查或统一V0–V4对比。

用人物结构/运动量/方向与切换延迟/边界退化联合评价，标明不可判定与相机运动混淆；光流非完整动作准确率。长时测试待短时基线稳定后再扩10秒。

## Gate C：普通FM因果适配

先准备32–128条可靠的真实124帧训练片及独立验证源，确保动作变化覆盖、时间对齐与原生Single I0编码。现有四56帧片不足以支撑该规模结论。固定一套adapter覆盖、rank、优化器与logical batch，不先做大扫参。基于已发布Action LoRA适配V3，teacher forcing训练历史由当前权重clean commit。缓存与权重匹配。

先真实profile少量完整forward/backward/update，记录每次有效样本数、tokens、GPU小时、allocated/reserved/CPU内存，随后确定首档预算与验证检查点。更新数不是收敛保证；训练集拟合、独立验证与自身历史续写三者分开判断。训练集都不改善才查容量/优化；只训练集改善则优先看覆盖和历史分布。明确允许失败，避免无依据从500自动扩到10000。

## Gate D：训练后失败诊断及方法选择

GT/self-generated history × constant/switching，增加动作时间偏移对照；两种历史均以该模型当前权重重建KV，不跨模型直接复用。自生成状态不同是混杂，不把差异自动等同exposure bias。只有稳定且可复现的机制证据后，再选一个SGF/角色LoRA/Action Context/Memory方向；AnyFlow与普通FM保持匹配NFE及历史条件。

## SolarWM独立参考

本地已有models/SolarWM/SolarWM-h3-33B-sgf-stage2-158f-fix目录，具体完整性/EMA身份/数据/环境仍需核查。只读准备可与主线并行；标准官方相机推理按独立任务计费。不重训Stage0.5，不把本地目录存在视为官方能力已经复现。训练型Stage2留待明确用途与资源profile。
