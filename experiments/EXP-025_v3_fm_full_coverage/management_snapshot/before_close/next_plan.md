# 当前可执行任务：EXP-025/v1

EXP-024/v2已accepted：真实全层C1/C2 profile完成，3DiT/2backward/2updates，76.367GPU秒，峰allocated24.163GiB、reserved27.508GiB；质量未测。

现授权原Worker立即实现并执行全层rank8/C1+C2普通FM小数据拟合：32train×每block4遍=256updates，固定前后各6视频及C1/C2同状态probes。CPU自检、零adapter回归和真实梯度门控通过后自动连续执行。

单空闲L40、总13800GPU秒/576顶层DiT/256backward/256updates/12decode；0新编码；失败不自动重试，不扩展新分支。[完整任务书](submission/experiments/EXP-025_v3_fm_full_coverage/taskbook_v1.md) · [授权](submission/experiments/EXP-025_v3_fm_full_coverage/AUTHORIZATION.json)。Worker按阶段消息回报，Judge审核/push。

正式V3保持冻结。EXP-023视频收益未证实，不再微调旧tail8配置；本轮容量与训练块覆盖共同改变，不能作单因素归因。

交接与发布确认：固定Worker回复EXP-025/v1现在开工，按CPU/回归门控通过后自动连续执行。EXP-024最终证据与EXP-025授权已push `48f2bc93af04af6fc285367cc79efad632dbe33e`，独立远端main与HEAD一致。

EXP-025启动实物核对：GPU0/PID3409453；102个输入tensor SHA及实现/依赖SHA一致；零adapter 6forward回归velocity逐值相等、完整KV摘要相同，进入前评测。训练与效果尚未验收。

EXP-025首个真实C1/C2梯度门控已Judge复核：两块50层梯度finite非零、路由完整、冻结参数无梯度；前评测6片实物SHA与16probes齐全。256updates训练正在GPU0继续，质量待后评测。

EXP-025 64updates checkpoint已Judge独立复核SHA、50层有限参数/100项optimizer状态、RNG与训练顺序；32clip每block各一遍。训练继续256，阶段证据已push aa681bb，远端SHA核对一致。

EXP-025 128updates checkpoint已Judge独立复核SHA、50层有限参数/100项optimizer状态、RNG与顺序；32clip每block各两遍。核查时131updates，按已测均速剩余训练约4211秒，仅估算，预算内继续。阶段证据已push 3d0966c，独立远端核对一致，效果仍待post。

EXP-025全部256updates已完成并Judge独立核对最终checkpoint SHA/50层有限参数/optimizer与RNG/256条训练顺序；32clip每block各4遍。训练9666.122秒（161.10分钟），低于11000秒阶段上限；post评测正在进行，质量待审。阶段证据已push 8768a3f，独立远端核对一致。
