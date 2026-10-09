# 保留clean-history梯度：CPU与真实33B探针已通过

当前全覆盖AnyFlow16四条39f视频全部未过action gate。这里检查一个尚未补齐的训练差异：SolarWM Stage1把clean/noisy两流一起送入可训练模型，prediction的梯度可经过clean-history特征；本地teacher forcing一直把历史K/V detach，只训练当前chunk分支。

新增可选 `--history-gradient-mode full`，探针通过后已作为可选路径合入主代码及submission，默认仍为detached；完整主源码77项CPU检查通过，实时训练继续使用冻结的独立runtime。每次带梯度的prediction之前重新计算有计算图的clean history，以CPU保存raw K/V；每个历史chunk读独立、不可变的cache快照，后续chunk与checkpoint重算不能覆盖之前读到的entries。三个AnyFlow目标前向仍no_grad。默认detached和推理commit仍保留严格no_grad检查。

这保持了现有cached forward、RGB dual、action、audio与window语义；**不是官方融合两流算子**，也没有证明它是画面退化原因。官方参考：SolarWM/src/solarwm/backends/minimax_h3/stage1.py 的 clean_rows+noisy 拼接和同一model prediction；anyflow_loss.py中的stop-gradient目标。

## 已验证

- 真实小H3，FM/AnyFlow、FP32/BF16、checkpoint/offload选项：前向和每层历史K/V逐元素相同，历史teacher输入收到梯度，未来teacher帧无梯度，基座仍冻结，cache在backward后保持原快照；连续独立loss不需要retain_graph。9项专项通过。
- 选择full-minus-detached参数梯度方向，直接扰动权重并通过原no-grad推理cache重算数值差分：0.00122190；full梯度方向导数0.00121648；detached为0.00020518。它验证这一小模型方向上的缺失梯度，不代表33B缺失梯度百分比，也不是画质度量。
- 在扩充FM专项前，完整套件72 passed；扩充后只重跑9项相关检查，其他源码数学逻辑未改。
- 全梯度CPU训练连续4次vs2+恢复到4次：visual/time/full-bank、Adam、CPU/logical RNG、sample loss和gradient完全相同；改变history-gradient mode不能伪装成exact resume。额外历史前向次数依次0/4/8/0，分别覆盖chunk0/1/2/0。

早期CPU fixture的history_protocol文字标签仍写detached（报告标签遗漏）；其config明确full且记录了额外历史前向。最终源码已更正标签，不更改这些历史fixture的实际计算或权重。

## 已完成的真实GPU检查

同容量FM16及四条视频退出后，已在GPU0、reserve6完成一个A/chunk2探针：同一模型/权重/noise/times，先算detached梯度、再算full梯度，比较loss和梯度，测allocated/reserved peak及CPU KV；最后做一次full-gradient更新并核对visual/time/action冻结。

全梯度prediction同时持有detached目标cache和graph cache，CPU KV计量会包含两者；checkpoint保存的其他CPU激活不算KV。真实33B的allocated峰值40973.35MiB、reserved峰值43176MiB，前向loss相同而梯度不同；单次更新及冻结检查通过，详见[GPU结果](GPU_RESULTS.md)。探针产物不是可精确续训的生产checkpoint，不直接拿来扩124帧或作为最终Demo。后续fresh16训练已在独立full_history候选目录启动，画质仍待评测。

实现补丁为clean_history_gradient.patch（基于当前full-scope/FM主代码）。数学与恢复证据见preflight.json、cpu_resume_audit.json及测试日志。Stage1效果仍未通过，Stage2暂缓。

临时capture收集器已在每个前向后seal并清空；checkpoint backward不再重建CPU KV副本或通过收集器持有自身计算图。相关9项与1项专门检查通过（共10项）。修正后的连续4次、旧step2恢复到4次，均与此前完整4次的参数/Adam/RNG/loss/gradient完全相同。最终fixture的history_protocol标签正确，见cpu_sealed_resume_audit.json。只停止并重建了尚未开始GPU探针的等待控制器，FM进程未受影响。
