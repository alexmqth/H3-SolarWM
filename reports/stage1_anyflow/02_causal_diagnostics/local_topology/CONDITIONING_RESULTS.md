# E1 条件校准：完整短窗口动作正控恢复，分块能力尚待检验

2026-10-09 05:08 完整收尾。**在 Original 权重上，恢复原始 text/action 时间条件，再移除重复第二首帧，完整 39f 当前窗口恢复了明确 A/D 响应。不能把它当作 5-latent causal 或 Stage1 已通过。**

[三列 A/D 诊断视频](conditioning_window12_AD.mp4)：每行分别为 A、D；三列依次恢复时间条件和单首帧。三列使用同一首图、prompt、seed13、初始 video/audio noise、Original H3＋released action LoRA、native 30-step video solver、h3_fp32 和 Original directed SDPA，均无训练。动作在整个 12-latent 当前窗口预先已知，分辨率 832×480，39RGB/24fps。它与后面的 5+5+2 局部对照分开解释。

| 12-latent 当前窗口条件 | A flow | D flow | A−D |
|---|---:|---:|---:|
| clean text time＋双首帧 | -0.745426 | -0.767189 | 0.021764 |
| 原生 text time＋双首帧 | -0.000594 | -0.799118 | 0.798524 |
| 原生 text time＋单首帧 | +1.250421 | -0.977313 | 2.227735 |

两个连续的单变量对照表明：在这个固定停车场样本中，原型的条件修改可以显著压低动作响应，即使使用 Original 的局部 directed attention。恢复单首帧后的 A 与 D 运动/转身明显不同，全部39帧静态图中人物及停车场结构可辨，未见严重人物分解。此处是完整静态帧检查，不冒充实时播放评审；单一seed也不支持普遍结论。不是一次完整 factorial 实验，不能将时间与 anchor 的交互影响直接相加。

## 为什么要先校准参照

H3-World 的 `model_fn_minimax_h3` 默认 `fixed_prefix_timesteps=False`，text/action 时间跟随 video，即 native `1−sigma`。原型显式设 True，将所有 text/action 时间固定为1；这独立于 attention mask，会改变逐token的时间调制。SolarWM 的 Stage0.5 采用 video time、Stage1 采用 clean text time，是其需要训练适配的阶段设计，不能说成其 bug。

旧 field-factorization 和真实FM geometry的双向参照也固定了text time。因此旧数值仍表示匹配原型时间条件下改mask的相对影响，但不是完整原生H3推理函数的对照。已给[机制总结](../../07_protocols/overviews/ACTION_MECHANISM_SUMMARY.md)和对应旧报告加限定，原数据不删除。

另外，所有本轮窗口校准都固定 audio noise/native audio time=0；原 H3 benchmark 联合去噪 audio/video。所以单首帧条件也不能称为逐bit复刻原始联合推理。当前已有可靠整个窗口正控，暂不新增 audio sweep。新单首帧分块必须保持 I0 原始位置，不能误用旧 dynamic-dual 逻辑把它 retime 成上一块末帧。

## 仅恢复时间条件的 5+5+2 局部结果

每个历史来源下当前块单独 fork A/D，30steps/chunk，未来 action/video 在refiner前物理移除。history来自旧 Original 生成，**不是GT**；两条分支使用同一份固定历史、相同初始噪声/anchor，仅替换当前chunk动作。

| 固定历史来源 | chunk | A flow | D flow | A−D |
|---|---:|---:|---:|---:|
| A | 0 | -0.920014 | -0.930697 | +0.010683 |
| A | 1 | +1.937947 | +0.525689 | +1.412259 |
| A | 2 | +1.672589 | +1.668780 | +0.003809 |
| D | 0 | -0.920014 | -0.930702 | +0.010688 |
| D | 1 | -0.956912 | -0.965282 | +0.008370 |
| D | 2 | -0.627791 | -0.628440 | +0.000648 |

只在 A 历史的第二块出现明显区别，两种动作依然都为正；D历史中仍近同向。首块与末块未见可靠区分，因此该条件版本未通过 E1。末块只有5RGB，光流与动作语义的可分性有限；这不是“完全没有任何动作信息”的证明。四条完整静态图中人物结构可辨，但17/34帧有oracle reset跳变，不能拿39f局部串接证明自由rollout连续。

首块真实33B T1/T2同输出、repeat均0；CPU验证不是预训练动作能力证据。T1后续块的新geometry尚未运行，失败双锚点参照的geometry gate保持关闭，不能让自比较cosine=1充当收益。

## 下一决策

恢复正控的native时间＋原始I0已完成5+5+2局部试验，全部12条局部片段和四条39f串接完整；两份历史各180次noisy forward＋3次首块诊断，GPU1/5原进程正常退出。新增单I0位置/未来隔离/只读CPU3项通过，单首帧首块真实33B的T1/T2/repeat误差0。**这次局部动作验收仍为No-Go。**

| 固定历史来源 | chunk | A flow | D flow | A−D |
|---|---:|---:|---:|---:|
| A | 0 | -1.035564 | -1.001839 | -0.033725 |
| A | 1 | +1.662819 | +0.534328 | +1.128490 |
| A | 2 | +1.650168 | +1.651914 | -0.001746 |
| D | 0 | -1.035564 | -1.000838 | -0.034726 |
| D | 1 | -0.969241 | -0.969736 | +0.000496 |
| D | 2 | -0.616305 | -0.609233 | -0.007072 |

A历史第二块的姿态/运动确有区别，但两者都为正；其余各块A/D几乎同向。全部四条39帧静态图中人物和停车场可辨，未见此前自由rollout的严重分解，但17/34的oracle重置不可作连续性证据。末块5RGB光流有限，判断是“未建立可信局部控制”，不是证明动作信息绝对为零。还没有新T1后续chunk同状态geometry结果，不能宣称完整T1/T2迁移比较已经通过。

[单I0：固定A历史下的局部A/D视频](native_single_positive_A/AD_local_forks.mp4) · [单I0：固定D历史下的局部A/D视频](native_single_positive_D/AD_local_forks.mp4)。均明确标注LOCAL ORACLE FORKS；其17/34帧的状态重置是实验设置。

成本：每份history整组wall为853.35/861.05秒，allocated peak均26409.54MiB，CPU raw-KV=0（T2逐sigma重算），latent history=1.71MiB。该wall含6条局部分支、诊断、解码/编码等，**不是一条自由生成39f视频的推理时间**。同主机单次值不用于speedup结论。

下一步仅推进E1的窗口粒度决策：保持原生时间＋单I0，当前窗口12latent，在现有124f Original A/D参考的多个历史位置fork动作；预先检查全37布局/noise与未来隔离、实际prefix解码范围。旧39f噪声/坐标不能冒充与37布局逐bit一致，需新匹配对照。最多两路、360次noisy forward、零更新，方案[已登记但尚未运行](coarse_window_protocol.json)。若更粗窗口在后续状态仍失败，停止继续扩窗口，回查history条件/动作表示和可靠动作后果监督。

E2/AnyFlow/Stage2未启动，局部geometry仍受可信参照门槛限制；新GT局部画面也尚待选定路线。I0适配源码隔离，305文件runtime保持不变。会议视频未替换，未推送。

所有完成视频均完整解码检查H264/yuv420p/24fps；每组`evaluation.json`、`video_audit.json`与独立review保留。采样耗时/峰值是共享主机单次可行性记录，不据此声明speedup。各paired局部实验为6×30=180 noisy forwards，另有3次首块诊断；每个12latent paired校准为2×30=60次；没有persistent KV复用或clean commits。这些都是T2重算试验。
