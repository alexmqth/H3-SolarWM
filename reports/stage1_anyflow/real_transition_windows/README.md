# E2：真实动作后果监督的4-update受控实验

以下准备记录已执行；训练与评测进度见当前状态。E1 已发现历史条件会影响后续动作响应，但局部人物重影未过验收。本目录把用户允许的“真实动作后果”转换为可审计输入，用于后续 FM-only / FM+action 受控比较。它不是已经成功的 E2，也不是 AnyFlow 或 Stage2。

## 数据与动作合同

- 从已有六个 ABot 原片取 **6 train + 2 validation**，每段真实124RGB/37latent、832×480、24fps；无新下载。训练/验证按episode隔离，完整124帧在同一episode内不重叠。lossless RGB/action/pose约748 MB，留在outputs、不打入submission。
- 当前窗口固定12latent，三段对应RGB `[0,39)`、`[39,81)`、`[81,120)`；第37latent/末4RGB不属于完整训练窗口，不把120输出写成124帧free rollout。
- 源视频索引、原始按键与COLMAP pose一一对应，保留W+A、S+D及IJKL。没有独立估计按键到视觉后果的延迟；不能把“同源帧索引”写成动力学时延已证明正确。
- 9bit接口不能表达Q/E/Space；对应负例会拒绝。相反方向落在同一latent格、或横向有效帧不足50%，也拒绝ranking负例。保留真实FM输入及拒绝原因，不能凭空补反事实视频。
- camera F来自本窗口观察到的yaw速度，是训练时的后果代理，不是独立记录的用户速度指令。A/D负例保持F及所有其他按键不变。原始translation仅留作溯源，episode未来统计不会进入模型条件。

## 两处未来信息审计

`action_script._camera_rate`原来可借用下一窄格：full37的latent35可能借到窗口外latent36。新模块按固定窗口生成并提交标签，邻格借用限定在窗口内部；未来动作扰动及可见前缀重算都不改变旧标签。停车场纯A/D没有相机项，不受此问题影响。

位置origin固定为已知head/horizon/首个动作的参照，并在完整layout上处理、再删未来行。第一次真实mixed-clause检查发现浮点平移残差约1e-13，已保留失败日志/冻结代码到`attempt1/`。修订直接从固定参照重建原生时间格，消除对未知文本长度的数值依赖；这不是已经找到人物崩坏根因。15项CPU检查通过。真实VAE前缀及文本编码仍以逐样本收据为准。

## 可训练路径

`local_transition.py`与上轮N条件前向数值一致：原生text/action time，single I0，同sigma临时加噪的detached历史；所有可见hidden每步重算，无persistent hidden KV。实际tiny-H3 FP32/BF16测试确认有限非零LoRA梯度、checkpoint梯度一致、历史detach、未来文本内容/长度不影响当前输出/梯度。

真实33B预检候选只开放 **released action LoRA末8层的QKV/out矩阵**，不加新模块；FP32 master保留原BF16初值，安装前后须输出完全一致。先证明history12/24的真实backward显存可行，预检不创建optimizer。训练之前仍需冻结LR、噪声分层、margin/lambda、采样表与评测表。

## 两臂实验目标

实际观察到`(H,a,x0)`，固定`zt=(1−sigma)x0+sigma*epsilon`和`u=epsilon−x0`。`e(a)=MSE(v(a),u)`。只交换当前窗口横向A/D得到负例`a'`，保持history、state、W/S及camera条件不变。

- FM-only：`e(a)`。
- FM+action：`e(a)+lambda*max(0,margin+e(a)−e(a'))`。

这是观察后果的统计对比，不是已知`a'`视频，也不相减不同history的teacher输出。必须同时报告正例绝对误差、负例误差、hinge和实际生成；只把负例loss推高不算通过。margin只允许用train校准，held-out不用于选数据/调参。原预算保留：先4更新、最多16/臂、保存0/4/8/16；不得自动扩训。

## 评测与局限

真实GT-history局部画面与停车场同状态A/D分开检验。自然场景有镜头和联合按键，不能套用纯A/D的flow符号。GT baseline先测两份验证history24的实际动作、30steps；输出是39帧局部片段，附8帧共同GT上下文，共47帧展示，不是自由生成。

数据量很小：train后续窗口中有7个ranking资格，history24仅1个且为A；validation history24提供A与D两场景，仍不足以支撑广泛泛化结论。若这两臂不能改善实际动作与结构，不再靠扩预算或追4步掩盖失败。E1完整/Stage1仍未过，AnyFlow与Stage2暂不启动。

## 真实33B梯度预检已通过
Original初始化，仅选released action LoRA tail8 QKV/out，共10,092,544参数。FP32 master安装前后输出差=0.0。3次真实反向传播均有限非零、未更新权重。作业总耗时89.66s，torch峰值allocated 27.81GiB；这不是端到端生成速度。
| history latent | 当前条件 | FM | 梯度范数 | 前向+反向秒 |
|---:|---|---:|---:|---:|
| 12 | positive | 0.16211884 | 0.0042447 | 16.40 |
| 12 | negative | 0.16219865 | 0.0042640 | 16.65 |
| 24 | positive | 0.15563034 | 0.0047138 | 27.04 |

同状态正确动作误差仅比交换动作低约0.00007981。这个差值可用于后续train-only校准的一个观测点；不能据此选定margin/lambda，更不能证明生成方向正确。局部生成基线仍须看实际视频。

## 编码与校准已完成

8段编码完成：24项仅观察RGB前缀检查、6项未来RGB反转检查、24项未来文本长度/内容输入检查通过，历史latent差均0。新的限制窗口相机速度标签实际改变了验证片段118eb5…的latent35；其他7段F标签不变。两个编码作业wall581.12/243.21s，峰值allocated约19.62GiB。

训练集两个片段×4sigma共8个同状态动作交换诊断已结束，16次实际前向/反向、零optimizer。5/8状态交换动作FM反而较低；这是局部拟合差异，不是反事实真实视频或动作正确性评分。按预登记规则选margin=9.30689275e-5、lambda=2.48589083，使动作项初始梯度规模约FM的25%，lambda上限10未触发。[完整分sigma读数](CALIBRATION.md)。

仅依据训练数据冻结4-update两臂采样表、LR2e-5、logical batch2、相同released tail8 QKV/out。序贯hinge反向与联合图梯度的两个实际tiny-H3测试通过，避免同时保留两套33B激活图。训练需等两条GT局部基线全部评审完成，再锁定正式协议；校准阶段没有optimizer更新。数据划分只对本次adapter适配按episode隔离，不保证原始H3预训练没有见过这些episode。

## 当前执行状态

2026-10-09 16:16：E2两臂各4更新、六组局部视频评测及48次held-out诊断全部完成。停车场两份历史的A/D符号均保留，但A分支重影仍在，FM+action没有一致优于FM-only；局部动作＋结构联合gate仍为No-Go。本轮不自动扩训，不进入AnyFlow/Stage2。

[完整结果与六条三列视频](FINAL_RESULTS.md) · [held-out分sigma诊断](heldout_fit/RESULTS.md) · [监督覆盖审计](SUPERVISION_COVERAGE.md)。训练、评测与完整静态帧评审均结束；本目录明确区分协议/早期准备记录与最终结论。下一步先核对可靠动作后果及输入到运动时序，不把仅有纯按键标签的片段直接用于新训练。
