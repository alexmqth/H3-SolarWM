# Stage1 验收：AnyFlow 与 causal H3 的实际效果

2026-10-09 07:14：E2真实后果数据与前置审计已完成（6train+2validation；24个真实前缀检查差0；17项CPU检查）。33B梯度预检峰值27.81GiB。两条真实GT-history24的30step局部基线人物大体完整，仍有边界跳变和运动偏离；GPU1/4现运行同初始化FM-only与FM+action，各固定4更新。动作+结构完整gate未过，不进入AnyFlow/Stage2。 [当前实验与视频](reports/stage1_anyflow/01_real_video/real_transition_windows/README.md)。

2026-10-09 05:59：E1粗窗口两历史/三窗口与VAE审计已全部结束。首窗A/D恢复，但A历史第二窗同正、D历史第三窗同负；第二窗有明显肢体重影，局部门槛仍No-Go。未来RGB干预8项past latent差均0，不支持VAE泄漏根因。停止窗口扩宽，下一项只审查历史噪声/时间条件，尚未启动新GPU/训练。 G1未通过。[完整证据](reports/stage1_anyflow/02_causal_diagnostics/coarse_window12/FINAL_RESULTS.md)。

2026-10-09 05:32：新full37匹配对照的12latent首窗恢复方向（全39RGB A=+0.863/D=−1.174；共同前17RGB A=+0.285/D=−1.198），5latent仍近同向。G1后续窗口、geometry、GT画面未齐；GPU1/5仍在运行。不能把本首窗结果升级为Stage1通过。[证据](reports/stage1_anyflow/02_causal_diagnostics/coarse_window12/FIRST_WINDOW_RESULTS.md)。

2026-10-09 05:09 E1终态：native＋单I0的5+5+2两history评审完整，动作仍No-Go，G1未通过。当前无本轮GPU任务。12latent更粗窗口协议已登记待执行，E2/E3继续不启动；[完整终态](reports/stage1_anyflow/02_causal_diagnostics/local_topology/CONDITIONING_RESULTS.md)。

2026-10-09 04:57 E1更新：完整39f已知动作窗口在恢复原生text/action时间和单首帧后，A=+1.250421、D=−0.977313，外观静态逐帧检查正常。**仅窗口正控成立，G1多chunk动作/GT画面仍未通过。** native时间但保留dual的两历史局部对照失败；native＋单I0的5+5+2对照正在GPU1/5运行。G0已通过的CPU/真实VAE/首块identity证据与其布局范围限制见[报告](reports/stage1_anyflow/02_causal_diagnostics/local_topology/CONDITIONING_RESULTS.md)。当前不启动E2/E3。

2026-10-09 02:57：**当前验收以[Next Plan v4](NEXT_PLAN.md)及[完整协议](reports/stage1_anyflow/07_protocols/three_experiments/PROTOCOL.md)为准。** 旧A–D不再自动推进。新E1先验证Original初始化下的局部动作/画面，30steps、同状态反事实、GT/teacher-history分列；当前第一门槛未过。T1/T2已经隔离实现并经过CPU/首块GPU检查，参见下方最新执行状态。

| 当前门槛 | 通过条件 | 未通过时的动作 |
|---|---|---|
| G0 因果与状态协议 | 全执行路径无未来泄漏，cache失效/replay、GT/anchor时域审计 | 修复协议，停止效果训练 |
| G1 局部动作与画面 | 首块及后两块响应可信、实际A/D方向正确、GT history充分采样下无严重分解 | 正控可信则做有限E2；否则先修局部条件/表示 |
| G2 Few-step | 4/8接近已经可信的同路线30步，动作/画面/finite-map共同检查 | 留在AnyFlow诊断，不用Stage2掩盖 |
| G3 Generated history | 自由rollout明显优于自身基线，无GT重置 | 在G1/G2可信时研究on-policy Stage2 |
| G4 效率与最终交付 | 整视频/首块/每块延迟、重复测量、完整质量/动作及124f验收 | 不以step比或低MAD宣布成功 |

旧density全部收尾且FAIL，见[终态](reports/stage1_anyflow/01_real_video/fm_density_control/FINAL_RESULTS.md)。允许窗口内双向、窗口间causal作为候选；若采用它，显式报告重算成本，不能标成persistent-KV等价。下方是保留的历史验收记录，冲突时本段优先；旧39f全片A−D>1不能直接挪到新局部正控，不能要求Stage1先完成20秒稳定才允许Stage2。

2026-10-09 02:32：FM密度对照六条自然39帧视频已全部评审，generated30/8仍出现人物分解或背景重影，未通过。固定generated历史12点delta cos仍仅0.035019。停车场A30 flow−0.190239，D及8步原队列继续；完整A–D未完成。 旧FM48全部失败结果与新密度分支局部动作/自然视频负结果均保留；[六视频评审](reports/stage1_anyflow/01_real_video/fm_density_control/NATURAL_VIDEO_RESULTS.md)、[完整阶段状态](reports/stage1_anyflow/07_protocols/overviews/ABCD_STATUS.md)。下文日期较早的运行描述为历史记录。

2026-10-08 20:15补记：真实数据FM仍在20/48；零更新完整视频已发现第二场景30步generated-history人物分解，GT18同状态动作差分cos=0.017516，无历史首块也低。新图Original纯A/D是弱正控，停车场回归另列。见[完整基线评审](reports/stage1_anyflow/01_real_video/real_abot_fm/BASELINE_COMPLETE_REVIEW.md)。以下无新训练/视频等旧措辞仅指当时诊断快照，不能代表当前状态。

2026-10-08最新：受控field对照定位到AnyFlow更新之前的动作几何失配；真实ABot普通causal FM桥接已启动，尚未验收。完整目标A–D的当前状态与前置门槛见[ABCD_STATUS](reports/stage1_anyflow/07_protocols/overviews/ABCD_STATUS.md)，不能将数据准备或只读诊断当作Stage1完成。

当前目标：完成 Stage1 的 AnyFlow 与因果改造，效果验证后才进入 Stage2。代码能运行、单次 loss 下降、生成文件可解码，都不能单独证明目标完成。

当前Stage1尚未通过。136训练/4与8步视频及54个双局部指标case已收尾，后段重影未修复。最新同generated-state当前chunk A/D机制诊断12点全部完成：匹配条件下student r=t与Original teacher动作速度差分cosine均值0.0382（范围−0.1023–0.2050），动作幅度为teacher的0.79–3.56倍。时间索引/实际attention路由、KV只读与未来内容负对照检查通过，支持优先调查动作条件函数迁移不足；双向teacher重算历史与student固定KV的结构差异仍需拆分。见[机制结论与归因边界](reports/stage1_anyflow/02_causal_diagnostics/generated_action_geometry128/INTERPRETATION.md)、[136完整结果](reports/stage1_anyflow/03_anyflow_trials/interval_consistency_candidate/FINAL_RESULTS.md)。该18:09机制诊断当时已结束且没有训练；随后已开展真实ABot的普通FM48桥接和完整零更新视频/36点机制评测，当前结果见页首链接。新的Stage2仍未启动；最多3张项目GPU。

## 阶段边界更正：长时稳定性不是进入Stage2的全部前提

Stage1应先在首块和受控干净历史下表现出合理的少步画质与动作响应；Stage2再针对自身rollout的分布偏移。**不要求Stage1先消除全部长时漂移，才允许Stage2。** 同一history下的当前action反事实对照仍有必要，避免oracle历史本身携带的动作结果造成误判。

下表同时包括局部Stage1能力和项目最终交付证据。39f自生成的`A>0,D<0,A−D>1`、切换与124f门槛保持为最终效果验收，不能全当作Stage2启动前必须完成的条件。已有step16的teacher-history改善仍不充分，且首块已偏离；不能因此声称当前只缺Stage2。完整证据和决策分支见[Stage1/Stage2边界说明](reports/stage1_anyflow/07_protocols/overviews/STAGE_BOUNDARY.md)。

## 必须提供的证据

| 项目 | 验收证据 | 当前状态（2026-10-08） |
|---|---|---|
| AnyFlow 实现 | 官方 loss/gradient 对照、目标时间条件、有限步更新、BF16 模型与 FP32 轨迹兼容 | CPU专项通过；真实33B训练已到128并回载完成4/8步评测，generated 8步分离度0.759660且视觉失败；同权重r=t画质改善但分离度0.662048，正常有限步收益尚未证明 |
| causal 训练 | 在第 0/1/2 个 chunk 上真实反传；历史cache分离且只读；QKV有非零梯度，官方策略的时间MLP应严格冻结 | 旧trainable-time变体已覆盖A/D三个chunk；官方冻结time-MLP策略的真实33B两个分支均完成64次，shift12进一步完成128次；真实恢复/四replica/参数策略已核对 |
| 对照公平性 | 相同首帧条件、prompt、action 行、seed 与 video/audio noise；记录 solver 网格和计步口径 | Original与已完成16/32/64的4/8及96的8步评测conditioning逐张量一致；时长/显存仍非隔离硬件重复均值 |
| 少步效果 | 同初始化普通 FM、AnyFlow step00、训练后 AnyFlow 4/8 steps/chunk 与 H3 30-step 比较；看完整视频与每类 raw loss | AnyFlow16和FM16的4/8-step已完成，动作均失败；step00/学习曲线/clean-history已完成且未通过；uniform对照已失败；frozen-time训练及6条视频已完成，全部未过动作gate |
| A/D 动作与画面 | 39 帧 A>0、D<0、A-D>1.0，同时人物/场景结构保持；不可凭单一光流指标通过 | 未通过；旧trainable-time4/8-step A-D=-0.091/0.304；legacy frozen-time4/8-step为0.003/0.229；native-FP32 tail16为−0.039/0.145，全覆盖detached rank8为0.044/0.191；完整历史梯度16次为−0.024/0.183，训练shift12的16次为0.005/0.180、64次为0.150/0.623；96次只评测8步，分离度0.246且D后段恶化；128次4/8步分离度0.134/0.760，8步A仅+0.040且视觉仍失败 |
| 验证噪声 | 至少一个未参与 checkpoint 选择的 inference seed 复验 | 尚未运行 |
| 时间绑定 | 39 帧 A→D→A、D→A→D 切换与逐 chunk 观察；未来动作不能污染之前的 chunk | 待短片 gate 通过 |
| 最终长度 | 124 帧 W/S/A/D 和动作切换；人物不分解、无突发换场，长时漂移完整保留并报告 | 新 AnyFlow checkpoint 尚未验证 |
| 交付 | 原始/causal 并排可播放视频，标注 steps/chunk、总 forwards、耗时；显存、CPU KV、MAD、boundary、action response 表 | 当前旧展示不包含 AnyFlow；新交付待验收 |

原始 39 帧参考重新计算：A=+1.181253、D=-0.842124、A-D=2.023377。光流沿用原有 Farneback 测法；MAD 是运动活跃度指标。最终视觉判断必须查看完整视频，contact sheet 只能辅助。

## 不能省略的解释

- AnyFlow 的非 diffusion loss 按本批 FM loss 自适应缩放。因此 weighted total 不能代表 endpoint/general-map 的残差已变小；必须保留每类 raw loss、sigma/r 和相同 held-out noise 的前后验证。
- 旧pilot额外训练目标时间MLP，因此与FM的可训练参数量不同；官方冻结策略下可训练参数量可以相同，但AnyFlow每个loss样本的四次前向仍使计算量不同。对照需要分别报告参数策略与训练算力。
- 旧 teacher 的 offload reserve 未记录，前期 GPU 2 使用 reserve=20 GiB且被共享，之后曾迁至独占GPU0，当前因新任务占用使用GPU4–7；reserve=6 GiB且不限制25GiB。旧 teacher 与本轮的 allocated peak 不能直接解释为 causal 节省了多少显存；单次耗时也不构成隔离环境下的 speedup 证据。
- 16 次更新是初始学习曲线，达到这个次数不自动等于训练充分。若结果未过 gate，继续基于证据定位 Stage1；不得把既有 Stage2-lite 视频重新标记为 AnyFlow 结果。

## 与官方 H3 Stage1 的已知差异

参考 SolarWM revision `ce1da4e7705391eda8eeda6016c0fd3f614b975e`：

| 维度 | 官方 H3 Stage1 | 当前受控 pilot |
|---|---|---|
| Loss 与时间条件 | AnyFlow v1.5，固定gate=0.25，克隆且冻结target-time MLP | loss/gradient对照通过；旧pilot额外训练time MLP，冻结分支已完成真实33B 16次更新，尚未通过视频gate |
| 数据与容量 | 多来源 158f 数据；全 block QKVO/FFN rank384；global batch128 | A/D两条39f teacher伪真值；已评测tail16 QKV rank8；全50+2块QKVO/FFN rank8已完成16次及四条评测并失败，43,237,376参数，logical batch4 |
| 历史训练 | clean/noisy 两流 teacher forcing | detached控制已完成；可选full-history梯度通过CPU与真实33B探针，fresh16及全部评测完成并失败；仍非官方融合两流算子 |
| 训练时间采样 | shift=12 | 已完成基准保持2.22；独立16次shift12训练及全部评测完成并失败，公共validation/inference仍2.22 |
| Stage1 验证网格 | `stage1_sampling.py` 明确使用 sigma=[1,.75,.5,.25,0] | 旧pilot沿用H3 shifted网格；显式uniform独立对照已完成且失败 |
| 计算精度 | 六组输入输出/时间线性层与时间混合为FP32，主block BF16 | 旧路径主要为BF16；计算精度与输入扰动的33B诊断均完成、结果混合。native-FP32已恢复原生12个F32权重，完成16次及全部4/8-step评测，仍失败。全覆盖LoRA候选已完成真实GPU16次更新与4/8评测并失败 |
| 音频条件 | encoded silence 与噪声插值，训练有独立音频时间，验证按音频 schedule 插值 | 固定 audio noise/native t=0，保证当前 cache 的 prefix contract |
| 图像与控制 | 单首帧、几何相机条件 | RGB dual anchor、H3-World 键盘 action 行及固定 action residual |

这些差异不是已证实的失败原因。FM/AnyFlow、均匀网格、冻结时间MLP16次及legacy32次训练对照均完成，未通过。legacy64次分支在第33次更新前暂缓，优先native-FP32候选；输入/边界诊断完成且结果混合。native-FP32完整评测也已失败；现在保持初始预测、数值/数据/anchor/chunk/action协议，验证全QKVO/FFN的覆盖与参数化候选。其GPU16次更新及全部4/8评测已完成并失败。同容量FM16四条评测完成，4/8-step A-D=0.2137/0.3209，A仍错误；4-step FM人物/场景比AnyFlow完整，8-step无明确AnyFlow收益。保留clean-history梯度的CPU与真实33B探针均通过（allocated40973.35MiB），full-history16训练及四条视频完成，4/8-step分离度−0.023916/0.182509，仍失败。两条分支的32/64次训练和全部4/8评测已完成；shift2.22的64次8步分离度0.215531、shift12为0.623096，A均错误。shift12四卡128与完整评测已完成仍失败；teacher-history与固定历史动作干预已完成，相对响应存在但保真未证实；同权重对角条件视觉更完整，正常finite-map不足待修。同checkpoint的teacher-history A/D8也未通过；首5个latent在两种history中逐元素相同，而第0块动作已偏离，不能全归因于生成历史累积误差。两支尚无新通过结果。

运行证据见源目录 `H3-World/outputs/2026-10-08-02/stage1_anyflow39_pilot/`：`run.json` 是队列状态，`report/REPORT.md` 与 `report/contact_{A,D}.jpg` 是已完成视频的快照，不代表队列全部完成。

参数策略更正及排队对照见[Stage1参数策略复核](STAGE1_PARAMETER_POLICY.md)。时间MLP能反传并不等于官方要求它进入optimizer。
