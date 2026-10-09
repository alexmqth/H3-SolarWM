# E2 真实动作后果监督：4-update受控结果

**本轮不通过局部动作＋画面联合验收；不扩至16更新，不进入AnyFlow/Stage2。** 两臂完成固定4更新及全部预登记评测。停车场两份历史均保留A正/D负，但A分支重影仍在，FM+action没有一致优于FM-only。四步小试不能证明动作后果监督不可行，也没有识别人物重影的唯一成因。

## 可播放三列对比

列顺序：**Original权重下的局部N / FM-only4 / FM+action4**。左列也是局部T2/N推理，不能标成Original全长双向生成。全部30steps/current window，832×480源片；展示缩为每列416×240，24fps。三列视频不含GT上下文、不补帧、不平滑。

- [固定A历史 → 当前A](review_step4/parking_historyA_currentA_comparison.mp4)：42当前RGB，全球索引39–80；历史来自Original生成。
- [固定A历史 → 当前D](review_step4/parking_historyA_currentD_comparison.mp4)：42当前RGB，全球索引39–80；历史来自Original生成。
- [固定D历史 → 当前A](review_step4/parking_historyD_currentA_comparison.mp4)：42当前RGB，全球索引39–80；历史来自Original生成。
- [固定D历史 → 当前D](review_step4/parking_historyD_currentD_comparison.mp4)：42当前RGB，全球索引39–80；历史来自Original生成。
- [真实GT-history：W+A+J及观测F](review_step4/118eb5d8b75e1b8ac23a4e9ae77af9a9_D_1015_comparison.mp4)：39当前RGB，索引81–119；文件名动作是旧首窗标签，以此处当前条件为准。
- [真实GT-history：D+L及观测F](review_step4/dfec8ed3237860eba14d67c089ecd041_A_1080_comparison.mp4)：39当前RGB，索引81–119；文件名动作是旧首窗标签，以此处当前条件为准。

这是固定reference/GT history上的局部测试，**不是124f自由生成**。两份GT例子含相机联合控制，F由当前观察速度派生；它们是oracle条件结构检查，不证明实时交互时可取得该速度。停车场纯A/D没有F。

## 同状态动作响应

| 固定历史 | 方法 | A flow | D flow | A−D |
|---|---|---:|---:|---:|
| A | zero | +1.645495 | -1.546733 | 3.192228 |
| A | fm_only | +1.657549 | -1.395918 | 3.053467 |
| A | fm_action | +1.678612 | -1.376481 | 3.055094 |
| D | zero | +0.614104 | -0.673822 | 1.287926 |
| D | fm_only | +0.699057 | -0.667578 | 1.366635 |
| D | fm_action | +0.657532 | -0.616613 | 1.274146 |

Farneback沿用416×240中心80%的相邻帧平均水平flow；单位px/frame。两个history分别评价，不与旧full-horizon teacher的2.023直接比较。符号与separation不是画质分数。

D-history的分离度从1.287926变为FM-only1.366635、FM+action1.274145；A-history则由3.192228降到两臂约3.05。动作项没有一致增益。人物方面，D-history/currentA约RGB55–68仍多重手臂/躯干残影；A-history/currentA也仍有肢体残影。当前D的两组结构相对完整。

## 真实GT局部结构与完整帧检查

| GT场景 | 方法 | Frame MAD | 首边界MAD |
|---|---|---:|---:|
| 118eb5 | zero | 19.1331 | 24.1643 |
| 118eb5 | fm_only | 19.4456 | 20.8947 |
| 118eb5 | fm_action | 19.4573 | 20.8450 |
| dfec8e | zero | 12.3277 | 24.0523 |
| dfec8e | fm_only | 12.1902 | 24.1816 |
| dfec8e | fm_action | 12.2052 | 24.1592 |

所有MAD均从同编码H264 MP4重算。Frame MAD是活动量；GT首边界为当前第一帧与同一rawGT RGB80的灰度差。旧运行收据中的未压缩/decoded-GT MAD定义不同，不混算。没有报告FVD、LPIPS、GT PSNR或warmup均值。

- **parking_historyD_currentA**：All preserve parking scene and person, with positive horizontal response. Repeated arms/torso around RGB55–68 remain. FM-only appears less duplicated than zero around RGB59–62 but does not remove ghosting; FM+action retains substantial multi-arm artifacts and is not clearly better than FM-only.
- **parking_historyD_currentD**：All retain negative horizontal response and recognizable walking figure through RGB39–80. No severe person disintegration or scene switch evident; small limb blur remains. FM+action reduces flow magnitude and is not visibly superior.
- **parking_historyA_currentA**：All retain positive background response and scene. FM-only is similar to zero. FM+action has visible duplicated forearm around RGB59–62; late leg transparency/smearing remains. Initial boundary displacement persists in all; small MAD reductions do not establish visual recovery.
- **parking_historyA_currentD**：All retain negative horizontal response and identifiable walking figure/parking scene through the full current window. Mild leg blur and early pose transition remain, without sustained severe multi-body breakup. Both trained arms weaken negative flow magnitude; no clear FM+action structural benefit over FM-only.
- **118eb5d8b75e1b8ac23a4e9ae77af9a9_D_1015**：Person and outdoor scene remain recognizable in all39 frames of each method. Both trained arms reduce first-frame background smear/boundary MAD compared with zero and look nearly identical to each other. Initial scale/pose/view transition is still abrupt from RGB81 to82; generated path is not proven faithful to observed joint controls. No sustained severe body breakup but no added action-loss quality benefit.
- **dfec8ed3237860eba14d67c089ecd041_A_1080**：All39 current frames keep the character and dark cobbled-road scene recognizable. Zero, FM-only and FM+action are very similar; initial pose/view jump versus the same GT history remains, and later trajectory does not establish observed-action fidelity. No clear added structural benefit or gross new degradation from action ranking.

评审覆盖全部738个当前RGB（三种参数bank、六个局部case），包括全帧sheet、预先固定位置的原尺寸人物crop及历史边界；属于静态逐帧检查，不声称实时播放评审。[完整记录](review_step4/visual_review.json) · [全指标](review_step4/METRICS.md)。

## 正确动作绝对拟合与错误动作排序

| 参数bank | 正确动作FM均值 | 交换动作FM均值 | gap | 正确排序状态 |
|---|---:|---:|---:|---:|
| zero | 0.25976668 | 0.25974831 | -0.00001837 | 5/8 |
| fm_only | 0.25969061 | 0.25967603 | -0.00001458 | 4/8 |
| fm_action | 0.25969900 | 0.25967836 | -0.00002065 | 4/8 |

正例均值改善不足0.03%，FM+action不比FM-only更好。两验证状态×四sigma不是八个独立视频样本；没有用此验证结果调lambda/margin/noise分布。[分sigma完整表](heldout_fit/RESULTS.md)。错误动作没有反事实真实视频，排序不能当方向真值。

## 计算与可复核性

| 方法 | 评测组（每组两条） | 整组wall s | 每条采样 s | 峰值allocated GiB | CPU hidden KV | noisy/identity forwards |
|---|---|---:|---|---:|---:|---:|
| fm_only | parking_A | 619.85 | 262.06/269.56 | 25.96 | 0 | 60/2 |
| fm_only | parking_D | 622.69 | 261.56/271.92 | 25.96 | 0 | 60/2 |
| fm_only | gt | 954.67 | 413.19/431.62 | 25.70 | 0 | 60/2 |
| fm_action | parking_A | 627.47 | 265.78/273.65 | 25.96 | 0 | 60/2 |
| fm_action | parking_D | 632.02 | 264.58/277.25 | 25.96 | 0 | 60/2 |
| fm_action | gt | 970.44 | 421.90/438.39 | 25.70 | 0 | 60/2 |

这是并行共享主机上的单次局部作业成本，整组含加载/两视频/解码/诊断；不是整视频warmup后延迟，也不提供speedup。T2逐sigma重算可见hidden，CPU hiddenKV为0但仍存放raw history/noise。权重、临时KV、激活的峰值分解没有独立测量，不能相减推算。

两臂训练各4更新、logical batch2、released action LoRA tail8 QKV/out共10,092,544参数，LR2e-5；只有objective不同。训练wall284.76/416.62秒，峰值allocated均27.91GiB。初始参数bank完全一致、更新有限非零、梯度replay误差0。六组视频合计360采样＋12identity forward，held-out另48forward。Original→step0输出identity均0，停车场旧N场重放均0；冻结backbone参数版本保持，307文件runtime核验。

## 决策与下一步

本轮停止在4更新。它支持“当前小样本动作排序项尚未显示额外收益”，不支持“所有动作监督都无效”，也不支持用Stage2替代局部能力问题。

[监督覆盖审计](SUPERVISION_COVERAGE.md)发现本次8个microbatch无纯A/D，且3个混有相机控制。CPU扫描训练原片后找到A/D各一段更纯候选，原片人物完整，但A存在待核查的输入到运动时序；仅完成源数据静态评审，没有训练准入、VAE编码或新optimizer。先核对可靠动作后果与时序、同状态配对来源，再冻结一个受控E2修订；不在当前验证集上扫loss系数或平移标签。

阶段顺序保持：可信causal30 → AnyFlow4/8 → on-policy Stage2。局部生成仍未联合通过，会议主demo不以本轮短片替换。
