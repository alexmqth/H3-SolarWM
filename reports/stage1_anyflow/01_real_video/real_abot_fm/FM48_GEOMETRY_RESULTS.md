# FM48 同状态动作机制：GT 与 fixed generated 双对照

2026-10-09 归因补充：本页 `geometry_primitives.py` 的 Original 参照也使用 text/action clean time=1，区别于 H3 原生 `1−sigma`。本页仍证明匹配该条件时 FM0→48 没有恢复参照动作差分；不能把此参照无保留称为原生 H3 推理函数，或把所有失配唯一归因于 attention。见[新 E1 校准及限定](../../07_protocols/overviews/ACTION_MECHANISM_SUMMARY.md)。

2026-10-08 22:29；两组各18个状态均完整，step00→48比较器已验证匹配。

结论：普通真实视频 FM48 没有恢复当前 A/D 的 velocity geometry。整体 velocity 的相似度有所增加，但动作差分几乎正交，不能把前者当作 action preservation。

| 固定状态来源 | 整体 velocity cosine 0→48 | A/D delta cosine 0→48 | delta相对误差 0→48 |
|---|---:|---:|---:|
| GT clean history / GT endpoint加噪 | 0.988193→0.989033 | 0.017516→0.017641 | 1.196923→1.203667 |
| 固定step00 generated history / endpoint加噪 | 0.994733→0.995210 | 0.012540→0.009336 | 1.214097→1.205989 |

不是把不同checkpoint各自产生的视频状态混在一起。每组都固定相同raw history、当前noise/endpoint插值、source、action pair与chunk/sigma；过去和未来动作不变，只替换当前chunk。每个checkpoint按自己的权重重建KV，A/D内部共享只读KV。完整18点由两个validation scene、3个chunk、3个sigma组成。

当前动作与video latent跨度的对应、直接own-frame反馈和当前设计的mask已核查；真实H3的chunk1下，A/D分别重建的历史KV哈希相同。当前预测不改cache内容、commit计数或参数version，重复预测RMSE=0。因此不是“完全没送入动作”或“干预顺手改了历史KV”的证据。首块也失配，不能只归因于长时generated-history漂移。

仍然存在重要结构差异：Original teacher双向重算历史，student使用过去clean commit的raw KV；公共prefix/video反馈和past-action可见性也不同。RGB anchor按相同源码构造但旧receipt没有保存逐tensor hash，teacher旧receipt也没有完整输出hash，故不声称所有中间tensor逐bit一致。不能把低cosine单独锁定为某一层权重错误，也不能由48更新判定普通FM不可能。

[GT完整比较](geometry/compare_00_48_gt/RESULTS.md) · [固定generated完整比较](geometry/compare_00_48_step00_generated/RESULTS.md)

## 与实际视频相互验证

停车场30 steps/chunk完整A/D：Original为A+1.181253、D−0.842124、分离度2.023377；causal0为−0.008565；FM48为A−0.182680、D−0.163995、分离度−0.018685。静态检查所有39帧及0/8/16/24/30/38对齐帧：人物和车库大体保留，A/D大多共同向场景深处移动，未恢复左右控制；不是实时播放评审。30步每条90 noisy forwards+3 clean commits，不能按30/8解释加速。

[停车场完整三列A/D视频与表](report/parking_step48_30step/README.md)。Original是旧legacy精度及原条件协议，causal0/48才是配置匹配的训练前后对照。

两个自然场景的GT-history30已完整。第一个人物保留但建筑/树林漂移及边界重置；第二个也保持人物轮廓，但约17帧GT历史切换产生位置重置，后段轮廓模糊。FM0和48没有明确视觉质变。第二个边界MAD28.7011→27.5534，均高于Original13.8119；该指标是帧差，不等于画质。GT-history拼接是oracle诊断，不是自由rollout。

[两个GT-history全39帧对照](report/trained_complete_gt30/README.md)。generated30/8及停车场8步仍等待完成，不能提前替它们下结论。

## 下一受控机制实验

首块2×2消融提示公共prefix读取当前video、own-action绑定共同影响Original动作场；仅恢复公共prefix边的delta cosine为0.240543，仍不够。隔离候选让每chunk有自己的当前video反馈prefix，video只直接读own action，历史video仍为persistent raw KV；prefix不读历史video KV，不共享可被未来chunk更新的prefix状态。

10项tiny-H3 CPU FP32/BF16检查已通过，真实33B跨chunk只读探针已在释放的GPU1启动（PID421741），和原GPU0/4评测合计3张卡。它没有训练、没有接入生产默认或meeting，也没有视频效果结论。首块identity只能做正控。
