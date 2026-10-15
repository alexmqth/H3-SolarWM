# V3 普通 FM 小数据训练：128updates / FM-8NFE

**目的：** 用真实视频训练严格因果V3，检查局部续写和自生成历史下的画面/动作能否改善。EXP-023已验收任务完成；**视频收益未证实，V3正式版本不变**。

## 名称与设置

FM = Flow Matching（流匹配）；128updates = 累计128次optimizer训练更新；FM-8NFE = 每个生成chunk采样调用网络8次，clean KV commit另计。GT = Ground Truth（真实视频）；C1/C2 = 第一/第二个生成块。LoRA = Low-Rank Adaptation（低秩适配）。tail8 QKV rank8 = 仅最后8层的query/key/value投影新增rank8 adapter，共1,720,320参数。它不等于训练了AnyFlow。

32train×4遍，2dev来自已观察episode；只训练GT-history C2，C1仅作为无梯度clean KV历史。原H3及released Action LoRA冻结。56帧、24fps、832×480；推理全部FM-8NFE。

## 结论与成本

8个固定FM probe均值MSE 0.154220→0.152322，下降1.23%；训练前后视频接近，运动滞留、路径偏差和边界变化仍存在。相邻帧MAD下降不代表动作/连续性提高。当前证据不足以认定容量不足或H3架构失败。

成功训练44.17分钟，峰allocated24.540GiB、reserved28.252GiB，CPU KV6.332GiB。全任务含编码/评测/失败共6222.051GPU秒（1.72835GPU小时）。评测峰allocated25.342GiB；单片C2采样36.6–49.5秒、自生成C1+C2采样81.2–105.0秒，均非完整端到端用时。全部18个MP4独立SHA/完整解码/56帧/24fps/PTS通过。

## 六组三列对照视频

每条依次为 **真实源视频 / 训练0updates / 训练128updates**。GT-history片前39帧为真实历史，后17帧为生成；self-history片全56帧生成，前后各自生成C1，因此历史状态并不相同。

- [dev constant / GT history](videos/dev_constant_gt_history_source_vs_0_vs_128.mp4)
- [dev constant / self history](videos/dev_constant_self_history_source_vs_0_vs_128.mp4)
- [dev switch / GT history](videos/dev_switch_gt_history_source_vs_0_vs_128.mp4)
- [dev switch / self history](videos/dev_switch_self_history_source_vs_0_vs_128.mp4)
- [train constant / GT history](videos/train_constant_gt_history_source_vs_0_vs_128.mp4)
- [train switch / GT history](videos/train_switch_gt_history_source_vs_0_vs_128.mp4)

## 证据与下一步

[Judge最终审核](../../../../experiments/EXP-023_v3_fm_small_data/judge/FINAL_REVIEW.md) · [Worker报告](../../../../experiments/EXP-023_v3_fm_small_data/WORKER_REPORT.md) · [前后数值](../../../../experiments/EXP-023_v3_fm_small_data/POST_COMPARISON.json) · [成本账本](../../../../experiments/EXP-023_v3_fm_small_data/CUMULATIVE_BUDGET.json) · [视频来源SHA](VIDEO_MANIFEST.json)。大checkpoint、encoded tensor、RGB数组留本地，不随Git发布。

停止当前配置的短预算微调；EXP-024先补齐C1/C2训练覆盖并核实容量与实测profile方案。它是CPU准备，尚无新GPU训练结果。[返回FM→AnyFlow](../README.md)。
