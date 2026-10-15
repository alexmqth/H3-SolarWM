# EXP-023/v1 Judge 最终审核

2026-10-15。**决定：accept（有限普通 FM 训练与评测任务完成）；视觉质量/动作 PARTIAL，新增能力收益未证实，不升级 V3。** 不继续当前 tail8/rank8/C2-only 配置的短预算微调。

## 独立核查

[CPU复核](FINAL_AUDIT.json)与[复核脚本](final_audit.py)：checkpoint128 SHA匹配，128更新记录与保存的训练顺序逐项一致，32clip各4遍，adapter参数有限、记录梯度非零，optimizer与RNG状态齐全；最新恢复清单47个文件SHA一致。独立相加8阶段账本与累计完全一致；失败保留，未超授权。全部18个MP4重新完整解码、56帧/24fps/尺寸/PTS与SHA通过。

真实首步重算门控[证据](../GRADIENT_PROTOCOL_CHECK.json)：forward50层，backward重算49→42保持current-prefix，冻结base/released action LoRA无梯度，新adapter有有限梯度。历史cache为**签名不变**检查，不夸称本次对全部KV逐字节复核。checkpoint0入口回归另有150项KV/RoPE逐值相等证据。EXP-022旧forward-only作用域的梯度PASS撤回继续有效，本轮不追认旧step02。

## 效果与视觉审阅

8个固定GT-history FM probe均下降。均值0.1542203575→0.1523224935，**两个均值之比下降1.230618%**；逐probe相对变化再平均为**下降1.343662%**，统计口径不同。四clip、两个σ、已观察dev，不能推断充分训练或泛化效果。

Judge审阅26张连续帧页：四条GT-history对照的RGB35–55（覆盖全部17新帧和边界），两条self-history对照的全部RGB0–55；非仅首末帧抽样。审阅方法是解码后逐帧图序列，非实时播放评分；所有页见FINAL_AUDIT中的visual_review_sheets。三列为GT/0updates/128updates。

- train constant山路与train switch草地：人物可辨、场景延续，训练前后姿态/位移基本近似，不能确认控制增益。
- dev constant岩石路：GT-history续写仍有运动节奏/细节差异；self-history运动明显不足，姿态重复/滞留，训练后未恢复GT运动轨迹。
- dev switch城镇：GT-history在约40帧附近有视点/人物构图变化，前后均存在；self-history从C1开始相机路径偏离GT，后段人物偏低、背景漂移，前后趋势接近。
- 未见训练后新的大面积人物崩解，但不能把“没有更坏”写成新增能力PASS。低分辨率帧页不用于证明细微ghosting全无；原片保留用于细节查看。

GT-history前39帧为展示的真实帧，只生成后17帧；self-history全56帧生成，其checkpoint0/128的C1状态不同，是端到端对照而非同状态C2消融。当前无同状态A/D反事实，不能给方向正确率。相邻帧MAD下降也可能来自运动减少，不作画质或连续性提升证据。与GT轨迹不完全相同本身并不证明动作错误，本轮结论是缺乏清楚的改善证据。

## 成本

| 项目 | 实测 |
| --- | ---: |
| 全阶段含失败GPU进程wall相加 | 6222.051秒（1.72835 GPU小时） |
| 成功训练128updates | 2650.296秒（44.17分钟） |
| 训练峰allocated / reserved | 24.540 / 28.252 GiB |
| pre / post评测阶段wall（各6片+probe等） | 766.373 / 660.890秒 |
| 两评测阶段峰allocated / reserved | 25.342 / 28.102 GiB |
| 每步C1 CPU raw KV | 6.332 GiB |
| 顶层DiT预算计数 / 上限 | 444 / 449 |
| backward与optimizer updates | 128 / 128 |
| VAE encode / decode；text | 70 / 14；151 |

444含一次失败C2入口reserve，不写成444次成功网络调用；activation checkpoint内部重算不另计顶层DiT。训练阶段秒数含该进程加载/保存等，不等同纯kernel时间。单片采样用时见POST_COMPARISON，GT-history C2约36.6–49.5秒、self-history C1+C2约81.2–105.0秒，均不含完整编码/加载/写片。前后运行耗时差不是算法加速证据。

## 研究决策与后续

本轮验证了修复后真实视频普通FM、128updates保存恢复、匹配评测的闭环；1.23%的代理指标收益不值得围绕当前配置反复做微调。只训GT-history C2、1.72M adapter、128updates并不足以否定FM或H3架构，也不能确认“参数量太小”就是原因。

EXP-024/v1已独立下发并消息通知原Worker，Worker确认开工：CPU检查C1/C2训练覆盖、adapter实际变化、全层QKV容量与有限GPU profile入口。CPU预算4线程/30分钟/额外8GiB，GPU尚未放行。不跳到AnyFlow/DMD/新Memory。公开SolarWM Stage2-fix官方相机基线仍待准备和独立预算，未声称已复现。

权重、encoded tensor和RGB中间数组留本地；Git发布代码、元数据、日志、正式MP4、审核图及报告。历史失败attempt与授权补充保持原样。
