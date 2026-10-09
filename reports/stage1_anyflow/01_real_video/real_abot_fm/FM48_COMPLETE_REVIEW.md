# 实验B FM48完整验收：有限预算未通过

2026-10-08 22:48。48次真实ABot普通causal FM更新、6条自然验证视频、4条停车场A/D视频、GT/generated共36个匹配反事实状态全部完成。GPU/CPU自动队列已正常结束；本报告不由loss或单个内部指标代替画质/action验收。

**结论：这次FM48没有建立可用于继续C/D的可信causal局部动作/视频基线。** 它不证明普通causal FM不可能，也不能用48次更新宣称完成官方Stage1训练。预算不延长。

## 1. 同状态动作geometry

| 固定状态来源 | 整体cos0→48 | A/D差分cos0→48 | 动作差分范数比0→48 |
|---|---:|---:|---:|
| GT（18状态） | 0.988193→0.989033 | 0.017516→0.017641 | 0.6433→0.6539 |
| 固定step00-generated（18状态） | 0.994733→0.995210 | 0.012540→0.009336 | 0.6914→0.6732 |

[完整干预协议与限制](FM48_GEOMETRY_RESULTS.md)。Raw history/noise/当前state固定，仅替换当前chunk A/D；动作是有响应但差分方向失配。Teacher双向重算历史与student persistent KV的结构差异明确保留。

## 2. 真实验证场景：逐帧观察

静态检查了所有6条FM48视频的0–38全帧contact sheet，并检查GT/Original/FM0/FM48的0/8/16/24/30/38对应帧；这不是实时播放。

| 配置 | 第一场景 118e…A_1140 | 第二场景 dfec…D_1750 |
|---|---|---|
| GT history，30/chunk | 人物保留，建筑/树林漂移；边界历史重置；无明确质变 | 人物轮廓保留，17帧位置重置、后段模糊；无明确质变 |
| generated history，30/chunk | 人物保留、运动变小，环境仍有形变 | 约23帧出现红色残影，26–35人物分解，末段近乎消失；未修复 |
| generated history，8/chunk | 人物大体保留，建筑/树林出现横向拖影与模糊 | 早段已模糊；约23帧后红色残影/人物分解，末段严重虚化 |

GT历史边界重置是oracle chunk拼接诊断，不能作为自由rollout的因果证据。generated30里边界MAD下降而人物仍分解，说明该指标只能衡量帧差。自然动作含联合键盘/镜头，不能把两场景之间的flow当作纯A/D保真。自然8步没有step00同协议视频，只有Original参考与FM48，不作8步训练前后提升结论。

[GT30完整对照](report/trained_complete_gt30/README.md) · [generated30完整对照](report/trained_complete_generated30/README.md) · [generated8完整对照](report/trained_complete_generated8/README.md)

## 3. 已知停车场纯A/D正控

| 方法 | A | D | A−D | 验收 |
|---|---:|---:|---:|---|
| Original30整段 | +1.181253 | −0.842124 | 2.023377 | 原始动作正控 |
| FM0 30/chunk | −0.195516 | −0.186950 | −0.008565 | FAIL |
| FM48 30/chunk | −0.182680 | −0.163995 | −0.018685 | FAIL |
| FM0 8/chunk | −0.021397 | −0.031908 | 0.010510 | FAIL |
| FM48 8/chunk | −0.025848 | −0.043304 | 0.017456 | FAIL |

30步人物/车库大体保留但A/D几乎同向；8步约22帧后人物透明叠影和车库重影仍在。[完整停车场评审](PARKING_FM48_REVIEW.md)。Original旧legacy精度及原条件，causal0/48才是匹配配置训练对照。

## 4. 训练、执行和范围

训练固定48updates/192噪声样本，中/高噪声raw validation loss分别下降1.42%/5.11%；低噪声validation未覆盖，训练低sigma只有4/192。这个覆盖缺口仍是后续可检验假设，没有证据允许直接认定低噪声upweight就是解法。[训练审计](FM48_TRAINING_RESULTS.md)。

每条39RGB、12latent、5+5+2chunks、RGB dual、history5、CPU KV。30步90noisy+3commits；8步24noisy+3commits。各条完整时间/峰值/首块延迟/CPU KV/MAD/boundary/flow在链接CSV中；共享主机单次执行，不能作warmup均值或速度提升结论。

[8个对比MP4完整解码收据](trained_complete_archive_video_validation.json)：39帧、H264/yuv420p、24fps、faststart；这些是诊断材料，不替换meeting主demo。没有新增全量数据下载、模型副本或Stage2训练。

接续工作是同状态路由/历史表示机制诊断，已有单独隔离的current-prefix候选；它必须经过后续chunk验证，不能把首块Original identity当作完整恢复。C的finite-map可信性和D的teacher/critic/FMBS训练仍未完成，A–D总体目标未完成。
