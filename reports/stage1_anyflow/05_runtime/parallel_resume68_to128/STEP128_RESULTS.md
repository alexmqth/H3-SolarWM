# AnyFlow128完整评测：8步动作分离增加，视觉仍未通过

128训练及A/D39f、4/8 steps/chunk全部完成。原控制器2446816正常结束，不再自动追加训练。当前目标未达成，未进入Stage2。

| updates | A 8/chunk | D 8/chunk | A−D |
|---:|---:|---:|---:|
| 16 | −1.312275 | −1.492164 | 0.179888 |
| 32 | −1.305473 | −1.478164 | 0.172691 |
| 64 | −0.589203 | −1.212299 | 0.623096 |
| 96 | −0.483647 | −0.729654 | 0.246007 |
| 128 | +0.040414 | −0.719245 | 0.759660 |

128的8步A全片flow刚变为正，但接近零，分离度仍低于1.0；4步A−0.751431/D−0.885575、分离度0.134144，仍失败。不能把128称为视觉最优：D后段比64更加重影和雾化，64保留作视觉参考，128保留作更强的动作数值参考，两者都未验收。Original30为A+1.181253/D−0.842124、分离度2.023377。

## 完整画面与可播放对照

- [Original / 64 / 128，8步A/D](original_anyflow64_anyflow128_8step_AD.mp4)
- [Original / 64 / 128，4步A/D](original_anyflow64_anyflow128_4step_AD.mp4)

四条完整0–38帧contact sheet，以及Original/64/128在12/24/30/38帧对应画面均已静态查看。4步两动作约18–20帧起重影，22帧后严重雾化/人物轮廓分解；8步A保留主体与车库，但人物退远、约22帧后透明/重影，缺乏与Original一致的A运动；8步D在22帧后人物/柱子叠影、后段画质明显差于64。静态全帧复核不是实时播放或VBench。

两条grid均保留全部39帧、24fps、H264/yuv420p/faststart；完整解码和第30帧标签检查通过。均为诊断，不替换meeting。

## 已有指标

| action | steps/chunk | flow | E2E s | peak allocated MiB | CPU KV MiB | noisy+commit | gray MAD | boundary RGB MAD |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| A | 4 | -0.751431 | 215.15 | 38984.16 | 6484.13 | 12+3 | 4.0733 | 6.4479 |
| D | 4 | -0.885575 | 220.53 | 38984.16 | 6484.13 | 12+3 | 4.1985 | 7.5259 |
| A | 8 | 0.040414 | 289.27 | 38984.16 | 6484.13 | 24+3 | 3.0477 | 4.1016 |
| D | 8 | -0.719245 | 276.56 | 38984.16 | 6484.13 | 24+3 | 3.2189 | 4.3577 |

步骤按每块计数，4/8步×3块=12/24 noisy forwards，另有3 clean commits。CPU KV不等于整进程RSS；MAD是运动活动量，不能作视频质量分数。相同首帧/prompt/各自action/seed/video-audio noise与Original逐张量核对。共享主机单次计时、Original offload配置不同，不作公平speedup或GPU节省结论。

## Stage1与历史分布诊断

[同一128模型的teacher/generated-history完整对照](../../03_anyflow_trials/history128/FINAL_RESULTS.md)已完成。teacher history A+0.504958/D−0.828582、分离度1.333539，后段人物/环境更清晰，但A有17/34帧状态重置。首块5 latent两种history逐元素相同；是历史影响的实测证据，不是free-running PASS。oracle含动作后的场景信息，[固定同一历史改变当前动作的独立诊断](../../02_causal_diagnostics/counterfactual128/FINAL_RESULTS.md)已完成：相对响应方向一致，同历史A−D约0.314/0.427，但尚未验证Original同状态的控制保真。

128生成A的RGB区间[0,17)/[17,34)/[34,39) flow为−0.074203/−0.114319/+1.039100，前两个区间并未整体转正；D为−1.206855/−0.527455/+0.186407。末区间只有4个transition且有temporal-VAE跨帧影响，不能用末段或全片正数宣称动作已恢复。完整逐transition在history128/history_response.json。

真实训练/恢复/四replica/精度检查见[TRAIN128_RESULTS.md](TRAIN128_RESULTS.md)及[固定学习曲线](training_curve_00_16_32_64_68_96_128.json)。没有AnyFlow128与同剂量FM128对照，不能拿FM32宣称AnyFlow少步收益。下一步从同history动作响应定位，不自动继续加AnyFlow训练次数，不要求Stage1先消除全部长时漂移；具体阶段口径见[STAGE_BOUNDARY.md](../../07_protocols/overviews/STAGE_BOUNDARY.md)。
