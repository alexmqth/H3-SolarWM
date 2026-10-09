# 同一AnyFlow128：干净历史明显改善后段，仍有oracle重置

[完整39帧Original / generated / teacher history，A/D两行](original_generated_clean_history_AD.mp4)。右列明确标注TEACHER history，不是free-running结果。H264/yuv420p/24fps/faststart、39帧完整解码和第30帧标签检查通过。

同一shift12 step128，8 steps/chunk，除了commit和下一块RGB anchor使用teacher历史，模型/image/prompt/action/seed13/video-audio noise、RGB dual、CPU raw KV、action rows与feedback全部相同。真实tensor检查：A/D首块5 latent均逐元素相同（max_abs=0），全部conditioning tensor与packed tensor一致，后续latent才不同；证据[history_response.json](history_response.json)。首块latent相同不保证靠近边界的decoded RGB相同，temporal VAE读取邻近latent。

teacher history分离度 **1.333539**，generated history **0.759660**。数值改善和后段画质改善支持历史来源是失败因素之一；不能据此单独确认当前action binding，也不是最终验收通过。

| history | action | flow | E2E s | gray MAD | boundary RGB MAD |
|---|---|---:|---:|---:|---:|
| generated | A | 0.040414 | 289.27 | 3.0477 | 4.1016 |
| generated | D | -0.719245 | 276.56 | 3.2189 | 4.3577 |
| clean | A | 0.504958 | 253.77 | 4.5154 | 13.5048 |
| clean | D | -0.828582 | 246.73 | 3.7793 | 9.8650 |

两条teacher-history全0–38帧和Original/generated/clean在12/24/30/38帧对应画面均静态复核。Teacher history下人物/车库到末帧保持明显更完整；A在17和34帧出现人物位置/朝向重置，D也有边界状态跳变。因此边界MAD明显高于generated，oracle拼接不连续不能称为稳定生成。静态全帧检查非实时播放。

| action / history | RGB[0,17) | RGB[17,34) | RGB[34,39) |
|---|---:|---:|---:|
| A generated | −0.074203 | −0.114319 | +1.039100 |
| A teacher | −0.051449 | +0.692521 | +1.542388 |
| D generated | −1.206855 | −0.527455 | +0.186407 |
| D teacher | −1.213613 | −0.854157 | −0.526932 |

上述沿用Farneback，分段排除跨边界transition并另行记录；末段只有4个transition，RGB分段受temporal VAE影响，不是独立latent chunk隔离。A后续方向改善并非全靠边界尖峰，但teacher历史本身含A的运动结果，仍需同一history上的当前动作对照。

每条24 noisy+3 clean commits、allocated峰值38984.16MiB、CPU KV6484.13MiB。共享主机并行单次时间，不声称加速或显存节约。

下一步固定相同teacher prefix/past action，在chunk1分别输入A或D。已有held-A/held-D作为参考，仅新增A→D→D与D→A→A两条，归因仅针对chunk1；chunk2 cache可能受改变后的动作commit影响。详见相邻counterfactual128报告。Stage1尚未完成，未开始Stage2，保持[阶段边界说明](../../07_protocols/overviews/STAGE_BOUNDARY.md)。

固定历史动作干预现已完成，见[完整结果](../../02_causal_diagnostics/counterfactual128/FINAL_RESULTS.md)：A/D相对响应存在，但同历史分离度仅0.314/0.427，不能用oracle 1.334直接当控制保真。
