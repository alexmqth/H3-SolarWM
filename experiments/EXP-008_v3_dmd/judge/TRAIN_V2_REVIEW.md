# EXP-008/v2 DMD有限延续训练审核

2026-10-11 04:21 HKT，Judge。**接受累计8个student DMD cycle的训练工程结果；生成质量待视频检验。** PID1017778已退出，result为complete_8_pending_judge，budget关闭。

新增cycle2–8全部完成，按AD/AA交替。实际99forward、14backward、14optimizer update（student7、fake7）、0VAE，完整墙钟1673.640974秒，三卡保守1.394700812GPU小时。Teacher/fake/student峰值allocated分别25.07593/25.88506/28.10182GiB，未随轮次明显增长，低于原2.25GPUh/44GiB预算。

[独立审计](train_v2_audit.json)全部通过：29项来源未改、全轮次精确角色/call顺序、native8map及同sigma评分、各角色cache重建/只读和base冻结、8map梯度及参数更新；实际CPU/三卡CUDA/两个generator恢复正确；从pilot噪声RNG重放全部7轮训练噪声逐值一致；cycle4/8的四文件SHA、metadata、student/fake optimizer步数及噪声RNG正确。最终student step8、fake step9，不是重新初始化训练。

最终checkpoint为`H3-World/outputs/EXP-008_v3_dmd_train_v2/cycle_08/`。Student QKV SHA `49534764770bd2cdc182624bd6f1c947daed1406824cc201b52d01897015bc06`，target SHA `ad5bce068b92653d1b84946fef1377d3ed264e9b9c0f0cb320f9aec40efb6130`，fake SHA `e2d9b5f09ab2d0087eeebfbeac59b3b0bec9bd1a2ae3fba201a5ab1852dc1ef2`，optimizer/RNG SHA `f83524a2da23821b0bb4040a660b5dfeaf79328f97a2cb23d8b5e51f494f0d3f`。

## 质量风险

Fake FM loss从cycle2约0.155、cycle3约0.411，cycle4升至31.535，之后约2.5；相应normalizer在cycle4升至2.625。DMD surrogate下降不能被解释为生成改善，后期QKV梯度也明显变小。这些是需要视频验证的风险信号，不单凭loss宣布通过或失败，也不为追loss自动追加训练。

本实验仍是固定FM8 C1下的current-block on-policy训练，teacher为冻结V3 causal模型，fake仅有限更新；不代表完整SolarWM Stage2、充分拟合的score或跨场景泛化。

## 下一阶段

最终cycle8评估CPU preflight已由Judge独立运行通过：21项来源、配套checkpoint、native8sigma及公共FM8首窗匹配。按预定义最多35forward/4VAE/.35GPUh进行视频；先检查AA/AD C2，若出现持续主体/场景崩溃，停止C3并将本配置作为负结果收口。普通边界/残影缺陷仍按PARTIAL，不机械卡死。具体GPU放行见独立EVAL_RELEASE.md与marker。
