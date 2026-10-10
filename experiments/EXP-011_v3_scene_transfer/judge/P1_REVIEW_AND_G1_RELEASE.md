# EXP-011 P1验收与G1首窗放行

2026-10-11 06:02 HKT。P1输入编码接受，生成能力未评估。Judge独立CPU审计确认新两fixture来源、原生CPU噪声逐值重建、Single I0、A/D非动作条件和所有保留Global坐标；Worker检查own-action mask及未来action删除通过。真实6text/2image encode、0denoiser/0decode，含两次失败启动累计70.564697678秒，allocated峰40.576426GiB。失败与修复证据保留，输入/模型协议没有改变。

现在批准GPU0顺序执行industrial/village × FM30/FM8四个G1首窗，合计最多76 sampling forward、0commit、4VAE decode，沿用推理累计4500秒与09:00截止。启动环境按G1_APPROVED.json，尤其ABOT_VRAM_RESERVE_GIB=18保持已验收推理offload设置，显式模型根禁止下载。每个首窗从各场景相同seed13噪声独立生成，全部39帧需审查，严重结构崩溃的配置不继续G2。无自动重试、无新训练。

G2仍未放行；P1工程成功不代表画质/动作迁移通过。正式V3结果保持冻结。
