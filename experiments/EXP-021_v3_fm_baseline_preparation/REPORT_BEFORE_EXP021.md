# EXP-019 当前执行状态

2026-10-15 HKT。P0 CPU 预检与 Judge 来源核查 PASS，R 阶段已获 Judge 正式接受。[R Worker 报告](../../../submission/experiments/EXP-019_v3_mixed_action_validation/R_WORKER_REPORT.md)记录 canonical A 旧/新入口在 C1/C2、σ=1.0/0.5 四组 velocity 与 50 层 150 组历史 KV/RoPE 逐值相同；10 forward、379.384 秒、allocated 峰 23.891 GiB。σ=0.5 是噪声缩放探针，不是采样轨迹。

E 阶段已获 Judge 接受：四个训练场景现场编码真实 I0/caption 与变长混合动作句，14 text encoder 调用、0 DiT/VAE/训练，完整 wall 192.620 秒、allocated 峰 26.483 GiB。四份 fixture 的新 I0/head、37 个连续 action spans/tags、真实已编码未来动作内容/长度干预后的可见前缀隔离通过。冻结配置字段 `imgvid_cond_noise_aug=0.999` 在 bfloat16 实际舍入为 1.0，四个 anchor 等于 clean I0 patch rows。见 [E Worker 报告](../../../submission/experiments/EXP-019_v3_mixed_action_validation/E_WORKER_REPORT.md)和[CPU 审计](../../../submission/experiments/EXP-019_v3_mixed_action_validation/E_CPU_AUDIT.json)。

**G1 首场景 `43866101` 已获 Judge 验收；G2 余三场景已按独立 marker 和并行补充授权全部完成，等待 Judge 最终审核。** 真实录屏前39帧作为 GT history，只生成 RGB39–55 的17帧；同一 C1 cache/初噪/真实复合动作的 FM30 与普通 FM8 共 39 DiT forward（含1 commit）、2 VAE decode、0训练，完整 wall 332.172 秒，allocated 峰 24.945 GiB，CPU raw KV 6.332 GiB。两条新增片人物/山路持续可辨，无持续肢体分解或换场；中心区域水平光流代理均为负且接近真实后续，但人物朝向和路径与 GT 有差异，单脚本不能证明动作语义或反事实控制。六条相关视频各56帧完整解码、全部17新帧已目视，边界和逐帧指标见 [G1 Worker 报告](../../../submission/experiments/EXP-019_v3_mixed_action_validation/G1_WORKER_REPORT.md)、[17帧三列图](../../../submission/experiments/EXP-019_v3_mixed_action_validation/G1_ALL_NEW17_CONTACT.jpg)与[CPU 审计](../../../submission/experiments/EXP-019_v3_mixed_action_validation/G1_CPU_AUDIT.json)。GT-history 局部结果不能推广为 free-running 或跨独立验证集成功。


G2 三场景 `7199292c`、`9dc2e588`、`b784d995` 均完成 39 forward（各含 1 commit）、2 decode、0训练；GPU 分别为 0/2/3，wall 分别 361.165/391.863/395.777 秒，峰 allocated 24.945/24.951/25.294 GiB。全部六条相关视频每场景均为完整 56 帧、24fps 且 PTS 递增；前39帧仅用于 GT history 与展示，新增 RGB39–55 在逐帧图中可检查。草地两场景人物结构可辨；工业道路 FM8 背景/塔架和路面构图变化较明显，未见持续人物分解。四场景光流仅为画面运动辅助，不证明角色/相机动作语义。b784d995 的真实脚本包含左平移→静止→前进，但单条轨迹没有同状态反事实，切换控制仍未通过验收。见 [G2 Worker 报告](../../../submission/experiments/EXP-019_v3_mixed_action_validation/G2_WORKER_REPORT.md)、[逐场景视频入口](../../../submission/experiments/EXP-019_v3_mixed_action_validation/README.md)和[机器指标表](../../../submission/experiments/EXP-019_v3_mixed_action_validation/METRICS_TABLE.csv)。

EXP-019 总账本为 166 DiT forward（含6次commit）、14 text encode、8 VAE decode、0训练，独立进程 wall 相加 2052.980 GPU 秒（0.5703 GPU 小时），最高同时占用2张项目卡；并行经过时间不抵扣 GPU 成本。[总账本](../../../submission/experiments/EXP-019_v3_mixed_action_validation/CUMULATIVE_BUDGET.json)已生成。R/E 结论保持已验收状态；G2 为 Worker 判断，待 Judge 正式审核。


## EXP-020 Original H3 真实混合动作参考（Worker 完成，待 Judge 审核）

在独立[冻结任务书](../../../submission/experiments/EXP-020_original_mixed_action_reference/taskbook_v1.md)和 GPU0 授权下，仅对 `b784d995` 运行一次 Original H3 + released Action LoRA 原生双向 pipeline，真实I0/原caption/17句左移→静止→前进、seed13、30NFE、56帧。实际30 DiT、4text、1首帧encode、1视频+1音频decode、0训练；完整wall 282.237秒，峰 allocated 26.049GiB。原片与[GT并排片](../../../submission/experiments/EXP-020_original_mixed_action_reference/artifacts/b784d995/gt_vs_original56.mp4)均完整解码56帧/24fps。人物大体可辨，但约RGB20以后天空有明显放射状结构伪影；单轨迹无法确证晚段动作切换。见[Worker报告](../../../submission/experiments/EXP-020_original_mixed_action_reference/WORKER_REPORT.md)和[CPU审计](../../../submission/experiments/EXP-020_original_mixed_action_reference/CPU_AUDIT.json)。Original 从I0生成全56帧，与EXP-019借用前39帧GT历史只续写17帧条件不同，不能直接作单因素画质/速度比较。外层zsh收尾误用只读变量导致shell返回码1，模型自身正常完成并落盘，无GPU重试。
