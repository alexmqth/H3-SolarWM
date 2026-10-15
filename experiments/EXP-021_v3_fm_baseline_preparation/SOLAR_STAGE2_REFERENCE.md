# SolarWM 官方 H3 Stage2-fix 标准相机推理准备（未运行）

本地已有 `models/SolarWM/SolarWM-h3-33B-base` 及 `SolarWM-h3-33B-sgf-stage2-158f-fix/ema.pt`；只读来源清单在 `../EXP-019_v3_mixed_action_validation/judge/SOLAR_REFERENCE_INVENTORY.json`，EMA SHA-256 为 `92e8f4b07aa92b0453a2f63954db1421caa3b8cac83ece706535db1a91493a05`、step 3900。存在权重不等于官方推理通过。

官方 `SolarWM/configs/examples/minimax_h3/infer-stage2-158f-sp4.yaml` 要求 8 GPU、SP4、158 RGB 帧/47 latents、1344×768、4 NFE、预编码 WebDataset 的 test index、silence latents 与 encoder contract。官方命令见 `SolarWM/docs/backends/minimax-h3.md` 的 “Stage2 SGF”。本地在 `/home/qma/work` 的文件名清单中**未发现** `test-index.jsonl.gz`、`h3_silence_153_158_170.safetensors`、`encoder_contract.json`，也没有发现 `SolarWM-Data` 根目录；正式运行前必须核对或补齐这些输入及 dataset shard，并检查环境/backend import。不得绕过缺失的相机轨迹输入，用 EXP-020 文本动作片冒充官方标准相机样本。

后续独立任务草案：CPU 校验 base、EMA、官方配置和预编码测试样本/相机轨迹/支持文件的 hash 与 schema；8 张确实空闲 GPU 且仍在允许时段时，先一次 `torchrun --standalone --nproc-per-node=8 -m solarwm infer` smoke，记录首个样本完整加载、模型峰值显存、进程 RSS、失败调用、NFE/总 wall、输出帧数与逐帧解码；超时/资源越界立即停。需 Judge 单独签发 GPU 预算，且不能延续到次日 09:00 后的 3 卡限制。当前 EXP-021/v1 只准备文档，不执行该命令。

EXP-020 是 Original H3 的 56 帧/17 action spans、832×480、30 NFE、seed13、原生双向文本动作全片生成；官方 Stage2 为 158 帧/47 latents、1344×768、4 NFE、相机轨迹、不同 checkpoint 与数据编码协议，精度配置为 bfloat16。两者不具备公平的画质或速度直接比较条件。EXP-020 的天空伪影也不构成针对官方 Stage2 调参的理由。
