# EXP-014/v1 T1 Worker 报告：首张训练初图的 FM30 教师反事实

2026-10-11 HKT。按 Judge [T1 marker](judge/T1_APPROVED.json)，GPU0 对固定 `s0_43866101/A_1245` 完成 V3 原生 FM30 C1 12 latent/39 RGB，然后用**同一份自己生成的 C1**作 sigma0 clean commit，分叉 AA 和 AD 的 C2 5 latent/17新RGB。该图真实录屏动作是 A+S+L；此处 A/D 是合成模型控制，不能解释为录屏 GT。

T1 的实际模型账本为 **90 sampling forward + 1 commit forward、3 VAE decode、0 backward/update**，墙钟 GPU 秒 **532.261 / 1350**。C1 wall 203.271秒、C2过程320.214秒；allocated峰38.181GiB/卡。C1后 `gc.collect()+empty_cache` 的残余 allocated 0.0089GiB，避免同进程第二次加载33B时保留上一份模型。C1 cache 50层各一个 `index=0` 项，raw K/V/RoPE 合计 **6,799,104,000 bytes**，AA/AD 采样过程 cache 身份和版本保持不变。完整 [逐调用账本](artifacts/T1/budget.json)、[C1结果](artifacts/T1/C1/result.json)、[C2结果](artifacts/T1/result.json)、[模型生命周期](artifacts/T1/model_lifecycle.json)、[原日志](artifacts/T1/T1_s0.log)保留；6.4GB cache和 latent endpoint 原件仍在 `H3-World/outputs/EXP-014_v3_multiscene_teacher/`，由各结果SHA及 [目标索引](artifacts/T1/target_manifest.json)关联。

T1 stdout显示部分模型文件由 editable DiffSynth 的旧快照路径解析；[模型文件身份审计](model_alias_audit.json)确认旧路径与冻结路径下29个 `.safetensors` 全部为相同设备/inode，实际读取的是同一批权重文件。关键DiT/pipeline源码SHA也与冻结版一致。

Worker [CPU协议审计](artifacts/teacher_audits/s0_43866101.json) PASS：首窗和两条56帧视频均为832×480/24FPS、完整可解码、PTS唯一；AA/AD C2噪声、sigma网格、全局位置相同，只在第12–16个 action span 不同；两条前39RGB和已发表C1逐值相同，50层实际cache都为index0，C2 endpoint finite且shape `[1,24,5,30,52]`。AA/AD 新17帧平均像素绝对差28.307，说明生成结果确实分开。联系表和原视频均已存入本目录，Judge另行完成独立KV/逐帧验收。

| C2分支 | 中央区域Farneback水平光流和 | 边界MAD(38→39) | C2段内相邻帧MAD均值 | 视觉观察 |
|---|---:|---:|---:|---|
| AA | +42.147 | 7.450 | 10.063 | 人物、道路、树木保持可辨；有局部软化/姿态变化 |
| AD | −27.459 | 5.494 | 9.818 | 同一C1后产生相反运动，人物结构仍可辨；有局部软化 |

光流是画面运动的辅助数值，不独立证明角色运动语义正确。Judge 已逐帧查看39+17+17帧和关键原分辨率细节，接受首图为**有限可用的teacher目标、quality PARTIAL**；没有把它升级为长时稳定或统计泛化。正式T2只有在Judge另行签发marker后才能做。

快速查看：[AA/AD同历史并排56帧](artifacts/teacher_comparisons/s0_43866101_AA_vs_AD_56.mp4)；[C1 39帧](artifacts/T1/C1/first39.mp4)；[AA 56帧](artifacts/T1/AA/rollout_56.mp4)、[AD 56帧](artifacts/T1/AD/rollout_56.mp4)。所有视频都是实际保存的 H3 输出，未重生成或覆盖原件。
