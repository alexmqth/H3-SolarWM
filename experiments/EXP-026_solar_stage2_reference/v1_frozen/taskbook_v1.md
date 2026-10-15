# EXP-026/v1：官方 SolarWM-H3 Stage2-fix 最小推理准备

2026-10-16 HKT。Judge授权原Exp Worker立即执行CPU/有限下载准备；本版本GPU预算为0。EXP-025已accepted收口，收益未证实，不再自动扩训。

## 研究问题与来源

能否用官方Stage2-fix权重、官方真实相机测试输入与原推理协议，准备一个可追溯、可实际启动的基线实例？本轮不是再写一份建议书，必须补齐可获得的输入、执行CPU reader/config检查，并留下精确命令和依赖证据。

Research Track：Baseline Validation / SolarWM相机参考。Parent：公开sgf-stage2-158f-fix EMA step3900。Hypothesis：官方预训练基线可帮助区分当前少量语言动作适配与原生相机模型的能力边界，尚未验证。Baseline为官方配置，保留H3-World语言动作参考，不把相机控制当ABot动作控制。

先读guideline/progress/next_plan、EXP-021的SOLAR_STAGE2_REFERENCE.md、EXP-019的SOLAR_REFERENCE_INVENTORY.json，以及SolarWM/docs/backends/minimax-h3.md、docs/latent-wds.md、docs/data-access.md、官方infer-stage2-158f-sp4.yaml与实际loader/rollout。

已有models/SolarWM/SolarWM-h3-33B-base及sgf-stage2-158f-fix/ema.pt。EMA SHA为92e8f4b07aa92b0453a2f63954db1421caa3b8cac83ece706535db1a91493a05；复用既有完整校验，核对文件身份，不重复下载几十GB权重。

## 立即执行的步骤

1. 固定当前官方checkout revision、工作区diff、环境版本；核对真实L40硬件及可用CPU RAM/disk。只做nvidia-smi查询，不初始化GPU模型。
2. 查询官方发布索引并实际下载所需小文件：clean-158f-h3 recipe test-index.jsonl.gz、H3 silence support safetensors、encoder_contract.json，以及**固定1条测试样本**实际依赖的最少latent-WDS shard/member。记录来源URL/revision、byte数/SHA、sample_id、完整索引行、源路径；优先复用本地数据。
3. 默认采用官方158f Stage2 latent-only验证路径（官方docs称Stage2无需raw-WDS）。读取loader核实能否固定sample_count=1；不得默认跑64样本。首个满足合法索引/长度/相机的样本按固定seed42选定并锁定，不根据画面质量挑样本。若latent访问不可得，可CPU评估官方standalone test-set路径，保留完整行及camera c2w/内参；不隐式切成source-length长推理。
4. CPU实际读取所选记录，检查latent形状/scale合同、caption/text条件、I0、camera姿态/内参、158RGB与47encoded/50rollout/decode47的对应关系；明确prefix/audio silence/MM-RoPE/PRoPE/Sliding Local/W6 raw KV、cache_mode exit、4NFE及time shift。写出来源代码位置，不复制我们V3的时间或action协议替代。
5. 冻结一份仅必要路径/单样本/资源变化的候选配置及和官方的结构化diff；可执行launch脚本默认拒绝未授权GPU。CPU解析并检查所有文件存在、包依赖和入口；不偷偷pip升级原环境。发现缺依赖可在隔离环境准备有限依赖并记录，不更改正在使用的环境。
6. 提交下一阶段具体GPU建议：一条158帧/24fps、4NFE官方相机视频，最小可行SP/FSDP/CPU offload布局、内存风险、加载/采样/commit/decode总时间及逐rank allocated/reserved峰与CPU KV采集方法，完整调用次数预算、超时和停止条件。官方SP并不自动分片所有参数，必须确认33B是否在每rank常驻以及L40能否容纳。不得把8卡等同于已解决参数显存。

## 预算与停止条件

本阶段0GPU forward/backward/encode/decode，0updates；CPU并发≤8，峰RSS≤96GiB；新下载累计≤20GiB、仅必要support/index与至多1个payload shard，新增磁盘≤30GiB；工作窗口90分钟，网络无进展15分钟或连续两次同类失败即报告。下载前查大小，未知大小必须流式计数并在上限前停；不全量snapshot下载。至少保留200GiB可用盘。

凭已授权任务直接执行CPU与有限下载，无需再问用户。若访问需要人工账户条款/权限、最小shard超预算、协议/权重不匹配、依赖冲突或GPU预算未定，记录具体阻塞与最小解决步骤并消息Judge；不伪造输入、不以ABot条件代替相机、不偷偷缩分辨率或重训。

## 交付与验收

README、WORKER_REPORT、INPUT_MANIFEST.json、CPU_PREFLIGHT.json、冻结config与官方diff、默认无GPU的launch入口、GPU_PROPOSAL.md及命令日志。实际缺失项必须清楚标记，不能只把旧缺失清单重新包装为完成。schema/reader/config可用与生成能力分开；本阶段没有生成视频，README明确原因，不占用正式视频结果位置。

资料齐全则Judge快速审核并以v2给有限GPU预算，再向原Worker发送立即执行消息；若关键输入无法获得则停止该阻塞分支，不进行无休止准备。Worker完成后更新根report.md并主动消息Judge，不操作Git。

日间09:00后项目最多3卡、夜间最多8空闲卡为全局上限；本v1授权0GPU，不自动获得全部空闲卡。训练、AnyFlow/DMD更新、SGF+、Memory、批量30–50场景全部不在本轮范围。
