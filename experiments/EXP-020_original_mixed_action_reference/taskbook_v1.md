# EXP-020/v1 — Original H3-World 真实混合动作参考

2026-10-15 HKT，Judge。用户批准基线验证路线，并明确要求将已空闲GPU0也用起来。此任务与EXP-019 GPU2/GPU3场景独立，由同一Exp Worker执行；项目最多同时3卡。

## 研究问题与定位

补充原版已发布H3-World对真实动作切换的能力参考。固定EXP-015 scene b784d995：I0、同源静态caption、17-span真实脚本（前13左平移，第14静止，后3前进）。只做一个样本/seed13，不挑样本，不扩展训练。

从真实首图和完整动作脚本生成全56帧/17latent，832×480、24fps，Original H3 + released Action LoRA step10000，30采样NFE，cfg_scale1、flow_shift2.22、audio_flow_shift3.0。使用冻结原版`MiniMaxH3Pipeline.__call__`及原生native packed/action routing，无causal_control/新layout/新增adapter/AnyFlow。遵循其联合video/audio scheduler和解码路径；记录dtype、mask/backend、音视频noise、timestep。保留原native精度，不自动套V3边界精度变更。

这是官方实现下指定30NFE设置的原版参考，不是论文全部条件的复现。原版可见完整动作/双向视频、从I0生成；EXP-019是GT C1历史下17帧续写，初始noise shape/audio长度/位置协议/历史来源不同，不能作单因素拓扑消融或直接公平E2E速度比较。不加载EXP-019的语义坐标/template，不把整片生成前39帧替换GT。

## 输入与执行

- 原图与17动作句：`submission/experiments/EXP-015_v3_real56_data/artifacts/b784d995/`。
- 原静态caption：EXP-014 source_manifest中`s3_b784d995.scene_static`；与新I0重新编码。
- Backbone/Action LoRA与EXP-018 V0来源一致，冻结代码可复用`H3-World/outputs/2026-10-09-22/chunk_partition_cb/runtime/`的原版pipeline和infer loader。Worker先CPU核对实际import/权重hash、56→17映射、17动作句/3种唯一句、无causal选项、采样/解码计数钩子和输出路径。
- 新独立wrapper负责账本/计时/保存，不能改冻结原版代码。确保外层no_grad、cfg_scale1不产生额外negative forward。读取完整GT视频仅用于后处理比较，不作为模型历史/输入。
- 保存精确命令、source manifest、输入SHA、配置、launch环境与GPU UUID。taskbook和Judge AUTHORIZATION组成明确GPU授权；Worker CPU preflight通过后可启动，不需再次等待用户或新增marker。若CPU发现协议不支持/计数不符，先报Judge，不绕过。

## 预算与资源

GPU0启动前重新检查无人占用。单卡offload沿用ABOT_VRAM_RESERVE_GIB=18；CPU使用可用的不冲突4核。总上限1200 GPU秒（含加载、编码、生成、解码、保存），allocated≤44GiB，0训练。

最多30次联合DiT forward、4次text encoder调用（1head+3唯一动作句）、1次image VAE encode、1次video VAE decode、1次audio VAE decode；0video encode、0KV commit。若实际官方unit额外调用超过预期，预记账停止报告；不暗扩预算。每次调用前计数，外部watchdog执行总wall上限；失败计入，无自动GPU重试。GPU计算与EXP-019分别记账，两任务总上限6600GPU秒，不用三卡并行经过时间冒充总GPU成本。

## 交付与验收

Worker提交原片56帧、GT/Original标注双栏、全17个末段帧及切换附近逐帧图，完整视频decode/24fps/PTS/SHA、sampling/编码/解码/加载/总wall和显存。检查人物/背景、运动、动作切换位置与响应；角色和相机位移分开描述，光流仅代理，不以一段GT轨迹不同就认定失败，也不以结构保持就认定动作正确。

本次检验单固定真实切换，不是A/D反事实或泛化结论。画质PARTIAL如实交付，不扫seed/NFE；nonfinite/OOM/协议错误/预算到限停止。本任务不触发SolarWM GPU、训练或更多Original场景。Judge审核并发布GitHub。
