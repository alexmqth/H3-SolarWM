# EXP-019/v1 — V3 固定动作布局真实模型验证与混合动作 GT-history 续写

2026-10-15 HKT，Judge。用户已批准执行基线验证→普通FM因果适配→失败诊断路线；本轮实现第一阶段，不重复已验收VAE编码，不启动AnyFlow/DMD/SGF+或Memory。

## 问题、来源与控制变量

EXP-017固定语义坐标候选是否在真实33B上保持canonical A协议输出，并能支持真实变长混合动作？复用EXP-015四个固定train episodes的56帧/17个真实动作span，以及EXP-016独立验收的I0/prefix12/full17编码。EXP-014同场景canonical full37模板只提供先验固定位置；真实I0/head必须重新编码。固定Original H3 + released Action LoRA、h3_fp32、Single I0及.999 noise augmentation、native time、Global RoPE、current-prefix/own-action反馈、clean commit、CPU raw KV max_history5、seed13、832×480/24FPS、shift2.22、12+5分块。仅采用EXP-017候选动作布局，不改冻结runtime。

## 分阶段执行与预算

- P0：CPU来源SHA、实物/路径/布局/导入核查，签发后进入GPU。
- R：s0 canonical A，C1/C2×sigma1/.5×旧/新两入口=8forward；旧/新各自从冻结generated C1 clean commit=2forward（旧KV已删除，真实重建），共10forward，0decode，≤900GPU秒。保存输出差异和50层KV差异，不保存大KV。精度/backend匹配；应逐值相等，若不等先报告定位，不把容差放宽掩盖变化。
- E：四场景新I0/原caption head与独立动作句编码，最多14 text encoder调用，0VAE/DiT，≤900GPU秒。复用EXP-016 I0 latent并按冻结原生.999/seed13公式生成anchor。真实embedding裁剪后检查未来内容/长度隔离、历史语义坐标稳定及token-refiner动作span。
- G1：首scene 43866101，在真实C1 GT-history clean commit后，用相同C2初噪、真实动作分别FM-30NFE/FM-8NFE，39forward/2decode，≤900GPU秒。Judge视觉评估后决定后续。
- G2：余三scene 7199292c/9dc2e588/b784d995，每scene39forward/2decode/≤900GPU秒，依次执行，最多总156forward/8decode。先验scene顺序固定，不换seed筛选结果。

总上限166forward、14text encode、8decode、0训练、5400GPU秒=1.5GPU小时；失败调用计入，无自动重试。单空闲L40串行，allocated≤44GiB，新增磁盘≤10GiB，剩余≥40GiB。每阶段启动前检查实际空闲卡；不操作其他进程。09:00后最多3卡，此任务始终1卡。每次模型调用预记账，保存sampling/commit/decode/模型加载/墙钟/GPU与CPU KV峰。显式stage marker绑定source manifest；本任务书基于用户批准授予Judge逐阶段放行权限，不需再次向用户请求同范围许可。

## 验收和停止

R差异、未来隔离失效、NaN/OOM/预算耗尽立即停止相应阶段，保留证据后Judge决定一次明确修复或归档，不自动重复。G1若持续结构崩坏/动作明显失败，先停止扩场景；细节PARTIAL不机械阻断。四段真实数据没有完整D反转GT，本任务不声称验证反转或泛化。

视频需明确前39帧为真实历史展示，只有后17帧新生成；与原视频及VAE重建分别标注，不把借用首窗称完整生成。展示时间56/24秒，生成成本对应17新帧。所有新增帧审阅人物、背景、动作可辨性、ghosting、边界和运动；光流仅辅助。GT-history局部续写通过不等于free-running通过。冻结V3及旧结果不覆盖。

## 后续路线

本轮结论决定下一编号任务：固定6–8场景且按episode隔离的评测集，Original/冻结V3动作切换比较；普通FM因果适配的小数据拟合与实测吞吐预算。公开SolarWM Stage2-fix少量官方相机样例独立准备，接口与训练不混用。训练更新档位按实测成本和验证趋势决定，不预先承诺10,000更新或四周完成。
