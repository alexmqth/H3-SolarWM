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

## 2026-10-15 执行分工纠正

用户明确要求由既有Exp Worker执行，Judge仅负责审核与放行。已把运行中的R及源码/预算移交Worker，禁止重复启动。具体见submission/experiments/EXP-019_v3_mixed_action_validation/judge/WORKER_HANDOFF.md。E/G1/G2等待R结果和Judge逐阶段marker。

## R验收 / E放行（2026-10-15）

R四组velocity与50层150组KV/RoPE逐值一致，10forward/379.384秒，Judge已独立核对原始证据。E_APPROVED.json已签发：仅14text/900秒，Worker执行；G1/G2待E报告审核。

## E验收 / G1放行（2026-10-15）

E四fixture来源/编码/未来隔离通过，14text/192.620秒。仅43866101_APPROVED.json已签发给Worker，39forward/2decode/900秒；G2待首图质量审核。

## G1验收 / G2放行（2026-10-15）

首scene视觉局部可行、动作语义PARTIAL，39forward/332.172秒。已签发剩余三scene marker按固定顺序串行，每scene39forward/2decode/900秒，严重失败停止后续；不扩展原任务预算，无新训练。

## 用户批准并行调度（2026-10-15 03:06 HKT，优先于上述串行条款）

已运行7199292c保持GPU0；剩余9dc2e588→GPU2、b784d995→GPU3，Worker重查空闲后可并行启动。完整实验数量/166forward/5400 GPU秒上限不变；GPU时间按各卡wall相加，项目同时最多3卡。详见submission/experiments/EXP-019_v3_mixed_action_validation/judge/G2_PARALLEL_ADDENDUM.md。

## 并行任务EXP-020（2026-10-15，用户要求GPU0也使用）

EXP-019 GPU2/3保持；另将GPU0分配给EXP-020/v1：b784d995原版H3-World真实动作切换参考，30NFE全56帧，4text/1image encode/1video decode/1audio decode，1200GPU秒、0训练。Worker CPU preflight并冻结来源后可按AUTHORIZATION启动。两个任务合计≤6600GPU秒，最多同时3卡。任务书：submission/experiments/EXP-020_original_mixed_action_reference/taskbook_v1.md。
