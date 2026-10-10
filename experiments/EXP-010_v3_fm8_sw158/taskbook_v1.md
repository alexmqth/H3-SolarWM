# EXP-010 / v1 — 全程FM8与SW-G的158帧有限联合验证

Judge正式CPU任务，2026-10-11 05:00 HKT。**当前仅授权CPU实现与准备；各GPU阶段须独立Judge marker。** EXP-009已完成审核：FM4有限可用、AF4 AD C3失败，当前AF4配置收口。 不修改正式V3 Baseline，不继续FM4/AF4步数扫描，不挽救当前DMD配置。

## 核心问题

Mainline；Parent为EXP-006从首窗开始的普通FM8，以及EXP-005已验证的SW-G位置/真实淘汰协议。研究普通FM8能否从已有73帧继续到124帧，并在发生真实最近5祖先淘汰后达到158帧，保持基本人物/场景与晚期A/D切换。它弥补当前“FM8只到73、SW-G158只测30步”的证据缺口。

假设为无新训练的8步推理与已验收Global滑窗协议可组合；不预设质量成功。30-step SW-G现有158帧A继续/D切换为参考，不重跑。各自生成历史不同，属于完整方案对比，不声称同raw KV状态单变量消融或公平E2E speedup。

## 固定协议与输入

Original H3 + released Action LoRA、Single I0、native video timestep/audio1000、own-action/action feedback、current non-action prefix feedback、fixed_prefix_timesteps=False、Global RoPE、strict causal、detached persistent CPU raw KV、max_history5、h3_fp32边界与既有attention backend不变。所有新增块8-step native scheduler、shift2.22，不安装AnyFlow或DMD模块。

从EXP-006 AA73接续：first12、AA C2与C3端点、AA cache_through17（由相同冻结权重建立）、published_73 RGB。不重新生成首窗，不借30步latent或hidden KV。新提交C3 clean KV之后继续12+5+5+5+5+5+5+5，共47latent/158RGB。

C1–C6全部A，达到124帧后共享这段全FM8历史。C7开始分叉A继续/D切换，分别至C8 158；这与EXP-005 G1晚切换条件一致，但不宣称FM8早期AD分支已经长时测试。

使用EXP-005冻结long47 fixture，保持0:37的所有条件与seed13初噪声，37:47使用已有extension_seed130005的相同保存噪声；禁止为“native长输入”重建并改变旧prefix/I0/action/video坐标。旧冻结fixture已明确不是默认47latent packed builder，不隐去这一限制。

## 阶段一：CPU准备

复用已验收interval_stage2及Global分支，显式ChunkPlan与精确indices；冻结代码/config/输入/原模型来源。核对EXP-006 AA73的latents、RGB、cache来源和hash，old/full fixture裁剪到12/17/22/27/32/37时layout/prompt逐值一致；噪声0:37不变，长尾与G1相同。禁止变更冻结入口或正式结果。

支持每块检查：history/cache sampling前后不变、clean commit之后精确祖先indices、旧RGB不变、sigma schedule、已保存端点与源文件hash。预算在forward/decode前扣减，失败记账无自动重试；实际空闲GPU、互斥锁、绝对时间与09:00截止。GPU只有Judge CPU审查与marker后执行。

## 阶段二：C4–C6到124帧（GPU另放行）

加载自有C1/C2 raw KV并sigma0提交已有C3一次。C4、C5、C6各8sampling+1clean commit+1decode，总28forward（24sampling+4commit）、3VAE。C6 commit实际淘汰C1，验证留下C2–C6，不在这一步把历史显示帧重写。

先交124帧原片、C4–C6全部新增帧与每块metrics，Judge看是否仍有人物/场景与动作，再放行后两块。普通形变/残影PARTIAL；持续全噪声、主体/场景消失立即停止后续。

## 阶段三：C7/C8与真实滑窗（另marker）

以同一FM8 AA124/cache分叉A继续和D切换。C7各8sampling+1clean commit+1VAE，C8各8sampling+1VAE，总34forward（32sampling+2commit）、4VAE。

C7读取indices1–5（C2–C6），commit后只留2–6；C8读取2–6（C3–C7）。所有50层检查，不仅数量。历史video KV容量14,164,800,000 bytes有界；prefix/latent/RGB/VAE全前缀成本仍可能增长，不能据此宣称整体内存恒定。

## 总预算、评价与停止

新增上限62forward=56sampling+6clean commit、7VAE、0backward/0update；累计≤.75GPUh，单卡allocated≤44GiB，绝对截止2026-10-11 09:00 HKT。预算含加载、cache保存、等待和失败；若改为多卡必须合并GPU小时。选择实际空闲卡，不占GPU3/4他人进程，09:00后项目≤3卡。

检查全部新增119帧（共享C4–6共51，两个C7/C8各34）、旧73RGB逐值不变、C7两路共享相同124RGB、A/D方向与切换、人物/场景、ghosting与边界；结合完整视频，不仅finite/光流判断。记录每块sampling/commit/decode/总耗时、GPU峰值、CPU KV bytes/indices、实际NFE与forward/VAE。与既有30步Global A/D158提供并排视频，清楚标注不同generated history与计时范围。

OOM/nonfinite、输入/权重/cache/时间协议不一致或预算触顶立即停，保留负结果，不重试/改种子/调参。阶段已出现持续结构崩溃即停止，无C9、无更多步数/seed/scene扫描、无训练。目标是决定全FM8滑窗是否具有限可行性，普通缺陷不阻止有边界的PARTIAL结论。

## 交付

EXP-010中保存冻结任务/config/source与input manifest、CPU结果、分阶段GPU marker、实际命令/原始日志/账本、指标和视频、Worker报告与Judge审核。大型tensor保留外部路径与SHA，不重复基础权重。结果通过后更新V3-FM8报告及主线导航，与正式30步V3及SW-G证据并存。论文级泛化/长期稳定/无限长度不在此有限测试结论内。
