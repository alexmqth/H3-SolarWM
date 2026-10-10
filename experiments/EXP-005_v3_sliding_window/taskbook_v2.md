# EXP-005 / v2：V3 Sliding Window有限GPU验证

2026-10-10 HKT。用户明确批准“开始进行实验”。Judge据已提交GPU_PLAN发布本轮执行授权；取代v1的CPU-only现行限制。v1代码/报告/审核仍为冻结阶段一证据。

## 目标与固定协议

Parent：V3 Original Feasibility Baseline（EXP-002/003），Original H3 + released Action LoRA、Single I0、native video sigma×1000/audio1000、own-action与action feedback、current非action prefix feedback、sigma0 clean commit、12后5分块、30步native FM shift2.22、原噪声和h3_fp32边界精度保持不变。SW-G仅增加真实五祖先淘汰；SW-L独立改变video位置，prefix/time/mask不变。正式Baseline不覆盖。

## 执行阶段与授权

| 阶段 | 内容 | sampling | commit | diagnostic | VAE | GPU小时 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| G0（当前批准） | 2个相同C6 noisy state旧/新入口对照，SW-G重放C6对冻结endpoint/RGB | 30 | 0 | 4 | 1 | 0.30 |
| G1（G0审核后由Judge放行） | 同AA124历史，继续A/切D各自C7/C8 | 120 | 3 | 0 | 4 | 0.70 |
| L1（G1审核后Judge独立放行） | 相同C7历史/动作/噪声的Local两路径及2次C8同历史诊断 | 120 | 2 | 2 | 4 | 0.70 |
| 核心总上限 | 含失败尝试 | 270 | 5 | 6 | 9 | 1.70 |

最多281完整forward，9VAE，0训练；1张实时确认空闲的GPU顺序使用（初选GPU0），项目合计≤3，allocated peak≤44GiB。占卡时间含加载/保存/等待，达到任一阶段上限立即停止；不自动重试或借未批准阶段额度。核心elapsed≤3小时，自首个GPU进程计。C9默认不授权；额外训练、AnyFlow/DMD/anchor替换/新场景seed/参数扫描均不执行。

G0仅使用冻结37latent输入，不依赖新长fixture；允许先完成G0，避免CPU长输入准备阻塞有效GPU回归。G1前必须完成真实47latent输入认证，显式保留旧37noise/prompt/坐标/anchor/audio并冻结新增action及噪声。默认重建长packed禁止。

## 运行与审计要求

- 新runner、配置和输入hash冻结在stage2；逐调用尝试前落盘预算，单进程锁和绝对wall限制，无自动重试。只加载必要cache，不复制多份18GB历史。
- G0比较相同history/noise/sigma的旧与SW-G输出maxabs/relativeRMS，CPU一致不能代替真实模型；fp32参考atol/rtol1e-5，超阈值由Judge判断来源，不静默放宽。C6重放检查endpoint、完整新增17RGB与旧107RGB不可变。
- G1先提交旧C6一次，实际淘汰C1；所有50层C7祖先[1..5]、C8祖先[2..6]。记录缓存字节/indices、采样只读、commit/decode/采样耗时、GPU peak、CPU KV/RSS、完整命令和环境。
- 保存原片、代表帧/全新增帧contact sheet、A/D flow辅助指标；Judge检查人物结构、ghosting、方向/切换与boundary。仅finite/CUDA无报错/KV数正确不作能力PASS。
- 输入错误、cache错误、nonfinite/OOM或预算用尽立即停并报告。普通细节缺陷不阻断可行性；持续严重人体崩坏或动作失效则停止该方向，不追加微小收益消融。
- Worker完成一个阶段及时提交报告/日志；Judge负责阶段验收和后续授权，不再要求用户重复批准已同意的核心任务。

## 交付

保持EXP-005，plan_version=2；新增证据归入stage2与对应artifacts。Worker负责runner/实施与实际报告；Judge负责输入协议审阅、预算授权、视觉/指标审核、progress/next_plan与Git归档。最终说明工程、生成可行性、长时能力各自边界。完整细则沿用GPU_PLAN，冲突以本v2任务书为准。

## Judge状态

G0 approved（须先冻结可运行源码/config/input manifest和预算入口）；G1 pending G0+fixture review；L1 pending G1 review。当前实际GPU调用尚未开始。
