# C12→5：自生成历史后的第二块得到局部正控；B7首窗方向未过

2026-10-09。**C在same-sigma历史协议N下通过本轮“第二块、单停车场/seed”的局部动作＋人物结构检查；clean历史在D→A失败。B首7latent画面基本完整，但A方向未过。** 这支持继续研究C候选，不等于完整124帧、persistent-KV、few-step或整体Stage1已经通过。

## 直接播放

- **[C：四条56帧完整串接，A→A / A→D / D→A / D→D](C_selfhistory_N_56.mp4)**。均为N历史；上排先A、下排先D，左续A、右续D。RGB39（从0计数）切入第二chunk。56帧/24fps=2.33秒，没有慢放、循环填充、平滑或GT重置。
- [C：两份历史的clean/N局部总览](C_second_chunk_summary.mp4)。25帧=8已显示历史＋17当前帧；左右大区是两份历史，每区左clean/右N、上当前A/下当前D。
- [A历史：clean与N的完整56帧对比](C_A/comparison_rollout.mp4)；[D历史：同样对比](C_D/comparison_rollout.mp4)。每片左clean/右N，上当前A/下当前D。
- **[B7与C12首窗，统一显示前22帧](B7_vs_C12_first22.mp4)**。左B首7，右旧C首12；右侧生成时看到了39RGB对应的当前窗口，所以不是相同上下文或相同交互延迟的比较。
- [B首22帧A/D](B_first/comparison_rollout.mp4)。人物完整但A未达到左向要求，作为保留的负结果。

全部32个MP4已完整解码，H264/yuv420p/24fps。[逐视频SHA与帧数、完整数值](summary.json) · [指标CSV](metrics.csv)。

## 这次到底改了什么

用户指定优先C=`[12,5,5,5,5,5]`，其次B=`[7,5,5,5,5,5,5]`。本轮只执行C的`[12:17)`续写和B的`[0:7)`首窗；C后续四块与B后续六块未执行。

Original H3＋released action LoRA，无新增训练adapter、AnyFlow、DMD；单原始I0、native text/action时间、h3_fp32、30steps/shift2.22/seed13、相同full37固定布局及video/audio noise。当前chunk内部使用Original directed action路由，未来action/video在refiner前物理删除；历史只读，T2每sigma重算可见hidden，**CPU persistent hiddenKV=0**。

新增显式`start/stop`区间诊断接口，未修改生产cached默认值。C首12来自此前`coarse_A/window0_A.pt`和`window0_D.pt`，是同种零训练局部生成器自己的endpoint，**不是Original全视频伪标签截取，也不是GT history**。各分支固定这份自身生成历史，只替换当前5latent的A/D。首窗没有历史，clean/N语义相同；旧首12实际solver状态在三个新进程中重放velocity误差均为0。

为明确历史处理影响，预登记两种条件：

- **clean**：已生成history以干净latent和干净video时间输入。
- **N / sigma_noised**：临时构造`(1−sigma)*history + sigma*fixed_history_noise`，对应video使用当前sigma。只积分当前chunk，不修改保存的历史、首段动作或已经显示的RGB。这沿用已有协议，并非本轮新增训练模块。

两个历史各4条续写分支共240次采样forward，B两分支60次，共**300采样＋8诊断forward、0 optimizer**。最多3张L40（GPU2/5/7），进程均正常结束。每条C完整两块路径概念上为30首块＋30续块=60采样forward；本轮首块复用旧结果，不能把整条56帧标成30次新生成。

## 动作与视觉结果

下表flow沿用固定416×240、中心区域Farneback平均水平光流。A应正、D应负仅是本停车场的方向约定；不等于3D角色控制准确率。

| 测试 | 当前A flow | 当前D flow | A−D | 结论 |
|---|---:|---:|---:|---|
| C，自生成A历史，clean | +1.506964 | −1.026712 | 2.533676 | 本历史局部方向与人物结构成立 |
| C，自生成D历史，clean | −0.176696 | −1.312233 | 1.135537 | A失败，不能仅以separation>1算通过 |
| **C，自生成A历史，N** | **+2.146440** | **−1.751855** | **3.898295** | **本轮第二块局部门槛通过** |
| **C，自生成D历史，N** | **+2.607594** | **−1.414301** | **4.021894** | **本轮第二块局部门槛通过** |
| B首7，无history | −0.152628 | −1.178848 | 1.026220 | A更像向前行走，方向门槛未过 |

检查了C两份首段共78RGB、八条续写的全部136RGB、B的全部44RGB静态帧；N分支另看原分辨率人物细节RGB39/43/47/51/55及RGB36–41边界。N的A/D具有不同转向/运动响应，人物保持单一躯干与可辨肢体，停车场结构保留，未见此前严重多躯干、半透明分解或瞬间scene switch。存在普通运动模糊和细节形变，不能说无瑕疵或达到Original质量。

D历史下N/currentA会明显转身，与currentD继续右向形成区别；这是局部定性动作响应，不宣称精准世界坐标strafe保真。首段到续段未见突兀场景重置，N的边界MAD有所增加。

视觉评审记录：[A-history](C_A/visual_review.json)、[D-history](C_D/visual_review.json)、[B首窗](B_first/visual_review.json)。这是完整静态帧和细节评审，不冒称实时播放人工评审。只有单场景/seed和一个后续17RGB（0.708秒），不代表长期泛化。

## 耗时、显存与连续性

| 分支 | 新chunk采样 s | 峰值allocated GiB | Frame MAD | 边界MAD |
|---|---:|---:|---:|---:|
| C A历史，N→A | 195.48 | 25.424 | 5.900 | 4.384 |
| C A历史，N→D | 195.29 | 25.424 | 5.296 | 6.487 |
| C D历史，N→A | 194.77 | 25.424 | 6.619 | 4.851 |
| C D历史，N→D | 194.60 | 25.424 | 3.969 | 3.854 |
| B首窗A | 117.02 | 24.763 | 2.876 | 不适用 |
| B首窗D | 117.91 | 24.763 | 3.956 | 不适用 |

采样列为30次solver循环，扣除额外同状态D诊断时间；不含模型加载、VAE和视频封装。A/D历史两个完整作业分别858.2/856.6秒，B作业277.7秒，均含其多分支及评测。C旧首窗A/D采样153.89/152.64秒来自另一次共享机器运行，不把不同日期相加冒充本次实测端到端延迟。首39/22帧覆盖的1.625/0.917秒是**生成内容的时间跨度**，不是交互响应耗时。

所有分支CPU hiddenKV为0；C原始history latent为1.714MiB，CPU权重offload另计，不能把0 hiddenKV理解为0 CPU内存。torch峰值包含权重/激活等，未分拆归因。无warmup多次均值，不提供speedup结论。

Frame MAD是0–255灰度相邻帧差；边界MAD是同样灰度定义下“冻结首段RGB38→新RGB39”，不是质量评分。没有与更早不同帧数/历史条件的MAD或flow直接排名。

## VAE显示协议：已实测而非只靠公式

仅加载真实VAE，耗时70.64秒、峰值6.465GiB，对Original A/D完整latent做5/7/10/12/17prefix以及future12+干预。7/12/17prefix的前17/34/51RGB与完整解码逐元素相同，末5RGB可回改；future12+改变RGB34–38但不改变RGB0–33。[全部实测](vae_audit.json)。

C显示规则是保留首12prefix已经输出的39RGB，再从17prefix只追加RGB39–55，不回改先前画面。实际续写时，对旧末5RGB重新解码所产生的差异平均为0.61–0.68/255（量化RGB），**记录但没有应用到已显示帧**。没有额外后处理smooth/soft overlap；VAE自身内部overlap属于原decoder。

这次正结果不能唯一归因于“VAE相位修好了”：首窗口长度、历史来源与较早失败实验不同；本轮同一partition的clean/N对照也证明历史协议有影响。

## 保存的候选与下一步边界

候选名：`C12_then5_N_30step_selfhistory`。它属于**Original H3＋causal routing（零新增训练）**，是推理配置而非新训练checkpoint。关键里程碑另保存了视频、协议及首段/续段latent状态；不复制33B权重。

若继续，先固定C＋N，用各自第二块生成结果接第三/第四块，检查动作切换与人物结构能否持续，再讨论124帧和KV迁移。B不自动延长，A不启动；本轮没有新增训练、AnyFlow或Stage2。只有第二块通过，不撤销以前完整长rollout的失败结论，也不替换会议124帧主片。

## 可复现性与外部依赖

`protocol.json`在新采样前固定输入hash、budget和验收条件；`launch.json`固定执行源码和前置审计；`cpu_tests.xml`的4个FP32/BF16×历史条件检查覆盖5种区间，对旧路径逐元素一致，并检查未来隔离/当前干预/历史只读。运行过程中源文件与冻结runtime不变，参数version和history/noise hash均保持。

本档案是原始输出脚本的源码快照。直接执行还需原项目中的`source_coarse`、`source_history`、`runtime`及53MiB×2条件输入；它们不是本Git目录自动包含的全量模型包。对应源路径与SHA在`protocol.json`、各`evaluation.json`和外部依赖清单中。原始运行目录为`H3-World/outputs/2026-10-09-22/chunk_partition_cb/`；不要在已存在case目录上覆盖重跑。

CPU整理入口为`report.py`、`review_assets.py`、`render_c_positive.py`；原推理入口`run.py`会核对前置VAE/CPU和源码hash再加载模型。重复新采样需建立新的时间目录和预登记，不把本轮冻结收据修改成下一轮。
