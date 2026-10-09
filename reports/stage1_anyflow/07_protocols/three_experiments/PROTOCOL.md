# Next Plan v4：先恢复 chunk-local action information flow

2026-10-09 16:16：E2两臂各4更新、六组局部视频评测及48次held-out诊断全部完成。停车场两份历史的A/D符号均保留，但A分支重影仍在，FM+action没有一致优于FM-only；局部动作＋结构联合gate仍为No-Go。本轮不自动扩训，不进入AnyFlow/Stage2。 [最终E2结果](../../01_real_video/real_transition_windows/FINAL_RESULTS.md)。下方初始协议及历次执行记录保留用于溯源，当前状态以此处为准。

2026-10-09 06:30：E1历史条件对照已全部结束。窗口1两history均恢复A正/D负，但D历史的A分支RGB59–68仍有明显躯干/多重手臂重影；画面门槛未过，按协议停止，不跑窗口2。56诊断＋120采样前向，零optimizer。下一步准备真实动作后果监督的数据/梯度审计，不进入AnyFlow/Stage2。 [E2真实后果准备](../../02_causal_diagnostics/history_conditioning/E2_REAL_TRANSITION_PREPARATION.md)为当前数据来源补充，取用户允许的真实动作后果路线；不把它称成真实反事实视频配对。以下初始paired-teacher来源设计保留，不再强行使用失真分支。

2026-10-09 05:59：E1粗窗口两历史/三窗口与VAE审计已全部结束。首窗A/D恢复，但A历史第二窗同正、D历史第三窗同负；第二窗有明显肢体重影，局部门槛仍No-Go。未来RGB干预8项past latent差均0，不支持VAE泄漏根因。停止窗口扩宽，下一项只审查历史噪声/时间条件，尚未启动新GPU/训练。 [完整终态](../../02_causal_diagnostics/coarse_window12/FINAL_RESULTS.md) · [后续协议](../../02_causal_diagnostics/coarse_window12/NEXT_HISTORY_CONTROL.md)。以下初始协议保留，原dual/fixed-time已由记录明确的校准对照覆盖；过去运行状态不代表当前状态。

2026-10-09 05:32执行更新：新的full37匹配5latent对照已FAIL，12latent首窗方向恢复；相同前17RGB区间也有区别。GPU1/5两历史的后续窗口继续，G1未通过，E2/E3不前移。[证据](../../02_causal_diagnostics/coarse_window12/FIRST_WINDOW_RESULTS.md)。原360次coarse前向预算保持；为控制fixture差异追加的一次matched首5latent60前向已完成，不追加其他候选。

2026-10-09。按用户最新要求，以三个有决策意义的实验替代继续扩展旧 A–D 工作流。本文件是当前研究协议，旧证据保留，旧执行指令不再自动生效。**初始状态（02:57）：设计完成，当时E1尚未实现。当前执行状态以上方最新记录为准；该段仅描述初始状态。**

核心顺序：可信的 causal H3 → AnyFlow 少步 → on-policy Stage2。可信的含义是：同状态当前动作干预有效，GT history + 充分采样下局部人物/场景基本正常；不要求提前解决 20 秒自由生成漂移，也不要求 4 步达到 Original。

## 为什么收敛

- 同权重受控 causalization 的整体 velocity cosine 约 0.996，但 A/D 差分 cosine 约 0.060；共同外观场相似不等于动作信息流保留。
- 旧 `own action + current-video prefix feedback` 候选恢复了首块 Original identity，但后续有历史的 12 点 delta cosine 从 0.028777 降到 0.008548。它没有修复后续块，不重新包装成新方案。
- 真实 ABot FM48、随后单变量 sampling density 对照均未通过动作/画面验收。新密度停车场 30 步 A/D flow 为 −0.190239/−0.158556，8 步为 −0.042918/−0.116288；完整视频结果不足以支持继续加预算。
- 旧单步 paired-velocity 诊断仅完成 tiny-H3 CPU 梯度等价检查，没有 GPU optimizer update。按照新路线搁置；这个 loss 不自动成为 E2 的主目标。

资料：[机制解释](../overviews/ACTION_MECHANISM_SUMMARY.md)、[旧 prefix 候选](../../02_causal_diagnostics/current_prefix_candidate/INTERPRETATION.md)、[密度实验收尾](../../01_real_video/fm_density_control/FINAL_RESULTS.md)。

## 统一实验条件与证据类型

1. E1 从 Original H3 + 已发布 H3-World action LoRA 出发；不加载后训 FM/AnyFlow/visual/action-residual 权重，不做 optimizer update。保持同精度、同 attention backend、同 prompt、首帧、video/audio noise、动作行与全局位置；对比记录张量哈希。
2. 保持现有 39 RGB / 12 latent / 5+5+2 三块，history cap=5，flow shift=2.22，主视频 30 steps/chunk。末块只有 2 latent，单列且按帧数归一化，不把三块当等时长。4/8 步不进入 E1。
3. 继续固定 `h3_fp32` 与 RGB dual 协议，首图 + 上一块末帧条件；各方法在同一个局部测试中使用相同 anchor 张量。它不是 latent dual 变体；本轮不顺便换 anchor、窗口大小、LoRA 容量或精度。若需改变条件修复正控，先修订协议，重建所有配对参考，不与旧数值混算。
4. 停车场是强动作正控，但现有停车场历史来自 Original **生成**，不是数据 GT。它用于同一历史状态下 fork A/D。真实 GT-history 局部画面另外使用既有两个 held-out ABot 场景；自然场景含联合按键/相机，不把其整体光流当纯 A/D 标签。
5. 每个 chunk 的 A/D 分支使用同一历史、同一当前 noisy latent，只替换该 chunk 的动作。过去 action 不改。先固定 common history 再 fork，禁止用已经分叉的整段 A、D 轨迹相减作为同状态标签。
6. 局部测试分块恢复同一 reference history 属于 oracle 条件测试；后续自由生成不重置、不拼接成“自由 rollout”。停车场历史可取两条预先冻结的参考轨迹，但每对 A/D 必须共用其中同一份状态。
7. 单一首图/seed13只支持可行性结论。先用 seed13 开发；入选方案在预留 seed29 上复验，失败完整记录。第二场景必须先验证参考方法确有可辨动作响应，弱正控保留为 inconclusive。

## E1：局部双向交互能否在不读取未来的条件下保留动作

### 两个候选及必要对照

| 标识 | 当前窗口的信息流 | 历史与 cache 语义 | 用途 |
|---|---|---|---|
| R-full | Original 39 帧双向生成 | 可以读取整个生成区间 | 展示参考；不是局部无未来 teacher，也不是同状态监督 |
| R-prefix / T2 | Original directed action mask，输入只有已知历史 + 当前块 | 每个 sigma 全部重算可变表示，历史 latent 固定 | 合法条件下的局部正控；也是下述 T2 完整历史窗口版本 |
| C0 | 现行 causal rows + feedback + persistent raw KV | 历史 KV 冻结 | 同权重失败基线 |
| C1 | 旧 own-action + common prefix 只读当前 video | 历史 KV 冻结 | 复用已有候选定义；明确新 T1 相对它改变了什么 |
| T1 | 当前块保留 Original 必要双向路由，prefix 可读取允许的历史 video KV | 过去已提交 video KV 永不被当前 action 改写 | 严格 chunk-level cache 候选 |

R-prefix 与 T2 在本轮相同输入窗口下是同一计算，不重复运行两套、也不把 T2 与自身 cosine=1 当改进证据。T2 必须靠实际局部动作/画面和成本验收。旧全长 Original 的 A−D≈2.023 不能直接充当 R-prefix 的局部阈值。

**T1：冻结过去 KV，恢复允许的局部 H3 路由。**

- 当前 video queries 直接读取自己对应的 action sentence，保持 Original own-frame 绑定；不额外开放全部 past-action 的直接入口。当前块 video 内双向，video 可读取过去已提交的 video KV。
- 非 action prefix queries 可读取可见的历史与当前 video；action sentence queries 读取其自己对应帧的 video（该帧若在历史中，读取对应历史 KV），同时保留 Original 的 prefix/action 隔离规则。
- 本次 forward 的 prefix 每层重新计算，不持久化 action-dependent prefix KV。当前块输出经这些 prefix 获得上下文；未来 action rows 从可达输入中移除或证明完全不可达。
- 历史 video KV 在其所属块完成时，用当时的已知 action/history、clean latent、规定的 anchor 另做 sigma=0 commit。commit 时仅记录当前块 video raw KV。未来块不覆盖它们；A/D fork 只读同一份过去缓存。
- 与 C1 相比，新因素是允许 prefix 在历史 KV 上重新获得 grounding，包含 common-prefix 和 own historical-action 的必要边；属于一个声明完整的路由组合，不能声称已经识别唯一致因边。参数、solver 和条件保持不变。

**T2：局部双向窗口，每个去噪步重算。**

- 保留历史 clean latent 与过去已知 action，将 `[history, current noisy chunk]` 按 Original directed mask 一起计算；只有当前块 latent 由 solver 更新。历史 latent/time=clean，当前块 time=当前 sigma。
- 当前 action 可以通过当前 video、common prefix 间接改变历史 token 的本次 hidden state。这不修改已输出的历史画面，也不读取未来块；它符合窗口之间的生成因果性，但不符合严格逐 token 冻结 KV 的自回归假设。
- 只要某个 KV 依赖当前 noisy latent/action，它在下一 sigma、A/D fork 或条件变化时就失效。最小实现每个 sigma 重算完整可见窗口，禁止只在块边界刷新；禁止复用 T1 历史 KV 冒充 T2。
- clean commit 持久保存本块 clean latent、动作与条件来源；如计算 clean KV，只能作相同依赖快照的可丢弃工作缓存，不声称可供下一块永久复用。分别记录 latent 存储、临时 KV、额外重算及 commit 开销。
- 39 帧最多只有 2 个历史块，先使用与 T1 相同全部可见历史，暂不增加窗口长度扫描。延长后才按相同 cap=5 比较丢弃策略；不从 39 帧推断长时成本。

### 无未来信息泄漏与 cache 检查先行

必须检查**多层信息可达性和完整执行路径**，不能只看单层 mask。

1. 对已知动作前缀相同、未来动作任意置换/删除/改变长度的输入，当前及已输出结果不变。检查 text/action refiner、公共 prompt、audio、RGB anchor、位置编码、padding、长度归一化与所有 cache；不能提前编码整段未来 action 后只遮一条 attention 边。
2. 扰动未来 video latent 时，当前结果不变；显式检查未来 token 根本未输入还是已被隔离。共用静态 prompt 不得包含未来动作安排。若采用随机 kernel，先独立重复测误差底噪，再设容差；不能随结果放宽。
3. T1 持久 KV 与按原始过去条件逐块重建的 replay 对齐；A/D 中过去 KV 的值/版本/hash 不变，扰动过去 value 能改变当前输出，证明实际使用历史。
4. T2 比较无缓存完整重算与任何拟复用策略；缓存身份必须含 action、sigma、state、anchor、全局位置与依赖窗口。改变当前 action/sigma 不得命中旧可变缓存，已提交历史 latent 始终不变。T2 的输出不要求与 T1 相同。
5. 检查 VAE encoder/decoder 的时间感受野：GT-history latent、上一块 RGB anchor 不能由看过当前/未来真实帧的全片编码泄漏答案。优先 prefix-only 编码并记录映射；如现有全片编码未证实因果，仅作 oracle latent 诊断。streaming 解码也不能用未来块回改已展示帧，flush/delayed frames 计入首块延迟。
6. chunk 内允许双向，意味着开始当前块前必须拿到当前块覆盖的 action segment。可以使用当前按键持续保持的约定，不要求未来块动作；报告控制响应粒度，不能将它描述成每 RGB 帧即时响应。

先 CPU 小模型/依赖图验证，再做真实 33B 首块 identity、两后续块、future negative control。CPU PASS 不代替 33B 检查；首块 Original identity 也不代替后续块收益。

### 固定预算的测量顺序

1. **正控校准**：R-prefix 在停车场同一已知历史下 fork A/D，30 步生成当前块。逐块确认运动方向与人物结构。如果合法的 R-prefix 本身没有响应，不能把全长 R-full 当有效局部 teacher；结论是当前局部条件/窗口还不足，先检查 reference 与 conditioning，不启动训练。
2. **同状态筛查**：seed13 的 2 份固定历史 × 3 chunks × 3 个预注册 sigma。sigma 取实际 30-step 网格的索引 0/15/27，生成输入前保存完整数值与哈希。比较 R-prefix、C0、C1、T1；T2 复用 R-prefix。当前 noisy latent 与 history 对每种拓扑相同；后两块分开报告，首块 identity 单列。
3. 保留 `cos(delta_v, delta_v_ref)`、delta norm ratio、whole-field 距离与逐帧结果，重复测量校准数值误差。近零 reference delta 标为不可判，不用 epsilon 放大成方向证据。端点插值探针与实际 solver states 分开标记；至少在入选路线复核真实 30-step solver 轨迹。
4. **局部视频验收**：通过机械检查的 T1/T2 与 C0 各做停车场 A/D 三个局部 fork，完整保留所有当前块输出；两条真实 ABot GT-history 39f 拼接诊断沿原动作评估画面。明确块边界、哪些输入来自 reference，检查全部帧及末段，不用接缝重置掩盖自由生成问题。
5. 入选路线只补 seed29 和 39f A→D→A / D→A→D 的时间绑定；此时只看动作切换与局部能力，不要求已经解决长期 generated-history。最多两个候选，不自动增加第三种 mask、LoRA/anchor/solver sweep。

### E1 决策

局部动作 PASS 必须同时满足：无未来泄漏；首块和两个后续块的同状态响应均有依据；在 R-prefix 有可信符号的停车场局部区间，A 正、D 负且肉眼运动方向匹配；局部人物/停车场结构无严重分解。保留原 Farneback 测法，逐块与全片分别报告，排除 GT-reset 拼接处光流；低纹理/符号不稳定区间报告不可判。

对于 T1，相对 C0/C1 的后续块 delta cosine 改善必须超过重复/后端数值不确定性，不能只凭首块或三块平均值通过。预注册的筛查规则是：后两块各自 6 个配对点中至少 4 点改善，并报告各块 median 与完整值；这是筛查规则，不是视频质量的替代指标。T2 的 delta identity 不参与筛查分数，直接用局部结果证明其能力。

- **T1 通过**：保留 persistent KV 路线；如果 T2 更好，报告质量/成本取舍，不只按 cache 是否存在选型。
- **仅 T2 通过**：允许选择窗口间 causal 路线作为主原型，完整披露重新计算成本；不把它标成严格 persistent-KV 等价方案。
- **机械检查通过、参考有响应、候选仍弱**：进入 E2 的固定预算动作监督试验；仅说明结构合法，不能说 action 能力已成立。
- **合法参考也无响应，或有泄漏/条件不匹配**：暂停模型效果训练，先定位局部 conditioning/窗口/动作表征；不靠 AnyFlow 或 DMD 掩盖无效正控。

## E2：监督同一个状态下动作造成的运动后果

触发条件为 E1 拓扑的可达性/条件协议可信，且能构造可靠的局部 action consequence reference。E1 已充分通过则不强制先加 E2 模块，可直接进入 E3；E2 仅针对尚未恢复的动作能力。

**配对来源。** 冻结同一初始状态/历史、首帧、prompt、noise，从只含已知历史和当前窗口的合法 Original/R-prefix 生成 A、D 两个完整局部 transition。pair identity 明确记录共同输入，除当前动作外完全相同。逐对检验参考的方向与外观；无响应、严重失真样本保留 rejection 原因，不把它们当可靠标签。划分 train/validation 的起始状态，seed29 只验证；同历史的两个分支必须在同一划分。真实 ABot 的不同轨迹不具备 counterfactual pairing，不能直接相减。

**主目标先用局部 transition 的 paired FM 与交换动作负例。** 令合法 teacher 产生当前块后果 `x0^A, x0^D`，两者共有起始历史。分别构造 `z_t^a=(1−t)x0^a+t*eps`、`u_t^a=eps−x0^a`，定义：

```
e(a,b) = mean((v_theta(z_t^a, H, action=b, t) − u_t^a)^2)
L_FM  = 0.5 * (e(A,A) + e(D,D))
L_action = 0.5 * sum_a max(0, m + e(a,a) − e(a,opposite(a)))
L = L_FM + lambda * L_action
```

这是 action-swapped transition ranking：对**同一个 noisy transition**，正确动作应比错误动作更能解释已验证的后果。它没有把不同历史下的 teacher velocity 相减，也不要求 causal student 拟合一个看过未来块的 bidirectional teacher。两条 endpoint 轨迹彼此不同，不能把它包装成前述同 noisy state 的 delta 测试；验收仍使用独立的同状态干预和真实生成。

该目标仍以 FM 参数化训练，并不保证语义控制。存在仅恶化错误动作 loss 来满足 margin 的漏洞：同时报告四个 `e(a,b)`、正确动作绝对误差、真实生成方向和画质；margin 降低但正确动作/视频不改善，判 FAIL。不开 adversarial discriminator；如需视觉 action probe，必须先证明其 held-out 方向识别可靠，再单独修订协议，不能悄悄并入本轮。

**有限对照而非新增大训练。** 先建立至多 8 个 train + 4 个 held-out 起始状态的配对小集；实际可靠配对不足如实报告。固定 E1 选定拓扑与 Original 初始化，从旧数据中继承的任何权重都要另列。两组使用完全相同、预先声明的 action-pathway trainable bank、LR、batch、sigma 序列和 logical updates：FM-only 与 FM+action。最小先做 4 更新趋势检查，上限 16 更新，保存 0/4/8/16；若双方视频/正确动作拟合均退化，不继续。参数组、margin、lambda 在数据校准后、optimizer 前写入单独 run protocol；本设计不伪称这些未校准数值已确定。

margin 用训练集可靠 transition 的误差尺度校准，lambda 用训练集初始梯度量级选一个固定值，不扫验证集；以后不根据 seed29 调参。FM-only 可以复用相同四次 forward 测量以匹配计算统计，但只有正确分支进入 FM 反传；报告实际训练 wall/forwards/backwards，不能因 optimizer steps 相同声称 FLOPs 相同。

验收仍是 E1 的逐 chunk A/D 方向 + 局部人物结构 + held-out 状态/seed，不是 loss 或 margin。若 16 次没有有意义改善，停止该目标，重新审视表示/可训练 action pathway；不自动扩预算或转 DMD。

## E3：通过局部能力门槛后，分开研究 AnyFlow 与 Stage2

| 子阶段 | 唯一主要问题 | 匹配对照与门槛 |
|---|---|---|
| E3a AnyFlow | 可信 30-step causal generator 能否减少每块采样次数 | 冻结拓扑/条件/选定初始化；30/8/4 steps/chunk；检查 diagonal 保持、finite endpoint 正确性、自适应权重、scheduler 覆盖、composition；4/8 的方向/结构接近同路线 30 步，而不是只看 residual |
| E3b on-policy Stage2 | 自己生成的 history 会否使局部能力失效 | 同一 AnyFlow checkpoint，在 GT/reference-history 与自身 rollout 分布上比较；frozen 合法 teacher + trainable fake score/critic + DMD，必要时评估 FMBS；保留动作配对验收，先 39f，再 124f |
| 最终评测 | 动作、稳定性与效率能否一起成立 | 同首图/prompt/actions/seed/noise/分辨率/帧数的 Original30 与选定路线；W/S/A/D、切换、124f，通过后才补 10s/20s |

若采用 T2，训练与推理必须匹配窗口重算语义。共享双向 prefix 所造成的教师强制泄漏须单独防止；可以逐窗口训练，不能直接套融合全序列两流 mask 并假定安全。

Stage2 的前提是局部 action/结构与基本有限映射可信，**不是**已解决全部 generated-history 漂移。AnyFlow/Stage2 已有 CPU 数学/图安全检查可以复用，但不等于新拓扑的模型效果证据。旧失败模型不能因接口通过而自动进入 E3。

## 统一输出与停止条件

每个实验保留：协议/源码/输入哈希、全部成功与失败结果、逐 chunk 同状态差分、可播放 MP4、逐帧图、人工评审、决策表。MAD 是活动量；boundary score 是连续性诊断，均不当作画质。Farneback 符号只在有强参考的同场景成立，不对所有 W/S/A/D 或所有视角泛化。

效率表测整视频端到端（包括 conditioning、传输、重算、commit、VAE decode）、首个可显示块、后续每块延迟、吞吐；另列 model-load cold start。预热一次、相同 GPU/offload 设置下至少 3 次正式测量，报告均值/离散度及共享负载。短探针只记成本，不据它宣称加速。峰值 allocated/reserved、resident weight、KV/CPU latent bytes 分开记录；不能用简单相减伪造 activation 峰值。计步写清 30×3=90 noisy forwards、额外 commits 与窗口重算，Original30 为 30 full-sequence forwards。

项目同时最多 3 张 GPU，先用 1 张做 E1 机械/真实路径预检，通过后最多 2 路做配对评测，第三张保留按需使用；不占他人进程。没有实验因机器空闲就自动扩大预算。

初始协议登记时，旧FM48 density队列已结束、E1 GPU尚未启动；这段历史安排现已执行。下一实际动作以上方最新历史条件协议为准。没有新证据前保持原会议视频和结论，不能提前写“action control 已保留”。
