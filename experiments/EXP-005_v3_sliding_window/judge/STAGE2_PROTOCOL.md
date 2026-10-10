# EXP-005/v2：三个候选的实际协议与比较边界

## 共同冻结条件

Original H3 + released Action LoRA，零新训练；Single I0、native video timestep `1000*sigma` / audio `1000`、own-action routing、action feedback、current non-action prefix feedback、strict causal、persistent **raw** video KV、sigma0 clean commit、首12后每块5 latent、30-step native FM shift2.22、原37-latent初始噪声均保持。实验使用一张L40，CPU最多4线程；h3_fp32边界精度、Transformer bf16与默认SDPA调度沿用冻结路径，没有固定唯一attention kernel。

| 属性 | V3 Baseline | V3-SW-G | V3-SW-L |
| --- | --- | --- | --- |
| 冻结正式参考 | EXP-002/003，124帧 | 独立EXP-005候选 | 独立EXP-005候选 |
| 历史 | 旧实验最多5祖先，末C6未commit | 最近5祖先，真实淘汰 | 同SW-G |
| 位置 | Global | Global | 仅video Sliding Local |
| C7输入历史 | 未测C7 | 冻结AA124 + C6首次clean commit | 同一份G1 Global历史cache |
| C8输入历史 | 未测C8 | 各动作自己的G-C7 | 各动作自己的L-C7 |
| 已显示RGB | 已验收124 | 追加17帧，旧帧逐值保留 | 同SW-G |
| 首次淘汰前回归 | 原输出 | G0真实模型两状态及完整C6重放 | 首淘汰前复用Global实现；未额外重复GPU回归 |

## 显式长输入

首37个latent的packed条件、prompt、噪声、I0/audio及坐标逐值保留；新增A/D action embedding重复其真实来源中已验证相同的10-row模板，前37动作均A，新增10个latent对应继续A或切D。尾部噪声由独立CPU generator seed130005一次生成，同动作/位置方案共享。

Video origin为786，action origin约585，保留约201的原生偏移。两者各自延伸原生非均匀时间网格；新action坐标会越过原I0坐标，I0/audio/text不移动。这是显式冻结prefix的长输入外推，**不等于默认H3整段长packed重建**。默认重建会改变旧位置，已排除。所有模型调用前按当前可见stop物理裁剪未来action/video行。

[输入认证](../stage2/long_fixture_certificate.json) · [Judge输入审阅](../stage2/long_fixture_review.json)

## Local的完整定义

令原生时间网格 `tau(0)=0`，增量为 `(5/3) × [1,4,4,4,4]` 循环。Global视频latent j使用 `O+tau(j)`；Local在当前窗口最老历史latent b处重新起算，使用 `O+tau(j-b)`，空间坐标不变。C7的b=12，C8的b=17；这通常不是全体时间位置减去一个常数。

- 历史raw K/V不原地修改；读取时仅将历史video K与当前video Q/K的RoPE映射到当前local grid。
- text/action/I0/audio prefix坐标保持Global，native timestep、attention mask和路由不变。
- clean commit仍保存canonical Global RoPE元数据。Local生成的raw KV隐藏状态具有Local计算历史；Global标签不意味着Global隐藏状态来源。
- 冻结DiffSynth H3路径没有SolarWM camera PRoPE输入，未新增camera模块。此候选仅测试实际存在的MM-RoPE。
- 历史raw KV已依赖此前层和上下文，重映射不等价于用新Local位置重新计算整段历史。固定Global prefix与非均匀网格也破坏了简单的统一平移假设。

L1额外在同一G1-A C7缓存、同一C8 noisy latent、sigma0.5、同一A条件上分别调用Global/Local。它回答固定历史下的位置数值影响；两条完整C8视频的差异还包含各自C7历史，不能直接称同历史消融。

## 缓存与成本边界

真实C6 commit淘汰C1；C7读取C2–C6，C7 commit后C8读取C3–C7。每层精确校验indices、shape和完整50层。每份有效历史video KV为14,164,800,000 bytes，9750 tokens/layer；不同分支共享未修改的历史entry引用，提交替换其列表。

仅历史video KV容量有界。action prefix、全量latent/RGB输出、全前缀VAE解码及临时复制仍有成本，进程RSS远大于KV字节数。阶段GPU占用账本含加载与保存；每块sampling含CPU KV传输，commit和decode另记，没有单独测传输时间。不得把sampling成本当完整158帧从零生成耗时，也不得据此宣称公平端到端加速。

## 授权与停止

用户批准后Judge按G0→G1→L1逐段绑定源码/config/manifest批准。核心上限281forward、9VAE、1.70GPU小时、elapsed3小时；G0/G1/L1分别34/123/124forward，阶段超限即停，无自动重试。C9、训练、AnyFlow、DMD、新scene/seed及扫描均不在本任务内。按可行性记录普通画质缺陷；持续结构/动作失败才停止相应方向，不追加实验追求微小收益。
