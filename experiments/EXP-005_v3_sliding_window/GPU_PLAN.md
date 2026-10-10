# EXP-005：阶段二有限GPU验证提案

**状态：DRAFT / NOT AUTHORIZED。当前GPU额度=0。** 本文给出可审核的上限，不因EXP编号存在、CPU通过或用户提出研究方向而自动生效。Judge需单独记录批准阶段、代码/config/input hash、起始时间及预算后才可启动。优先G，L独立放行；本轮不训练。

## 1. 固定协议与输入就绪条件

Baseline：EXP-002/003已验收Original H3 + released Action LoRA，30步native FM/Euler、shift2.22、native Single I0、own-action/action feedback/current non-action prefix feedback、video sigma×1000/audio1000、sigma0 clean commit、h3_fp32边界精度、相同SDPA backend/offload。Partition为`[12,5,5,5,5,5,5,5]`，0-based index0–7；第7/8块分别latent[37,42)/[42,47)，RGB124–140/141–157。最多158帧/24fps。5是祖先chunk数量，不是固定25帧原始视频。

复用EXP-003 AA的124帧生成历史、C6 endpoint和`cache_through32.pt`（C1–C5）。commit C6恰好一次后才得到C2–C6，真实触发首次淘汰；**不能把未提交C6的旧cache直接称为第7块已就绪**。这组共享原始历史用于C7的A继续和A→D切换。

审批前必须完成：

1. 实际入口、冻结依赖、权重、首窗/历史endpoint/cache/RGB与新fixture逐一hash；长fixture的前37latents及原prefix/视频坐标和旧输入完全一致。
2. 后10latent噪声独立CPU固定Generator一次生成并保存，同动作/方法共享；旧37噪声直接复制，禁止以同seed重新randn更长tensor冒充前缀一致。说明新增噪声seed、生成器/版本和SHA。
3. 新action rows由同一冻结A/D条件来源延伸，并显式记录新增行/位置。保持旧非action文本、I0、audio张量与全局坐标不变；future rows在text refiner前物理删除。不能默认重新build整段47latent的packed等价原37，因为text_len和action mirrored origin会变。
4. 冻结`global`与`sliding_local`配置，逐层精确indices检查、canonical-global RoPE provenance与mode lineage，禁止把L生成的raw KV当G cache默默使用。
5. 预先锁定SDPA实现/torch版本、精度、误差阈值；入口增加GPU授权文件必需项、逐调用尝试前记账、单进程锁、无自动重试。CPU-only代码通过不意味着生产GPU runner就绪。

## 2. 分阶段上限（须分别放行）

| 阶段 | 问题 / 执行 | Sampling forwards | Commit | 固定状态diagnostic | VAE | GPU小时上限 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| G0 首淘汰前回归 | 冻结旧入口与SW-G在2个相同C6 noisy state比较；SW-G完整重放C6一次与保存endpoint/RGB比 | 30 | 0 | 4 | 1 | 0.30 |
| G1 SW-G C7/C8 | 同AA124历史分为继续A、切D；各自C7后commit并续自己的C8 | 120 | 3 | 0 | 4 | 0.70 |
| L1 SW-L C7/C8 | 共享G1已提交C6历史，两个动作各自续C7/C8；另做C8同G历史G/L固定状态对照 | 120 | 2 | 2 | 4 | 0.70 |
| **核心总上限** | 仅上述三段分别获批后 | **270** | **5** | **6** | **9** | **1.70** |

**总281次完整denoiser调用**，含失败尝试，无新首窗生成/训练/reencode。G1的3commit=共享C6一次+两条G路径C7各一次；L1的2commit=L两条路径C7各一次。C8末块不做无用commit。共享C6缓存只读；磁盘复制/加载/解码/检查等待均计占卡时间。阶段小时上限包含加载与失败；核心总elapsed≤3小时，从G0首GPU进程起算。每阶段达到任一上限即停止，不借未批准阶段额度补跑。

最多1张空闲GPU顺序运行，项目合计≤3张；每卡allocated峰值≤44GiB，峰值不reset，单机CPU原始cache副本最少化。以实际nvidia-smi确认空闲，不干预他人进程。

### 可选C9：默认关闭

只有C7/C8出现“瞬态异常后恢复还是持续恶化”这类可能改变决策的明确问题，Judge另行激活指定路径C9。每条上限30sampling+1commit、1VAE、0.175GPU小时；最多4条合计124forward/4VAE/0.70GPU小时。不得为微小flow改进或占满预算自动扩展。**含全部可选分支的绝对上限405forward、13VAE、2.40GPU小时、总elapsed≤4小时；不是当前授权。** 通常C8已足以检查第二次淘汰及容量，不需要C9。

## 3. 回归与比较口径

- G0首先用相同cache、current、sigma、prefix、dtype、backend比旧入口和SW-G的velocity tensor；报告relative RMS/maxabs、finite、cache只读。预期同backend结果逐元素一致；若仅浮点差异，采用预登记fp32 `atol=1e-5, rtol=1e-5`参考阈值并报告实际量级，超阈值暂停定位，不靠放宽阈值直接继续。不能改采样输入或mask来“复现”。
- 30步C6重放报告每步、末endpoint relative RMS/maxabs与冻结末17 RGB，旧107RGB逐像素不变。GPU全模型实际误差尚未知，CPU一致不能提前填PASS。若同协议出现可解释低精度/backend差异，Judge先判断是否足以排除附带改动，再决定放行；不无限重跑追求压缩MP4字节一致。
- C7四个完整生成结果形成同历史/同噪声的`G-A/G-D/L-A/L-D`对照。A/D只变当前动作，G/L只变video位置处理。
- C8各模式接各自C7输出，是闭环比较，历史已不同。为满足同可见历史的位置隔离检查，另在**同一G-A C7 cache/同一C8 noisy input/sigma=0.5**上各做G/L一次只读forward，报告差异。这两个调用只是固定raw历史的直接位置效应，不能冒充L自己的历史，也不能替代两段完整视频。
- 两条路径表示A历史后继续A或切D，仅验证这一次切换；不宣布任意A/D长时切换通过。

## 4. 验收、测量与停止

1. 实际cache indices：C6采样[0,1,2,3,4]；C6commit后/C7采样[1,2,3,4,5]；C7commit后/C8采样[2,3,4,5,6]，所有实际层一致。错误indices、缺层、未来信息、非法sigma/重复commit立即停。
2. 采样期间raw K/V的identity/storage/version和内容摘要不变；Local的临时rope只读映射不回写历史。commit只追加当前endpoint；旧RGB124与随后141前缀在`.npy`层逐像素保持，MP4全帧可解码。
3. 原始缓存容量：首次淘汰前C1较长（12latents），C1–C5共32latents；淘汰后最近5个5latent祖先共25latents。按旧50层、390tokens/latent精度，估计由18,130,944,000降为14,164,800,000 bytes，以实际账本核对；C8/C9应保持同阶/同尺寸。区分current cache、峰值、序列化与临时拼接分配。
4. **只宣称历史video KV有界。** 原生保留已知action prefix、完整历史latent/RGB和整前缀VAE解码仍可能增长；本任务不通过悄悄裁prefix/解码协议来取得“整体无限长度内存常数”结论。
5. 每块记录sampling NFE、commit/diagnostic/VAE尝试次数、采样/搬运/commit/decode/wall、GPU allocated/reserved峰值、CPU KV字节/indices与进程RSS。搬运若未拆分说明计入sampling，不重复相加。
6. 完整新增帧、边界和必要native细节检查：人物结构、ghosting、瞬态/持续分解、场景漂移、动作及切换，flow仅辅助。普通细节/小幅proxy异常按可行性尺度接受；持续严重人体崩坏或明显动作失效则停相应路径，不追加训练救结果。
7. NaN/OOM、预算/资源超限、基线回归明显失败、cache/输入协议无效立即停；保存失败日志。仅CUDA无报错、finite forward或5个cache条目不能判生成能力PASS。

## 5. 交付与决策

独立配置与sources/artifacts manifest；完整命令/环境/模型hash、只读cache与RGB检查、分阶段预算账本、逐块指标、原片和左G右L同动作对照、必要Baseline C6回归片。历史与时间口径标明。Baseline只作冻结参考，不覆盖旧文件。

G0过关才考虑G1；G1已足以否定实用SW时停止并归档，不强行做L。L失败可接受为位置协议负结果，不自动触发adaptation。Judge逐项区别“工程正确、有限生成可行、长期能力”；最后决定接受某个候选或维持Baseline。
