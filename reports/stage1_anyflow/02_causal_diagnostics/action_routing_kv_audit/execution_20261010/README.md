# Action Routing / KV：固定状态审计完成

2026-10-10。**P0在统一数值执行下通过：6状态×A/D，50层历史K/V、RoPE及最终velocity逐元素相同。P1/P2证实：公共prefix、严格video因果图、历史提交时间冻结和额外past-action直读都会改变动作差分；不能把它们全部称为KV实现错误。**

本轮零训练、零新视频、零AnyFlow/DMD。所有新数字都是同状态velocity诊断，不授予视频质量、左右方向或长视频PASS。C+N旧第二块局部正控保留，未扩第三/第四块。

## P0：先排除数值执行差异，再检查缓存

| 执行版本 | 历史KV最大相对RMS | velocity最大相对RMS | 说明 |
|---|---:|---:|---|
| run_01 | 0.03315813 | 0.00913808 | 默认GEMM/SDPA，首状态失败后停 |
| run_02 | 0.02934631 | 0.00845389 | 禁止BF16 reduced-precision reduction；仍停 |
| run_03 | 0.00000000 | 0.00627897 | 再统一attention可见key分组；历史KV全相同，当前输出仍不等价 |
| run_04 | 0.00000000 | 0.00000000 | 再统一线性/LoRA的GEMM行数；全部6状态通过 |

第一处差异的定位：历史input/embedding/norm/QKV输入相同，默认BF16投影输出因矩阵行数变化而不同。禁用reduced reduction后第0层历史K/V相同；canonical SDPA后全部历史K/V相同，但裁剪当前分支的投影仍有小差异；固定GEMM行数后最终输出也逐元素相同。见numeric_debug.json、numeric_layer1.json、numeric_current.json及其原始脚本。

canonical SDPA只按相同布尔mask聚合query、按原顺序提取可见key；canonical GEMM使用8/256固定行、补零并丢弃补零输出。参数、dtype、时间、可见边不变，但浮点运算路径改变。仅用于隔离数值混杂，不是生产默认或加速优化，不能宣称默认部署已bitwise等价。

**R3-match与R3-commit0分开：** P0每个sigma用同一native prefix条件、clean history重建cache，与R2比较；它不是复用sigma0 cache。P1的R3使用真正sigma0 clean commit，再冻结到三个当前sigma，专门测提交上下文变化。P0通过不意味着P1这项也应为0。

P0覆盖两份自身A/D历史、当前[12,17)、三sigma(.939540/.689441/.240781)。每个状态的A/D都只读同一cache；完整hash不变，RoPE为0误差。只覆盖一份12latent历史＋当前5latent，不扩大为全部滑窗/eviction/旧adapter配置验证。原production recompute_forward的参数/feedback局限没有被悄悄修改。

[逐状态/逐层P0原始收据](run_04/evaluation.json) · [包含失败轮次的CSV](P0_metrics.csv)

## P1：每次只改变声明的因素

R1=同可见范围Original双向；R1-P=仅关闭非action公共prefix读取video；R2=在R1-P上只切断历史video读取当前video；R3=own+feedback及sigma0持久历史；R4在同一R3 cache上切动作直读范围。各对照对照对象不同，下面的cosine不可相加为归因比例。

| 干预（被比较者/参照） | 状态数 | Δv cosine均值 [min,max] | 范数比均值 | 相对L2均值 |
|---|---:|---:|---:|---:|
| 关闭公共 prefix 的 video feedback | 6 | 0.3354 [0.0233, 0.8720] | 0.9183 | 1.0634 |
| 仅将 video/video 改为严格 chunk-causal | 6 | 0.1175 [-0.0961, 0.2617] | 0.9981 | 1.3311 |
| 当前时间重算 → sigma0 clean commit 冻结 | 6 | 0.6400 [0.4409, 0.8718] | 1.0791 | 0.8546 |
| 仅增加当前 chunk 内较早 action 直读 | 6 | 0.4113 [0.2241, 0.7185] | 1.1464 | 1.1754 |
| 增加全部 past action 直读 | 6 | 0.0645 [-0.0971, 0.2382] | 0.7222 | 1.1982 |
| 仅增加历史 chunk action 直读（中sigma） | 2 | 0.0226 [-0.0035, 0.0487] | 0.5121 | 1.1204 |

另外统一与R1 Original-visible比较，防止把“偏离已经失配的own”误解成“更偏离Original”：

| 角色 | 与Original的Δv cosine均值 | 范数比均值 | 相对L2均值 |
|---|---:|---:|---:|
| R1P | 0.3354 | 0.9183 | 1.0634 |
| R2 | 0.0570 | 0.9163 | 1.3188 |
| R3 | 0.0669 | 0.9927 | 1.3661 |
| R4_within | -0.0069 | 1.1376 | 1.5377 |
| R4_full | 0.0717 | 0.6912 | 1.1741 |

失配在没有persistent KV的R2已出现。R4-full与Original的平均cosine并未比R3更低，且A-history高sigma点从R3的0.0244变为R4-full的0.3237；不能宣称每次增加past-action都更坏。它没有一致恢复Original，去掉该直读也不足以修复严格causal图。


六点来自同一停车场/seed的两份history×三sigma，不是六个独立场景。差分均非零；幅度接近也不表示方向一致。没有按cosine选择新视频，没有据此声称角色方向恢复。逐sigma见[P1_metrics.csv](P1_metrics.csv)，不要只读均值。

R1-P/R2共用forward函数和同一full输入：当前latent、clean raw history、action feedback、native text/action time、Single I0、audio noise/time、全局位置都相同。真实832×480布局CPU检查确认两mask仅差9,126,000条“历史video query读取当前video key”边，prefix/action反馈行完全相同。见[真实布局验收](real_layout_audit.json)。

## P2：块内与跨块action直读

within=own＋当前chunk内较早latent的action；cross=own＋历史chunk的action；full=两者并集，仍不直接读未来latent动作。真实布局额外直连分别39,000/234,000/273,000条，交集恰为own，full逐元素等于生产causal模式。不是只换名称。

| history / 中sigma | within vs own cosine | cross vs own cosine | full vs own cosine |
|---|---:|---:|---:|
| A | 0.2241 | -0.0035 | -0.0043 |
| D | 0.4421 | 0.0487 | 0.0340 |

这两个中sigma状态中，相对own的跨chunk直读方向变化比within更大；within本身也明显改变差分。这不等于相对Original更差，也不等于视频质量受损。是边组干预结果，尚未定位到单条最小破坏边，不推广成所有sigma/场景的贡献排序。本轮R4固定同一own-cache，测即时直读；没有声称测过按R4规则重新生成整段历史的累积效应。

## C+N的同状态桥接

中sigma另比较R1/N、R1-P/N、R2/N。三者使用完全相同的noised history和current state，防止把N重算与clean cache冒充单因素。两份独立进程R1/N重复最大误差均为0。

| history | R1-P/N vs R1/N cosine / norm ratio | R2/N vs R1-P/N cosine / norm ratio |
|---|---:|---:|
| A | 0.1469 / 0.7672 | 0.0195 / 0.8428 |
| D | 0.0861 / 0.7026 | 0.0649 / 0.8506 |

C+N所用的局部双向图不能被严格causal图视为等价替换；N历史本身也没有消除这些依赖差异。这里没有重新生成C+N视频，不把速度差分直接翻译成画面崩坏或光流符号。

## 冻结协议、资源与复现边界

权重为Original H3＋released action LoRA，未加载我们新增训练adapter；Single I0/native text time、full37固定布局/noise、同prompt/seed13。当前state来自预选endpoint/noise插值，并非实时solver中间状态；A/D在每一组共用同一state/history，未把不同轨迹输出直接相减。

R1只是相同可见prefix的Original参考，没有未知未来video；audio固定noise/time0，不冒充完整124f联合audio/video采样。这些差异对所有主对照一致。原始C+N视频由旧数值后端生成，不能冒称本轮数值后端已通过视频验收。

主成功进程106完整前向，耗时1120.5s，GPU allocated peak 25.677GiB，CPU raw KV 6.332GiB。N桥接另6完整前向；前三轮失败共20完整前向；数值定位另1完整历史prefill和8次第0/1层截断检查。合计133完整33B调用＋8截断检查。主体72次A/D对照包含在这些数字内，其余是严格P0、prefill、重放、P2及排错开销。零optimizer、零新视频。最多GPU2/4两卡，其余卡未占用。

这是诊断成本，不是视频推理速度或warmup后均值；CPU KV与CPU模型权重offload分开。9项CPU测试通过，另有真实layout审计；历史模型参数version/hash保持。没有新的训练质量结论。

原始运行目录：`H3-World/outputs/2026-10-10-00/routing_kv_audit/`。该归档是源码/原始JSON/CSV/log快照；runner需要原工作树中的冻结runtime、Original基础权重、released LoRA和预处理conditioning。各protocol保留输入与源码SHA，source_runtime_manifest.json定位原冻结依赖；不是在本Git子目录直接运行的独立模型发行包。

在原工作树中先执行三个CPU测试文件，再用h3world环境运行`run_fixedgemm.py --gpu <idle_gpu> --output <new_directory>`。该入口强制先P0再P1；不要覆盖run_01至run_04。N桥接入口固定读取本轮run_04，复现实验应复制到新时间目录并重新登记协议。临时velocity tensors留在原工作区，路径/哈希见local_velocity_tensor_manifest.json；没有复制模型权重、KV或数据集。

## 技术判断与下一边界

已有证据支持：Single-Egress配对索引未被这次审计推翻；缓存基础操作在匹配图/时间/受控数值条件下可完全等价；原模型动作函数对公共prefix反馈、video双向回路及额外动作直读敏感，提交时间冻结也有独立影响。

不能从这些局部点唯一排序整段视频失败原因，也不能直接证明任何新配置恢复了动作与人物结构。下一步保留own及自身反馈，保留C+N独立正控；在决定严格因果适配前先明确prefix/历史时间契约。没有自动启动新的LoRA、AnyFlow、DMD或多窗口视频。
