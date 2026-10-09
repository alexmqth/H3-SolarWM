# EXP-003 continuation preflight review

2026-10-10 04:27 HKT，Judge。此为代码审阅，非视频验收。

已核对interval从EXP-002扩展到third5提交及三个新5-latent区间，全局start和cache index分离，Single I0/native时间/全局RoPE保持。metered_prefix只增加历史读取、搬运计时/字节记录；attention计算与已验收current-prefix相同，CPU测试覆盖计时代码输出一致。保留5个祖先，最后第六块读取全部前32latent缓存；最后不额外commit。

Runner读取各自EXP-002 third endpoint、through17缓存与73帧RGB，核对输入摘要；third5用原stop22 prompt/native sigma0提交一次，后续只追加自己的端点。sampling内存cache identity/version未变、历史和noise不变均有断言，RGB只追加。预算计调用前记账；占卡等待也计入GPU-hours。通知Worker把reload计时改成增量并保存，块wall和人工review等待分开，不为计时补跑GPU。

接下来只评估同配置自身历史至124帧和实际成本，普通质量缺陷保留；不重复短片能力诊断。

## AA至90帧（04:32 HKT）

独立完整解码90帧/24fps/PTS，前73逐像素冻结，RGB/endpoint摘要、内存cache不变、模型参数版本均通过。新增RGB73–89及边界72全静态查看，native末89与82/85细节查看。flow+1.4629、sampling148.45秒，cache22→27由12.465GB→15.298GB。72→73姿态跳变明显；中段约79–86存在人体扭曲/局部游离肢体残影，超过普通细节软化，末89恢复单体可辨，场景持续。此时不作最终PASS；按原预算观察后续是否持续恶化，不增加诊断/训练。已通知Worker如实描述，并在严重结构问题持续时停止对应路径。

## AA至107帧（04:37 HKT）

独立完整解码107帧/24fps/PTS、前90RGB逐像素冻结、摘要/endpoint/内存cache/参数版本通过。静态看新增90–106全部帧与native末106：开头运动姿态和边界仍不自然，此后人物恢复完整，场景连贯，前块的严重扭曲未持续到本块主体。flow+0.9529，sampling176.43秒，boundaryMAD9.69；through32 cache18,130,944,000 bytes，50层每层12480tokens。继续最后授权块；不把前一块的瞬态结构缺陷抹去，也不以该缺陷自动否定已恢复的整体路线。

计时/峰值解释补注：源码third5初始commit后调用peak(torch)验证≤44GiB，但局部变量third_peak未持久化，随后逐块reset_peak。最终峰值只能称“已记录的新采样块/commit/VAE峰值”，不能冒称完整进程最大值；初始提交有上限断言而无具体值。无需为此重跑GPU，保留测量范围即可。每块sampling已含历史KV搬运，transfer事件累计是sampling的组成部分，不能再相加到总成本。

## AA至124帧（04:41 HKT）

完整解码124帧/24fps/PTS，前107RGB冻结、RGB/endpoint摘要、cache不变与参数版本检查通过。查看107–123所有新增帧与native末123：人物与停车场基本保持，动作姿态偏僵、局部背景纹理/瞬态亮痕，未延续第四块严重人体扭曲。flow+1.33545，sampling200.42秒，boundaryMAD2.91。最后未多余commit，cache仍through32/18.131GB。AA三新块flow+1.463/+0.953/+1.335，sampling148.45/176.43/200.42秒；90sampling+3commit=93forward、3VAE、798.95GPU秒（含检查等待）。短暂明显结构缺陷后恢复，支持有限长片可行性，但不能称画质无退化或全帧结构稳定；最终判定等AD及报告。

## AD至90帧（04:46 HKT）

独立完整解码90帧/24fps/PTS、前73RGB逐像素冻结、摘要/endpoint/内存cache/参数版本通过。静态查看73–89全部新增帧及native末89：单体人物/停车场保持，运动偏僵、普通细节软化，未见AA第四块的明显人体扭曲。flow−1.30403，sampling148.94秒，boundaryMAD7.18。cache和token规模与AA相同，实际内容来自AD自己的历史。继续原授权后两块。

## AD至107帧（04:50 HKT）

独立完整解码107帧/24fps/PTS、前90RGB冻结、摘要/endpoint/内存cache/参数版本通过。静态看90–106全部新增帧及native末106：开头91–92附近有短暂肢体/姿态异常，随后单体人物和场景保持可辨，没有持续分解。flow−0.64824，sampling176.66秒，boundaryMAD3.34。继续最后授权块；保留动作僵硬与局部瞬态缺陷。

## AD至124帧与预算结清（04:55 HKT）

完整解码124帧/24fps/PTS、前107RGB逐像素冻结、摘要/endpoint/内存cache/参数版本通过。静态查看107–123全部新增帧及native末123：单体人物/停车场可用、动作略僵，未见持续结构崩坏，flow−1.24023、sampling199.43秒、boundaryMAD2.50。AD三新块flow−1.304/−0.648/−1.240。两条新102帧均逐帧静态审阅，首73证据继承EXP-002；不声称原速播放。实际账本180sampling+6commit=186forward、6VAE、0训练、1572.266GPU秒=0.436741GPU-hours，两个GPU0进程均结束，未超预算。成本逐块核对见costs_checked.json。
