# EXP-003：同一严格缓存候选的124帧与效率可行性

发布：2026-10-10 04:23 HKT，Judge。EXP-002已验收归档并推送GitHub：bdcb43a5bf7568b6182b8563a9b08a8391fa53fb。本文件是当前唯一GPU任务授权。

## Task / Track / Parent

EXP-003 / v1；Worker Status: running（单GPU0，AA先行）；Judge Acceptance: pending；Mainline / V3 feasibility。Parent为EXP-002 native Single I0/current-prefix/clean-commit persistent-KV候选，Original H3 + released action LoRA，零新增训练。

## 核心问题与假设

同一候选在自己的历史KV下延伸至124帧，是否仍具备可辨动作、基本人物/场景结构，并体现真实历史复用与可解释的推理成本？假设是EXP-002短窗口能力可延续，不以正式V3通过为预设结果。

已有回答与本轮价值：EXP-002检查同一个A首窗历史下当前A/D（AA/AD）并续至73帧；本轮只增加后续三块，验证增长后的自身历史及成本，不重复短窗口和18点诊断。正结果可用于同配置V3可行性验收；严重持续失效则记长度边界，停止该无训练方向，不自动加训练挽救。

## 保持条件 / 唯一变化 / 输入

保持Original/released LoRA、停车场832×480、seed13、full37相同noise、Single I0、native时间与全局RoPE、固定audio条件、30steps/chunk、shift2.22、own action/current-prefix连接、sigma0/native clean commit、原精度/backend/offload与CPU raw KV。AA全A，AD首12为A、其后均D。

从各自EXP-002完成的first12_A + second5 + third5 endpoints与cache_through17续接。先用自己的third5按原图/条件提交一次得到cache_through22，不能重建于当前新prefix下。首73已发RGB保持原样。继续latent[22,27)、[27,32)、[32,37)，累计90/107/124RGB，完整partition[12,5,5,5,5,5]。

本轮只扩大生成范围；不改变mask、anchor、history sigma、窗口淘汰、精度、权重或动作协议。六块范围内保留全部历史；cache index与全局start分开，历史K/V是祖先提交的原表示，不能每步重算历史或用V2b/teacher reset。

## 执行步骤

1. 冻结EXP-002源码/输入/缓存/endpoint/RGB hash；隔离EXP-003入口，扩展显式区间到37latent。CPU验证新增索引/位置、缓存覆盖及预算；复用已有动作/未来隔离证明，不重跑模型诊断。
2. AA/AD各自续写三个chunk至124帧，保留中间checkpoint和真实帧；一个有限任务里可连续执行，无需每块提交正式验收。发现明确严重结构或持续控制失败时停止受影响路径，保留结果；普通细节、节奏或边界问题不取消所有路径。
3. 每块只读历史KV采样30步；需要进入下一块时当前endpoint sigma0 commit一次。最后124帧无须多余commit。生成帧只追加，不对过去末5帧进行回改或平滑。
4. 交付候选AA/AD完整124帧、Original持续A vs candidate AA124、V2b AA vs candidate AA124，以及V2b AD/candidate AD共有73帧。Original A→D匹配视频缺失则标记；不使用Original持续D冒充，也不新生成基线。各片标明动作、范围、协议与计时口径。
5. 提交report后停止GPU扩展，Judge判断是否可以发布正式V3可行性版本并同步GitHub。

## 预算与资源

0训练；两路径、单scene/seed。6个新chunk×30=180 sampling；续接third5提交2次、90/107帧后提交各2次，总commit≤6；诊断0；完整denoiser总≤186。VAE≤6、RGB re-encode0，失败/重试计入。

全任务≤2 GPU-hours，首个GPU进程起≤3h elapsed；allocated每卡≤44GiB。项目在2026-10-10 09:00 HKT前最多8卡，之后最多3卡；本任务最多2卡且可顺序运行，08:30后不增加超过3卡占用，09:00前释放多余卡，不自动恢复8卡。

建议一个进程连续处理同一路径剩余chunk，保留必要offload；若逐块重启，明确加载/保存cache成本，不将其混入纯sampling比较。所有活跃进程、加载、提交、VAE、失败都计入GPU-hours。

## 效率与验收

记录每块sampling、clean commit、cache读取/传输/保存、VAE和wall时间、实际前向次数、cache层/token/bytes、GPU allocated峰值；核对只对新chunk做Transformer计算和实际历史cache未变。首73复用时，时间明确为增量；首屏/完整E2E未测则写未测，不能补造。

对照优先复用EXP-001 AA对应三块实测：310.26/377.53/447.19秒sampling；记录硬件/offload/步数及运行时间差异。可描述已记录条件下的观察成本，不能当严格单变量KV收益或统一重复均值speedup。Original整片30步和本候选每块30步不直接求加速比。

同一个明确配置同时满足以下条件时，Judge可验收V3的可行性版本：strict chunk-causal、真实persistent KV复用；EXP-002同history A/D有可辨条件响应；本轮124帧自身历史里人物/场景基本可用且动作无明显持续失效；代码/输入/视频可追溯，效率测量足以说明历史复用与实际成本。保留单场景/seed、切换范围、质量、无整体Original speedup和未训练等限制。不要求成熟产品画质、10/20秒、多场景泛化或AnyFlow/DMD已经成功。

若只有一条可用则能力PARTIAL；若关键机制错误则INVALID；有效负结果可接受任务执行但不能宣布V3。普通模糊、肢体细节、短暂边界和小幅proxy反号不独立判失败。

## 输出与停止

独立EXP-003目录代码/config/manifest/metrics/日志/真实视频/效率表、根report.md；最终说明同一配置解决什么和还缺什么。协议泄漏、历史回改、NaN/OOM、预算超限、严重质量/控制失败时停止对应运行并报告。无训练、额外seed/场景、参数搜索、额外诊断、AnyFlow/DMD或更长视频授权。

本任务授权限于上述186次前向与资源范围；完成或提前停止后提交report，不自动扩展。
