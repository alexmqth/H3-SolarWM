# EXP-001 / v3：Judge正式验收

日期：2026-10-10，Asia/Hong_Kong。**Judge Acceptance: accepted；Decision: accept / close。**

## 结论与范围

接受本任务执行，并接受**单停车场、seed13、持续A/D、124 RGB帧（24fps，5.17秒）的V2b多窗口可行性**。两条路径人物保持单体可辨、停车场结构连续，呈现不同方向的持续运动；辅助flow与这一观察一致。当前是未经额外大量训练的原型验证，不要求成熟产品画质。

| 维度 | 判定 | 边界 |
| --- | --- | --- |
| 执行与证据 | PASS | v1四路第三块、v3仅AA/DD三块续写，零训练，输入/源码/历史可追溯 |
| 持续A/D动作可行性 | PASS，限定本场景/seed/124帧 | 非动作准确率、非同状态反事实、非跨场景泛化 |
| 多窗口视觉基本结构 | PASS，可行性范围 | AA RGB72→73明显姿态跳变；动作节奏不均/细节软化；DD末段靠近画面下边缘 |
| 动作切换 | PARTIAL / 未通过联合验收 | AD/DA停于73帧；DA第三块A的flow −0.175，原四路径符号门槛FAIL保留 |
| 效率测量 | 已记录；公平speedup NOT_TESTED | 每步重算全部可见历史，persistent hidden KV=0 |
| V3 | NOT_COMPLETED | 本配置缺少strict chunk-causal历史表示与真实persistent KV复用 |

不为普通画质缺陷追加实验；也不把本轮成功范围扩展到切换、10/20秒、泛化或V3。

## 独立审核与可复现性

Judge已读取最终Worker报告、任务v3与用户ROI/可行性补充、runner/config/来源manifest、原始逐块JSON、预算和对比来源。四路前73帧的完整静态帧序列及第三块全部人物原尺寸裁剪已审阅；随后AA/DD每段新增17帧静态序列全部查看，另检查关键边界和末帧原尺寸画面，覆盖两条完整124帧。没有声称正常速度实时播放。Original匹配参考完整解码，抽看12帧/路径；对比片查看排版/标签并完整解码。

独立核对：v1/v3冻结worker文件hash均匹配；两条124帧、24fps、单调PTS；每个已发布RGB前缀、生成endpoint hash均匹配；三条并排片的hash、124/124/248帧、1664×570与PTS均通过。见[最终技术检查](final_technical_checks.json)、[对比片检查](comparison_checks.json)及[逐阶段审核](REVIEW.md)。保留旧源码/config/旧四路径门槛，未追溯改写旧FAIL。

Original为2026-10-01-21的已存A/D参考，输入审计确认初始video/audio noise、anchor、prompt、位置等相符。Original联合音视频整段去噪，V2b固定audio条件且逐块重算自身历史，属于协议能力比较，不能作单变量归因。

## 成本与限制

全任务300 sampling +12 diagnostic =312 denoiser forwards，18 VAE decodes，0训练；3704.853 GPU-seconds（1.029126 GPU-hours）；allocated峰值26392.86MiB。10个登记GPU运行均已结束，v3只用两卡，未接近09:00截止。CPU截止/预算逻辑有检查，不能宣称实际经历跨09:00切换。历史资源字段与重复配置字段按各自版本解释，实际执行满足最新8/3授权和v3更紧预算。

V2b完整六块协议每条180 sampling forwards；本次前两块复用。73→124三块增量wall为AA1216.825s/DD1194.907s，Original全片E2E为478.447s/454.570s，口径不同，无公平加速比。CPU offload主机内存峰值未单独量化，不据此宣称内存优势。全片flow为V2b A +1.034788/D −1.049849，Original A +1.076725/D −1.601923，仅作辅助。

历史CPU测试日志有8项/当前收集4项的数量差异，Worker已披露；不以此阻止已有代码/输出证据充分的任务。开跑时v3测试首项依赖state=22；Worker在结案前将原文件及manifest保留为test_v3_contract_at_run.py/source_manifest_v3_at_run.json，再将当前test_v3_contract.py适配到完成态37，manifest记录了两份hash。Judge已检查该改动仅覆盖完成态断言，并运行当前三项CPU检查，3 passed；生成runner/config未变。

## 研究决定

V2b完成所需的持续动作多窗口对比，停止在本方向追求边界或小指标收益。下一项采用native Single I0/12→5的既有current-prefix严格缓存候选，以有限短视频判断能否同时保留基本动作/结构与真实KV。候选失败时停止无训练拓扑微调，只有存在明确可修复信号才考虑有限适配，否则换路线。下一任务须单独发布，不由本验收自动授权GPU。
