# EXP-014/v1 — 四训练初图的原生V3 FM30教师目标

2026-10-11，Judge。**当前状态：P0与P1已验收，T1首图已验收；T2余三图已获Judge marker，可在空闲GPU0/1/2并行生成。新训练更新始终为0。** 用户夜间研究授权内的独立任务。正式V3、FM8、AF2结果保持冻结。

## Research Track / Parent / Question

Branch C训练数据准备；Parent冻结V3 Baseline与EXP-011 native输入/生成协议、EXP-013已验收候选。核心问题：四个确定性train初图在冻结V3 FM30下能否提供具有基本人物结构和A/D响应的C1/C2反事实教师目标，供后续独立AnyFlow训练？假设待验证，不因teacher身份默认正确。

## 输入、比较和冻结条件

严格使用EXP-013 candidate_manifest.json SHA `b6af2b0208a5dcf83c02f76fa826e5a03ba52fa9bd83de357f9474b5efb257b6`中四张PNG及caption.scene_static，顺序43866101/A1245、7199292c/A1670、9dc2e588/A1410、b784d995/A1635。完整episode/PNG摘要按manifest，不只匹配缩写。排除两个validation episode；不换图/seed/动作。

Original H3 33B+released Action LoRA，native Single I0/process_image=True、full37/124-frame packed位置、seed13同原生video/audio噪声构造、30步native shift2.22、Global RoPE、strict causal、own-action/action feedback/current non-action prefix、sigma0 clean raw KV commit、50层CPU cache和12+5 latent分块均沿用EXP-011冻结协议。只有初图与静态描述换为train候选。A/D是合成控制条件，不是原录屏GT。

本任务比较同一场景、同一teacher C1与C2噪声下AA/AD分叉，不新增FM8对照或比较跨场景flow大小。不引入LoRA训练、AnyFlow、DMD、Dual Anchor、Local RoPE、长时扩展。

## P0 — CPU实现与来源冻结（立即授权）

在独立实验和原始输出目录复用EXP-011已验证runner/encode代码，保留旧文件不修改。实现四candidate配置、阶段marker、每次GPU调用前账本、失败计费、预算/截止时间/磁盘闸门。冻源码/配置/输入/模型来源摘要，显式携带模型根环境，避免重复已知加载错误。

CPU验证候选PNG/静态prompt来源、train/validation隔离、future action/video rows的物理裁剪、native37位置与seed13噪声构造、C1/C2分块、每层祖先索引0、预算数学与无marker拒绝。不要为已冻结通用核心重复大量测试；只验证新增入口的真实风险。提交可运行命令、CPU报告、代码和source manifest，等待P1 marker。

## P1 — 四图native Single I0编码（待marker）

每图3文本static/A/D+1 image VAE，合计**12 text forward、4 image encode、0 video encode、0 denoiser、0 decode**；总上限600 GPU秒，allocated≤44GiB。只编码PNG，不复用旧Dual Anchor .pt。保存四native fixture，CPU审计Single I0、A/D差异范围、37个span/Global位置、native噪声、来源摘要。fixture审计通过后才进入T1。

## T1 — 第一候选完整教师目标（待marker）

固定43866101/A1245，先A C1=39RGB/12latent，以其sigma0 clean KV分叉AA/AD C2到56RGB。每图**30+1+30+30=91 denoiser forward，3 decode**。C1与两路C2都保存endpoint、旧RGB/完整视频、逐块指标与账本。保存第一图实际50层C1 cache供独立审核。

Judge检查全部39+17+17帧和关键原分辨率细节、动作分叉、人物结构/ghosting/边界；核对同C1/noise、独立分支、旧39RGB不变。普通细节软/瞬态重影可PARTIAL；持续人物瓦解、全帧噪声或两动作完全失效则停在该门，不自动运行余下三图。

## T2 — 剩余三候选（待T1验收及marker）

固定剩余三图，同协议各91forward/3decode；新增**273forward/9decode**。可按episode用实际空闲卡独立执行，预计三任务并行以缩短等待；每个任务内C1/commit/C2保持依赖顺序。并行任务使用独立账本，聚合GPU秒，避免共享文件覆盖。必须逐scene验收，不因一图通过就把全部teacher作为合格目标。

## Resource Budget / Deadline / Stop

- 全任务最多12text+4image encode、364denoiser（360sampling+4commit）、12decode、0backward/0update。编码≤600GPU秒，教师生成≤5400GPU秒，失败启动/模型加载/保存时间计入阶段墙钟GPU总成本。每scene教师≤1350GPU秒，无自动重试。
- 只用启动时实际空闲卡，不动他人VLLM的GPU3/4。09:00后项目总占卡≤3；本任务争取08:50前GPU完成，08:40后不启动新scene，未完成部分如实交接，不超预算补齐。空闲卡不构成新增样本授权。
- allocated峰≤44GiB/卡；CPU线程每进程≤4；新增磁盘≤40GiB，剩余磁盘≥60GiB。最多暂存四C1 cache约27.2GB供审计，保留/清理由Judge之后决定；不保存全量权重副本。
- 来源或协议错配、NaN/OOM、错误KV或写入历史RGB、超预算/截止时间立即停止；已失败scene不自动换样本。持续结构/动作失败报告Judge决定余下阶段，不靠调参修复本轮teacher。

## Deliverables / Acceptance

独立目录：代码/config/source hashes、CPU报告、marker、完整实际命令/环境/日志/原始账本、四fixture来源审计、C1与AA/AD端点和视频、逐scene人物/动作/边界检查、forward/commit/decode/总耗时/GPU峰/CPU KV bytes、可重建训练manifest（teacher C1 latent与C2 AA/AD endpoint）。大tensor仅原路径+SHA索引，Git只提交代码、日志、小证据与视频。

本任务验收是“有来源且质量边界明确的教师目标准备”，不是AnyFlow训练完成。后续student应使用冻结teacher C1 latent并按当前student权重重建KV；从student自己C1采样属于另一个评估条件。不得因目标生成结束自动训练32update或启动DMD。最终所有失败样本与有效样本均记录，Judge决定下一任务。

## P0通过与P1放行

Judge独立CPU预检和实际runtime摘要核对PASS。已签发EXP-014/judge/P1_APPROVED.json，仅GPU0编码四图，12text+4image encode/600GPU秒；0denoiser/0decode/0训练。T1/T2仍未授权，四fixture需独立审核。

## P1通过与T1放行

四native fixture Judge独立CPU审计PASS，12text/4image编码实际110.214GPU秒、峰40.578GiB。已签发T1_APPROVED.json，仅首图s0_43866101、GPU0、91forward/3decode/1350GPU秒。T2仍待第一图完整视觉和缓存验收；训练0。

## T1通过与T2放行

首图全部73张唯一帧观察（39+17+17）、实际50层KV及完整视频核查PASS，人物/场景/动作分叉有限可行，quality PARTIAL。91forward/3decode/532.261GPU秒。T2_APPROVED.json授权余三固定图，各91forward/3decode/1350秒，GPU0/1/2实时空闲时执行；无训练，逐图验收。
