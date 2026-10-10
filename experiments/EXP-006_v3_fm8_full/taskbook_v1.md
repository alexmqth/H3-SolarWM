# EXP-006 / v1：V3-FM8全程普通FM减步验证

2026-10-11 01:36 HKT，Judge。**APPROVED：立即实施并执行有限GPU任务。** 用户授权Judge监控报告、完成后布置后续任务，持续至2026-10-11 09:00 HKT；期间不刻意限制GPU数量。此授权覆盖本任务，后续AnyFlow/DMD由Judge按证据分阶段立项，不再请求用户重复批准。

## 研究问题与固定协议

Mainline效率可行性，Parent为冻结V3 Original Feasibility Baseline。唯一研究问题：首窗也用普通FM8步后，是否仍有基本结构与A/D控制？EXP-004只测30步首窗后的8步续写，不能代替本轮。

Original H3 + released Action LoRA；Single I0、native timestep、own-action routing/action feedback/current non-action prefix feedback、Global RoPE、strict chunk causal、persistent raw KV、sigma0 clean commit、首12后5分块、seed13及已保存相同noise/audio/prompt、native FM shift2.22不变。无新训练/adapter、Local或anchor替换。首窗index0、空cache必须按同一原生路由正确处理；不要复用30步首窗输出冒充从零生成。

## 执行步骤

1. 复用EXP-004/005已验证实现，新增独立EXP-006 runner/config；CPU核查index0空cache、prefix反馈、visible action裁剪、原噪声、8步schedule、budget拒绝超限。不要修改冻结Baseline源文件。
2. 冻结runner/config/input/source manifest，明确来源SHA。必要CPU修复允许；生成前自行核对hash与本任务书预算，记录实际命令/PID/GPU。此任务书即Judge GPU授权，无需再次等待用户或额外形式审批。
3. 首12latent从原始noise生成8步A首窗39RGB；保存完整帧/endpoint；clean commit一次构建该新首窗的raw KV。首窗明显持续结构崩坏应停并提交，不机械消耗后续预算。
4. 同一新首窗cache/相同尾部噪声分叉AA与AD，各生成第二、第三个5-latent块到56/73RGB；各第二块commit一次。第三块接自己的第二块历史，不GT重置。
5. 比较已保存匹配30步参考和EXP-004混合首窗结果；注明闭环history差异，不能当同raw KV单状态消融。不新增30步参考GPU，若输入不匹配如实报告。

## 资源和停止

**最多40sampling+3commit=43完整forward、5VAE、0训练、合计0.75GPU小时，失败也计数，无自动重试。** 单任务先用一张实际空闲L40（建议GPU0），CPU最多4线程，allocated peak≤44GiB，绝对elapsed≤90分钟且不超过09:00。逐调用前记账、单进程锁、超时退出，未用额度不转作新实验。

今夜项目不设人为总卡数上限，可在不同已批准任务之间利用所有实际空闲卡；不要触碰别人的GPU3/4进程。增加本任务GPU数没有必要，后续训练按实际显存/并行收益决定。09:00后恢复项目最多3张卡，所有任务需支持截止/缩容，训练保存checkpoint后退出。

nonfinite/OOM/输入或cache错误/预算超限立即停止报告。普通ghosting或边界缺陷按可行性接受；持续主体崩坏/动作失效则停止，不扫描steps/shift或新增scene/seed。

## 验收与交付

- 完整首39帧和四段17新增帧，人物/场景基本可用，A/D方向和切换可辨；光流仅辅助。
- 精确chunk indices/全50层、采样历史不变、clean commit、旧RGB不可变；无未来action/video泄漏。
- 逐块sampling/commit/decode时间、forward计数、GPU peak、CPU KV/RSS、首屏和从零路径计时；共享首窗成本与分支分别列出，不称公平speedup。
- 独立目录保存代码/config/manifests/logs/metrics、原片、完整新增帧contact sheet、带清晰历史标签的代表性并排视频；大KV/latent不入Git。
- 完成后更新root report.md（task_id/plan_version/worker_status/PID/真实结果），并冻结worker_report.md。Judge独立检查视频/账本后更新progress/next_plan并发布Git。Worker不自行升级正式版本。

## 后续衔接

V3 Baseline124与SW-G158可行性已验收。EXP-006回答普通FM8首窗问题后，转入V3-AF的独立CPU/初始化验证和小预算finite-map训练；参考EXP-005/FUTURE_ANYFLOW.md，但使用新target-time student与student自身KV，旧AnyFlow权重只作历史资料。DMD在student基础能力与cache/time协议可用后另立有限任务，不能仅训练能跑就宣布成功。

当前唯一GPU任务为EXP-006/v1；后续任务由Judge按报告追加明确编号/预算。当前任务预算不包含AnyFlow/DMD。
