# EXP-027/v1：回归 ABot 主线，复用证据确定下一项基线实验

2026-10-16 HKT。用户决定当前只在 ABot 上研究。Judge 授权原 Exp Worker 立即执行本轮 CPU 盘点与具体实验方案；SolarWM 官方参考支线停止，不继续查权限/下载/编译。保留已完成的 EXP-026 环境/加载证据。

## 目标

围绕 H3-World 语言动作控制与 V3 causal/KV，使用已在本地的 ABot 数据，找出下一项有决策价值的训练/诊断实验。读取 EXP-023/025 已有训练/指标/视频结论，不把过去小预算负结果当充分训练失败，也不机械重启相同配置追加几轮。提交一个优先方案，避免分支膨胀。

## 立即执行的 CPU 工作

1. 复用 EXP-023 的32train+2dev数据与实际编码/动作清单，给出常量/切换动作、C1/C2时间覆盖及source episode分组。核对关键manifest与现有文件存在性；不重复读取所有大tensor或重编码。明确两个dev已观察、不作为独立盲测，任何同episode重叠须标明。
2. 从 EXP-023/025 的既有日志检查训练暴露次数、sigma覆盖、实际更新幅度/梯度和固定probe趋势，区分随机训练loss与固定评估。已有证据缺失就标未知，不凭loss小降证明学会动作。不要重新审阅所有视频；优先引用Judge已完成的质量审核，必要时只CPU查看代表性失败。
3. 判断下一项最高收益的问题：是先证明少量清晰动作样本可被有效拟合，还是已有证据足以直接做GT/self-history × constant/switching诊断？两者择一给出优先推荐，并说明相对EXP-025究竟新增什么信息。若选择小样本拟合，采用真实多样sigma/noise与固定未用于更新的probe，不能把固定噪声记忆当泛化。若选择历史诊断，真实历史只能注入已完成块，不泄漏当前或未来目标。
4. 提交可执行GPU任务草案：固定样本ID与动作时间、checkpoint初始化、新训练参数/冻结项、检查点/更新数、采样NFE、明确的训练阶段与评测阶段预算、forward/commit/codec计数及已有实测吞吐估算。优先单卡L40与原h3world环境，能复用的视频不重生成。先一个方向，不同时做rank/learning-rate/AnyFlow/DMD搜索。
5. 验收以可行性/学习信号为主：动作响应或主体运动有清楚改善可继续；若无明确趋势则及时停止，不围绕1%左右MSE收益追加实验。人物细节有缺陷不单独否决；稳定性改善须同时检查运动是否减少。正式V3继续冻结。

## 预算与限制

CPU准备≤20分钟，≤4 workers/RSS≤32GiB、新增磁盘≤1GiB，0网络下载/0GPU/forward/codec/训练。新GPU预算由Judge根据可执行草案签发，用户已确定回归ABot方向，不重复向用户询问方向许可。原32clip实验不自动续训；不得加载SolarWM权重替代H3-World Action LoRA，不改原冻结协议以掩盖失败。

## 交付

DATA_SUMMARY.json、EVIDENCE_REVIEW.md、一个优先GPU_PROPOSAL.md、WORKER_REPORT.md及根report.md。存在不确定性如实简述，20分钟内给出可决策结果，不做新框架/大数据工程。消息Judge，Worker不操作Git或progress/next_plan。
