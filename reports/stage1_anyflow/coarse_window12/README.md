# E1：12-latent局部窗口跨历史验证

2026-10-09 05:59：E1粗窗口两历史/三窗口与VAE审计已全部结束。首窗A/D恢复，但A历史第二窗同正、D历史第三窗同负；第二窗有明显肢体重影，局部门槛仍No-Go。未来RGB干预8项past latent差均0，不支持VAE泄漏根因。停止窗口扩宽，下一项只审查历史噪声/时间条件，尚未启动新GPU/训练。 [完整终态](FINAL_RESULTS.md) · [下一有限协议](NEXT_HISTORY_CONTROL.md)。下方运行信息为历史快照。

**2026-10-09 05:32新结果：** [同输入、同前17帧区间的首窗报告](FIRST_WINDOW_RESULTS.md)完成。匹配5latent失败；12latent首窗方向恢复且人物可辨。GPU4短对照已结束，GPU1/5后续窗口继续。不能把首窗通过当作多窗口接受。

2026-10-09 05:25。准备与CPU检查完成，三路真实H3已启动，尚未验收。[实验协议](protocol.json)、[公平性补充与预算](launch_protocol.json)。本组继续Original初始化、native时间、单I0、30steps，零训练；不进入E2/E3。

现有Original124f A/D的37latent输入经hash核验，首图/主prompt/布局/audio+video初始noise相同。全局位置固定且与未来A/D内容无关；每次在refiner前物理移除未知动作/video行。此fixture与旧39f的noise/layout不同，因此额外运行一次同fixture的首5latent A/D作为匹配控制，60次forward；coarse两history各三窗共360次，诊断共9次。第一5latent没有历史，只跑一份，不复制A/D reference。

真实VAE只读取12/24/36已知latent，实测分别输出39/81/120RGB；可见输出区间为[0,39)、[39,81)、[81,120)。每次追加新RGB，不回改已发出的帧。最终120f拼接是**oracle local forks**，历史来自Original生成而非GT，边界恢复reference；不能作124f自由rollout或长时稳定性证据。

CPU实际tiny-H3在37latent布局、三个12latent窗口、FP32/BF16上的未来动作/视频隔离、单I0位置、历史只读、当前动作干预和首窗T1/T2identity通过（2项参数化测试）。真实33B三路的首窗口T1/T2、repeat均0。它们只证明接口/执行语义，不证明动作能力。

GPU0因他人任务在free-memory guard处退出，未加载模型/未创建评测目录/没有forward；[转GPU4记录](first5_gpu_reassignment.json)保留。实际GPU为1/4/5，最多3张，source/runtime/运行中协议冻结。进程PID＋start_ticks已核实，快照见[launch_verified](launch_verified.json)；不能仅根据快照推断实时存活。

本组源码依赖上一轮冻结的305文件runtime（输出目录中为symlink）。提交包内复验源码时可令本目录`runtime`指向`../local_topology/runtime`；基础权重、cache和大tensor输入未打包。实际GPU运行仍使用项目outputs下的完整输入与runtime。结果将保存每窗A/D视频、全部静态帧、实际A分支solver0/15/27状态。以后geometry须在同一捕获状态fork两动作，不能相减A/D各自轨迹。

尚缺完整三窗口视频/动作评审、匹配短窗效果、后续同状态geometry和GT局部画面。T2没有persistent hidden KV，需计入历史/prefix重算成本；不拿step数或当前单次wall宣称加速。旧会议视频未更换，未推送。
