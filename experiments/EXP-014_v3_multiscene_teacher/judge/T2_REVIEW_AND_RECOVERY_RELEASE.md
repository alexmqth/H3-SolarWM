# EXP-014 T2审核与v2恢复放行

2026-10-11，Judge。T2三图首39+AA各17帧、s2 AD17帧及原分辨率细节已看。s2实际G1/G2 cache/视频审计PASS、结构可用，但AA/AD背景同向，flow+68.385/+50.074，D切换PARTIAL。s1/s3的AD无端点/视频，原会话exit143（Worker报告）、进程确实不存在，故NOT_EVALUATED，不能判作画质失败。可见cgroup OOM计数0，终止来源未知。原账本和保守耗时上界保留，见中断审计。

50个T2原始副本逐字节相同、16个报告链接、s2成对视频56帧/24FPS/1664×552及原片内容对位通过，不以发布完整性代替生成能力。

v2恢复入口已独立CPU审阅与实物核查PASS：两套真实50层C1 cache各6,799,104,000 bytes、C1/AA/原中断记录摘要、原AD noise/prompt/Global位置/native30网格一致；不重建C1、不commit、不覆盖旧AD。恢复复用原v1采样与解码函数，仅输出和账本独立，SIGTERM捕获写入失败记录。无marker拒绝通过。

审核后Worker只修改CPU预检报告的重复运行行为及增加code核对；Judge逆向该小补丁可精确重现先前runner SHA，证明采样/缓存/账本代码未改变。首次签发因此被SHA检查拒绝，未创建marker也未运行GPU。现绑定新manifest b294c0d1a55a81c374a0c1741f50e59127bad1c4102c0211a55ca0bef7c416f4签发RECOVERY_APPROVED.json。

仅GPU0顺序s1再s3，各30sampling/1decode、600GPU秒，总60forward/2decode/1200秒、0commit/0训练。第一条成功退出后才启动第二条；任何再次中断或协议错误即停止，不做第二次恢复。全任务预记账402forward/12decode、累计teacher5400秒上限。GPU0启动前确认实际空闲。

这次补齐只回答原固定数据问题；s2动作PARTIAL不重跑，后续训练未授权。
