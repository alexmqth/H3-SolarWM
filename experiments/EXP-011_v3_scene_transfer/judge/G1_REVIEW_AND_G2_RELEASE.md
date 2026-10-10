# EXP-011 G1首窗验收与G2逐配置放行

2026-10-11 06:15 HKT。四个G1首窗已完成，Judge看过全部156帧和必要原分辨率细节。工业FM30/FM8人物/场景与运动可辨；村落两方法中段树遮挡后人物重现，FM8的颗粒感、树叶与人物交叠残影更明显。四配置首窗有限可行接受，quality PARTIAL；本阶段不宣称A/D切换或persistent KV续写通过。

独立CPU核查四条实际native sigma、源fixture/初噪声/prompt/Global坐标、FP32边界、端点/RGB/video SHA、39帧24fps全解码及PTS全部PASS。实际76sampling、0commit、4decode，推理账本577.338099311秒，allocated峰25.090662GiB；P1另外70.564697678秒。详细逐配置审计和visual_notes见judge目录。

**现在分别批准industrial/village × FM30/FM8的G2。** 每配置提交自身C1 clean KV一次，再共享同C1 KV/尾噪声生成AA与AD各17新帧至56。两个FM30各61forward/2decode，两个FM8各17forward/2decode，G2合计156forward=152sampling+4commit、8decode。全任务推理232forward/12decode、4500GPU秒及09:00截止不变；GPU0/ABOT_VRAM_RESERVE_GIB=18等环境与G1保持一致，真实空闲检查、44GiB显存和60GiB磁盘底线。

每个配置有独立G2 marker，绑定已验收首窗row/endpoint/RGB SHA。不覆盖首39帧，检查实际内存KV签名和全50层indices。看完整新增画面与A/D差异，光流仅辅助。一般缺陷PARTIAL；协议/数值/资源问题或持续结构崩溃停止后续启动，无C3/新训练/调参/自动重试。
