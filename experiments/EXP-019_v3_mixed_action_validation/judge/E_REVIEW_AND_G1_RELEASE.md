# E验收与首scene G1放行

2026-10-15，Judge。**ACCEPT条件编码**。独立核对归档文件SHA/字节、实际14次text调用/192.620秒、四fixture与真实I0/脚本及finite embedding、stop12/17未来embedding干预；阅读Worker的实际Qwen句子替换/变长future及历史坐标审计源码和结果。FFmpeg二进制SHA与CPU完整解码/短片编码回读通过。

新head来自真实新I0/caption，不能复用旧fixture的Qwen head；17句真实脚本及未知尾部固定占位均明确。配置.999在BF16实际=1.0，anchor逐值等于clean I0 rows；保持冻结数值行为，不声称存在0.1%注噪。E只验证条件，尚无真实混合动作生成结论。

## G1授权

仅scene43866101：真实prefix12 GT-history，用同一新fixture/初始噪声/真实动作分别FM-30NFE和FM-8NFE续写C2；39forward含1clean commit，2decode，0训练，单空闲卡，900GPU秒，allocated≤44GiB。Worker启动时前置已验收任务bin/ffmpeg并记录环境；外部watchdog执行wall上限。只前39GT展示+后17新生成，保存原decode及标注比较视频。

审核人物/场景结构、动作可辨性、ghosting与边界，允许普通细节PARTIAL；动作不可判定须如实写，不以单一光流代替。Worker提交所有新帧和资源报告后，Judge决定是否扩展余三scene。G2无marker，不自行启动。
