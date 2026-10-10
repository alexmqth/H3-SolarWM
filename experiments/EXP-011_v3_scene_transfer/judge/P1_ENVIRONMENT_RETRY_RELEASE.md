# EXP-011 P1模型目录修正与attempt3授权

2026-10-11 06:00 HKT。attempt2因独立P1没有import infer所设置的模型根目录而解析出空权重列表，0模型调用，已终止；其日志/result/预算和marker原样存attempt2_archive。累计P1账本8.667172311秒，所有模型调用计数仍0。账本包括Worker明确标注的保守失败启动开销，原始账本保留。

Judge CPU使用相同ModelConfig、禁止下载、显式DIFFSYNTH_MODEL_BASE_PATH核查通过：14个已存在文本权重分片、1个VAE文件、processor目录正确指向既有H3模型。证据P1_MODEL_PATH_CPU_CHECK.json。

批准attempt3一次编码启动，命令环境必须按P1_APPROVED.json显式指定冻结模型根。代码manifest不变（11363412837dfdca85a611761fc1a59b730cb2c541117942fbbcf14f24866976），同6text/2image encode、0denoiser/0decode、累计900秒；不重置失败账本、不改图/文本/协议。G1/G2仍未授权。此修复只处理资源解析，不能作为模型能力证据。
