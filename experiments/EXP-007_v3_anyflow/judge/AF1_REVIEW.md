# EXP-007/v2 AF1 Judge审核

2026-10-11 02:05 HKT。**接受真实模型初始化与单步训练工程可行性；AnyFlow视频能力尚未测试。**

attempt1在模型加载前路径类型错误，0forward/0backward/0update，3.538686514秒。Judge批准明确修复后attempt2完成22forward/4backward/1update/0VAE，用时183.042843103秒。两次累计186.581529617秒（0.051828203GPU小时），allocated peak26.631452084GiB，低于0.75GPUh/44GiB限制。进程已退出。

真实33B初始化对角输出maxabs=0、relativeRMS=0；target/QKV梯度范数0.143264964/0.000082038889，均发生参数变化；基础参数版本不变。反向后cache身份检查通过，更新后同C1重新构建的KV变化。样本为2diagonal+1endpoint+1general map，遵循3 detached+1gradient调用。

Judge独立CPU核对step0/step1的六份checkpoint SHA、target/QKV/optimizer元数据配对、RNG、20项optimizer state及实际账本，全部通过，见af1_audit.json。CPU审计首次使用当前虚拟环境缺少的hashlib.file_digest报错，改为标准流式SHA后通过；未启动任何额外GPU任务。

接受有限训练工程基础，不由一次更新、loss或finite forward宣称生成能力。发布EXP-007/v3：新增最多31updates，匹配8NFE评估分阶段执行。
