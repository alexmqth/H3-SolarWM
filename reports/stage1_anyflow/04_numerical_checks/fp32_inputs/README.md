# AnyFlow有限差分：保留FP32输入扰动的独立诊断

官方Stage1的stage1.py把clean/anchor/noise设为FP32；anyflow_loss.py的noisy/plus/minus也保持FP32，直到FP32输入投影之后才进入BF16主block。本地旧路径在loss回调、chunk_forward及_embed的输入处会转BF16。前一个precision_probe只改六组线性层及time混合，仍保留这个输入舍入，因此不能覆盖完整的输入精度差异。

本目录使用独立runtime副本，不修改正在运行的冻结对照或共享DiffSynth源码。只给chunk_forward和_embed添加可选的输入dtype标志，默认路径完全保留。诊断绕过loss内部的BF16样本转换，直接在相同noisy/plus/minus上比较：

| Profile | 六组线性层/time混合 | noisy/plus/minus到输入投影 |
|---|---|---|
| legacy | 旧BF16路径 | 旧BF16舍入 |
| boundary_fp32 | FP32计算 | 仍先BF16舍入 |
| boundary_fp32_inputs | FP32计算 | 保持FP32扰动 |

三者复用同一个frozen16 checkpoint、旧clean CPU KV、同一噪声幅值和同一有限差分方向。保持原来已BF16舍入的base权重值；没有重新加载原生FP32权重，也没有把旧BF16噪声替换成另一条随机序列。这是输入精度的诊断，不是完整官方recipe复现，尚未用于训练或视频交付。

CPU实际小H3与offload wrapper验证已通过：给定1+1e-4的FP32视频状态，原路径和仅FP32边界计算保留0/1920个原始扰动值，新输入路径保留1920/1920；三个分支均有有限非零QKV梯度、history cache只读、退出profile后权重和输出逐bit恢复。这证明显式小扰动不会在该入口被提前抹掉，不代表真实训练的全部有限差分都曾丢失。既有AnyFlow集成回归6 passed；官方loss公式6个参数化检查未重复执行。

GPU0真实33B诊断已完成；24项测量、8次逐bit恢复、16项重叠control测量完全复现。结果混合，详见[RESULTS.md](RESULTS.md)。32次训练已完成，64次续训已暂缓，本诊断随后串行执行。patch_receipt.json与manifest.json记录隔离副本及checkpoint来源；模型/数据路径依赖当前工作区，提交包中的脚本是运行证据。
