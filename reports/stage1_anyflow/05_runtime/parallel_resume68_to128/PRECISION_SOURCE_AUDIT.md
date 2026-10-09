# 残差与时间调制精度：只读源码核查

2026-10-08。核查期间既定96→128训练仍运行；未改变任何运行时、权重或配置。

本次未发现“官方使用FP32残差流，而当前错误地使用BF16”的依据。SolarWM的H3 adapter导入Diffusers的`MiniMaxH3TransformerBlock`；本机`solarwm-h3`环境的对应实现将文本、视频、音频投影结果打包为文本流dtype，原checkpoint中为BF16。主block的scale/shift、gate乘加和残差均没有显式升到FP32；token refiner也是普通残差相加。

官方的区别在输入/输出/时间边界：六组projection保持FP32，time embedding及其SiLU在FP32计算，然后才转为AdaLN线性层的BF16。当前`h3_fp32`策略已恢复原生12个FP32权重张量，保留FP32 noisy input/time/SiLU，再在projection边界转换；主transformer保持载入dtype。因此此次源码核查没有支持再做一个“全FP32残差”训练分支。

这只是精度策略的源码核对，不是两套33B模型逐张量数值等价的证明；不同attention/融合算子与H3-World action扩展仍可产生差异。此前真实33B精度探针及视频评测保留，未重复运行，也不将本结论视为画质修复。

定位：

- `SolarWM/src/solarwm/backends/minimax_h3/model.py`：导入原生block、模态打包、调用block。
- 本机`/home/lpeng/miniconda3/envs/solarwm-h3/lib/python3.10/site-packages/diffusers/models/transformers/transformer_minimax_h3.py`：103–158时间调制；250–277 refiner；319–373主block；625–642打包与时间条件。
- `H3-World/code/causal/h3_precision.py`：原生FP32恢复及AdaLN输入hook。
- `H3-World/DiffSynth-Studio-h3-v2/diffsynth/models/minimax_h3_dit.py`：`_modulate_scale_shift`、`_modulate_gate`、`_embed`。

对应源文件哈希见`precision_source_hashes.json`。
