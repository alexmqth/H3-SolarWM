# Frozen-time AnyFlow step32：未通过，legacy64暂缓

39帧native8 A=-1.332148，D=-1.474533，A-D=0.142385；step16同设置分离度0.229330。A方向仍错误。两套都有后段模糊/重影，未见明确视觉修复。[对照抽帧](frozen16_vs32_AD8.jpg)保留同样的0/10/20/30/38帧。

从step16保留Adam/RNG续训16次，含准备/验证1316.6秒，allocated峰值38619.2MiB（37.71GiB），reserve6实际可运行。相同验证noise下A weighted loss0.116573→0.115909，但endpoint raw146.207→186.948；D endpoint17.867→18.614。不能用total下降宣称AnyFlow学好。

06:21，自动64次分支已在第33次更新前终止，step32权重/Adam/RNG和两条完整视频保留。因官方FP32输入协议差异已确认，优先做输入精度诊断及新的native-FP32训练。停止的是本实验自己的训练/等待控制器，见precision_priority_receipt.json；不是暂停整个Stage1目标。
