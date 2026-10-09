# 匹配训练量的FM32对照（full history / train shift12）

GPU3/reserve6，从与shift12 AnyFlow相同的原始visual/action/full-bank初始化开始，训练普通FM到32次，随后A/D39f的4/8 steps/chunk。没有新增模型模块；FM不安装AnyFlow目标时间MLP，使用普通FM loss/Euler，而AnyFlow使用时间区间条件/有限差分loss/finite map。这是需要比较的目标与采样差异。

保持全50+2块rank8 QKVO/FFN、43,237,376 trainable parameters、native-FP32、full clean-history gradient、teacher A/D两条39f、logical batch4、LR3e-5、训练shift12、validation/inference2.22、原image/action/prompt/noise seed13、RGB dual、chunk5/history5、CPU raw KV、causal prefix/feedback。

先做32次，不自动扩64。CPU重放预测完整32次sigma/noise序列，前16次sigma/r已对照真实AnyFlow记录；运行中核验三套初始adapter及CPU/CUDA/logical RNG，训练结束后核验全部32次action/chunk/sigma与预定序列。对FM只匹配sigma，target_sigma本来就是sigma。预测noise hash只作计划记录，不能冒充实际noise hash检查。

比较时必须使用AnyFlow step32；与step16/64结果只能作不同训练量诊断。相同更新数仍不等计算量，AnyFlow有额外target/finite-difference前向；不宣称等计算budget。训练/视频检查不自动构成Stage1验收。

该实验是补齐效力对照：不能用AnyFlow32/64对FM16的差异宣称AnyFlow收益。无124f扩展、无Stage2、无新anchor/架构。控制器依赖源机器冻结runtime，跨机器复现使用通用trainer/benchmark入口。
