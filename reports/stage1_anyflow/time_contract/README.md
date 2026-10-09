# C准备：真实H3时间权重与逐token时间映射核查

2026-10-09 00:57，CPU检查完成。**只验证初始化时间条件和生产预处理；不是训练后velocity保持、可靠causal checkpoint、视频效果或C阶段完成。** 未启动新GPU/训练任务，没有改变正在运行的B密度对照。

## 实际检查范围

从本机Original H3 safetensors只读取四个FP32时间MLP张量，维度为256→5376→2688。发布action LoRA没有time_embedder键。四个张量hash与正在运行的真实FM分支precision收据相同。相关源码中4项与其冻结runtime逐字一致，`h3_cached.py`仅注释/docstring不同、去除docstring的完整计算AST一致；[来源核验](training_provenance.json)和[文本差异](h3_cached_comments_only.diff)保留。首次按完整源码hash相同断言未通过，检查差异后改为上述有范围的核验，没有忽略计算变化。这不是读取活跃GPU模型内存。

使用真实ABot验证片段`118eb5d8b75e1b8ac23a4e9ae77af9a9_A_1140`的causal双RGB-anchor packed layout，通过真正的`chunk_forward`和`model_fn_minimax_h3`构造时间条件，在进入transformer前截取：native shift2.22和uniform两种网格×4/8步×3chunks×diagonal/finite共144例，另加4例完整10-latent clean-history前缀与sigma=0端点。合计148例，属于同一数据布局的协议用例，不是148个独立视觉状态。

逐token检查包括text/action、双RGB anchor、audio、clean history、当前video和padding；分组无重叠且覆盖全layout。缓存路径的历史不在当前输入行中；额外full-prefix用例只验证显式clean-history行时间，不证明两种attention/KV语义等价。

| 行类型 | 当前native time | 目标native time |
|---|---:|---:|
| 当前video | 1−sigma | 1−rho |
| prompt和action text | 1 | 1 |
| RGB anchor | 0.999 | 0.999 |
| 固定audio noise | 0 | 0 |
| 显式clean history | 1 | 1 |
| padding（如果存在） | 1−sigma | 与当前时间相同 |

在sigma=1时audio与video的当前time均为0，但有限映射下目标时间不同；实际代码按(t,r)二元组去重，未将两者误并成一个时间条件。

## 数值结果

- 所有native current/target时间处于[0,1]，没有把模型的输入错误扩大成[0,1000]。
- 43种唯一时间对的实际MLP norm/混合变化见[CSV](embeddings.csv)。base embedding范数范围0.473853–1.466413；这里只是数值记录，不是质量阈值。
- target MLP初始化逐tensor等于base；本次FP32数值中，r=t的混合embedding与base最大绝对差为0。
- 使用SolarWM实际`H3AnyFlowConditioningMixin._time_condition`和同一个集成H3 embedding作公式对照，所有混合输出最大绝对差为0。该项检验clone/gate混合；原生Diffusers正弦实现另有下方独立数值对照。
- 故意将native time乘1000的负对照，相对embedding误差189.426，说明本检查能区分该尺度错误；这是故意错误输入，不是当前模型存在该错误。

[完整收据](audit.json)包含源码hash、真实权重hash、逐例时间对/行数/时间hash和norm；[检查脚本](audit_time_contract.py)可从本机源数据重现，提交包不复制原始权重和latent。

## 01:06补充：原生Diffusers正弦投影与MLP数值对照

检查SolarWM环境安装的`transformer_minimax_h3.py`，其文档明确指定unscaled `[0,1]`，构造器使用`Timesteps(freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0)`。从同环境`embeddings.py`提取并原样执行`get_timestep_embedding`、`Timesteps`、`TimestepEmbedding`三个AST节点，未改函数体，避免导入完整transformer/CUDA依赖。

将同四个真实FP32权重严格映射到原生MLP的`linear_1/linear_2`，在前述网格覆盖的全部17个唯一native时间上，与项目集成的`MiniMaxH3TimeEmbedder`比较：**输出逐元素相同，最大绝对差0**。实际SiLU类型、无额外condition/post-activation及constructor参数均检查。

[数值比较收据](native_time_comparison.json)保存原生源码hash、17个时间值和结果；[脚本](compare_native_time.py)保留原始AST执行方法。这为初始化时间投影尺度增加独立证据，不声称BF16/GPU/训练后数值同样逐bit一致。

## 初次fixture修正与证据边界

第一次检查误用了Original单anchor packed布局配合causal双anchor张量，在进入transformer前触发shape error。修正检查器为训练/评测实际使用的`raw['causal_packed']`后通过；原训练入口本来就使用这一字段，无生产修复。[初次失败](initial_fixture_failure.json)与当时源保留。首版通过后补齐padding及全layout覆盖断言再次通过；旧收据与对应源仅作审计历史。

这里没有执行transformer、attention、KV生成、VAE或optimizer。因此没有测u(z,t,t)是否在训练后保持，也没有测finite-map composition、目标正确性、动作响应或画质。新C训练仍须等B给出可信局部field；这些项目没有被本报告替代。
