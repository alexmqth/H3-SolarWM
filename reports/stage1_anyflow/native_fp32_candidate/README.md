# Stage1 native-FP32 candidate：单次GPU更新完成，初始化视频仍失败

本分支针对已确认的数值协议差异。官方H3 Stage1保留输入/输出投影及时间MLP的FP32权重和计算，并将noisy/plus/minus保持FP32；旧本地路径将这些base权重及输入扰动提前转为BF16。前面的同状态精度诊断结果混合，因此没有把精度修改称为画质修复。

这里使用独立runtime和显式`--precision-profile h3_fp32`：

- 从原始MiniMax-H3 safetensors恢复六组线性层的12个FP32权重/偏置，而不是只把已舍入的BF16值转回FP32。记录逐张量SHA256。
- 时间MLP、AnyFlow `(t,r)` 混合及SiLU保持FP32；AdaLN投影前转为其BF16计算dtype。
- 训练clean/noise、插值和有限差分扰动保持FP32，直到FP32输入投影完成；文本/packed hidden和主transformer保持原加载dtype。
- CPU offload的offload/onload/preparing/computation dtype同时设置，避免VAE/DiT切换后丢失精度策略。
- frozen target-time MLP、tail16 QKV rank8、action residual、RGB dual anchor、chunk5、history5、shift2.22、A/D39f等保持当前协议。没有新增anchor或attention结构。
- 原有`legacy`路径保留，旧checkpoint缺少精度字段时解释为legacy。改变精度必须建立新实验；精确恢复拒绝精度变化。推理还核对原生FP32权重哈希。

FP32训练保留了原随机种子的底层抽样，但不再将训练噪声幅值舍入成BF16，因此不是每个训练张量都与legacy位相同。推理首帧/prompt/action rows/video/audio initial noise仍按同一旧流程生成并逐张量对照teacher。

CPU验证：17项专项通过；覆盖原生FP32值恢复、实际offload wrapper、BF16主干的FM/AnyFlow反传、FP32扰动保留、KV只读、官方loss/gradient对照和错误精度拒绝。另验证旧legacy checkpoint恢复与旧连续训练逐bit相同，以及FP32分支连续4次vs2+恢复至4次的adapter/Adam/RNG/loss/gradient逐bit一致。目标时间MLP保持冻结。证据为preflight.json和cpu_training_validation.json。这不是33B GPU或视频质量通过。

06:21调整后的有限队列：legacy训练到32及A/D评测已结束且失败，64次分支在第33次更新前停止；等待独立FP32输入诊断完成后，本分支从相同visual/action初始化进行1次真实更新→step00 A/D4诊断→精确恢复本分支Adam/RNG到16→step16 A/D4和A/D8。若已有duration control通过numeric gate则先等待人工视觉审查，不继续加训。任何错误或输入/解码/原生权重校验异常都会停止。

全部使用独占GPU0、reserve6、CPU raw KV。当前benchmark的FP32分支仅支持`--modes cached`，原始双向baseline继续保留旧复现入口；不能宣称已验证原始pipeline的全FP32运行。结果不自动替换会议demo、不自动接受画质，也不启动Stage2。

最新真实GPU与视觉记录见[GPU_RESULTS.md](GPU_RESULTS.md)，run.json为保存时的队列快照。源实验runtime/保持隔离；native_fp32_runtime.patch是合入前相对旧主代码的差异。可选数值策略已合入主源码与submission，默认legacy不变，运行中副本没有改动。复现时在AnyFlow patch之后应用code/diffsynth_native_fp32.patch，训练/评测显式设置--precision-profile h3_fp32。
