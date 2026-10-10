# V3 DMD真实模型pilot草案 — 尚未授权GPU

2026-10-11 Judge。先完成AF3视频审核；本文件仅将已通过的CPU协议转化为可预算的后续实验。编号/最终配置/预算由下一任务书确定，不能凭此启动。

## 问题与角色

首先只验证真实33B在当前V3 causal协议下，是否能对8个finite maps组成的生成链完成一次DMD更新，测显存与成本。teacher是冻结V3 causal模型，不能称原生双向Original H3 teacher；fake-score是独立普通velocity模型及新QKV adapter，无target-time模块；student使用AF3评估过的配套AF2 target/QKV。

三个角色的基础权重/released Action LoRA冻结。C1为EXP-006 FM8 clean first12，动作/I0/音频条件使用冻结输入；当前C2训练initial noise用独立CPU generator seed170008新生成，score-noise使用另一独立CPU generator seed170108。后续视频评估仍使用冻结seed13 slice，避免pilot直接训练评估噪声。pilot可固定AA，只证明当前块on-policy；没有对完整自生成历史分布训练。每角色必须以其自己的当前权重prefill C1；不能跨模型复用raw KV。

## 设备与计算图

优先三个实际空闲GPU分别放teacher、fake、student，单进程显式设备上下文或独立进程传递detached评分均可。GPU3/4他人进程不动。48GiB卡的配置沿用现有offload和≤44GiB allocated限制；每卡独立峰值计数，不重置。CPU权重可复用文件，不能复制checkpoint本体。

student真实8map链在其设备上保留完整计算图，沿用checkpoint/offload；fake训练只消费其detach endpoint。teacher/fake评分结果detach后转回student设备形成surrogate。不要在student checkpoint反向前替换其共享模型权重，不可为了省显存断开map间梯度。

三个占用设备的GPU时间按各设备实际占用累计；若从启动到结束一直保留三个角色，则保守计3×wall time，不能把三卡墙钟时间当单卡GPU小时。预算需包含加载、角色等待、保存及失败。首次只一cycle，建议≤30min wall/≤1.5GPUh，是否合适以正式任务书决定。

## 一个pilot cycle的可核对账本

1. 三角色C1 prefill：3forward。
2. student同native shift2.22的8map生成C2：8forward，保留图。
3. fake用detach endpoint做2次普通FM warmup/update；每次更新后重建自身C1 KV：2FM+2prefill=4forward，2backward/2update。teacher/学生不在这两次更新中改变。
4. teacher/fake对同一个detach+renoise endpoint、相同sigma=.6及动作条件评分：2forward。
5. DMD loss沿原8map图反向，student一次optimizer update：1backward/1update。

合计**17完整forward、3backward、3update、0VAE**。模型内部checkpoint recompute不另冒充采样NFE；应单独记录checkpoint设置与反向耗时。外层forward、反向、更新都预记账。

方向使用`fake_x0-real_x0=sigma*(v_teacher-v_fake)`，normalize沿用已测试实现。fake与student优化器相互独立；基础参数不动，scores/noise/normalizer detach。保存各角色权重/optimizer/RNG和父checkpoint身份；不保存每步巨大KV。第一次失败即交Judge，不自动追加重试。

## 完成后的决策

必须同时检查8map各中间velocity的有效梯度、student target/QKV更新、fake普通FM更新、角色隔离、cache来源/不变性、finite数值与显存/时间。工程成功只批准进一步考虑有限cycles和视频，不宣布画质提升。

若pilot成本合理，后续最多约8个追加cycles可作为候选：每cycle一次fake更新，每次角色权重更新后刷新自身KV，teacher cache可复用。实际调用数、lr、动作顺序和停止条件须重新冻结，不从此草案自动执行。匹配AF3条件的视频检验仍为能力判断依据；无可辨收益就收口，不进行lr或critic比例扫描。
