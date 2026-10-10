# DMD真实模型pilot代码准备初审

2026-10-11 02:32 HKT，Judge。当前为CPU准备；没有DMD GPU授权、实际33B调用或显存结论。

已读run_pilot.py和config_pilot_prep.json，并核对其执行顺序与草案：三个独立角色/设备，student保持真实8map生成图，fake只使用detach端点做2次普通FM更新，teacher/fake在同renoise端点评分；合计17forward/3backward/3update/0VAE。各角色自身clean KV、fake更新后重建、student与teacher历史只读，target/QKV参数变化和角色梯度隔离均有检查。

实现已补上固定adapter/RNG初始化、实际导入DMD模块路径和SHA、角色设备下的反向/optimizer、clip1、分组梯度/参数变化、末尾显存/时间预算检查。预算保守按三卡占用的3×wall累计；未设置任何授权marker。

Judge独立以CUDA_VISIBLE_DEVICES空运行--preflight，真实结果为WAITING_FOR_AF2_AND_AF3，gpu_calls=0。这表示正确等待最终checkpoint和视频证据，不应写成真实模型pilot通过。数学和小模型协议证据在同目录check_protocol.py/result.json；实际多卡加载与8map反向的显存、耗时仍需下一独立任务验证。

下一步：AF2完成、AF3视频经Judge审核后，决定是否签发独立DMD任务编号、冻结最终config/source/input与单cycle预算。当前目录是准备位置；正式实验需明确归档路径，不能覆盖V3/AF结果。普通FM8已可行，应以真实视频及训练成本判断AnyFlow/DMD的ROI，不仅看loss。
