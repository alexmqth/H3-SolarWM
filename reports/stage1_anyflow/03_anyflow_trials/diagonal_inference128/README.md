# AnyFlow128同权重：有限步预测与对角速度条件

已有正常推理使用模型v(z,sigma,target_sigma)，target_sigma等于下一求解点。本诊断只把模型看到的target_sigma设置成当前sigma（r=t），使用同一网络的瞬时速度条件。实际积分仍走原来的8步native sigma网格，更新到原来的next_sigma；所有128权重、initial image、动作、seed/video-audio noise、RGB dual、generated history、CPU KV不变。

不是重新训练FM，也不等于匹配FM训练对照；是同一个AnyFlow模型的推理条件消融。若对角版本改善而有限步版本退化，有限步映射学习/使用是候选原因；若两者都差，则不能只归因于off-diagonal条件。不把对角版本的成功标为AnyFlow finite-map成功。

wrapper复用原冻结benchmark，只替换其当前chunk前向的target_sigma，并显式记录新的sampler名称和27条实际时间对。clean commit保持sigma=target_sigma=0；步骤/前向数应保持24+3，source不修改、weights不修改。

A使用GPU0，D使用GPU3，启动前重新检查空闲，reserve6，不限25GiB。两个推理完成后停止；不启动新训练，不开始Stage2。run_A.json/run_D.json记录当前状态。
