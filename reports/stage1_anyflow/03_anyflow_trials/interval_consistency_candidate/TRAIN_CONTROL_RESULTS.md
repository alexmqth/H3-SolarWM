# 原AnyFlow loss控制组136训练完成；视频评测进行中

这是128→136的8次原loss更新，无一致性辅助项。训练完成不等于Stage1画质/动作验收。

新增更新本身共1512.06s；含模型/anchor准备和前后验证的总wall 1888.10s；GPU allocated peak 40666.14MiB。共享主机单次结果，不作吞吐量比较。

更新部分128 current forwards + 8 physical clean commits + 32 differentiable-history forwards，不含validation或backward checkpoint重计算。全52块rank8 bank更新；visual/action/time保持冻结。

| action | sample type | raw128 | raw136 |
|---|---|---:|---:|
| A | diffusion | 0.06069667 | 0.06045168 |
| A | diffusion | 0.13547640 | 0.13505462 |
| A | endpoint | 34.42242050 | 22.48710632 |
| A | flow_map | 0.15970805 | 0.15901309 |
| D | diffusion | 0.09205692 | 0.09181879 |
| D | diffusion | 0.19222862 | 0.19186658 |
| D | endpoint | 15.18542194 | 14.10613537 |
| D | flow_map | 0.21104014 | 0.21045838 |

以上固定验证的sigma/r/type/weight一致；初始验证与源128相同。不能用raw loss改善替代自由生成视频。

预更新权重/Adam/RNG核查见control_preupdate128_audit.json；step132冻结/更新策略见control132_parameter_audit.json；sample_audit.json重建了129–136逻辑样本与CPU noise，并逐项核对时间对、动作/chunk以及128/132/136实际保存的logical RNG。实际GPU noise未在训练时直接记录，不能称为GPU逐张量比较。

运行中的控制器684856正在按既定顺序评测A8/D8/A4/D4。辅助组仍在同卡队列中，尚未训练，不能据此判断辅助项有效性。没有新训练预算或Stage2。
