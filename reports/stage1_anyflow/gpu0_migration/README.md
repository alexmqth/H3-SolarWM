# GPU2 → GPU0 迁移与后续显存预算

用户指定GPU0已空闲。原GPU2分支在step08完整保存后停止本实验的训练/等待控制器，从同一step08在GPU0恢复到总16次更新。预留显存仍为20GiB，保持这一轮对照配置。真实恢复审计确认weights、Adam、logical/CPU/CUDA RNG、更新历史和teacher身份逐项相同；这证明载入状态一致，不是完整后续CUDA轨迹与未中断训练的对照。

GPU0已完成第9–16次更新和最终验证，目前正在做A/D视频评测。frozen16 native4 A=-1.218440、D=-1.221512、A-D=0.003072，画面后段雾化，仍未通过。当前主运行目录：H3-World/outputs/2026-10-08-05/stage1_anyflow39_gpu0/。

用户进一步允许使用GPU0更多显存。下一轮32/64-update条件续训及其评测设为ABOT_VRAM_RESERVE_GIB=6，约38.4GiB模型驻留预算。旧的reserve20等待队列已终止并标记superseded；新等待队列为H3-World/outputs/2026-10-08-05/stage1_anyflow39_gpu0_extend64_fullmem/。它仍在等待step16结果，未实际测得更高预算下的峰值。

源控制器作为运行证据归档，含源实验相对路径；通用复现使用STAGE1_ANYFLOW.md的训练/评测入口。所有短片质量gate仍待验证，Stage2保持暂缓。

05:29更新：高显存续训前插入独立precision probe，等待当前6条评测全部退出后串行运行；CPU真实小H3/offload/梯度/回滚检查通过，GPU结果待测。详见../precision_probe/README.md。

05:59更新：frozen16六条评测已完成，三组均未过动作gate。计算精度诊断已完成且结果混合，真实reserve6 allocated峰值36.85GiB。高显存队列已实际从step16继续训练到32，随后按gate决定是否64；独立FP32输入诊断排在有限队列结束后。参见../frozen_time_snapshot/README.md及../precision_probe/RESULTS.md。
