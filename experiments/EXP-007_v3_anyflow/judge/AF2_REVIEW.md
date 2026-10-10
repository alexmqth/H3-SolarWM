# EXP-007/v3 AF2 Judge审核

2026-10-11 03:21 HKT。**接受累计32updates的有限训练工程结果，生成能力交AF3视频检验。** GPU1 PID4087679已退出，Worker命令exit0，账本正常关闭。

AF2新增31updates，527显式forward、124backward、0VAE，GPU占用4287.860705614秒（1.191072418GPU小时），peak allocated26.767441273GiB，reserved27.318359375GiB。低于2GPUh/44GiB限制。内部gradient-checkpoint重算不单独计为采样NFE，反向及其耗时已包含。

Judge运行audit_af2_completed.py独立核对：完整step2至32顺序；偶数D/奇数A；每update17forward/4backward；4样本类型/合法时间/finite loss与梯度；历史KV重建与只读；运行源码摘要未变；step8与step32三文件SHA、元数据、optimizer step及RNG配对。全部通过，见af2_completed_audit.json。没有因中途loss波动追加配置或筛选checkpoint。

最终checkpoint：H3-World/outputs/EXP-007_v3_anyflow_af2/step_32。target SHA `ceb7d62324834a5b2be9fd6a47bfe4b5a9d810e49d1bb33889af0fb788940f70`，QKV SHA `07c8e5e68d1c947e60217e326ca8dd4c00222a555542b286dfc03514e27e916e`，optimizer/RNG SHA `2fda88175634075b1b4feda209337ff1281e4007ce25dc540eb1992704c523b1`。

本结果仅支持有限训练流程正确执行。训练数据是冻结V3生成的C1/C2 AA/AD，单scene；不能由32updates或loss声明画质提升。现在批准预定义AF3最终checkpoint匹配8NFE评估，35forward/4VAE/0update/0.35GPUh；前39帧为共同FM8历史，不称全程AF首窗已验证。
