# EXP-014 P0验收与P1放行

2026-10-11，Judge。P0独立CPU测试和import preflight PASS；实际运行Python的DiT/pipeline/text encoder/video VAE/scheduler加载路径虽指向旧editable checkout，五项实际源码摘要全部与冻结代码清单一致。来源四train图、排除validation、future rows裁剪、own-action/cache50层index0、分scene账本和无marker拒绝通过。

审阅与EXP-011编码入口的diff：仅来源数量/预算/命名/资源入口改变，原生编码计算不变。G1/G2同进程边界补充gc和empty_cache，并检查残余GPU allocation，避免重复模型生命周期。代码清单SHA `5afe075f4db7e5b6b107ffabbfcd1feb408dbccd173649b6548c74f876173331`。

现仅批准GPU0 P1：12text+4image encode、600GPU秒、0denoiser/0decode/0训练；当前GPU0空闲。四fixture必须独立CPU审计后再发T1 marker。见[P1授权](P1_APPROVED.json)、[独立P0证据](P0_INDEPENDENT_AUDIT.json)。
