# AnyFlow128：动作仍有独立响应，teacher-history分离度不能直接当控制保真

[完整39帧固定历史动作干预视频](fixed_history_action_intervention_39.mp4)：列为当前A/D，行为先前A/D。仅chunk1用于同历史归因；保留全部前后画面并标明teacher history，非free-running。两条新增switch视频以及四种组合的12/20/26/33帧已静态查看，MP4完整39帧H264/yuv420p/24fps解码与第26帧标签检查通过。

## 第二块的受控动作响应

| 同一teacher prefix和先前动作 | 当前A，RGB[17,34) flow | 当前D，RGB[17,34) flow | A−D |
|---|---:|---:|---:|
| A history | +0.692521 | +0.378091 | +0.314431 |
| D history | −0.427330 | −0.854157 | +0.426827 |

同一A历史把当前A换成D，flow下降；同一D历史把D换成A，flow上升。两种干预均有方向一致的相对影响，动作不是完全失效。实际chunk1 latent也变化，max_abs分别2.450238/2.399653。

但两个动作在同一历史下尚未产生相反的光流符号，画面也以原历史构图/人物状态为主，变化幅度有限。缺少Original H3在同一历史上的counterfactual参考，不能将动作惯性与控制不足严格区分；不能单靠绝对符号直接认定切换错误。**已有不同teacher-history A/D的1.333539包含历史状态差异，不能用来宣称当前动作的同状态保真已恢复。**

## 实际控制变量核查

128同权重、8步/块、sigma网格、CPU KV、RGB dual、action feedback保持。新增A→D→D与D→A→A分别对照已有A→A→A与D→D→D。实际conditioning：initial/video-audio noise、anchor、基础prompt及前5行action embedding相同，后续action embedding确实改变；packed tensor相同。第0块5个latent在每对分支中逐元素相同，max_abs=0。证据evaluation_A_to_D.json和evaluation_D_to_A.json。

这是clean prefix相同、前一动作相同的对照，不仅是从两条不同世界状态生成A/D。只对chunk1归因：生成chunk1后用改变后的action作clean commit，chunk2的历史KV可能不同。没有实测KV tensor哈希，不将首块latent相同扩大成直接KV逐张量审计。RGB区间仍受temporal VAE影响，不能当独立latent chunk隔离；跨边界transition单列，整片flow只作描述。

完整静态帧显示teacher状态约束下人物/车库更完整，switch与held分支有可见但有限的姿态/位置差异；两者仍有teacher-history边界重置，不能称连续交互rollout通过。完整逐transition与视频哈希见[counterfactual_response.json](counterfactual_response.json)。

## 新增两条的开销与常用指标

| schedule | E2E s | allocated MiB | CPU KV MiB | gray MAD | boundary RGB MAD |
|---|---:|---:|---:|---:|---:|
| A_to_D | 264.68 | 38984.16 | 6484.13 | 4.3005 | 13.4327 |
| D_to_A | 255.57 | 38984.16 | 6484.13 | 3.8825 | 11.4229 |

每条24 noisy+3 clean commits；共享主机单次时间，不作speedup声明。MAD是活动量，不能当画质。

Stage1目标仍未完成。本轮没有模型/训练修改，也没有Stage2。下一项同权重诊断仅把AnyFlow模型的目标时间条件r改为当前t，实际8步积分网格不变，用来区分有限步预测的不足与共有causal/generated-history问题；它是推理消融，不可将其结果冒充正常AnyFlow finite-map收益。
