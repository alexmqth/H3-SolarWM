# EXP-012/v1 Judge 最终验收

2026-10-11 07:06 HKT。**任务完成并停止扩展；工程/协议PASS，当前AF2 step32未通过两个固定其他场景的联合迁移验证。** 工业动作切换PARTIAL，村落AA后半C2人物/前景结构FAIL；村落AD有限可用。没有稳定优于普通FM8的联合收益，优先保留普通FM8作为低成本候选。不改写停车场AF8的旧有限可行结论，不推广为全部AnyFlow方法失败。

## 实际范围与成本

唯一AF2 step32配对QKV/target-time（blocks42–49/rank8/scale.125、gate.25），零新增训练。每场景复用EXP-011 FM8首39RGB/12-latent C1，以AF自身权重重建clean C1 KV，再匹配8NFE分别AA/AD续写17帧到56。没有AF首窗、C3、重试、额外场景或编码。

两场景总 **34forward=32sampling+2clean commit，4decode，354.403244 GPU秒=0.098445345 GPUh**。工业168.848117秒，村落185.555127秒；allocated峰26.229722GiB。加载、commit、sampling、decode、保存分开记录，阶段账本包含其外层输入准备成本；不能与常驻服务E2E混为一谈。见[最终预算审计](FINAL_BUDGET_AUDIT.json)。

## 独立审计

Judge独立CPU预检通过后逐场景放行。最终两套真实AF cache均加载，全部50层精确index0、max_history5、detached CPU、每套6,799,104,000 bytes；对比各自FM8真实cache，全部50层K/V不同，RoPE逐值一致。代码确保先安装AF权重、再sigma=target_sigma=0 clean commit，采样不修改历史。checkpoint/source/代码摘要冻结，无未来动作或prefix/time协议变化。

共同C1/噪声/action prompt/native8网格/video位置与FM8对照匹配；AF采样使用相邻source/target sigma。两场景四条视频旧39RGB逐值不变，端点finite、56帧/17帧24fps完整解码、递增PTS均通过。详细见[G1工业审计](G1_industrial_audit.json)、[村落审计](G1_village_audit.json)。这些是实现证据，不能代替能力判断。

## 视觉和动作判断

Judge检查全部68新增帧、两个场景FM8/AF8四格原分辨率末帧，以及村落AA frame47原分辨率成对比较。见[逐帧记录](visual_notes.json)。

| 场景 | FM8 AA / AD辅助flow | AF8 AA / AD辅助flow | 结论 |
| --- | --- | --- | --- |
| 工业 | +28.20 / −38.27 | +28.14 / +11.07 | 画面与持续A可用，AA/AD有差异；AD在当前17帧内没有清楚反向响应，切换PARTIAL |
| 村落 | +87.82 / −44.23 | +70.33 / −68.63 | 动作方向分叉可辨；AD人物/场景可用，AA后半块人物/前景出现大片分叉透明条纹，结构FAIL |

flow沿用EXP-011中央ROI帧间中值位移在16次转移上的累计，仅辅助语义观察；不按绝对值排名、不混用停车场指标。工业弱切换不是靠一个反号机械判定：完整AD新增段和末帧对照均不支持与FM8同等清楚的反向响应。村落AA的失败来自持续结构破碎，区别于FM30树木完整遮挡后人物重现；建筑背景仍在，不称全画面崩溃。

边界RGB MAD：工业AF8 AA12.46/AD13.16（FM8约11.90/11.02），村落AF8 AA12.40/AD18.19；只是辅助，不以边界小幅变化独立否定路线。

## 决策与边界

当前停车场有限训练的step32在两个固定其他场景存在退化。冻结该checkpoint的本轮负结果，不增加C3、训练步数、seed/prompt或checkpoint搜索。普通FM8已在同场景同C1提供更稳妥的有限基准；AF8不升级为正式主线。

本轮是共同FM8 C1后的匹配历史模型比较，两模型各自KV，不是从首窗全程AF、相同raw KV消融或广泛统计泛化。只有两个预先固定validation初图和seed13；不能据此否定多样化数据、充分训练后的AnyFlow。

下一项优先做**多场景训练数据与独立验证方案的CPU准备**，查清可用数据、episode隔离、native Single I0重编码与teacher目标构建预算，再决定未来独立训练。保持当前AF4/DMD失败配置关闭，不在本轮追加GPU。
