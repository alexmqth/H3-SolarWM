# 冻结time-MLP16次更新：39帧评测完成，未通过

| Sampling | A flow | D flow | A-D | Gate |
|---|---:|---:|---:|---|
| 4 steps/chunk, native | -1.218440 | -1.221512 | 0.003072 | FAIL |
| 8 steps/chunk, native | -1.250413 | -1.479744 | 0.229330 | FAIL |
| 4 steps/chunk, uniform | -2.632457 | -2.310146 | -0.322311 | FAIL |

native4/uniform4后段严重雾化和人物退化；native8结构更完整但仍有模糊/重影，且A方向错误。冻结时间MLP没有修复本轮短片。详见[frozen16_all_vs_teacher.jpg](frozen16_all_vs_teacher.jpg)和[观察记录](visual_review_all.json)。

训练覆盖A/D三个chunk；step08从GPU2迁到GPU0，恢复Adam/RNG后完成16次更新。QKV变化，time参数严格不变。GPU0后8次更新及其准备/前后验证用时933.1秒，allocated峰值25,063.4MiB；不是完整1–16训练合计。

6条视频的帧数、分辨率及保存输入一致性检查通过。旧teacher的offload reserve不明；本轮也不据此声称效率收益。
