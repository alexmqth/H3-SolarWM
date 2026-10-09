# Current-prefix候选：真实33B同状态路由诊断

冻结Original H3+released action LoRA，零训练更新；2个scene×3chunks×3sigmas。每个状态只替换当前chunk A/D。

**首块候选=Original是身份正控，不能当作修复收益。后续chunk单独列出。没有生成候选视频，也没有action/画质PASS。**

| 分组 | 点数 | 整体cos baseline→candidate | A/D delta cos baseline→candidate | delta相对误差 baseline→candidate |
|---|---:|---:|---:|---:|
| all | 18 | 0.994733 → 0.998509 | 0.012540 → 0.339032 | 1.214097 → 0.962156 |
| first_chunk_identity | 6 | 0.991830 → 1.000000 | -0.019933 → 1.000000 | 1.251932 → 0.000000 |
| later_chunks | 12 | 0.996184 → 0.997763 | 0.028777 → 0.008548 | 1.195180 → 1.443233 |
| chunk_1 | 6 | 0.995811 → 0.998334 | 0.028335 → 0.033839 | 1.198970 → 1.388669 |
| chunk_2 | 6 | 0.996556 → 0.997193 | 0.029219 → -0.016742 | 1.191389 → 1.497798 |
| sigma_0.240781 | 6 | 0.993634 → 0.999143 | 0.005900 → 0.347629 | 1.261968 → 1.016873 |
| sigma_0.689441 | 6 | 0.997096 → 0.998955 | 0.027272 → 0.347391 | 1.228000 → 0.928254 |
| sigma_0.939540 | 6 | 0.993468 → 0.997430 | 0.004449 → 0.322077 | 1.152323 → 0.941340 |

后续12点中7点delta cosine增加；这不是光流方向正确率。

控制：baseline重跑与旧probe逐状态delta cosine差<1e−8；input/history/pair匹配；当前anchor及teacher完整输出hash已保存。两种causal路由各自重建KV，A/D内部共享只读cache。chunk1分别以A/D重建历史KV一致；当前预测不修改KV/参数；重复RMSE=0。

解释边界：候选的own直接绑定和current-video公共prefix两项改变在先前首块2×2实验中已分开；本次测试它们组合到persistent cache的行为。Teacher仍双向重算历史，不能把候选当作Original完整等价模型。状态是固定endpoint加噪，不是在线solver采样到的中间状态，也不是候选自生成分布。

真实执行：120 noisy/reference forwards、16 clean-history forwards；1285.45s、allocated peak26983.03MiB，单次共享主机不作speedup声明。

[逐状态指标](metrics.csv) · [完整probe](probe.json) · [CPU协议](README.md)
