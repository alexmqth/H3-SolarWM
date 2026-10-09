# GT-history同状态A/D：整体拟合改善未恢复动作差分方向

2026-10-09。两个held-out真实ABot场景×3chunks×3sigmas共18个完整反事实状态。固定原始history、当前noisy latent与动作layout，只替换当前chunk的A/D；旧shift12与新shift2.22均为48-update普通FM，权重函数固定shift12。

| 等权均值 | 旧shift12 FM48 | 新shift2.22 FM48 |
|---|---:|---:|
| 整体velocity cosine | 0.989033 | 0.989268 |
| A/D delta cosine | 0.017641 | 0.008237 |
| Student/teacher delta norm ratio | 0.653948 | 0.631022 |
| Relative delta error | 1.203667 | 1.195712 |

动作差分仍明显非零，但与Original方向几乎不相关。relative error微降伴随差分范数比下降，不能当方向恢复；整体cosine约0.989也不能替代动作差分评价。首块delta cosine0.030018→0.029546，chunk1为0.011285→−0.013008，chunk2为0.011621→0.008172；没有跨chunk一致改善。

所有18点state/history/action-pair/endpoint标识匹配；六个history cache检查通过、A/D共用只读KV、chunk1独立按D重建历史的完整KV与A一致、重复teacher/student输出RMSE=0，参数version保持。两个入口仅使用经反向恢复校验的路径/来源变更。

此处仍有明确限制：GT历史不同于自生成历史；这些是endpoint-noise插值而非真实solver中间状态；Original teacher双向重算历史而student使用已commit KV，内部依赖不同；旧原始收据缺完整anchor/teacher输出tensor哈希，标量teacher范数一致不能宣称逐bit输出相同。18点来自两个相关场景，不作独立样本显著性推断。

因此本组支持“噪声采样密度单因素尚未修复局部action geometry”，不证明FM永远无法适配，也不将低cosine唯一定位到某一层权重。固定step00-generated的匹配组及完整视频另行验收。
