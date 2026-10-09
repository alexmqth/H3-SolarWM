# 固定generated history的当前A/D机制：密度对照完整结果

**本轮仍未恢复动作差分方向。** 新shift2.22在有generated history的12点delta cosine为0.035019（旧shift12为0.029239），幅度约为teacher的67%；GT有历史12点反而由0.011453降到−0.002418。整体velocity cosine很高，不能据此判定action保真。

两组GT/generated各18状态已经完整完成。主表只取后两个有历史chunk：2场景×2chunks×3sigmas共12点；首块6点另存。所有FM模型使用相同固定状态来源，generated取旧step00轨迹，未换成各自新生成状态。

| 历史 | 模型 | Whole velocity cos | A/D delta cos | Delta norm ratio | Relative delta error |
|---|---|---:|---:|---:|---:|
| gt | 未训练causal FM0 | 0.987984 | 0.009196 | 0.539531 | 1.141850 |
| gt | FM48 shift12 | 0.988751 | 0.011453 | 0.555337 | 1.149230 |
| gt | FM48 shift2.22 | 0.988892 | -0.002418 | 0.521629 | 1.137637 |
| step00_generated | 未训练causal FM0 | 0.996184 | 0.028777 | 0.671606 | 1.195180 |
| step00_generated | FM48 shift12 | 0.996389 | 0.029239 | 0.645296 | 1.179819 |
| step00_generated | FM48 shift2.22 | 0.996377 | 0.035019 | 0.672755 | 1.192429 |

完整18点（含首块无历史对照）汇总：

| 历史 | Delta cos FM0 / old48 / new48 | Whole cos FM0 / old48 / new48 |
|---|---:|---:|
| gt | 0.017516 / 0.017641 / 0.008237 | 0.988193 / 0.989033 / 0.989268 |
| step00_generated | 0.012540 / 0.009336 / 0.013624 | 0.994733 / 0.995210 / 0.995354 |

## 机械检查与解释边界

- 当前chunk A/D编码采用完全相同layout；当前chunk以外文本逐元素保留原值，历史联合动作不被替换。时间映射仍为latent5–9→RGB[17,34)、latent10–11→RGB[34,39)。
- 每个checkpoint由相同raw history重建自己的KV，A/D分支共用并保持只读；六次完整cache digest核查通过，chunk1以D独立重建与A完全相同。中sigma重复teacher/student输出RMSE=0，参数version不变。不同checkpoint的KV不要求相同。
- 直接attention路由的实际mask证据见此前generated_action_geometry128；本轮没有另改mask或architecture。本轮主要新增相同状态下的真实FM48训练密度对照，不能把接口检查当语义恢复。
- 全部state/history/action-pair/endpoint哈希跨三个checkpoint匹配；入口源仅经过审计的路径适配。旧收据没有完整anchor或teacher-output张量hash，只核验teacher范数，不能夸大为完整tensor逐bit证明。
- Original teacher双向重算历史；student依赖clean-committed raw KV。相同raw state和动作条件不意味着内部条件函数完全相同。这里的noisy latent为endpoint-noise插值，不是实际solver轨迹捕获。
- 动作delta与整体velocity必须分别看；delta非零不等于方向正确。这里只判断局部函数响应，不直接把latent cosine当视频左右方向，也不作18或12个独立样本的显著性推断。

generated有历史12点中新student动作delta约占自身整体velocity的0.94%，teacher约1.47%；公共分量占主导，因此整体cos0.996与差分cos0.035并不矛盾。小差分仍受BF16数值影响；统一后端和重复误差0只限制一部分数值混淆，不能称为完全排除精度影响。

[GT逐点报告](report/geometry_gt/README.md) · [generated逐点报告](report/geometry_step00_generated/README.md) · [全部汇总数值](geometry_summary.json) · [图](action_geometry_density.png)

噪声采样密度对照不自动建立可信Stage1。完整generated30/8与停车场A/D视频仍须单独验收；未通过前不据此开展C/D效果训练或替换meeting。
