# V2b · Same-σ History / C12→5 H3-World

## 1. Version Name / Research Objective

Current Best Local Causal Rollout Candidate：验证首12latent正控接自身history后，第二个5latent的动作与结构。

## 2. Parent Version / Baseline

**从Original H3 + released action LoRA恢复原生条件**，针对V1暴露的问题探索另一条并列路线。没有使用V2a visual adapter，不是V2a checkpoint续训。

## 3. Main Changes

Single I0、native action/time、固定全局RoPE；首12后续5；历史按当前sigma与固定历史噪声临时加噪；T2每步联合重算可见history/current。未来action/video在refiner前物理移除。

## 4. Model and Inference Configuration

30steps/chunk，shift2.22，seed13；39帧自身生成首段+17帧续写=56f/24fps。T2窗口内video双向；只积分当前chunk，不更新保存的过去。没有persistent hidden KV或GT reset。

## 5. Representative Videos

- [V1 vs V2b：共同前56f，协议不同](videos/V2b_vs_V1.mp4)
- [四路径完整56f；RGB39边界高亮](videos/V2b_four_paths_56.mp4)
- [Original vs V2b，AA/DD共同前56f](videos/V2b_vs_original.mp4)
- [V2a vs V2b，并列能力比较，非升级关系](videos/V2a_vs_V2b.mp4)
- [A→A原片](videos/V2b_AA_full.mp4)
- [A→D原片](videos/V2b_AD_full.mp4)
- [D→A原片](videos/V2b_DA_full.mp4)
- [D→D原片](videos/V2b_DD_full.mp4)

## 6. Quantitative Results

第二块：A-history下当前A/D为+2.1464/−1.7519；D-history下+2.6076/−1.4143。每条续写30 forwards，sampling约194.6–195.5s，GPU peak25.42GiB，CPU hidden KV=0。首窗重用，不能将续写时间当全56f E2E。

## 7. What Was Improved

**Local Visual and Action Fidelity Demonstrated — 仅56f第二块，未验证长期稳定。**

两份自身history下第二块A/D方向正确，人物结构保持，无瞬间换场；相较clean history在D→A的方向失败，这是明确的局部正结果。

## 8. What Still Failed

只有停车场、seed13、第二块；没有第三/第四块、跨场景或完整124f验收。不是strict chunk-causal mask，更没有高效persistent KV证明。

## 9. Lessons Learned

V2b是历史协议/可见窗口的新候选，不是Same-σ单因素修好了V2a。decoder重解码可能修改过去末5RGB；展示冻结已发39帧再追加17帧，边界不能隐藏。

## 10. Source Code / Checkpoint / Original Experiment References

H3-World/outputs/2026-10-09-22/chunk_partition_cb/；候选C12_then5_N_30step_selfhistory是推理配置，不是LoRA checkpoint。保存的first12/next5 .pt是生成latent endpoints。

[核心代码说明](code/README.md) · [代码SHA与源路径](code/SOURCE_MANIFEST.json) · [本版来源清单](PROVENANCE.json) · [返回汇报导航](../README.md)

指标均为历史记录；flow是运动proxy，MAD是活动量/连续性描述，不是视频质量评分。Original生成视频不是GT。

**MISSING_MATCHED_VIDEO**：Original/V2a的A→D和D→A匹配56f视频未找到；四路径只作V2b内部展示。AA/DD跨版本主片只用RGB[0,56)，V0/V2a完整124f原片仍保留。
