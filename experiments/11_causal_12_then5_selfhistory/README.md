# 11：12→5在自身生成历史上取得第二块局部动作＋结构正控

**解决的问题：首窗动作正确以后，第二个短chunk是否仍能响应A/D，并保留人物结构。** 本轮通过的是单停车场、seed13、17个续写RGB的局部门槛；不是完整124f、persistent-KV或Stage1完成。

方案：Original H3＋released action LoRA，零新增训练；首12latent使用已保存的自身生成endpoint，后续5latent只换当前A/D。单I0/native时间/30steps/shift2.22，T2每步重算可见窗口，未来动作和video物理删除。采用已有N协议：history临时按当前sigma和固定噪声加噪，只积分当前chunk，保存的过去不变。没有新anchor、AnyFlow或DMD。

| 相同自生成历史 | clean A / D flow | N A / D flow | N局部画面 |
|---|---:|---:|---|
| 首12动作为A | +1.507 / −1.027 | +2.146 / −1.752 | 人物完整、响应不同、无场景切换 |
| 首12动作为D | −0.177 / −1.312 | +2.608 / −1.414 | 换A会改变转向；人物结构保持 |

clean在D→A仍失败，因此不能将结果只归因于分块相位。B首7的A/D为−0.153/−1.179，画面完整但A方向未过，不自动延长。

- **[C：A→A / A→D / D→A / D→D，完整56帧](C_selfhistory_N_56.mp4)**。每条路径均39帧自身首段＋17帧新生成，24fps，共2.33秒；RGB39切块，没有GT重置。
- [clean与N两历史局部对比](C_second_chunk_summary.mp4)。每组前8RGB历史＋17RGB续写。
- [B7与C12首22帧对比](B7_vs_C12_first22.mp4)。两者当前可见上下文不同，不能称为等延迟比较。
- [详细实验、耗时/显存/MAD、全部分支与复现边界](../../reports/stage1_anyflow/02_causal_diagnostics/chunk_partition_cb/README.md)。

VAE边界也经实际权重核查：12prefix前34RGB与完整decode一致，末5RGB可能回改；本片冻结已显示39帧，再追加新17帧，并测量边界差。没有额外输出平滑。

候选`C12_then5_N_30step_selfhistory`是**推理配置，不是新训练checkpoint**。[manifest](manifest.json)保存两份首12和四份后续5的latent状态及SHA，供后续续写使用；这些`.pt`不是LoRA权重。33B底座、released LoRA、条件输入和冻结runtime仍为外部依赖。

下一步仅值得在相同协议下验证第三/第四块自己的history。当前只有两块、单场景/seed，不能推广到完整动作保真或长期稳定；本轮没有继续生成124帧或恢复训练。
