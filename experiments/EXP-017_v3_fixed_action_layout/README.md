# EXP-017/v1 · 变长动作的固定语义坐标候选

状态：**CPU布局候选及回归检查通过；模型forward、KV数值、训练和生成视频未测试。** 本目录不修改冻结V3，只用真实动作的tokenizer长度、EXP-014 canonical full37 fixture与冻结源码，验证未来动作的内容/长度不改变当前可见输入。

- [研究协议与位置字段](PROTOCOL.md)：固定head/I0/audio/video/action语义三轴坐标，允许真实action物理行变长；说明未知未来占位合同与未测试项。
- [候选实现](fixed_layout.py) · [CPU结果](CPU_RESULTS.json) · [实际运行日志](CPU_RUN.log) · [来源/源码SHA](SOURCE_MANIFEST.json)。
- [Worker报告](WORKER_REPORT.md) · [Judge独立审核](judge/INDEPENDENT_CPU_AUDIT.json)。
- [Judge最终验收](judge/FINAL_REVIEW.md) · [未授权的下一阶段草案](judge/NEXT_STAGE_DRAFT.md)。

四场景canonical A/D packed全部字段逐值回归，真实script下stop12/17的可见token和布局对未来长短句干预保持不变，已提交history语义坐标不变。Judge另做24次短/长/单token未来干预，均通过。最初实现中原builder的action-aware origin检查仍依赖未来总文本长度；[旧结果](CPU_RESULTS_attempt1.json)与[旧日志](CPU_RUN_attempt1.log)保留，正式候选改为仅复用其物理行构造，再赋予冻结语义坐标。这里没有启用SolarWM camera PRoPE，也没有证明实际DiT mask或混合动作画质。
