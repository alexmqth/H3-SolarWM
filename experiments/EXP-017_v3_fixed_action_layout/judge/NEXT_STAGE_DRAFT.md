# 下一阶段建议 — 尚未发布执行授权

目的：用真实模型验证EXP-017固定布局候选，然后判断真实混合动作/GT-history下V3 FM30与FM8是否具有基本结构/动作可行性。正式V3、已有FM8/AF/DMD证据冻结。可作为后续EXP-018任务基础，但本文件不是任务编号或GPU marker。

1. 先CPU集成审查：真实text encoder独立逐句规则、当前action行和Token Refiner分段、fixed template坐标、future物理删除、own-action/current-prefix和KV版本。新head用EXP-015新I0/原caption，不能复用旧I0/Qwen head embedding。EXP-016新I0 VAE latent可复用；已有大权重/cache不复制。
2. Canonical实际模型回归：固定原V3 C1/C2状态及sigma1/0.5，各旧/新输出的比较总调用数必须按实际是否复用旧输出明确列账。若两边均新算，2sigma×2chunk×2入口为**8forward**；只预算4forward须明确已有匹配旧输出可复用，不能少算一半。建议保守8forward、0decode、上限360GPU秒，零训练。输入/精度/backend一致，显著差异即停。
3. 真实mixed-action文本编码：四scene各1head；真实独特action句+固定A未知占位各2/2/3/3，共**14text encoder调用**（4head+10action）为不跨scene缓存的保守上限，0新VAE编码/0DiT。运行前实际tokenizer/句表和加载路径再核对，上限600GPU秒。独立编码不等于完整模型协议通过。
4. 首固定train scene：真实C1 GT-history teacher forcing，Original H3+released action LoRA同权重下共享一次clean C1 commit，固定相同C2 noise/真实动作，对比FM30与FM8，**39forward=1commit+30+8、2decode**。首图有持续结构失败或动作明显无效就停止。普通细节PARTIAL，不扫seed/更换scene。
5. 如首图有限通过，再考虑余3固定图；全四图**156forward/8decode**，采样/commit/decode/加载费用独立计账，上限2700GPU秒（0.75单卡GPU小时）。这是GT-history局部续写，不称free-running或D反事实。需完整逐帧、人物/ghosting/边界、真实动作响应和各阶段耗时/显存。

以上保守总预算8canonical forward+156teacher-forcing forward、14text encode、8decode、0训练，合计时间上限3660GPU秒约1.0167GPU小时。正式执行前仍应形成新任务书、冻结实现和分阶段Judge marker。09:00后项目≤3卡，优先单空闲卡；GPU3/4他人任务不操作。

若真实mixed-action条件仍不能提供基本结构/响应，则先决定是否值得普通FM真实转移适配。AnyFlow target-time finite-map训练单列对照和预算，不能将普通FM8减步混为AnyFlow，也不将当前teacher弱AD纳入正确监督；此前AF4/DMD失败配置不继续调参。
