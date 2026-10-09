# E1 下一有限对照：历史条件协议，而非更宽窗口

状态：**已设计，未实现/未启动新 GPU 作业，未训练。** 本文件是完成 coarse12 No-Go 后的后续协议，不修改已冻结的 `protocol.json`、`launch_protocol.json`、runtime 或现有视频。最多3张项目GPU的上限保持。

## 已知与未知

- 已知：恢复原生 text/action time 与单 I0 后，完整当前窗口有响应；同输入5latent首窗失败，12latent首窗改善。12latent跨历史依然动作/结构失败，不能再扩大窗口来追指标。
- 已知：CPU实际tiny-H3的9个窗口/噪声状态显示，clean历史video的模型时间为1，当前video与全部text/action为1−sigma；历史action与其own video的time存在sigma差。见 [实际执行审计](history_contract_audit.json)。
- 这沿用原生pipeline的denoise-mask/retake处理，不是已证实的实现错误。既有代码支持clean retake，不能说Original绝对不能处理混合时间。
- 未知：后窗失败来自历史条件分布、动作表征、位置/历史长度或它们的组合。当前证据不能定位唯一原因。VAE未来RGB干预全部为0，不支持历史编码泄漏解释本轮失败。

## 只比较两种历史条件协议

保持Original+released LoRA、native prefix time、单I0固定位置、12latent窗口、过去/当前动作、布局、audio、sigma序列、30steps、seed13和当前state不变。attention仍为当前T2，逐sigma重算，不改网络/LoRA，不复用可变hidden KV。

| 条件 | 传给DiT的历史video | 历史video时间 | 历史动作时间 | 原始历史保存 |
|---|---|---|---|---|
| C：现有clean-history对照 | H | 1 | 1−sigma | 只读 |
| N：与当前sigma一致的历史inpainting条件 | (1−sigma)H + sigma·epsilon_H | 1−sigma | 1−sigma | 只读 |

epsilon_H固定使用同一fixture已知前缀的初始noise，A/D共用，不能每步重抽；只在临时模型输入上加噪。solver仍只推进当前chunk，已生成的历史和已发出的RGB不更新。每步按同一H与epsilon_H重建临时历史，sigma=0回到H，不把模型预测的历史输出写回。将历史video输入与时间一起变化是一个**历史条件协议**干预，不能声称单独识别了时间或噪声的作用。

只输入已知历史/当前动作与当前窗口，未来action/video仍在refiner前删除。噪声来自预先冻结的外生noise，不取双向teacher的中间历史hidden state，避免引入未来视频/动作。反转未来输入不应影响当前结果。

## 执行与预算

1. 首先做CPU实际tiny-H3检查：无历史时C/N完全一致；sigma=0一致；所有已知历史tensor/hash只读；当前action可达；未来action/video删除/扰动不改变结果。核验实际输入video和action的逐行时间，不能只检查wrapper参数。通过前不加载33B。
2. 固定保存的coarse A-solver状态，窗口1/2、两history、实际solver索引0/15/27。在**同一当前noisy state**上分别计算C/N的A/D差分。总48次forward，加至多4次首窗identity和4次repeat，硬上限56。C的A输出须与原始保存值重放对齐，记录精度/执行误差；不相减两条solver轨迹。
3. 这些探针只能显示条件协议是否改变响应，不能用一个未经局部画面验证的delta当teacher方向。没有合法参照时不报告“恢复teacher cosine”，delta变大也不作为PASS。
4. 若执行语义正确且probe有可测变化，先做窗口1两history×A/D、30步，共120次noisy forwards。复用本轮C的视频作为已冻结参照，评估方向、人物结构及与同一历史末端的连接。两history都改善且不出现严重分解才评窗口2，额外120。否则停止这一因素；不跑124f、不再扫噪声强度。
5. 最多56诊断+240采样前向，零optimizer。需要新脚本/输入哈希、独立输出目录与启动记录，严禁改写本轮No-Go收据。两个后续窗口通过后，才补真实GT-history、seed29和动作切换验收；不因本小对照通过直接进入Stage2。

## 判定与风险

若N改善多窗口方向和结构，说明历史条件协议值得继续，仍需测history adherence与成本。高sigma下历史信号会变弱，模型可能忽略过去并从I0重新开始；即使A/D符号恢复，出现位置/人物重置或边界突跳也判失败。不能用重置换来“稳定”。

若N无改善，保留负结果，停止此条件因素。E2仍须获得可靠的同起始状态局部后果配对，重新审视action pathway/可训练表示，不能直接拿失真的R-prefix作强监督。E3的顺序保持可信causal→AnyFlow→on-policy Stage2。

该对照是定位历史条件分布的假设实验，不是SolarWM官方配方，也不提前宣称能修复动作控制。
