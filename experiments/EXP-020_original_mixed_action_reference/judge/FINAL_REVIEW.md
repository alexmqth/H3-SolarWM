# EXP-020/v1 — Judge 最终验收

日期：2026-10-15 HKT。决定：**accepted，有限原版参考任务收口**。工程执行 PASS，人物结构有限可行；视觉质量/动作语义 PARTIAL，动作切换尚未通过正式验收。不是官方全套能力复现，也不升级正式模型。

## 核查与视觉

独立核对run.py原生双向pipeline调用、released Action LoRA加载104对、causal_chunk_size=None、30NFE、真实17句脚本；8个来源SHA、6组实物/归档SHA与大小、2条输出MP4完整56帧/24fps/PTS通过，见 [FINAL_AUDIT.json](FINAL_AUDIT.json)。实际30DiT、4text、1image encode、1video decode、1audio decode，0训练。完整wall282.237秒，峰allocated26.049GiB/reserved26.840GiB，均在授权内。

Judge从实际MP4 CPU解码审阅全部56帧。人物大体完整且有步行，草地持续可辨；RGB20后天空出现明显放射状云纹，后段持续。不能称该样本画面保真通过，也不能仅以此认定原始H3架构缺陷。晚段仍有步态，但无法可靠确认左平移→静止→前进的时序响应，光流代理不能补足该证据。

外层zsh收尾误用只读变量`status`，shell返回1；模型完整落盘且账本完成，没有GPU重试。原日志不改写。后续launcher用专有变量名处理退出码。

## 比较口径与决策

Original从I0联合生成全部56帧；EXP-019借GT前39帧只生成后17帧。历史、噪声形状、精度/attention backend、音视频联合路径和生成范围有差异；两者是参考展示，不能据此作单变量因果化优劣归因或计算E2E加速。冻结V3精度协议不得为追求一致而偷偷修改。

保留负结果，不重复抽seed争取好看片段，不为此单样本追加GPU。后续CPU准备应核对原版56帧/17动作与官方标准长度、精度、路由的适用范围，将可能的协议混杂记入下一任务；目前没有证据确定伪影根因。公开SolarWM Stage2-fix相机参考仍未执行，应单列预算，不能用本实验冒充该项完成。
