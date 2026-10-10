# EXP-012/v1：冻结 AF8 在共同 FM8 首窗上的短程场景迁移

P0 与两场景 GPU 续写已完成；[Judge 最终验收](judge/FINAL_REVIEW.md)接受工程/协议 PASS，但当前 AF2 step32 **未通过动作与画质的联合迁移验证**。[Worker 最终报告](WORKER_REPORT.md)保留全部证据：工业 AD 的反向响应弱于普通 FM8，村落 AA 人物/前景后半块出现持续透明条纹并被判结构FAIL。评价只限两个固定 ABot validation 初图、共同 FM8 首窗、seed13、8NFE C2；不否定其他 AnyFlow checkpoint 或方法本身。

[任务书](taskbook_v1.md) · [配置](config.json) · [来源清单](source_manifest.json) · [代码冻结清单](code_manifest.json) · [CPU audit](P0_CPU_AUDIT.json) · [原始证据清单](artifacts/manifest.json)。P0 核对唯一 step32 QKV/target-time 权重、原生 Single I0 fixture、两个 FM8 C1 endpoint/RGB、A/D 当前动作行和 native8 sigma。两次 GPU 阶段各自有 Judge marker，原片和负结果均保留。

冻结入口为 [run_exp012.py](run_exp012.py)。实际每场景1次 AF-own C1 clean commit + AA/AD 各8次 finite-map + 2次 decode，到56帧；总34 forward/4 decode/354.403 GPU秒，无训练或新编码。原始 FM8 对照直接复用 EXP-011，评价为 matched-history model comparison，不是同权重或同 raw KV 消融。先看[工业 AD 对照](artifacts/comparisons/industrial_AD_FM8_vs_AF8_56.mp4)和[村落 AA 对照](artifacts/comparisons/village_AA_FM8_vs_AF8_56.mp4)。
