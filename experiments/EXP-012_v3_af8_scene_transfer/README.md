# EXP-012/v1：冻结 AF8 在共同 FM8 首窗上的短程场景迁移

当前仅完成 [P0 CPU 准备](P0_REPORT.md)。研究问题是：EXP-007 AF2 step32 在工业和村落两张已固定 ABot validation 初图上，接续 EXP-011 普通 FM8 生成的**同一首39帧历史**时，8NFE 的 AnyFlow C2 是否比普通 FM8 保留或改善动作/画质。AF 会用自己的权重重新提交 C1 KV；不能复用 FM8 的 hidden KV，也不能把这个结果称为全程 AF 首窗生成。

[任务书](taskbook_v1.md) · [配置](config.json) · [来源清单](source_manifest.json) · [代码冻结清单](code_manifest.json) · [CPU audit](P0_CPU_AUDIT.json)。P0 核对了唯一 step32 QKV/target-time 配套权重、原生 Single I0 fixture、两个 FM8 C1 endpoint/RGB、A/D 当前 action rows 与 native8 sigma。真实33B GPU 推理和视频质量**尚未执行**；每个场景需 Judge 独立 marker。

冻结入口为 [run_exp012.py](run_exp012.py)。预定每场景1次 AF-own C1 clean commit + AA/AD 各8次 finite-map + 2次 decode，到56帧；总预算34 forward/4 decode/≤1260 GPU秒，无训练或新编码。原始 FM8 对照直接复用 EXP-011，后续评价为 matched-history model comparison，不是同权重或同 raw KV 消融。
