# EXP-017/v1 — Judge最终验收

2026-10-11 08:57 HKT。**ACCEPT CPU CANDIDATE：固定语义坐标/可变物理行的独立布局候选通过CPU回归和未来隔离。** 0GPU、0模型forward/encode/decode/训练；正式V3 runtime和已验收结果未修改。不是mixed-action模型能力验收。

## 已执行及独立证据

四scene使用EXP-015真实17-span脚本和本地tokenizer；每句真实长度分别14、13、14，s3为前13句10/后4句8，canonical A/D每句10。模板只读取EXP-014冻结full37布局，不复用旧I0/embedding作新训练条件；已保存canonical A/D embedding只作回归。

候选物理row保持真实文本长度，无新增zero prefix或动作padding；head三轴/标签、每latent action三轴、SingleI0/audio/video三轴从预先固定canonical模板映射。使用原visible_inputs删除未来action/video，不改变attention代码。对应为H3 Global/MM-RoPE坐标，不启用SolarWM camera PRoPE。

Judge独立检查四scene：canonical A/D可见布局与prompt逐字段/逐值相等；所有text/image/audio物理indices联合互斥且在界内；head/action/I0/audio/video历史语义坐标从C1到C2不变。对stop12与17分别施加未来单token、1/2token交替以及40–42token长句干预，共24次，可见token IDs与**全部**packed字段均相同。[独立审计](INDEPENDENT_CPU_AUDIT.json)。

初版先调用原builder的action origin校验，对极短未来句仍会在固定坐标覆盖前拒绝。Judge指出该边界后，候选改为先只构物理layout，再显式填原action字段和固定坐标；canonical全字段仍exact。旧attempt1日志/结果保留，不把它冒充覆盖所有长度；当前候选SHA `fc4a47e8070aa0e9214c486396367227ca413e97e9056357dd318efc22d0d7d2`。

## 尚未验证与后续决策

- mixed真实句子的text encoder输出、Token Refiner/DiT forward及backend最终mask未执行；CPU使用真实token IDs/合成长度，不把ID sentinel当embedding。
- 坐标不变只支持raw KV位置解释一致的前提，不证明各层实际缓存值或新旧输出完全等价。
- 没有生成视频、动作控制或训练收益；四段真实数据缺少D后果，不作反事实GT或广泛泛化结论。
- full37、四个固定scene/head合同，未验证其它head/任意长度/长滑窗，也不称对原native mixed-action全段打包无损等价。

本任务CPU范围收口，08:57后不追加测试。后续建议见[NEXT_STAGE_DRAFT.md](NEXT_STAGE_DRAFT.md)：先canonical实际模型回归，再真实C1 teacher-forcing的FM30/FM8有限对照。草案没有GPU授权，不启动AnyFlow/DMD或新训练；保留失败配置归档和ROI原则。
