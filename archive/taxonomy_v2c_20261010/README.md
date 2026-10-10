# 2026-10-10：V2并行路线分类迁移

用户要求原V3 Efficient Causal归入V2c Strict Causal + Persistent KV。V2a/V2b/V2c是解决V1视觉与动作退化的三条并行方案，字母不表示权重继承。

- [新V2汇报入口](../../report/v2/README.md)
- [主线定义](../../mainline/v2/README.md)
- [新旧目录/文件映射](migration.json)
- [迁移前文件SHA](before_moved_files.json)
- [迁移前现行文档原文](before_documents/)
- [两段展示视频标题更新](display_video_relabels.json)
- [迁移核验](validation.json)

EXP编号与plan_version不变。EXP-004当前目录为`experiments/EXP-004_v2c_8step`，旧目录`EXP-004_v3_8step`保留相对符号链接，供冻结脚本与历史来源引用使用；EXP-002/003维持原中性目录名。实验原始代码、日志、任务/报告快照、逐块指标和原视频未因改名修改。mainline/report的当前摘要与导航已重写为V2c。

只更新两段8步展示视频的顶部文字为V2c，CPU重新编码，完整帧序列/24fps/底部条件说明保留；原标题视频在实验目录保留。其余移动的视频和代码均逐字节保留。没有新增GPU推理或训练。

`legacy_v3_planned/`是旧入口原文归档，其中相对链接按历史目录语境解释；当前入口以本页上方导航为准。以后有变体的版本按`report/vN/vNa_<method>/`分组，当前未创建尚无研究定义的V3版本。
