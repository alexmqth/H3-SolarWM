# 2026-10-11：V3 / V4报告分类整理

用户要求：V3下按Sliding Window（Global/Local）与FM → AnyFlow分组；DMD归V4 SGF路线；README解释名称、目的并提供实际视频。

- [迁移清单](migration_manifest.json)：迁移前建立，记录13个目录映射及原文件SHA-256。
- `before_documents/`：修改前的活跃管理/展示文档快照，不代表当前状态。
- `management_snapshot/`：本次整理后的根目录管理文档快照；项目根目录本身不在Git仓库中，随本次发布保存。
- [新增视频副本清单](video_copies.json)：原35个实验视频的展示副本；另补旧Stage2-lite的2条视频，来源在其报告video_manifest.json。
- [当前报告导航](../../report/README.md)：用新路径查找当前内容。

未迁移experiments原始证据，未改变实验编号、权重、配置、视频像素及冻结Judge审核。历史快照中的旧路径按migration_manifest映射定位；快照不作当前导航。正式Baseline的release_manifest保持原字节。V4名称不改变DMD失败验收。

仅本地文件整理、已有MP4复制与CPU媒体检查；无模型推理/训练。

## 整理后验证

[机器检查结果](verification.json)：673个活跃文档本地链接全部有效；V3/V4共50条视频通过容器信息/首帧CPU解码，41项视频来源清单逐字节/SHA-256匹配。13个目录迁移已核对，原有非Markdown媒体及清单不变；冻结Baseline release manifest与Git HEAD逐字节相等。实验代码、权重、原始证据未改，根目录仍为六份Markdown。网页播放器脚本通过node语法检查；本次没有重复全视频质量验收或GPU调用。
