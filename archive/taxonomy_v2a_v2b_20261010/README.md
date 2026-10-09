# V2a / V2b 并列路线修订

用户纠正：RGB anchor与Same-σ是同一条件/历史问题的两种研究路线，不是顺承版本。将原V2→V2a，原V3→V2b，原V4 Planned→V3 Planned，目录、文档和屏幕标签同步。

- [修订前文件SHA](before_files.json) / [git mv与路径映射](migration.json)。
- [本次来源映射](selected_sources.json)：保留legacy asset ID用于追溯，不与当前display_version混淆。
- [新视频配方](render_specs.json) / [编码与解码结果](render_results.json) / [汇报副本](gallery_copies.json)。
- [完整修订验收](validation.json)：原视频不覆盖、代码/测量不改、链接/解码/便携性。

旧编号MP4在[obsolete_presentation_v2_v3](../obsolete_presentation_v2_v3/README.md)归档；初次整理的原始审计收据保持当时路径，结合本次migration查新位置。本次只有CPU编码/文档操作，无模型训练或推理。

本次可重放的渲染源码：[render_parallel_gallery](../../scripts/reorganization/render_parallel_gallery.py)。迁移/文档脚本为一次性来源记录；现行验收入口为[validate_parallel_taxonomy](../../scripts/reorganization/validate_parallel_taxonomy.py)。旧prepare/finalize脚本描述初次编号，不再用于现行目录。
