# 2026-10-10 整理与CPU视频制作审计

先审计Git，再建索引/版本映射，之后复制、git mv历史文档、CPU制作对比。原始测量、checkpoint、视频与生产代码未改写。

- [before_tracked.json](before_tracked.json)：整理前2844个跟踪文件的SHA256；基准commit。
- [file_index.json](file_index.json) / [video_index.tsv](video_index.tsv)：提交包与两处outputs的文件/视频索引（只列物理树，不重复遍历目录软链接）。
- [version_map.json](version_map.json)：复制与渲染之前冻结的版本选择和缺失。
- [selected_sources.json](selected_sources.json)：主展示12条原片、原config/run与SHA。
- [copy_manifest.json](copy_manifest.json) / [layout_actions.json](layout_actions.json)：字节复制与文档移动/跳转。
- [render_specs.json](render_specs.json) / [render_results.json](render_results.json) / [gallery_copies.json](gallery_copies.json)：逐新MP4配方、decode/PTS/hash和独立汇报副本。
- [validation.json](validation.json)：完整性、链接、语法、视频及便携检查；[legacy_link_issues.json](legacy_link_issues.json)记录最终剩余断链（应为0）；[修复前快照](legacy_link_issues_before.json)与[修复清单](link_repairs.json)保留原路径和原Markdown字节，测量文件不改写。

CPU制作源码：[prepare_sources](../../scripts/reorganization/prepare_sources.py)、[render_gallery](../../scripts/reorganization/render_gallery.py)、[write_documents](../../scripts/reorganization/write_documents.py)、[finalize_layout](../../scripts/reorganization/finalize_layout.py)、[validate](../../scripts/reorganization/validate.py)。

这些脚本不是模型训练入口。prepare/write/finalize是本次整理的来源记录，finalize不是可重复执行的迁移器；render拒绝覆盖已有MP4。若需重做渲染，先复制仓库到单独目录并使用空gallery，不能覆盖原片。验证可独立重复：

```bash
/home/lpeng/miniconda3/envs/h3world/bin/python submission/scripts/reorganization/validate.py
```

PyAV调用FFmpeg/libx264，Pillow缩放与文字覆盖。只使用CPU；GPU模型参数从未加载。新视频完整解码为H.264/YUV420P/24fps，帧PTS逐项检查；旧文件按SHA比对。

[Git复核](git_review.json)记录原始patch/历史文档的空白提示；新撰写内容无diff-check错误。
