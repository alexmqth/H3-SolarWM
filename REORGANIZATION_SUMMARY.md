# 当前目录与分类整理记录

2026-10-10按用户要求，将原V3 Efficient Causal归为V2c Strict Causal + Persistent KV。V2a、V2b、V2c是回应V1视觉与动作退化的三条并行路线；没有按字母排列的权重继承关系。

## 当前结构

```text
report/
├── V0_original/
├── V1_native_causal/
└── v2/
    ├── v2a_rgb_anchor/
    ├── v2b_same_sigma_local_bidir/
    └── v2c_strict_causal_kv/
        ├── videos/
        ├── 73frame_evidence/
        └── 8step_continuation/
```

mainline采用同样的v2家族分组。未来有变体的版本继续采用`report/vN/vNa_<method>/`，各级README提供导航。当前没有另行定义未来V3。

## 实验与历史记录

EXP-002/003保持稳定的中性目录，现行标题和分类为V2c；EXP-004规范目录改为`experiments/EXP-004_v2c_8step/`，旧路径是相对符号链接，保持冻结脚本可访问。EXP编号、任务书修订号、原始协议、日志和证据不因研究分类改变。

两段8步展示视频仅通过CPU更新顶部V3文字为V2c，原比较视频保留在实验目录。其他移动的视频/源码逐字节保持；不新增模型推理、训练或实验消融。

[新V2导航](report/v2/README.md) · [迁移清单与核验](archive/taxonomy_v2c_20261010/README.md) · [本次之前的整理报告原文](archive/taxonomy_v2c_20261010/before_documents/submission/REORGANIZATION_SUMMARY.md)。

更早编号曾将Same-σ路线称为旧V3，它对应现V2b；近期Efficient Causal所用旧V3则对应现V2c。历史原文必须结合协议与日期阅读，不能做全局编号替换。
