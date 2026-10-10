# 当前目录与分类整理记录

2026-10-10按用户最新决定，恢复 **V3 Original Feasibility Baseline**，并新增独立候选V3-SW-G、V3-SW-L。此前V2c分类保留在历史档案；本次不改变冻结实验结果。

## 当前结构

```text
report/
├── V0_original/
├── V1_native_causal/
├── v2/
│   ├── v2a_rgb_anchor/
│   └── v2b_same_sigma_local_bidir/
└── v3/
    ├── v3_baseline/          # 正式参考，含73帧、124帧与8步续写证据
    ├── v3_sw_g/              # Global滑窗候选
    └── v3_sw_l/              # Local滑窗候选
```

mainline采用相同家族分组。后续版本如有变体，继续使用`vN/<variant>/`二级组织，实验原始证据仍按稳定EXP编号归档。

## 兼容与证据保护

- `mainline/V3_efficient_causal`指向`mainline/v3/v3_baseline`。
- EXP-004正式目录恢复`experiments/EXP-004_v3_8step/`；旧`EXP-004_v2c_8step`为反向兼容链接。
- 展示文件名恢复V3 Baseline。两段8步展示片复用实验目录原V3标题的冻结文件；此前CPU重标的V2c片保存在本次档案，不再次编码。
- 保留原始报告、任务书、代码、日志、视频和指标；只调整当前分类说明和导航。根目录继续保留六份Markdown。

[V3导航](report/v3/README.md) · [本次迁移与核验](archive/v3_baseline_restore_20261010/README.md) · [此前V2c迁移原文](archive/taxonomy_v2c_20261010/README.md)。

更早历史曾用旧V3指代Same-σ路线，需结合日期与协议阅读；本次正式V3 Baseline仅指EXP-002/003的strict causal/persistent KV版本。
