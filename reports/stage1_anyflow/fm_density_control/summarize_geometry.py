"""Summarize complete raw A/D probes, separating nonempty-history states."""
import json
from pathlib import Path
from statistics import mean

BASE = Path(__file__).resolve().parent
CONTROL = BASE.parent / "real_abot_fm"
if not CONTROL.exists():
    CONTROL = BASE.parents[1] / "2026-10-08-18/stage1_real_abot_fm"

def read(path):
    return json.loads(path.read_text())

def state_key(row):
    return row["clip"], row["chunk"], row["sigma"]

groups = {}
for history in ("gt", "step00_generated"):
    raw = {
        "fm0": read(CONTROL / f"geometry/step_00_{history}/probe.json"),
        "shift12": read(CONTROL / f"geometry/step_48_{history}/probe.json"),
        "shift2_22": read(BASE / f"geometry/step_48_{history}/probe.json"),
    }
    for data in raw.values():
        assert data["status"] == "complete" and len(data["records"]) == 18
        assert len(data["cache_audits"]) == 6
        assert all(a["read_only"] and a["shared_A_D_cache"] for a in data["cache_audits"])
        assert all(a["before_sha256"] == a["after_sha256"] for a in data["cache_audits"])
        assert all(a["rebuilt_D_history_equals_A"] for a in data["cache_audits"] if a["chunk"] == 1)
    indexed = {role: {state_key(r): r for r in data["records"]} for role, data in raw.items()}
    assert all(set(t) == set(indexed["fm0"]) for t in indexed.values())
    for state, reference in indexed["fm0"].items():
        for table in indexed.values():
            assert all(reference[k] == table[state][k] for k in
                       ("state_sha256", "history_sha256", "pair_sha256", "source_sha256", "endpoint_sha256"))
    groups[history] = {}
    for group, chunks in (("all18", (0, 1, 2)), ("with_history12", (1, 2)), ("first_chunk6", (0,))):
        groups[history][group] = {}
        for role, data in raw.items():
            rows = [r for r in data["records"] if r["chunk"] in chunks]
            delta = [r["action_delta"] for r in rows]
            groups[history][group][role] = {
                "n": len(rows),
                "whole_cosine": mean(r["absolute"][a]["cosine"] for r in rows for a in "AD"),
                **{metric: mean(d[metric] for d in delta) for metric in
                   ("cosine", "norm_ratio", "relative_delta_error", "student_response_relative_to_velocity",
                    "teacher_response_relative_to_velocity")},
                "cosine_range": [min(d["cosine"] for d in delta), max(d["cosine"] for d in delta)],
                "norm_ratio_range": [min(d["norm_ratio"] for d in delta), max(d["norm_ratio"] for d in delta)],
            }

result = dict(status="complete_matched_summary", groups=groups,
              primary_scope="chunks1/2 with nonempty history; empty-history controls separate",
              state_construction="fixed endpoint/noise interpolants, not captured solver intermediates",
              teacher_full_tensor_hashes_available=False, stage1_accepted=False)
(BASE / "geometry_summary.json").write_text(json.dumps(result, indent=2) + "\n")
lines = ["# 固定generated history的当前A/D机制：密度对照完整结果", "",
         "**本轮仍未恢复动作差分方向。** 新shift2.22在有generated history的12点delta cosine为0.035019（旧shift12为0.029239），幅度约为teacher的67%；GT有历史12点反而由0.011453降到−0.002418。整体velocity cosine很高，不能据此判定action保真。", "",
         "两组GT/generated各18状态已经完整完成。主表只取后两个有历史chunk：2场景×2chunks×3sigmas共12点；首块6点另存。所有FM模型使用相同固定状态来源，generated取旧step00轨迹，未换成各自新生成状态。", "",
         "| 历史 | 模型 | Whole velocity cos | A/D delta cos | Delta norm ratio | Relative delta error |",
         "|---|---|---:|---:|---:|---:|"]
for history in ("gt", "step00_generated"):
    for role, label in (("fm0", "未训练causal FM0"), ("shift12", "FM48 shift12"), ("shift2_22", "FM48 shift2.22")):
        g = groups[history]["with_history12"][role]
        values = [g[k] for k in ("whole_cosine", "cosine", "norm_ratio", "relative_delta_error")]
        lines.append(f"| {history} | {label} | " + " | ".join(f"{x:.6f}" for x in values) + " |")
lines += ["", "完整18点（含首块无历史对照）汇总：", "",
          "| 历史 | Delta cos FM0 / old48 / new48 | Whole cos FM0 / old48 / new48 |",
          "|---|---:|---:|"]
for history in ("gt", "step00_generated"):
    data = groups[history]["all18"]
    cells = [" / ".join(f"{data[r][m]:.6f}" for r in ("fm0", "shift12", "shift2_22"))
             for m in ("cosine", "whole_cosine")]
    lines.append(f"| {history} | " + " | ".join(cells) + " |")
lines += ["", "## 机械检查与解释边界", "",
          "- 当前chunk A/D编码采用完全相同layout；当前chunk以外文本逐元素保留原值，历史联合动作不被替换。时间映射仍为latent5–9→RGB[17,34)、latent10–11→RGB[34,39)。",
          "- 每个checkpoint由相同raw history重建自己的KV，A/D分支共用并保持只读；六次完整cache digest核查通过，chunk1以D独立重建与A完全相同。中sigma重复teacher/student输出RMSE=0，参数version不变。不同checkpoint的KV不要求相同。",
          "- 直接attention路由的实际mask证据见此前generated_action_geometry128；本轮没有另改mask或architecture。本轮主要新增相同状态下的真实FM48训练密度对照，不能把接口检查当语义恢复。",
          "- 全部state/history/action-pair/endpoint哈希跨三个checkpoint匹配；入口源仅经过审计的路径适配。旧收据没有完整anchor或teacher-output张量hash，只核验teacher范数，不能夸大为完整tensor逐bit证明。",
          "- Original teacher双向重算历史；student依赖clean-committed raw KV。相同raw state和动作条件不意味着内部条件函数完全相同。这里的noisy latent为endpoint-noise插值，不是实际solver轨迹捕获。",
          "- 动作delta与整体velocity必须分别看；delta非零不等于方向正确。这里只判断局部函数响应，不直接把latent cosine当视频左右方向，也不作18或12个独立样本的显著性推断。", "",
          "generated有历史12点中新student动作delta约占自身整体velocity的0.94%，teacher约1.47%；公共分量占主导，因此整体cos0.996与差分cos0.035并不矛盾。小差分仍受BF16数值影响；统一后端和重复误差0只限制一部分数值混淆，不能称为完全排除精度影响。", "",
          "[GT逐点报告](report/geometry_gt/README.md) · [generated逐点报告](report/geometry_step00_generated/README.md) · [全部汇总数值](geometry_summary.json) · [图](action_geometry_density.png)", "",
          "噪声采样密度对照不自动建立可信Stage1。完整generated30/8与停车场A/D视频仍须单独验收；未通过前不据此开展C/D效果训练或替换meeting。"]
(BASE / "GEOMETRY_RESULTS.md").write_text("\n".join(lines) + "\n")
print(json.dumps(groups, indent=2))
