"""Historical AnyFlow scalar/output-gradient audit; no H3 model or new training."""
import ast
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median

import torch

BASE = Path(__file__).resolve().parent
data = json.loads((BASE / "training128.json").read_text())
shapes = json.loads((BASE / "input_shapes.json").read_text())
assert data["status"] == "complete" and len(data["updates"]) == 128
assert [u["step"] for u in data["updates"]] == list(range(1, 129))
assert data["config"]["logical_batch"] == 4

# Execute the unchanged function AST from the actual frozen training runtime.
source = (BASE / "reference_anyflow.py").read_text()
node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "adaptive_scale")
namespace = {"torch": torch}
exec(compile(ast.Module(body=[node], type_ignores=[]), "reference_anyflow.py", "exec"), namespace)
adaptive_scale = namespace["adaptive_scale"]

rows = []
max_scale_error = max_weighted_relative_error = 0.0
for update in data["updates"]:
    samples = update["samples"]
    assert [s["sample_type"] for s in samples] == ["diffusion", "diffusion", "endpoint", "flow_map"]
    references = [s["raw_loss"] for s in samples[:2]]
    shape = shapes[update["action"]]["shape"]
    count = min(5, shape[2] - update["chunk"] * 5)
    elements = math.prod([*shape[:2], count, *shape[3:]])
    for index, sample in enumerate(samples):
        raw = torch.tensor(sample["raw_loss"], dtype=torch.float32)
        scale = adaptive_scale(raw, is_diffusion=index < 2, diffusion_losses=references)
        expected = float(scale)
        scale_error = abs(expected - sample["adaptive_scale"])
        assert scale_error == 0, (update["step"], index, scale_error)
        weighted = float(raw * scale * torch.tensor(sample["weight"], dtype=torch.float32))
        weighted_error = abs(weighted - sample["weighted_loss"]) / max(abs(weighted), 1e-30)
        assert weighted_error < 2e-6
        max_scale_error = max(max_scale_error, scale_error)
        max_weighted_relative_error = max(max_weighted_relative_error, weighted_error)
        # Target derivative and adaptive scale are detached in the real loss.
        # For e = prediction + stop_gradient(derivative term) - target,
        # l = alpha*w*mean(e^2)/B, ||d l/d prediction||_2 = 2*alpha*w/B*sqrt(L/N).
        no_adapt = 2 * sample["weight"] / 4 * math.sqrt(sample["raw_loss"] / elements)
        row = dict(step=update["step"], action=update["action"], chunk=update["chunk"],
                   sample_index=index, elements=elements, diffusion_reference=mean(references),
                   **sample, output_grad_l2=no_adapt * expected,
                   output_grad_l2_without_adaptive_scaling=no_adapt)
        row["noise_band"] = ("low" if row["sigma"] <= .24078089904785155 else
                             "mid" if row["sigma"] <= .6894410400390625 else "high")
        rows.append(row)

# Verify the analytical output derivative against autograd at actual logged
# scalar magnitudes, including the largest raw residual and shortest chunk.
# This does not recreate the H3 prediction vector or its parameter Jacobian.
checks = []
selected = [max(rows, key=lambda r: r["raw_loss"]),
            min((r for r in rows if r["sample_type"] == "endpoint"), key=lambda r: r["sigma"]),
            next(r for r in rows if r["sample_type"] == "flow_map" and r["chunk"] == 2)]
for row in selected:
    n = row["elements"]
    prediction = torch.full((n,), math.sqrt(row["raw_loss"]), dtype=torch.float64, requires_grad=True)
    raw = prediction.square().mean()
    scale = adaptive_scale(raw, is_diffusion=False, diffusion_losses=[row["diffusion_reference"]])
    loss = raw * scale * row["weight"] / 4
    grad, = torch.autograd.grad(loss, prediction)
    expected = 2 * float(scale) * row["weight"] / 4 * math.sqrt(float(raw.detach()) / n)
    assert math.isclose(float(grad.norm()), expected, rel_tol=1e-10, abs_tol=1e-14)
    # Negative control: accidentally differentiate through ref/(raw+eps).
    wrong_raw = prediction.square().mean()
    incorrect = wrong_raw * row["diffusion_reference"] / (wrong_raw + 1e-5) * row["weight"] / 4
    wrong, = torch.autograd.grad(incorrect, prediction)
    ratio = float(wrong.norm() / grad.norm())
    assert math.isclose(ratio, 1e-5 / (float(raw.detach()) + 1e-5), rel_tol=5e-6, abs_tol=1e-10)
    checks.append(dict(step=row["step"], sample_type=row["sample_type"], elements=n,
                       autograd_output_norm=float(grad.norm()), analytic_output_norm=expected,
                       incorrect_non_detached_norm_ratio=ratio))

groups = []
for kind in ("diffusion", "endpoint", "flow_map"):
    for band in ("all", "low", "mid", "high"):
        selected = [r for r in rows if r["sample_type"] == kind and (band == "all" or r["noise_band"] == band)]
        group = dict(sample_type=kind, noise_band=band, count=len(selected))
        for key in ("raw_loss", "weight", "adaptive_scale", "weighted_loss", "output_grad_l2",
                    "output_grad_l2_without_adaptive_scaling"):
            group[f"mean_{key}"] = mean(r[key] for r in selected)
            group[f"median_{key}"] = median(r[key] for r in selected)
        group["sum_output_norm_ratio"] = sum(r["output_grad_l2"] for r in selected) / sum(
            r["output_grad_l2_without_adaptive_scaling"] for r in selected)
        groups.append(group)

with (BASE / "samples.csv").open("w") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
result = dict(status="passed", optimizer_updates_added=0, gpu_used=False, historical_samples=512,
              scale_replay_max_error=max_scale_error, weighted_replay_max_relative_error=max_weighted_relative_error,
              checks=checks, groups=groups,
              limitation="Output-gradient norm only. H3 parameter Jacobians, gradient angles, clipping and Adam updates not reconstructed.",
              source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                             (BASE / "training128.json", BASE / "reference_anyflow.py", BASE / "reference_sample_parallel.py",
                              BASE / "input_shapes.json", Path(__file__))})
(BASE / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
lines = ["# C准备：已有AnyFlow128的自适应权重及输出梯度核查", "",
         "CPU-only只读核查，不启动训练或新的GPU工作。使用旧AnyFlow128的512个实际样本，不是当前真实ABot FM48，也没有宣称可靠causal checkpoint上的C已经完成。", "",
         "## 直接结果", "",
         f"512个样本的adaptive scale与冻结训练函数逐值重放误差为{max_scale_error:g}；weighted loss最大相对重放误差{max_weighted_relative_error:.3g}。", "",
         "| 类型 | noise | n | Raw loss均值 | Adaptive scale中位数 | 输出梯度L2均值 | 去掉adaptive时的L2均值 | 两列L2之和比 |",
         "|---|---|---:|---:|---:|---:|---:|---:|"]
for g in groups:
    lines.append(f"| {g['sample_type']} | {g['noise_band']} | {g['count']} | {g['mean_raw_loss']:.6g} | "
                 f"{g['median_adaptive_scale']:.6g} | {g['mean_output_grad_l2']:.6g} | "
                 f"{g['mean_output_grad_l2_without_adaptive_scaling']:.6g} | {g['sum_output_norm_ratio']:.6g} |")
lines += ["", "low≤0.240781，mid≤0.689441，high>0.689441。跨训练状态的描述性分组，不是固定输入验证或独立样本统计。", "",
          "## 计算的是什么", "",
          "实际loss为`alpha * w * mean(e²) / B`，其中B=4；e中finite-difference target和alpha均detach。因此对本次有梯度的预测velocity，`||d loss/d prediction||₂ = 2*alpha*w/B*sqrt(raw_loss/N)`。N由实际历史teacher latent shape与当前chunk长度得到，包含末chunk长度2，非统一假设为5。", "",
          "本表仅重建输出端梯度范数。真实参数梯度还要乘H3的Jacobian；不同样本可能方向冲突，之后还有clip/Adam。列中范数相加不等于合并后的梯度范数，不能把百分比叫做参数梯度贡献或训练预算占比。去掉adaptive的一列仅是同一保存状态下的数学反事实，没有运行无adaptive训练。", "",
          "3个实际标量/维度的autograd核查确认输出导数公式；故意不detach的负对照会额外乘`eps/(raw+eps)`，冻结源码实际没有此错误。这不是实际33B参数梯度实验。", "",
          "## 对后续的意义", "",
          "高raw residual的非对角样本会被自适应缩放压低输出梯度，weighted loss接近FM参考不能证明finite-map误差已小。这个行为符合现有实现，尚不能判断是否过度压制，更不证明它导致普通FM的action geometry失配。", "",
          "此处raw residual包含有限差分导数项，不是单纯velocity MSE或GT endpoint误差。高噪声endpoint的108样本raw均值64.381、weighted均值0.06978；同状态下输出梯度范数之和约为数学上去掉adaptive的0.935%。这既提示不能只看weighted loss，也说明直接移除adaptive可能放大高残差梯度，不能据此自动改训练。", "",
          "进入可信causal checkpoint的C验证时，继续同时记录raw residual、diagonal保真、teacher/GT端点距离与composition；若要调整adaptive策略，必须单变量比较真实参数梯度和完整视频，不能直接删除权重。B局部action前置仍未通过，本核查不授权自动扩训128/136。", "",
          "初版CPU负对照复用了已释放的计算图而失败；改为从同一prediction叶子独立重建raw loss后通过。首次源码/失败记录保留在initial_audit.py及initial_failure.json；没有改生产模型或运行中训练。", "",
          "[完整收据](audit.json) · [512样本](samples.csv) · [历史训练记录](training128.json) · [检查源码](audit.py)"]
(BASE / "README.md").write_text("\n".join(lines) + "\n")
print(json.dumps(dict(status=result["status"], samples=len(rows), checks=checks, groups=groups), indent=2))
