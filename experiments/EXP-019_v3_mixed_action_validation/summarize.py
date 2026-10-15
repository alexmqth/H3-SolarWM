"""Summarize completed EXP-019 stage ledgers and saved CPU audits; no model calls."""
from __future__ import annotations

import csv
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASE = ROOT / "H3-World/outputs/EXP-019_v3_mixed_action_validation"
SCENES = ("43866101", "7199292c", "9dc2e588", "b784d995")
FIELDS = ("stage", "gpu", "status", "wall_seconds", "forward", "commit", "decode", "text",
          "sampling_nfe_fm30", "sampling_nfe_fm8", "sampling_seconds_fm30", "sampling_seconds_fm8",
          "peak_allocated_gib", "peak_reserved_gib", "cpu_kv_gib",
          "gt_flow_sum", "fm30_flow_sum", "fm8_flow_sum",
          "gt_boundary_mad", "fm30_boundary_mad", "fm8_boundary_mad")


rows = []
intervals = []
for stage in ("R", "E", *SCENES):
    budget = json.loads((BASE / stage / "budget.json").read_text())
    assert budget["status"] == "complete", stage
    assert budget["elapsed"] <= 900, stage
    counts = budget["counts"]
    row = {"stage": stage, "gpu": budget["gpu"], "status": budget["status"],
           "wall_seconds": budget["elapsed"],
           "forward": counts["forward"], "commit": counts["commit"],
           "decode": counts["decode"], "text": counts["text"],
           "peak_allocated_gib": budget["peak_allocated_gib"],
           "peak_reserved_gib": budget["peak_reserved_gib"]}
    intervals.append((budget["started"], budget["started"] + budget["elapsed"], stage))
    if stage in SCENES:
        result = json.loads((BASE / stage / "result.json").read_text())
        assert result["status"] == "COMPLETE_PENDING_JUDGE"
        runs = {r["nfe_per_chunk"]: r for r in result["runs"]}
        assert set(runs) == {30, 8}
        audit_name = "G1_CPU_AUDIT.json" if stage == SCENES[0] else f"G2_{stage}_CPU_AUDIT.json"
        audit = json.loads((HERE / audit_name).read_text())
        motion = audit["motion_auxiliary"]
        row.update({"sampling_nfe_fm30": 30, "sampling_nfe_fm8": 8,
                    "sampling_seconds_fm30": runs[30]["sampling_seconds"],
                    "sampling_seconds_fm8": runs[8]["sampling_seconds"],
                    "cpu_kv_gib": result["cpu_kv_bytes"] / 2**30,
                    "gt_flow_sum": motion["source"]["central_median_horizontal_flow_sum_c2"],
                    "fm30_flow_sum": motion["fm30"]["central_median_horizontal_flow_sum_c2"],
                    "fm8_flow_sum": motion["fm8"]["central_median_horizontal_flow_sum_c2"],
                    "gt_boundary_mad": motion["source"]["boundary_mad_38_39"],
                    "fm30_boundary_mad": motion["fm30"]["boundary_mad_38_39"],
                    "fm8_boundary_mad": motion["fm8"]["boundary_mad_38_39"]})
    rows.append(row)

assert sum(r["forward"] for r in rows) == 166
assert sum(r["text"] for r in rows) == 14
assert sum(r["decode"] for r in rows) == 8
assert sum(r["commit"] for r in rows) == 6  # included in forward count
total_gpu_seconds = sum(r["wall_seconds"] for r in rows)
assert total_gpu_seconds <= 5400
events = sorted([(start, 1, stage) for start, _, stage in intervals] +
                [(end, -1, stage) for _, end, stage in intervals])
active = max_active = 0
for _, change, _ in events:
    active += change
    max_active = max(max_active, active)
assert max_active <= 3

with (HERE / "METRICS_TABLE.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
summary = {"task": "EXP-019/v1", "rows": rows,
           "totals": {"forward": 166, "sampling_forward": 160, "commit_forward": 6,
                      "text_encoder_calls": 14, "vae_decodes": 8, "train_updates": 0,
                      "summed_single_gpu_wall_seconds": total_gpu_seconds,
                      "summed_single_gpu_wall_hours": total_gpu_seconds / 3600,
                      "max_concurrent_project_gpus": max_active},
           "timing_note": "Sum of independent GPU process wall times, not concurrent elapsed wall or pure CUDA kernel time."}
(HERE / "CUMULATIVE_BUDGET.json").write_text(json.dumps(summary, indent=2) + "\n")
print("EXP-019 complete:", summary["totals"])
