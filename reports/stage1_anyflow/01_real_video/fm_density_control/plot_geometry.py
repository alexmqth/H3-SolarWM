"""Plot complete paired density-control geometry reports; CPU, no model."""
import json
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
histories = ("gt", "step00_generated")
reports = {}
for history in histories:
    report = json.loads((BASE / "report" / f"geometry_{history}" / "metrics.json").read_text())
    assert report["status"] == "complete_matched_statistics"
    assert report["source_adaptation_audited"] and len(report["rows"]) == 18
    reports[history] = report

fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.8))
colors = ("#3969ac", "#da7c30", "#3e9651")
for col, history in enumerate(histories):
    # The user's fixed-history mechanism concerns chunks with past video.
    # Keep all 18 points in the raw report, but exclude 6 empty-history controls here.
    rows = [r for r in reports[history]["rows"] if r["chunk"] > 0]
    assert len(rows) == 12
    for row_index, metric in enumerate(("whole_cosine", "delta_cosine")):
        ax = axes[row_index, col]
        for i, row in enumerate(rows):
            offset = (i - 5.5) * 0.012
            values = [row[f"{role}_{metric}"] for role in ("shift12", "shift2_22")]
            ax.plot([offset, 1 + offset], values, color="0.82", linewidth=0.65, zorder=1)
            ax.scatter([offset, 1 + offset], values, color=colors[row["chunk"]], s=18, zorder=2)
        averages = [mean(r[f"{role}_{metric}"] for r in rows) for role in ("shift12", "shift2_22")]
        ax.scatter([0, 1], averages, marker="D", color="black", s=40, zorder=3)
        ax.text(0.5, 0.05, f"Means: {averages[0]:.6f} -> {averages[1]:.6f}",
                transform=ax.transAxes, ha="center", fontsize=9,
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"))
        ax.set_xticks([0, 1], ["FM48 shift12", "FM48 shift2.22"])
        ax.set_xlim(-0.35, 1.35)
        ax.grid(axis="y", alpha=0.2)
        if row_index == 0:
            ax.set_ylim(0.978, 1.001)
            ax.axhline(1, color="0.5", linestyle="--", linewidth=0.8)
            ax.set_title("GT clean history" if history == "gt" else "Fixed step00-generated history")
            ax.set_ylabel("Whole velocity cosine vs Original")
        else:
            ax.set_ylim(-0.45, 1.06)
            ax.axhline(0, color="0.5", linewidth=0.8)
            ax.axhline(1, color="0.5", linestyle="--", linewidth=0.8)
            ax.set_ylabel("Current A-D delta cosine vs Original")
            ax.text(0.5, 1.0, "Original identity = 1", ha="center", va="bottom", fontsize=8)

handles = [plt.Line2D([], [], color=colors[i], marker="o", linestyle="", label=f"chunk {i}")
           for i in (1, 2)]
handles.append(plt.Line2D([], [], color="black", marker="D", linestyle="", label="12-state mean"))
fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.065), frameon=False)
fig.suptitle("Whole-field fit and action geometry are different acceptance criteria", fontsize=13)
fig.text(0.5, 0.035, "2 scenes x 2 chunks with history x 3 sigmas; first-chunk controls excluded. No significance claim.",
         ha="center", fontsize=9)
fig.text(0.5, 0.012, "Original recomputes bidirectional history; student uses fixed raw KV. Interpolated states, not solver captures.",
         ha="center", fontsize=8)
fig.tight_layout(rect=(0, 0.12, 1, 0.96))
fig.savefig(BASE / "action_geometry_density.png", dpi=170)
fig.savefig(BASE / "action_geometry_density.svg")
