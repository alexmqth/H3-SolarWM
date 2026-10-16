"""CPU-only summary of the frozen EXP-028 recovery output."""
from __future__ import annotations
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from statistics import mean
from datetime import datetime

HERE = Path(__file__).resolve().parent
OUT = HERE / "recovery01"
rows = json.loads((OUT / "ROLE_RESULTS.json").read_text())
ledger = json.loads((OUT / "GPU_LEDGER.json").read_text())
scheduler = json.loads((OUT / "SCHEDULER_STATUS.json").read_text())
first_attempt = json.loads((HERE / "attempt00/SCHEDULER_STATUS.json").read_text())
assert len(rows) == 8 and ledger["status"] == "COMPLETE_PENDING_JUDGE"
assert ledger["counts"] == dict(dit=14, backward=8, commit=2, identity=4, update=0, video=0)
assert scheduler["returncode"] == 0
group = defaultdict(list)
for row in rows:
    assert set(row["layers"]) == {"0", "49"}
    for layer, block in row["layers"].items():
        assert block["pass_role_sum"] and block["grad_a_structural_zero"]
        assert block["cotangent_invocations"] == 1
        for qkv, cosine in block["action_video_cos"].items():
            if cosine is not None:
                group[(row["graph"], layer, qkv)].append(cosine)
mean_cos = {f"{g}/L{l}/{q}": mean(values) for (g,l,q),values in group.items()}
paired = []
for clip in sorted({x["clip"] for x in rows}):
    for sigma in (.25,.75):
        orig = next(x for x in rows if x["clip"] == clip and x["sigma"] == sigma and x["graph"] == "original_visible")
        v3 = next(x for x in rows if x["clip"] == clip and x["sigma"] == sigma and x["graph"] == "v3")
        diffs = {}
        for layer in ("0","49"):
            diffs[layer] = {q: None if orig["layers"][layer]["action_video_cos"][q] is None else
                v3["layers"][layer]["action_video_cos"][q] - orig["layers"][layer]["action_video_cos"][q]
                for q in "qkv"}
        paired.append(dict(clip=clip, sigma=sigma,
                           original_fm_loss=orig["loss"], v3_fm_loss=v3["loss"],
                           v3_minus_original_cosine=diffs))
source = (OUT / "ROLE_RESULTS.json").read_bytes()
first_seconds = (datetime.fromisoformat(first_attempt["finished_utc"])
                 - datetime.fromisoformat(first_attempt["launched_utc"])).total_seconds()
recovery_process_seconds = (datetime.fromisoformat(scheduler["finished_utc"])
                            - datetime.fromisoformat(scheduler["launched_utc"])).total_seconds()
summary = dict(task="EXP-028/v1", status="COMPLETE_PENDING_JUDGE",
               sample_count=2, sigma_count=2, graphs=["original_visible","v3"],
               completed_rows=len(rows), layer_rows=len(rows)*2,
               max_role_sum_relative_rms=max(b["relative_rms"] for r in rows for b in r["layers"].values()),
               max_identity_relative_rms=max(r["identity_relative_rms"] for r in rows if r["identity_relative_rms"] is not None),
               mean_action_video_cosine=mean_cos, paired=paired,
               calls=ledger["counts"], ledger_elapsed_seconds_recovery=ledger["elapsed_seconds"],
               recovery_process_wall_seconds=recovery_process_seconds,
               ledger_vs_process_gap_seconds=recovery_process_seconds-ledger["elapsed_seconds"],
               prior_failed_startup_seconds_exact=first_seconds,
               prior_failed_startup_seconds_charged=4,
               cumulative_process_wall_seconds_exact=first_seconds+recovery_process_seconds,
               cumulative_process_wall_seconds_charged=4+recovery_process_seconds,
               peak_allocated_gib=ledger["peak_allocated_gib"],
               peak_reserved_gib=ledger["peak_reserved_gib"], peak_rss_gib_recorded=ledger["rss_gib"],
               shared_gpu=True, results_sha256=hashlib.sha256(source).hexdigest(),
               exact_role_dot_products_persisted=False,
               note="No video/behavior test; cross-graph gradient matrices and exact role dots were not persisted.")
(OUT / "SUMMARY.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False)+"\n")
print(OUT / "SUMMARY.json")
