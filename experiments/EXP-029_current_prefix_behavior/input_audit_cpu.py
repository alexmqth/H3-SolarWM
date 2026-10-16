"""Read-only CPU audit for a possible EXP-029 A/D behavior pilot."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
EXP14 = ROOT / "submission/experiments/EXP-014_v3_multiscene_teacher"
OUT14 = ROOT / "H3-World/outputs/EXP-014_v3_multiscene_teacher"
sys.path.insert(0, str(ROOT / "H3-World/code"))
from causal.local_topology import visible_inputs


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


p1 = json.loads((OUT14 / "P1_result.json").read_text())
rows = []
for scene in ("s0_43866101", "s3_b784d995"):
    source = p1["fixtures"][scene]
    fixture_path = ROOT / source["path"]
    assert sha(fixture_path) == source["sha256"]
    fixture = torch.load(fixture_path, map_location="cpu", weights_only=True)
    assert fixture["format"] == "exp014_native_single_i0_full37_v1"
    assert fixture["scene"] == scene and fixture["initial_noise"].shape == (1,24,37,30,52)
    first_dir = OUT14 / "G1" / scene / "FM30"
    g1 = json.loads((first_dir / "result.json").read_text())
    first_path = first_dir / "first12.pt"
    rgb_path = first_dir / "published_39.npy"
    assert g1["status"] == "complete_pending_visual_review"
    assert sha(first_path) == g1["endpoint_sha256"]
    assert sha(rgb_path) == g1["published_sha256"]
    first = torch.load(first_path, map_location="cpu", weights_only=True)
    assert first.shape == (1,24,12,30,52) and torch.isfinite(first).all()
    rgb = np.load(rgb_path, mmap_mode="r")
    assert rgb.shape == (39,480,832,3)
    spans = fixture["packed"]["action_text_spans_local"]
    assert len(spans) == 37
    a = fixture["prompts"]["A"]
    d = fixture["prompts"]["D"]
    ad = a.clone()
    for lo, hi in spans[12:17]:
        ad[lo:hi] = d[lo:hi]
    c1_a, p1_a = visible_inputs(fixture["packed"], a, 12, 390)
    c1_ad, p1_ad = visible_inputs(fixture["packed"], ad, 12, 390)
    assert torch.equal(p1_a,p1_ad)
    assert all(torch.equal(c1_a[k],c1_ad[k]) for k in ("img_position_ids","token_tags",
        "img_pos","text_pos","audio_pos","action_text_rows"))
    c2_a,p2_a = visible_inputs(fixture["packed"],a,17,390)
    c2_ad,p2_ad = visible_inputs(fixture["packed"],ad,17,390)
    assert all(torch.equal(c2_a[k],c2_ad[k]) for k in ("img_position_ids","token_tags",
        "img_pos","text_pos","audio_pos","action_text_rows"))
    assert c2_a["seq_len"] == c2_a["action_video_start"] + 17*390
    assert len(c2_a["action_text_rows"]) == 17
    local_spans = c2_a["action_text_spans_local"]
    allowed = torch.zeros(len(p2_a),dtype=torch.bool)
    for lo,hi in local_spans[12:17]: allowed[lo:hi] = True
    assert torch.equal(p2_a[~allowed],p2_ad[~allowed])
    assert not torch.equal(p2_a[allowed],p2_ad[allowed])
    # Changing future action embeddings cannot affect the physically cropped C2 input.
    future_changed = a.clone()
    for lo,hi in spans[17:]: future_changed[lo:hi] = d[lo:hi]
    future_layout, future_prompt = visible_inputs(fixture["packed"],future_changed,17,390)
    assert torch.equal(future_prompt,p2_a)
    assert torch.equal(future_layout["img_position_ids"],c2_a["img_position_ids"])
    rows.append(dict(scene=scene, fixture_sha256=source["sha256"],
        first12_sha256=g1["endpoint_sha256"], published39_sha256=g1["published_sha256"],
        generated_c1_shape=list(first.shape), published39_shape=list(rgb.shape),
        c1_prompt_exact_same=True, c2_only_action_rows_differ=True,
        c2_current_noise_same=True, future_rows_physically_removed=True,
        action_sentences=fixture["action_sentences"],
        c2_noise_sha256=hashlib.sha256(fixture["initial_noise"][:,:,12:17].contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()))
result=dict(status="PASS_CPU_READ_ONLY",task="EXP-029 proposal input audit",gpu_calls=0,
            samples=rows, note="Same C1 raw KV still requires runtime commit/reuse verification; no GPU call made.")
(HERE / "EXP029_INPUT_AUDIT.json").write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
print(result["status"],len(rows))
