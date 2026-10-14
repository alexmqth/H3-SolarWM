"""Independent CPU audit of EXP-019 E fixtures; no model inference."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = ROOT / "H3-World/outputs/EXP-019_v3_mixed_action_validation/E"
FROZEN = ROOT / "H3-World/outputs/2026-10-09-22/chunk_partition_cb/runtime"
FIX = ROOT / "H3-World/outputs/EXP-014_v3_multiscene_teacher/fixtures"
REAL = ROOT / "submission/experiments/EXP-015_v3_real56_data/artifacts"
VAE = ROOT / "H3-World/outputs/EXP-016_v3_real56_vae"
SCENES = ("43866101", "7199292c", "9dc2e588", "b784d995")
sys.path[:0] = [str(HERE.parent / "EXP-017_v3_fixed_action_layout"),
                str(FROZEN / "code")]

from fixed_layout import action_ids, candidate, frozen_cpu_functions, same_layout, semantic_parts, TOKENIZER_DIR
from causal.local_topology import visible_inputs
from transformers import AutoTokenizer


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def tensor_sha(t: torch.Tensor) -> str:
    t = t.detach().cpu().contiguous()
    return hashlib.sha256(str((tuple(t.shape), str(t.dtype))).encode() +
                           t.view(torch.uint8).numpy().tobytes()).hexdigest()


def patch(latent: torch.Tensor) -> torch.Tensor:
    b, c, ft, fh, fw = latent.shape
    return torch.einsum("nctrhpwq->nthwcrpq", latent.reshape(b, c, ft, 1, fh // 2, 2, fw // 2, 2)).reshape(
        b * ft * (fh // 2) * (fw // 2), c * 4).contiguous()


budget = json.loads((OUT / "budget.json").read_text())
result = json.loads((OUT / "result.json").read_text())
assert budget["status"] == "complete" and result["status"] == "PASS"
assert budget["counts"] == {"forward": 0, "commit": 0, "decode": 0, "text": 14}
assert len([e for e in budget["events"] if e["kind"] == "text"]) == 14
assert budget["elapsed"] <= 900 and budget["peak_allocated_gib"] <= 44
assert [r["scene"] for r in result["fixtures"]] == list(SCENES)

builder, visible = frozen_cpu_functions()
tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR), local_files_only=True)
rows = []
for i, (scene, recorded) in enumerate(zip(SCENES, result["fixtures"])):
    path = OUT / f"{scene}.pt"
    assert sha(path) == recorded["sha256"]
    f = torch.load(path, map_location="cpu", weights_only=True)
    old = torch.load(FIX / f"s{i}_{scene}.pt", map_location="cpu", weights_only=True)
    script = json.loads((REAL / scene / "action_script17.json").read_text())
    assert f["scene"] == scene and f["script"] == script
    assert f["source_I0_sha256"] == sha(REAL / scene / "I0.png")
    assert f["source_I0_sha256"] != old["source_png_sha256"]
    assert torch.equal(f["initial_noise"], old["initial_noise"])
    assert torch.equal(f["audio_noise"], old["audio_noise"])
    packed, prompt = f["packed"], f["prompt"]
    spans = packed["action_text_spans_local"]
    assert len(spans) == 37 and spans[0][0] == recorded["head_rows"]
    assert all(spans[j][1] == spans[j + 1][0] for j in range(36))
    assert torch.equal(packed["action_text_rows"], torch.tensor(spans, dtype=packed["action_text_rows"].dtype))
    assert spans[-1][1] == recorded["text_rows"] == len(prompt) == len(packed["text_pos"])
    assert [hi - lo for lo, hi in spans[:17]] == recorded["token_lengths"]
    assert prompt.shape[1] == 5120 and prompt.dtype == torch.bfloat16 and torch.isfinite(prompt).all()
    text_tags = packed["token_tags"][packed["text_pos"]]
    assert torch.all(text_tags[spans[0][0]:] == 1)
    assert not torch.equal(prompt[:spans[0][0]], old["prompts"]["A"][:spans[0][0]])

    i0 = torch.load(VAE / scene / "I0.pt", map_location="cpu", weights_only=True)
    clean = patch(i0).to(torch.bfloat16)
    noise = torch.randn((1, 24, 38, 30, 52), generator=torch.Generator("cpu").manual_seed(13),
                        dtype=torch.bfloat16)[:, :, :1]
    noise_rows = patch(noise)
    sigma = torch.tensor(.999, dtype=torch.bfloat16)
    reproduced = (sigma * clean + (1 - sigma) * noise_rows).contiguous()
    assert f["anchor"].shape == (390, 96) and torch.isfinite(f["anchor"]).all()
    assert torch.equal(f["anchor"], reproduced)
    assert tensor_sha(f["anchor"]) == recorded["anchor_sha256"]
    effective_clean = torch.equal(f["anchor"], clean)
    assert float(sigma) == 1.0 and effective_clean

    checks = []
    parts = []
    full_lines = script + [old["action_sentences"]["A"]] * 20
    for stop in (12, 17):
        current, current_prompt = visible_inputs(packed, prompt, stop, 390)
        assert same_layout(current, visible(packed, prompt, stop, 390)[0])
        prefix_end = spans[stop - 1][1]
        altered = prompt.clone()
        altered[prefix_end:] = 0
        alt_layout, alt_prompt = visible_inputs(packed, altered, stop, 390)
        assert same_layout(current, alt_layout) and torch.equal(current_prompt, alt_prompt)
        # Counterfactual future uses a *real already encoded Qwen action row*
        # from this same fixture; no new text model calls are needed.
        original_future = full_lines[stop]
        original_length = spans[stop][1] - spans[stop][0]
        alternatives = [(line, j) for j, line in enumerate(full_lines) if line != original_future]
        alternatives.sort(key=lambda item: (spans[item[1]][1] - spans[item[1]][0] == original_length, item[1]))
        alt_line, encoded_index = alternatives[0]
        encoded_lo, encoded_hi = spans[encoded_index]
        real_alt_embedding = prompt[encoded_lo:encoded_hi]
        alt_ids = action_ids(tokenizer, full_lines[:stop] + [alt_line] * (37 - stop))
        changed, _ = candidate(builder, old["packed"], alt_ids)
        changed_prompt = torch.cat([prompt[:prefix_end]] +
                                   [real_alt_embedding] * (37 - stop))
        assert len(changed_prompt) == len(changed["text_pos"])
        changed_visible, changed_rows = visible_inputs(changed, changed_prompt, stop, 390)
        assert same_layout(current, changed_visible) and torch.equal(current_prompt, changed_rows)
        parts.append(semantic_parts(current, 12))
        checks.append({"stop": stop, "future_embedding_values_ignored": True,
                       "future_actual_encoded_action_row_ignored": True,
                       "future_token_length_changed": len(real_alt_embedding) != original_length,
                       "future_token_length_ignored": True,
                       "counterfactual_action_embedding_source_index": encoded_index})
    assert all(torch.equal(parts[0][key], parts[1][key]) for key in parts[0])
    rows.append({
        "scene": scene,
        "fixture_sha256": recorded["sha256"],
        "source_I0_sha256": f["source_I0_sha256"],
        "new_I0_differs_from_old_canonical_I0": True,
        "head_reencoded_and_differs_from_old_fixture": True,
        "head_rows": recorded["head_rows"],
        "action_span_count": len(spans),
        "real_action_token_lengths": recorded["token_lengths"],
        "future_checks": checks,
        "semantic_history_stable": True,
        "anchor_formula_exact": True,
        "anchor_sha256": recorded["anchor_sha256"],
        "effective_sigma_bfloat16": float(sigma),
        "effective_anchor_equals_clean_I0_rows": effective_clean,
        "initial_video_audio_noise_unchanged_from_canonical_fixture": True,
    })

audit = {
    "status": "PASS",
    "scope": "CPU audit of saved E fixtures; no Qwen/DiT calls",
    "counts": budget["counts"],
    "elapsed_seconds": budget["elapsed"],
    "peak_allocated_gib": budget["peak_allocated_gib"],
    "peak_reserved_gib": budget["peak_reserved_gib"],
    "rows": rows,
    "limitation": "Future alternatives reuse real independently encoded Qwen action rows within each saved fixture; no second Qwen call for a new sentence. Configured .999 rounds to 1.0 in bfloat16, so effective anchor is clean I0 rows.",
}
(HERE / "E_CPU_AUDIT.json").write_text(json.dumps(audit, indent=2) + "\n")
print("E CPU audit PASS: 4 fixtures, 14 text calls, exact anchor and prefix isolation")
