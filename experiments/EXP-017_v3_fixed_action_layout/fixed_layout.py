"""CPU-only candidate: variable physical action rows, canonical semantic coordinates.

This does not patch the H3 runtime or run any model. Its only construction
primitive is the exact frozen PackedSequenceBuilder class extracted by AST.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RUNTIME = ROOT / "H3-World/outputs/2026-10-09-22/chunk_partition_cb/runtime"
BUILDER_FILE = RUNTIME / "DiffSynth-Studio-h3-v2/diffsynth/pipelines/minimax_h3_audio_video.py"
VISIBLE_FILE = RUNTIME / "code/causal/local_topology.py"
TOKENIZER_DIR = RUNTIME / "DiffSynth-Studio-h3-v2/models/MiniMax/MiniMax-H3/FL2VA/processor"
EXP014 = ROOT / "H3-World/outputs/EXP-014_v3_multiscene_teacher/fixtures"
EXP015 = ROOT / "submission/experiments/EXP-015_v3_real56_data/artifacts"
SCENES = ("43866101", "7199292c", "9dc2e588", "b784d995")
FIXTURES = ("s0_43866101.pt", "s1_7199292c.pt", "s2_9dc2e588.pt", "s3_b784d995.pt")
FRAME_ROWS = 390


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def frozen_cpu_functions():
    class PipelineUnit:
        def __init__(self, *args, **kwargs):
            pass
    ns = {"np": np, "torch": torch, "PipelineUnit": PipelineUnit,
          "MiniMaxH3Pipeline": object}
    for path, name in ((BUILDER_FILE, "MiniMaxH3Unit_PackedSequenceBuilder"),
                       (VISIBLE_FILE, "visible_inputs")):
        node = next(x for x in ast.parse(path.read_text()).body if getattr(x, "name", None) == name)
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), ns)
    return ns["MiniMaxH3Unit_PackedSequenceBuilder"](), ns["visible_inputs"]


def action_ids(tokenizer, lines: list[str]) -> list[list[int]]:
    return [list(tokenizer(line, add_special_tokens=False)["input_ids"]) for line in lines]


def candidate(builder, canonical: dict, ids: list[list[int]]) -> tuple[dict, torch.Tensor]:
    """Build physical rows, then assign fixed full37 semantic positions.

    The canonical full37 fixture is chosen before these action IDs exist.
    Only its layout metadata is read; no future actual script length chooses
    any semantic origin. A zero-valued head token sentinel is for CPU tests,
    not a substitute for actual text embeddings at model runtime.
    """
    if len(ids) != 37 or any(not row for row in ids):
        raise ValueError("Expected 37 nonempty independently tokenized actions")
    head = int(canonical["action_text_spans_local"][0][0])
    spans = []
    cursor = head
    for row in ids:
        spans.append((cursor, cursor + len(row)))
        cursor += len(row)
    audio_t = len(canonical["audio_pos"]) // 2
    # Ask the frozen builder only for physical rows. Its action-aware path
    # validates an origin computed from the (unknown) future total text
    # length before coordinates can be overridden; that check is invalid
    # for this fixed-coordinate candidate when future sentences are short.
    p = builder._build_packed_fl2va(cursor, 37, 30, 52, audio_t, [0],
                                   action_text_spans=None)
    p["action_text_rows"] = torch.tensor(spans, dtype=torch.long)
    p["action_text_spans_local"] = spans
    p["action_video_start"] = int(p["img_pos"][FRAME_ROWS])
    p["action_frame_rows"] = FRAME_ROWS
    p["action_real_used"] = int(p["cu_seqlens"][1])
    cp = canonical
    g, cg = p["img_position_ids"], cp["img_position_ids"]
    # Head includes Qwen image presentation rows: preserve its text/vision
    # tags and its semantic PRoPE coordinates, including all three axes.
    g[:, :head] = cg[:, :head]
    p["token_tags"][:head] = cp["token_tags"][:head]
    for (lo, hi), (clo, chi) in zip(spans, cp["action_text_spans_local"]):
        coordinate = cg[:, int(clo):int(chi)]
        if not torch.equal(coordinate, coordinate[:, :1].expand_as(coordinate)):
            raise RuntimeError("canonical action span has nonconstant semantic position")
        g[:, lo:hi] = coordinate[:, :1]
    if len(p["img_pos"]) != len(cp["img_pos"]) or len(p["audio_pos"]) != len(cp["audio_pos"]):
        raise RuntimeError("condition/audio/video row count changed")
    g[:, p["img_pos"]] = cg[:, cp["img_pos"]]
    g[:, p["audio_pos"]] = cg[:, cp["audio_pos"]]
    # Physical row positions are new and are intentionally not copied.
    # The prompt sentinel encodes the exact current action token IDs so
    # future-content tests compare more than metadata shapes.
    prompt_ids = torch.tensor([0] * head + [token for row in ids for token in row], dtype=torch.long)
    assert len(prompt_ids) == len(p["text_pos"])
    return p, prompt_ids


def same_layout(left: dict, right: dict) -> bool:
    if left.keys() != right.keys():
        return False
    return all(torch.equal(left[k], right[k]) if torch.is_tensor(left[k])
               else left[k] == right[k] for k in left)


def semantic_parts(packed: dict, stop: int) -> dict[str, torch.Tensor]:
    g = packed["img_position_ids"][0]
    spans = packed["action_text_spans_local"]
    return {"head": g[packed["text_pos"][:spans[0][0]]].clone(),
            "actions": torch.cat([g[packed["text_pos"][lo:hi]] for lo, hi in spans[:stop]]),
            "I0": g[packed["img_pos"][:FRAME_ROWS]].clone(),
            "audio": g[packed["audio_pos"]].clone(),
            "video": g[packed["img_pos"][FRAME_ROWS:FRAME_ROWS*(stop+1)]].clone()}


def main() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("EXP-017 is CPU-only")
    cpus = sorted(os.sched_getaffinity(0))[:4]
    os.sched_setaffinity(0, cpus)
    torch.set_num_threads(4)
    builder, visible_inputs = frozen_cpu_functions()
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR), local_files_only=True)
    rows = []
    for scene, fixture_name in zip(SCENES, FIXTURES):
        fixture_path = EXP014 / fixture_name
        script_path = EXP015 / scene / "action_script17.json"
        fixture = torch.load(fixture_path, map_location="cpu", weights_only=False)
        canonical = fixture["packed"]
        a_sentence, d_sentence = fixture["action_sentences"]["A"], fixture["action_sentences"]["D"]
        a_ids = action_ids(tokenizer, [a_sentence] * 37)
        d_ids = action_ids(tokenizer, [d_sentence] * 37)
        assert all(len(x) == canonical["action_text_spans_local"][0][1] -
                   canonical["action_text_spans_local"][0][0] for x in a_ids + d_ids)
        canon_candidate, _ = candidate(builder, canonical, a_ids)
        assert same_layout(canon_candidate, canonical)
        regression = {}
        for label in ("A", "D"):
            prompt = fixture["prompts"][label]
            assert len(prompt) == len(canonical["text_pos"])
            for stop in (12, 17, 37):
                old, old_prompt = visible_inputs(canonical, prompt, stop, FRAME_ROWS)
                new, new_prompt = visible_inputs(canon_candidate, prompt, stop, FRAME_ROWS)
                assert same_layout(old, new) and torch.equal(old_prompt, new_prompt)
                regression[f"{label}_stop{stop}"] = True

        real17 = json.loads(script_path.read_text())
        assert len(real17) == 17
        # Only a fixed, declared tail placeholder; no observed future
        # action or future text length enters the template.
        actual_lines = real17 + [a_sentence] * 20
        actual_ids = action_ids(tokenizer, actual_lines)
        packed, ids = candidate(builder, canonical, actual_ids)
        assert len(packed["action_text_rows"]) == 37
        assert all(int(lo) < int(hi) <= len(ids) for lo, hi in packed["action_text_rows"])
        assert torch.unique(packed["text_pos"]).numel() == len(packed["text_pos"])
        assert torch.unique(packed["img_pos"]).numel() == len(packed["img_pos"])
        assert torch.unique(packed["audio_pos"]).numel() == len(packed["audio_pos"])

        # Future interventions change both content and length. The first
        # deliberately shortens C2; the second lengthens unknown tail.
        changed_c2 = list(actual_lines)
        changed_c2[12:17] = ["the man stands still, camera holds steady"] * 5
        changed_c2[17:] = ["the man walks forward and strafes left while running, camera pans right sharply"] * 20
        other12, other_ids12 = candidate(builder, canonical, action_ids(tokenizer, changed_c2))
        changed_tail = list(actual_lines)
        changed_tail[17:] = ["the man walks backward and strafes right while running, camera pans left sharply"] * 20
        other17, other_ids17 = candidate(builder, canonical, action_ids(tokenizer, changed_tail))
        shortest = list(actual_lines)
        shortest[12:] = ["A"] * 25
        short_pack, short_ids = candidate(builder, canonical, action_ids(tokenizer, shortest))
        assert len(other_ids12) != len(ids) and len(other_ids17) != len(ids)
        assert len(short_ids) < len(ids)
        vis12, tok12 = visible_inputs(packed, ids[:, None], 12, FRAME_ROWS)
        alt12, alt_tok12 = visible_inputs(other12, other_ids12[:, None], 12, FRAME_ROWS)
        vis17, tok17 = visible_inputs(packed, ids[:, None], 17, FRAME_ROWS)
        alt17, alt_tok17 = visible_inputs(other17, other_ids17[:, None], 17, FRAME_ROWS)
        short12, short_tok12 = visible_inputs(short_pack, short_ids[:, None], 12, FRAME_ROWS)
        assert same_layout(vis12, alt12) and torch.equal(tok12, alt_tok12)
        assert same_layout(vis12, short12) and torch.equal(tok12, short_tok12)
        assert same_layout(vis17, alt17) and torch.equal(tok17, alt_tok17)
        assert len(vis12["action_text_rows"]) == 12 and len(vis17["action_text_rows"]) == 17
        assert vis12["seq_len"] == vis12["action_video_start"] + 12*FRAME_ROWS
        assert vis17["seq_len"] == vis17["action_video_start"] + 17*FRAME_ROWS
        parts12, parts17 = semantic_parts(vis12, 12), semantic_parts(vis17, 12)
        assert all(torch.equal(parts12[k], parts17[k]) for k in parts12)
        for stop, vis in ((12, vis12), (17, vis17)):
            ranges = vis["action_text_rows"].tolist()
            assert len(ranges) == stop and all(0 <= lo < hi <= vis["action_video_start"] for lo, hi in ranges)
            assert all(ranges[k][1] <= ranges[k+1][0] for k in range(stop-1))
            assert len(vis["img_pos"]) == (stop+1)*FRAME_ROWS
        rows.append({"scene": scene, "fixture_sha256": sha(fixture_path),
                     "action_script_sha256": sha(script_path),
                     "real_token_lengths_17": [len(x) for x in actual_ids[:17]],
                     "canonical_action_length": len(a_ids[0]),
                     "fixed_unknown_tail_contract": {"sentence": a_sentence, "spans": 20},
                     "physical_text_rows_actual": len(ids),
                     "physical_text_rows_changed_C2": len(other_ids12),
                     "physical_text_rows_changed_tail": len(other_ids17),
                     "physical_text_rows_shortest_future": len(short_ids),
                     "canonical_full_packed_all_fields_exact": True,
                     "canonical_visible_A_D_12_17_37_exact": regression,
                     "C1_future_content_and_length_isolated": True,
                     "C1_future_25_one_token_spans_isolated": True,
                     "C2_future_content_and_length_isolated": True,
                     "C1_history_semantic_coordinates_preserved_at_C2": True,
                     "future_action_video_rows_physically_removed": True,
                     "action_row_indices_nonoverlap_in_bounds": True})
        print(scene, "PASS", rows[-1]["real_token_lengths_17"], flush=True)
    manifest = {"task": "EXP-017/v1", "status": "CPU_CANDIDATE_CHECKS_PASS",
                "GPU_calls": 0, "model_forwards": 0, "training_updates": 0,
                "cpu_affinity": cpus, "source_sha256": {
                  "frozen_builder": sha(BUILDER_FILE), "frozen_visible_inputs": sha(VISIBLE_FILE),
                  "tokenizer_json": sha(TOKENIZER_DIR / "tokenizer.json")},
                "scenes": rows,
                "not_tested": ["actual text encoder embeddings for real mixed action",
                               "DiT/token-refiner forward and backend masks",
                               "raw KV numerical equivalence under new variable-row layout",
                               "generated video visual/action fidelity"]}
    target = HERE / "CPU_RESULTS.json"
    if target.exists():
        raise FileExistsError(target)
    target.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
