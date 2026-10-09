"""CPU checks for multiwindow action routing, visibility, and resume semantics."""

from pathlib import Path
import sys

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
FROZEN = Path(__import__("json").loads((HERE / "config.yaml").read_text())["frozen_source"])
sys.path[:0] = [str(FROZEN / "runtime/code"),
                str(FROZEN / "runtime/DiffSynth-Studio-h3-v2")]

from causal.h3_training import make_small_h3, synthetic_h3_batch
from causal.h3_precision import configure_precision
from causal.local_topology import visible_inputs
from interval_forward import interval_forward
from rollout_contract import PATHS, RANGES, prompt_for_path, rgb_sha, rgb_stop, stitch_immutable


def test_action_partition_and_resume():
    assert set(PATHS) == {"AA", "DD", "AD", "DA"}
    assert [rgb_stop(x) for x in (12, 17, 22, 27, 32, 37)] == [39, 56, 73, 90, 107, 124]
    assert [x for x, _ in RANGES] == [17, 22, 27, 32]
    rng = np.random.default_rng(15)
    prefix = rng.integers(0, 256, size=(56, 4, 6, 3), dtype=np.uint8)
    original = prefix.copy()
    full = rng.integers(0, 256, size=(73, 4, 6, 3), dtype=np.uint8)
    new = stitch_immutable(prefix, full, 17, 22)
    assert rgb_sha(new[:56]) == rgb_sha(original)
    assert np.array_equal(new[56:], full[56:])
    assert not np.array_equal(prefix[-5:], full[51:56])
    with pytest.raises(AssertionError):
        stitch_immutable(prefix[:55], full, 17, 22)


def test_real_fixture_history_actions_and_layout():
    data = {a: torch.load(FROZEN / f"source_coarse/inputs/parking_{a}.pt",
                          map_location="cpu", weights_only=True) for a in "AD"}
    for key in ("initial_noise", "audio_noise", "anchor"):
        assert torch.equal(data["A"][key], data["D"][key])
    for key in ("img_position_ids", "action_text_rows", "text_pos"):
        assert torch.equal(data["A"]["packed"][key], data["D"]["packed"][key])
    spans = data["A"]["packed"]["action_text_spans_local"]
    for path, (first, later) in PATHS.items():
        for stop in (17, 22, 27, 32, 37):
            text = prompt_for_path(data, path, stop)
            base = data[first]["pairs"][0]["prompts"][first]
            donor = data[later]["pairs"][0]["prompts"][later]
            assert torch.equal(text[:spans[11][1]], base[:spans[11][1]])
            for lo, hi in spans[12:stop]:
                assert torch.equal(text[lo:hi], donor[lo:hi])
            for lo, hi in spans[stop:]:
                assert torch.equal(text[lo:hi], base[lo:hi])
            visible, trimmed = visible_inputs(data[first]["packed"], text, stop, 390)
            assert len(visible["action_text_rows"]) == stop
            assert trimmed.shape[0] == int(visible["text_pos"].shape[0])


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_new_intervals_visibility_and_readonly(dtype):
    torch.set_num_threads(2)
    torch.manual_seed(1501)
    model = make_small_h3().eval().to(dtype).requires_grad_(False)
    configure_precision(model, "h3_fp32")
    fixture = synthetic_h3_batch(frames=37, seed=321)
    text = fixture["prompt_embeds"].to(dtype)
    spans = fixture["packed"]["action_text_spans_local"]
    with torch.no_grad():
        for start, stop in RANGES:
            layout, prompt = visible_inputs(fixture["packed"], text, stop, 4)
            assert len(layout["action_text_rows"]) == stop
            assert int(layout["img_position_ids"].shape[1]) == layout["seq_len"]
            history = fixture["clean_video"][:, :, :start].float().clone()
            noise = fixture["noise"][:, :, :start].float().clone()
            state = fixture["noise"][:, :, start:stop].float().clone()
            before = [x.clone() for x in (history, noise, state)]
            kw = dict(start=start, history=history, history_noise=noise,
                      mode="sigma_noised", full_packed=layout, prompt=prompt,
                      anchor=fixture["anchor_rows"].to(dtype),
                      audio=fixture["audio_latents"], sigma=0.6)
            output = interval_forward(model, state, **kw)
            assert output.shape == state.shape and torch.isfinite(output).all()
            # No future action may enter the visible text/packed graph.
            future = text.clone()
            for lo, hi in spans[stop:]:
                future[lo:hi] += 10
            alt_layout, alt_prompt = visible_inputs(fixture["packed"], future, stop, 4)
            assert torch.equal(alt_prompt, prompt)
            assert torch.equal(alt_layout["img_position_ids"], layout["img_position_ids"])
            assert torch.equal(alt_layout["action_text_rows"], layout["action_text_rows"])
            # The current action and generated history must affect the field.
            action = text.clone()
            for lo, hi in spans[start:stop]:
                action[lo:hi] += 3
            alt_layout, alt_prompt = visible_inputs(fixture["packed"], action, stop, 4)
            changed = interval_forward(model, state, **dict(kw, full_packed=alt_layout, prompt=alt_prompt))
            assert float((output - changed).abs().max()) > 1e-6
            changed = interval_forward(model, state, **dict(kw, history=history + 2))
            assert float((output - changed).abs().max()) > 1e-6
            for actual, frozen in zip((history, noise, state), before):
                torch.testing.assert_close(actual, frozen, atol=0, rtol=0)
