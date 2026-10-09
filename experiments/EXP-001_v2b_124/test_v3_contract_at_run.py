"""CPU-only tests for the narrowed EXP-001 v3 resource and resume contract."""

import json
from datetime import datetime
from pathlib import Path

import pytest

import run_rollout_v3 as runner


def ledger(epoch, active=(), sampling=120, closed_gpu_seconds=1257.1022913455963):
    return dict(first_gpu_start=epoch - 1000, gpu_seconds=closed_gpu_seconds,
                sampling_reserved=sampling, diagnostic_reserved=12,
                original_reserved=0, runs=list(active))


def test_active_paths_and_resume_are_frozen():
    assert runner.CONFIG["plan_version"] == 3
    assert runner.CONFIG["active_paths"] == ["AA", "DD"]
    for name in ("AA", "DD"):
        state = json.loads((runner.OUTPUT / name / "state.json").read_text())
        assert state["next_start"] == 22
        assert len(state["results"]) == 1
    assert runner.CONFIG["max_sampling_forwards"] == 300


def test_absolute_09_cutoff_and_active_gpu_accounting():
    cutoff = datetime.fromisoformat(runner.CONFIG["gpu_cutoff_hkt"]).timestamp()
    assert runner.gpu_limit_at(cutoff - 1) == 8
    assert runner.gpu_limit_at(cutoff) == 3
    current = ledger(cutoff, active=[dict(path="AA", gpu=0, status="running", start=cutoff - 100)])
    assert runner.budget_usage_at(current, cutoff) == pytest.approx(1357.1022913455963)
    runner.validate_v3_reservation(current, "DD", 1, cutoff, 0, 30)
    with pytest.raises(AssertionError):
        runner.validate_v3_reservation(current, "AA", 1, cutoff, 0, 30)


def test_budget_never_resets_and_no_extra_paths_or_diagnostics():
    cutoff = datetime.fromisoformat(runner.CONFIG["gpu_cutoff_hkt"]).timestamp()
    current = ledger(cutoff, sampling=300)
    for path, diagnostics in (("AA", 0), ("AD", 0), ("AA", 1)):
        with pytest.raises(AssertionError):
            runner.validate_v3_reservation(current, path, 0, cutoff, diagnostics, 30)
    current = ledger(cutoff, closed_gpu_seconds=runner.CONFIG["max_GPU_hours"] * 3600 - 500)
    with pytest.raises(AssertionError):
        runner.validate_v3_reservation(current, "AA", 0, cutoff, 0, 30)
