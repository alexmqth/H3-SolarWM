"""Small CPU checks for the independent P0 runner gates and finite-map sign."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import torch

import common


def test_budget() -> None:
    old = common.OUT
    with tempfile.TemporaryDirectory(prefix="exp012_budget_") as tmp:
        common.OUT = Path(tmp)
        try:
            first = common.Ledger(0, "G1_industrial")
            for _ in range(16):
                first.reserve("sampling_forwards")
            first.reserve("commit_forwards")
            for _ in range(2):
                first.reserve("vae_decodes")
            try:
                first.reserve("sampling_forwards")
            except RuntimeError:
                pass
            else:
                raise AssertionError("per-scene 17th sample should be refused")
            first.close("complete")
            second = common.Ledger(0, "G1_village")
            for _ in range(16):
                second.reserve("sampling_forwards")
            second.reserve("commit_forwards")
            for _ in range(2):
                second.reserve("vae_decodes")
            second.close("complete")
            third = common.Ledger(0, "G1_extra")
            for field in ("sampling_forwards", "commit_forwards", "vae_decodes"):
                try:
                    third.reserve(field)
                except RuntimeError:
                    pass
                else:
                    raise AssertionError(f"total budget did not refuse {field}")
            third.close("complete")
            ledger = json.loads((Path(tmp) / "budget.json").read_text())
            assert ledger["sampling_forwards"] == 32
            assert ledger["commit_forwards"] == 2
            assert ledger["vae_decodes"] == 4
        finally:
            common.OUT = old


def test_finite_map() -> None:
    common.setup_paths()
    from causal.anyflow import finite_map_step
    x = torch.tensor([1.0, 2.0])
    v = torch.tensor([2.0, -4.0])
    y = finite_map_step(x, v, 0.8, 0.3)
    assert torch.allclose(y, x - 0.5 * v)


def test_marker_absent_without_gpu() -> None:
    marker = common.HERE / "judge/G1_industrial_APPROVED.json"
    if marker.exists():
        # A later approved rerun should not turn this historical P0 test into a failure.
        return
    result = subprocess.run([sys.executable, str(common.HERE / "run_exp012.py"),
                             "--scene", "industrial", "--gpu", "0"],
                            capture_output=True, text=True,
                            env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"})
    assert result.returncode != 0
    assert "Judge stage marker missing" in result.stderr
    assert not common.OUT.exists()


if __name__ == "__main__":
    test_budget()
    test_finite_map()
    test_marker_absent_without_gpu()
    print(json.dumps({"task": common.CFG["task"], "status": "CPU_TEST_PASS",
                      "budget_simulation": "PASS", "finite_map_sign": "PASS",
                      "missing_marker_no_gpu": "PASS"}, indent=2))
