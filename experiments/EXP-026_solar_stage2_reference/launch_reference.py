"""Fail-closed launcher for the official H3 Stage2 reference.

The EXP-026/v1 authorization is CPU-only. This entry point intentionally
refuses GPU work until a later Judge authorization and a validated memory plan
are supplied. It never substitutes a fabricated input for an official sample.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def read_json(name: str) -> dict:
    path = HERE / name
    if not path.is_file():
        raise RuntimeError(f"required preflight artifact missing: {path}")
    return json.loads(path.read_text())


def preflight() -> list[str]:
    auth = read_json("AUTHORIZATION.json")
    manifest = read_json("INPUT_MANIFEST.json")
    cpu = read_json("CPU_PREFLIGHT.json")
    issues = []
    if auth.get("gpu_seconds", 0) <= 0 or auth.get("gpu_forwards", 0) <= 0:
        issues.append("EXP-026/v1 authorizes 0 GPU seconds and 0 GPU forwards")
    if manifest.get("sample_id") is None or manifest.get("selected_index_row") is None:
        issues.append("no official sample was selected from a readable test index")
    for name, value in manifest.get("required", {}).items():
        path = value.get("path")
        if not path or not Path(path).is_file():
            issues.append(f"missing official input: {name}")
    if not cpu.get("candidate_config_parses"):
        issues.append("candidate configuration did not parse")
    if not cpu.get("sample_reader_executed"):
        issues.append("official sample reader has not passed")
    if not cpu.get("initial_full_model_load_fit_l40"):
        issues.append("the current full-model pre-FSDP load cannot fit one L40")
    if not (HERE / "candidate_single_case.yaml").is_file():
        issues.append("candidate configuration is missing")
    return issues


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="requires later GPU authorization and all gates")
    args = parser.parse_args()
    issues = preflight()
    if issues:
        print("BLOCKED: official H3 Stage2 reference is not launchable", file=sys.stderr)
        for issue in issues:
            print(f"- {issue}", file=sys.stderr)
        return 2
    if not args.execute:
        print("Preflight passed; no GPU launched. Use --execute only under a later authorization.")
        return 0
    command = [
        str(ROOT / ".venvs/solarwm-h3/bin/torchrun"),
        "--standalone", "--nproc-per-node=8", "-m", "solarwm", "infer",
        "--config", str(HERE / "candidate_single_case.yaml"),
    ]
    return subprocess.call(command, cwd=ROOT / "SolarWM")


if __name__ == "__main__":
    raise SystemExit(main())
