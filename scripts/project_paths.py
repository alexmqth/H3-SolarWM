"""Resolve preserved historical GWM paths after the qma migration.

This changes path lookup only; it does not rewrite evidence or recreate pruned
KV/optimizer files. Use the historical record's hashes to verify file content.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HISTORICAL_PREFIXES = {
    '/home/lpeng/code/mq_PubDataset/GWM': PROJECT_ROOT,
    '/work/lpeng/qma/GWM_models': PROJECT_ROOT / 'models',
}


def resolve_recorded_path(value: str | Path, *, must_exist: bool = True) -> Path:
    path = Path(value)
    for prefix, replacement in HISTORICAL_PREFIXES.items():
        try:
            path = replacement / path.relative_to(prefix)
            break
        except ValueError:
            pass
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if must_exist and not path.exists():
        raise FileNotFoundError(
            f'{path}: see submission/archive/qma_migration_20261012/cleanup_plan.json.gz '
            'for deliberately pruned KV/optimizer state; rebuild it before reuse.')
    return path


def relocate_manifest(value):
    """Return an in-memory relocated view; leave frozen JSON bytes untouched."""
    if isinstance(value, dict):
        return {k: relocate_manifest(v) for k, v in value.items()}
    if isinstance(value, list):
        return [relocate_manifest(v) for v in value]
    if isinstance(value, str):
        for prefix in HISTORICAL_PREFIXES:
            if value == prefix or value.startswith(prefix + '/'):
                return str(resolve_recorded_path(value, must_exist=False))
    return value


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path')
    parser.add_argument('--json', action='store_true', help='Print a relocated view of a frozen JSON manifest')
    args = parser.parse_args()
    if args.json:
        print(json.dumps(relocate_manifest(json.loads(resolve_recorded_path(args.path).read_text())), ensure_ascii=False, indent=2))
    else:
        print(resolve_recorded_path(args.path))
