#!/usr/bin/env python3
"""Check eager/compiled directed H3 mask equivalence, including changed controls."""
import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "abot"))
import infer  # noqa: F401 -- selects and checks the patched DiffSynth checkout
import torch
from diffsynth.models.minimax_h3_dit import _build_action_block_masks
from torch.nn.attention.flex_attention import create_mask


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    n, device = 2176, "cuda"
    cu = torch.tensor([0, n], device=device, dtype=torch.int32)
    ann = torch.tensor([[8, 16], [16, 24], [24, 32]], device=device)

    def build(compiled):
        os.environ["H3_COMPILE_BLOCK_MASK"] = str(int(compiled))
        return _build_action_block_masks(ann, 128, 640, 3, cu, n, device, n_real=2113)[0]

    previous = None
    for assignment in ([[8, 16], [16, 24], [24, 32]], [[24, 32], [8, 16], [16, 24]]):
        ann.copy_(torch.tensor(assignment, device=device))
        eager, compiled = build(False), build(True)
        dense_eager = create_mask(eager.mask_mod, 1, 1, n, n, device=device)
        dense_compiled = create_mask(compiled.mask_mod, 1, 1, n, n, device=device)
        assert torch.equal(dense_eager, dense_compiled)
        for key in ("kv_num_blocks", "kv_indices", "full_kv_num_blocks", "full_kv_indices"):
            assert torch.equal(getattr(eager, key), getattr(compiled, key)), key
        if previous is not None:
            assert not torch.equal(previous, dense_compiled), "Changed actions must affect mask"
        previous = dense_compiled
    result = dict(status="pass", sequence_length=n, padded_real_length=2113,
                  action_rows=3, max_mask_difference=0,
                  check="Exact dense mask and sparse block metadata equality, including changed action assignment")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
