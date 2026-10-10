# EXP-005 stage-one source manifest

SHA-256 values below are for the files read or produced by this CPU stage. They identify the exact frozen dependencies used by the tests; no weights or large cache files were copied here.

| Role | Source path (relative to GWM root) | SHA-256 |
| --- | --- | --- |
| Frozen raw-KV attention/cache | `H3-World/outputs/2026-10-09-22/chunk_partition_cb/runtime/code/causal/h3_cached.py` | `629f8f26d912c188d7d7122c6106ff742f5b2a633dab5881acbb1eb5f77c806e` |
| Frozen current-prefix patch | `submission/reports/stage1_anyflow/02_causal_diagnostics/current_prefix_candidate/current_prefix.py` | `5d0cd21e2464fffe4a50782b6d4ebb8228117d61279790ee58f8ea13bbb68310` |
| EXP-002 accepted interval | `submission/experiments/EXP-002_native_cached/interval_cached.py` | `1d4885a837d8ae449c218f63a56deda05d21b436d162dd2abaffa742d59599d0` |
| EXP-003 accepted interval | `submission/experiments/EXP-003_native_cached_124/interval_cached.py` | `9882fc041310c16377be31bd3c3b80285d6df124ff842eecf9ef2cd1dad4dd37` |
| SolarWM attention reference | `SolarWM/src/solarwm/backends/minimax_h3/sgf_attention.py` | `3eacc2effd9cf2621337775d90735585633b11d09c2f9aa3d7c7e24a7f94f6d2` |
| SolarWM rollout reference | `SolarWM/src/solarwm/backends/minimax_h3/sgf_rollout.py` | `93b3405d1faa82ae47901400e26aad7e96f4c02854ddc2a2b979388be13be0db` |
| Frozen H3 DiT / MM-RoPE | `H3-World/outputs/2026-10-09-22/chunk_partition_cb/runtime/DiffSynth-Studio-h3-v2/diffsynth/models/minimax_h3_dit.py` | `68e9cb9336590fae7c53f2fac83c09aaf8d03fba1776777383a2b924eee40e38` |
| Frozen H3 packed builder | `H3-World/outputs/2026-10-09-22/chunk_partition_cb/runtime/DiffSynth-Studio-h3-v2/diffsynth/pipelines/minimax_h3_audio_video.py` | `e2291e331f284a22158878ed22a49de80a1df03b6cad5fd77e35b1f9615ac40b` |
| Frozen rollout utilities | `submission/experiments/EXP-001_v2b_124/rollout_contract.py` | `79b6e672cf8d97ba57f3d62e876a0e0bd901f78654cb2c1f9766818378ef2377` |
| New chunk plan | `submission/experiments/EXP-005_v3_sliding_window/chunk_plan.py` | `40e0cbfde20fcf3489e493f1c1dd6f5df76eacaa92229722f3fa10b9598b4e60` |
| New position candidate | `submission/experiments/EXP-005_v3_sliding_window/position_sw.py` | `c1d26591521d0efc4dcef656bd57550ab9132f3f53aa5cafdbd3bfa11b871f97` |
| New interval entry | `submission/experiments/EXP-005_v3_sliding_window/interval_sw.py` | `56bbd6229a9da44e00b4535f510028ba77c1709d66aff8ab2bea507990de9c17` |
| CPU contract tests | `submission/experiments/EXP-005_v3_sliding_window/test_contract.py` | `1902d88a07f727d3c1b0b757918833deb69ebb1b43fd751a04a7ff2c39638e87` |

CPU test fixtures: frozen `H3-World/outputs/2026-10-09-22/chunk_partition_cb/source_coarse/inputs/parking_A.pt` (`675f7c35a523f8b634a443b71f9108a4b945ecb9eb66fb6bb821c1829f3d5572`) and `parking_D.pt` (`f85db42ad83cbe1ce00612bc29ee41d6a1398be05d3a96c53cd3615925187d0e`), loaded on CPU via `torch.load(weights_only=True)`; no new model checkpoint or generated video was produced. The accepted checkpoint and cache/endpoint paths remain documented in EXP-002/003. Stage two must create a separate input/checkpoint/noise hash manifest before any GPU execution.
