"""CPU structural audit. No pretrained weights, sampling, or optimizer.

VAE temporal orchestration is real; its expensive neural decoder is replaced
by a shape-only stub. This establishes frame counts/control flow, NOT pixels,
quality, or numerical prefix equivalence. Existing real-VAE evidence is linked
separately. Production partitioning is neither patched nor executed here.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import inspect
import json
from pathlib import Path
import sys
import types


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', type=Path, required=True,
                        help='H3-World directory containing patched DiffSynth')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    submission = Path(__file__).resolve().parents[4]
    vendor = args.runtime.resolve() / 'DiffSynth-Studio-h3-v2'
    sys.path[:0] = [str(submission / 'code'), str(vendor)]
    import torch
    from abot.abot_action import frame_spans, latent_t_for
    from causal.h3_cached import slice_packed, ChunkAttention, H3ChunkCache
    from causal.local_topology import visible_inputs
    from diffsynth.models.minimax_h3_dit import patchify_video
    from diffsynth.models.minimax_h3_video_vae import MiniMaxH3VideoVAE
    from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Unit_PackedSequenceBuilder

    torch.set_num_threads(2)
    builder = MiniMaxH3Unit_PackedSequenceBuilder()
    spans = frame_spans(37)
    assert builder._FRAME_PER_TOKEN == (1, 4, 4, 4, 4)
    boundaries = [0] + [end for _, end in spans]
    torch.testing.assert_close(builder._video_t_grid(37, 0.),
        torch.tensor(boundaries[:-1], dtype=torch.float64) * builder._FRAME_RESCALE)
    action_spans = [(16+8*k, 24+8*k) for k in range(37)]
    prompt = torch.zeros(16+8*37, 32)
    full = builder._build_packed_fl2va(len(prompt), 37, 30, 52, 2, [0],
                                      action_text_spans=action_spans)
    rows = int(full['action_frame_rows'])
    assert rows == 390
    latent = torch.arange(37.).reshape(1, 1, 37, 1, 1).expand(1, 24, 37, 30, 52)
    patches = patchify_video(latent)
    assert patches.shape == (37*390, 96)
    assert torch.equal(patches[:, 0], torch.arange(37.).repeat_interleave(390))

    partitions = {
        'old_uniform5': [5]*7+[2],
        'old_uniform12': [12]*3+[1],
        'proposed_A': [2]+[5]*7,
        'proposed_B': [7]+[5]*6,
        'proposed_C': [12]+[5]*5,
    }
    result = dict(timestamp=datetime.now().astimezone().isoformat(), scope=__doc__,
                  gpu_used=False, weights_loaded=False, production_code_modified=False,
                  spatial_tokens_per_latent=rows,
                  supported_length_formula=[dict(rgb=n, latent=latent_t_for(n))
                                            for n in (5,22,39,56,73,124)],
                  partitions={}, source_sha256={})
    for name, sizes in partitions.items():
        assert sum(sizes) == 37
        start = 0
        records = []
        for size in sizes:
            stop = start + size
            visible, _ = visible_inputs(full, prompt, stop, rows)
            current = slice_packed(visible, start, stop, rows)
            v0 = int(current['action_video_start'])
            orig0 = int(full['action_video_start'])
            assert current['seq_len']-v0 == size*rows
            assert current['action_text_spans_local'] == action_spans[:stop]
            assert current['action_text_rows'].tolist() == [list(s) for s in action_spans[:stop]]
            torch.testing.assert_close(current['img_position_ids'][:, v0:],
                full['img_position_ids'][:, orig0+start*rows:orig0+stop*rows], atol=0, rtol=0)
            records.append(dict(latent_interval=[start, stop], rgb_interval=[boundaries[start], boundaries[stop]],
                end_phase_mod5=stop%5, complete_action_spans=True,
                complete_spatial_rows=True, future_actions_physically_removed=True,
                absolute_positions_preserved=True))
            start = stop
        result['partitions'][name] = records

    # Direct action visibility in the actual persistent-KV implementation.
    # Small spatial rows suffice to query the predicate without large masks.
    small = builder._build_packed_fl2va(len(prompt), 37, 4, 4, 2, [0], action_text_spans=action_spans)
    result['direct_action_visibility'] = {}
    for mode in ('own', 'causal'):
        attn = ChunkAttention(H3ChunkCache(), 1, int(small['action_video_start']),
                              4, 5, small['action_text_rows'], mode)
        _, mask = attn.masks(20, 20, torch.device('cpu'))
        visible_actions = [k for k, (lo, hi) in enumerate(action_spans)
                           if bool(mask[0, lo:hi].all())]
        assert visible_actions == ([5] if mode == 'own' else list(range(6)))
        result['direct_action_visibility'][mode] = dict(video_latent_index=5,
            visible_action_indices=visible_actions,
            preserves_original_direct_single_egress=(mode == 'own'))

    # Run the actual VAE orchestration, without allocating neural weights.
    vae = MiniMaxH3VideoVAE.__new__(MiniMaxH3VideoVAE)
    torch.nn.Module.__init__(vae)
    defaults = inspect.signature(MiniMaxH3VideoVAE.__init__).parameters
    vae.clip_length = defaults['clip_length'].default
    vae.token_drop = defaults['token_drop'].default
    vae.vae_ratio_t = 1
    for stride in defaults['time_down'].default:
        vae.vae_ratio_t *= stride
    vae.tokens_chunk_size = (vae.clip_length+vae.vae_ratio_t-1)//vae.vae_ratio_t
    vae.frame_pre_padding = (-vae.clip_length)%vae.vae_ratio_t
    vae.token_overlap = (-vae.token_drop)%vae.tokens_chunk_size
    vae.frame_overlap = max(vae.token_overlap*vae.vae_ratio_t-vae.frame_pre_padding, 0)
    calls = []
    def shape_decode(self, z):
        calls.append(z.shape[2])
        return torch.zeros(1, 3, z.shape[2]*self.vae_ratio_t, 1, 1)
    vae.tiled_decode = types.MethodType(shape_decode, vae)
    result['vae_schedule_shape_only'] = []
    for length in (1,2,3,5,7,10,12,17,24,36,37):
        calls.clear()
        z = torch.zeros(1, 24, length, 1, 1)
        if length == 1:
            try:
                vae.decode_temporal(z)
            except RuntimeError as exc:
                assert 'frame plan mismatch' in str(exc)
                result['vae_schedule_shape_only'].append(dict(latents=length,
                    decoder_call_latent_lengths=list(calls), error=str(exc)))
            else:
                raise AssertionError('Expected the current 1-latent temporal-path edge case')
            continue
        output = vae.decode_temporal(z)
        count = None if output is None else output.shape[2]
        result['vae_schedule_shape_only'].append(dict(latents=length,
            decoder_call_latent_lengths=list(calls), rgb_frames=count,
            returns_none=(output is None)))
        if length >= 3:
            assert count == boundaries[length]
        else:
            assert output is None
    result['two_latent_initial_chunk'] = (
        'Current decode_temporal returns None: num_chunks=0; decode_video then calls recon.float(). '
        'The 5 RGB / 2 latent length formula alone is not a supported streaming decode guarantee.')
    result['scope_limits'] = [
        'No new pretrained VAE pixel/quality comparisons; shape stub cannot measure receptive fields.',
        'Explicit interval slicing works, but variable-size production rollout/cache APIs are not implemented.',
        'Global positions are inherited from a fixed full37 layout; no dynamically rebuilt prefix origin.',
        'Audio rows stay in prefix as fixed noise; audio/video synchronized streaming is not established.',
        'No evidence that a partition change repairs action velocity geometry or generated-history drift.',
    ]
    sources = [Path(__file__), Path(inspect.getfile(MiniMaxH3VideoVAE)),
               Path(inspect.getfile(MiniMaxH3Unit_PackedSequenceBuilder)),
               Path(inspect.getfile(patchify_video)), Path(inspect.getfile(frame_spans)),
               Path(inspect.getfile(slice_packed)), Path(inspect.getfile(visible_inputs))]
    for path in sources:
        label = str(path.relative_to(submission)) if path.is_relative_to(submission) else str(path.relative_to(args.runtime.resolve()))
        result['source_sha256'][label] = hashlib.sha256(path.read_bytes()).hexdigest()
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n')
    print('PASS: logical action/video slices, global positions, native spans, routing predicates, VAE schedule.')
    print('CONFIRMED EDGE CASE: 2-latent prefix returns None; no neural decoder call.')
    print('No weights/GPU/optimizer; see scope limitations in receipt.')


if __name__ == '__main__':
    main()
