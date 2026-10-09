"""Teacher-forcing flow loss on the actual patched H3 DiT.

This is the minimum Stage1-style training interface, not AnyFlow or DMD.
Only current-chunk outputs receive a loss; history is clean, detached and
marked at the clean timestep through H3's existing retake input mask.
"""
from __future__ import annotations

import torch
from torch import nn


class QKVLoRA(nn.Module):
    def __init__(self, base: nn.Linear, rank: int):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.lora_A = nn.Parameter(torch.randn(rank, base.in_features) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))
        self.scale = 1.0 / rank

    def forward(self, x):
        return self.base(x) + ((x @ self.lora_A.T) @ self.lora_B.T) * self.scale


def make_small_h3(rank=4):
    """Same H3 code path, random small weights; does not load pretrained 33B."""
    from diffsynth.models.minimax_h3_dit import MiniMaxH3DiT

    # Keep the model tiny, but use a 16-wide attention head so CUDA
    # FlexAttention can compile it as well as the CPU eager fallback.
    hidden_size = 64
    dit = MiniMaxH3DiT(
        num_layers=2, token_refiner_num_layers=1, hidden_size=hidden_size,
        num_attention_heads=4, attention_head_dim=16, ffn_hidden_size=128,
        text_dim=32, timestep_input_dim=16, time_embed_hidden_size=64,
        time_embed_dim=32, adaln_out_features=6 * hidden_size * 3,
        final_adaln_out_features=2 * hidden_size, rope_inv_freq_len=2,
    )
    dit.requires_grad_(False)
    for block in dit.blocks:
        block.attn.qkv_proj = QKVLoRA(block.attn.qkv_proj, rank)
    return dit


def h3_teacher_forcing_loss(
    dit, *, clean_video, noise, sigma, chunk_index, packed, prompt_embeds,
    anchor_rows, audio_latents, chunk_frames=5, window_chunks=5,
):
    """One H3 teacher-forcing loss; sigma=0 is clean, sigma=1 is noise.

    Packed input contains history through the current chunk, with no future
    frames. Build it using H3's PackedSequenceBuilder and action spans for
    that prefix. H3's returned scheduler velocity is noise-clean (the native
    DiT predicts its negative); reusing model_fn handles that sign explicitly.
    """
    from diffsynth.pipelines.minimax_h3_audio_video import model_fn_minimax_h3

    if clean_video.ndim != 5 or clean_video.shape[0] != 1 or noise.shape != clean_video.shape:
        raise ValueError("expected matching [1,24,T,H,W] clean/noise tensors")
    start = chunk_index * chunk_frames
    if chunk_index < 0 or chunk_frames <= 0 or not start < clean_video.shape[2] <= start + chunk_frames:
        raise ValueError("input must end at the current chunk, without future frames")
    if not 0 < sigma < 1:
        raise ValueError("training sigma must lie in (0,1)")
    clean = clean_video.detach()
    noisy = (1 - sigma) * clean + sigma * noise
    update = torch.zeros_like(clean)
    update[:, :, start:] = 1
    prediction, _ = model_fn_minimax_h3(
        dit, noisy, audio_latents, packed, prompt_embeds,
        timestep_video=torch.tensor(sigma * 1000, device=clean.device),
        timestep_audio=torch.tensor(0., device=clean.device),
        keyframe_cond_anchor=anchor_rows,
        input_latents_video=clean, denoise_mask_video=update,
        causal_chunk_size=chunk_frames, causal_window_chunks=window_chunks,
    )
    return torch.nn.functional.mse_loss(prediction[:, :, start:], (noise - clean)[:, :, start:])


def synthetic_h3_batch(frames=10, seed=7):
    """Small real H3 layout, including independently refined action spans."""
    from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Unit_PackedSequenceBuilder
    g = torch.Generator().manual_seed(seed)
    head = 16
    # Eight action tokens per frame give the mirrored positions enough room
    # between the head and the video origin (H3's temporal stride is ~5.7).
    spans = [(head + 8*i, head + 8*i + 8) for i in range(frames)]
    text_len = head + 8*frames
    packed = MiniMaxH3Unit_PackedSequenceBuilder()._build_packed_fl2va(
        text_len, frames, 4, 4, 2, [0], action_text_spans=spans)
    clean = torch.randn(1, 24, frames, 4, 4, generator=g)
    return dict(
        clean_video=clean, noise=torch.randn(clean.shape, generator=g),
        packed=packed, prompt_embeds=torch.randn(text_len, 32, generator=g),
        anchor_rows=torch.randn(4, 96, generator=g),
        audio_latents=torch.zeros(2, 32, 2),
    )
