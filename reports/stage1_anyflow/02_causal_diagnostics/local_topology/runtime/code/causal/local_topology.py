"""Isolated no-grad E1 topologies on Original H3; not production defaults.

T1 keeps committed video KV immutable and recomputes a prefix grounded in
visible history/current video. T2 recomputes the entire visible window every
call. It deliberately has no persistent mutable hidden-state cache.

Packed positions must come from a FIXED, action-content-independent layout
contract. Removing unknown rows below preserves those global coordinates;
it does not undo positions previously computed from unknown action lengths.
"""
from contextlib import contextmanager

import torch
from torch.nn import functional as F

from . import h3_cached as hc


def visible_inputs(full, prompt, stop, frame_rows):
    """Delete future action/video rows before refiner or DiT execution.

    Input global coordinates are a declared immutable layout contract. Only
    prefix (head, known actions, anchors, fixed audio) and video [0:stop] remain.
    The returned layout contains no padding or future text segments.
    """
    spans = full['action_text_spans_local']
    if not 0 < stop <= len(spans):
        raise ValueError('visible stop is outside the declared layout')
    p = int(full['action_video_start'])
    text_pos = full['text_pos']
    if len(prompt) != len(text_pos):
        raise ValueError('prompt and packed text rows disagree')
    device = text_pos.device
    keep_text = torch.ones(len(prompt), dtype=torch.bool, device=device)
    keep_seq = torch.zeros(int(full['seq_len']), dtype=torch.bool, device=device)
    keep_seq[:p + stop * frame_rows] = True
    for lo, hi in spans[stop:]:
        keep_text[int(lo):int(hi)] = False
        keep_seq[text_pos[int(lo):int(hi)]] = False
    seq = torch.nonzero(keep_seq).flatten()
    remap = torch.full_like(keep_seq, -1, dtype=torch.long)
    remap[seq] = torch.arange(len(seq), device=device)
    local_map = torch.full((len(prompt),), -1, dtype=torch.long, device=device)
    local_map[keep_text] = torch.arange(int(keep_text.sum()), device=device)
    out = dict(full)
    out['img_position_ids'] = full['img_position_ids'][:, seq].clone()
    out['token_tags'] = full['token_tags'][seq].clone()
    for field in ('img_pos', 'text_pos', 'audio_pos'):
        mapped = remap[full[field]]
        out[field] = mapped[mapped >= 0].clone()
    out['action_text_spans_local'] = [(int(local_map[int(lo)]), int(local_map[int(hi)-1])+1)
                                     for lo, hi in spans[:stop]]
    # action_text_rows are absolute packed positions, not offsets in prompt.
    out['action_text_rows'] = torch.tensor([
        (int(remap[int(lo)]), int(remap[int(hi)-1])+1)
        for lo, hi in full['action_text_rows'][:stop].tolist()],
        dtype=torch.long, device=device)
    out['action_video_start'] = int(remap[p])
    out['seq_len'] = len(seq)
    out['action_real_used'] = len(seq)
    out['cu_seqlens'] = torch.tensor([0, len(seq)], dtype=torch.int32, device=device)
    return out, prompt[keep_text.to(prompt.device)].clone()


def annotation_ids(prefix, spans, device):
    ann = torch.full((prefix,), -1, dtype=torch.long, device=device)
    for frame, (lo, hi) in enumerate(spans.tolist()):
        if not 0 <= lo < hi <= prefix:
            raise ValueError('action annotation outside prefix')
        ann[lo:hi] = frame
    return ann


def sdpa(q, k, v, mask, scale):
    return F.scaled_dot_product_attention(
        q.transpose(0, 1).unsqueeze(0), k.transpose(0, 1).unsqueeze(0),
        v.transpose(0, 1).unsqueeze(0), attn_mask=mask[None, None],
        scale=scale).squeeze(0).transpose(0, 1)


class GroundedPrefixAttention(hc.ChunkAttention):
    """T1, or archived C1 with prefix_history=False. Inference-only."""
    prefix_history = True
    fixed_chunk_frames = 5

    def attend(self, q, k, v, *, rope_freqs, layer, apply_rope, scale):
        if torch.is_grad_enabled() or self.clean_graph_entries is not None:
            raise RuntimeError('E1 topology supports no_grad inference only')
        if self.action_prefix_mode != 'own' or not self.action_feedback:
            raise ValueError('E1 requires Original own-frame actions and feedback')
        if self.action_adapter is not None or self.action_prefix_adapter is not None:
            raise ValueError('E1 uses Original + released action LoRA only')
        p = self.prefix
        history = self.cache.history(layer, self.index)
        keys = [e.key.to(k.device) for e in history]
        values = [e.value.to(v.device) for e in history]
        ropes = [e.rope.to(rope_freqs.device) for e in history]
        kr = torch.cat([rope_freqs[:p], *ropes, rope_freqs[p:]], dim=0)
        kk = apply_rope(torch.cat([k[:p], *keys, k[p:]], dim=0), kr)
        vv = torch.cat([v[:p], *values, v[p:]], dim=0)
        qr = apply_rope(q, rope_freqs)
        hist_frames = []
        for e in history:
            if e.key.shape[0] % self.frame_rows:
                raise ValueError('history rows are not frame-aligned')
            hist_frames.append(e.index*self.fixed_chunk_frames +
                torch.arange(e.key.shape[0], device=q.device)//self.frame_rows)
        current_frames = self.frame_start + torch.arange(q.shape[0]-p, device=q.device)//self.frame_rows
        frames = torch.cat([*hist_frames, current_frames])
        ann = annotation_ids(p, self.action_rows, q.device)
        prefix_prefix = (ann[None, :] < 0) | (ann[:, None] == ann[None, :])
        prefix_video = (ann[:, None] < 0) | (ann[:, None] == frames[None, :])
        if not self.prefix_history:
            prefix_video[:, :sum(x.numel() for x in hist_frames)] = False
        prefix_mask = torch.cat([prefix_prefix, prefix_video], dim=1)
        video_prefix = (ann[None, :] < 0) | (ann[None, :] == current_frames[:, None])
        video_mask = torch.cat([video_prefix,
            torch.ones(len(current_frames), len(frames), device=q.device, dtype=torch.bool)], dim=1)
        out = torch.cat([sdpa(qr[:p], kk, vv, prefix_mask, scale),
                         sdpa(qr[p:], kk, vv, video_mask, scale)], dim=0)
        if self.commit:
            self.cache.commit(layer, self.index, k[p:], v[p:], rope_freqs[p:])
        return out


@contextmanager
def grounded_prefix(*, history=True, chunk_frames=5):
    """Scoped construction hook; restores on failure, no concurrent use."""
    original = hc.ChunkAttention
    if original is not GroundedPrefixAttention.__bases__[0]:
        raise RuntimeError('Another cached attention patch is already active')
    class ScopedAttention(GroundedPrefixAttention):
        prefix_history = history
        fixed_chunk_frames = chunk_frames
    hc.ChunkAttention = ScopedAttention
    try:
        yield
    finally:
        hc.ChunkAttention = original


class OriginalWindowAttention:
    """Original directed predicate on visible-only window, no hidden cache."""
    allow_grad_read = False

    def __init__(self, packed, frames, rows, frame_start=0):
        self.prefix = int(packed['action_video_start'])
        self.spans = packed['action_text_rows']
        self.frames, self.rows, self.frame_start = frames, rows, frame_start
        self.mask = None

    def build_mask(self, device):
        p, n = self.prefix, self.frames*self.rows
        ann = torch.cat([annotation_ids(p, self.spans, device),
                         torch.full((n,), -1, device=device, dtype=torch.long)])
        frame = torch.cat([torch.full((p,), -1, device=device, dtype=torch.long),
                           self.frame_start + torch.arange(n, device=device)//self.rows])
        aq, ak, fq, fk = ann[:, None], ann[None, :], frame[:, None], frame[None, :]
        same = (aq >= 0) & (ak >= 0) & (aq == ak)
        own = (fq >= 0) & (ak >= 0) & (fq == ak)
        return ~(((ak >= 0) & ~same & ~own) | ((aq >= 0) & (fk >= 0) & (aq != fk)))

    def apply_hidden(self, hidden, layer):
        return hidden

    def attend(self, q, k, v, *, rope_freqs, layer, apply_rope, scale):
        if torch.is_grad_enabled():
            raise RuntimeError('E1 window supports no_grad inference only')
        if self.mask is None:
            self.mask = self.build_mask(q.device)
        return sdpa(apply_rope(q, rope_freqs), apply_rope(k, rope_freqs), v, self.mask, scale)


def window_forward(dit, current, *, history, full_packed, prompt, anchor,
                   audio, sigma, index, chunk_frames=5, history_chunks=5,
                   anchor_slot=1, return_all=False):
    """T2: retain raw history, recompute visible window, return current field.

    Caller must pass already visibility-trimmed packed/prompt. History may
    contain the full known prefix; cap is applied here with global positions.
    Inputs and all previously emitted latents are read-only.
    """
    from diffsynth.pipelines.minimax_h3_audio_video import model_fn_minimax_h3
    if torch.is_grad_enabled() or hasattr(dit, 'anyflow_conditioner'):
        raise RuntimeError('E1 window requires no_grad and ordinary FM weights')
    if history.shape[2] != index*chunk_frames:
        raise ValueError('history must contain exactly the known clean prefix')
    if history_chunks < 1 or not 0 <= sigma <= 1:
        raise ValueError('invalid history cap or sigma')
    start = index*chunk_frames
    stop = start + current.shape[2]
    if len(full_packed['action_text_rows']) != stop:
        raise ValueError('trim future action rows before window evaluation')
    first = max(0, start-history_chunks*chunk_frames)
    dtype = getattr(dit, '_h3_input_dtype', prompt.dtype)
    whole = torch.cat([history[:, :, first:], current], dim=2).to(dtype)
    h = start-first
    rows = (current.shape[-2]//2)*(current.shape[-1]//2)
    packed = hc.slice_packed(full_packed, first, stop, rows)
    if start:
        packed = hc.retime_anchor_position(packed, full_packed,
            anchor_rows=anchor.shape[0], frame_index=start-1,
            frame_rows=rows, anchor_slot=anchor_slot)
    update = torch.ones_like(whole)
    update[:, :, :h] = 0
    control = OriginalWindowAttention(packed, whole.shape[2], rows, first)
    output = model_fn_minimax_h3(dit, whole, audio.to(dtype), packed, prompt,
        timestep_video=torch.tensor(sigma*1000, device=current.device),
        timestep_audio=torch.tensor(1000., device=current.device),
        keyframe_cond_anchor=anchor.to(dtype), input_latents_video=whole,
        denoise_mask_video=update, fixed_prefix_timesteps=True, causal_control=control)[0]
    return output if return_all else output[:, :, h:]
