"""Incremental H3 attention: raw K/V from clean history, fixed prefix times.

Uses the released H3 action-text conditioning and *global* H3 RoPE positions.
No SolarWM PRoPE, local position rebasing, AnyFlow or DMD is implied.
"""
from dataclasses import dataclass, field

import torch
from torch.nn import functional as F


def last_frame_anchor(latent: torch.Tensor) -> torch.Tensor:
    """Convert one clean latent frame to the H3 keyframe anchor rows.

    ``latent`` is expected to be ``[1, 24, T, H, W]`` in the same normalized
    latent space used by ``keyframe_cond_anchor``.  Keeping this conversion in
    the causal module makes the dynamic-anchor rule identical in training and
    rollout.  The returned rows are intentionally detached by the caller when
    they are used as a conditioning prefix; Stage1 teacher forcing can still
    choose to retain gradients outside this helper.

    This is the minimal diagnostic implementation: the rows occupy the same
    packed prefix positions as the original image anchor.  A production
    implementation would construct a local packed layout and mark the
    repeated anchor frame as non-denoised instead of replacing these values in
    place.
    """
    if latent.ndim != 5 or latent.shape[0] != 1 or latent.shape[2] != 1:
        raise ValueError(f"expected [1,C,1,H,W] latent frame, got {tuple(latent.shape)}")
    from diffsynth.models.minimax_h3_dit import patchify_video
    return patchify_video(latent).contiguous()


@torch.no_grad()
def last_frame_image_anchor(video_vae, history: torch.Tensor, *, dtype=None,
                            tile_size: int = 256, tile_overlap: int = 64) -> torch.Tensor:
    """Build a true H3 image-condition anchor from generated history.

    H3's keyframe path is ``RGB -> encode_video(process_image=True)``.  The
    earlier diagnostic ``last_frame_anchor`` only patchified a temporal latent,
    which skips the image branch of the VAE. Decode the available prefix with
    the temporal decoder, select its last RGB frame, re-encode that one frame
    through the independent image encoder, and patchify the normalized image
    latent. The prefix is required because decoding a lone latent token does
    not recover the last visible frame of H3's temporal VAE.
    """
    if history.ndim != 5 or history.shape[0] != 1 or history.shape[2] < 1:
        raise ValueError(f"expected nonempty [1,C,T,H,W] history, got {tuple(history.shape)}")
    out_dtype = dtype or history.dtype
    rgb = video_vae.decode_video(
        history, dtype=out_dtype, process_image=False, tiled=True,
        tile_size=tile_size, tile_overlap=tile_overlap)
    if rgb.ndim != 5 or rgb.shape[2] < 1:
        raise RuntimeError(f"image decoder returned unexpected shape {tuple(rgb.shape)}")
    rgb = rgb[:, :, -1:].contiguous().clamp(0, 1)
    image_latent = video_vae.encode_video(
        rgb, dtype=out_dtype, process_image=True, tiled=True,
        tile_size=tile_size, tile_overlap=tile_overlap)
    from diffsynth.models.minimax_h3_dit import patchify_video
    return patchify_video(image_latent.to(device=history.device, dtype=out_dtype)).contiguous()


def retime_anchor_position(packed: dict, source_packed: dict, *, anchor_rows: int,
                           frame_index: int, frame_rows: int,
                           anchor_slot: int = 0) -> dict:
    """Move a one-frame anchor to its real global H3 temporal position.

    The packed layout keeps condition rows in the prefix so the existing H3
    model can consume them as ``keyframe_cond_anchor``. Replacing only their
    values while leaving frame-zero grid positions creates a long-rollout
    mismatch. This helper copies the temporal/spatial grid from the referenced
    global video frame into the sliced prefix condition rows.
    """
    if frame_index < 0 or anchor_rows < frame_rows or anchor_rows % frame_rows:
        raise ValueError("retimed anchor requires complete latent-frame anchor rows")
    num_anchors = anchor_rows // frame_rows
    if not 0 <= anchor_slot < num_anchors:
        raise ValueError(f"anchor_slot {anchor_slot} outside {num_anchors} anchors")
    source_img = source_packed["img_pos"]
    current_img = packed["img_pos"]
    if source_img.numel() < anchor_rows + (frame_index + 1) * frame_rows:
        raise ValueError(f"anchor frame {frame_index} is outside the source packed video")
    if current_img.numel() < anchor_rows:
        raise ValueError("sliced packed layout has fewer rows than the anchor")
    source_row = source_img[anchor_rows + frame_index * frame_rows]
    source_grid = source_packed["img_position_ids"][0, source_row:source_row + frame_rows]
    target_rows = current_img[anchor_slot * frame_rows:(anchor_slot + 1) * frame_rows]
    out = dict(packed)
    positions = packed["img_position_ids"].clone()
    positions[0, target_rows] = source_grid
    out["img_position_ids"] = positions
    out["anchor_frame_index"] = int(frame_index)
    out["anchor_slot"] = int(anchor_slot)
    return out


def expand_packed_two_anchors(full: dict, *, frame_rows: int) -> dict:
    """Add a second H3 image-anchor block before audio/video rows.

    Existing conditioning files contain one image anchor.  The dual-anchor
    prototype keeps that original scene anchor and adds a second slot whose
    value can be replaced by the previous chunk's last frame.  The inserted
    slot initially receives the first global video-frame grid, which matches
    the duplicated first-frame value used for chunk 0.  Later rollout calls
    retime this slot to the predecessor's real global frame.
    """
    if frame_rows <= 0 or "action_video_start" not in full:
        raise ValueError("packed layout must include action video metadata")
    img_pos = full["img_pos"]
    anchor_rows = int(frame_rows)
    if img_pos.numel() < anchor_rows:
        raise ValueError("packed layout has no complete first anchor")
    audio_pos = full["audio_pos"]
    if audio_pos.numel() == 0:
        raise ValueError("dual anchor layout requires audio/video packed rows")
    insert_at = int(audio_pos[0])
    seq_len = int(full["seq_len"])
    if not 0 <= insert_at <= seq_len:
        raise ValueError("invalid audio insertion point")
    video_rows = int(img_pos.numel() - anchor_rows)
    if video_rows < frame_rows or video_rows % frame_rows:
        raise ValueError("packed video rows are not frame aligned")
    first_video_row = img_pos[anchor_rows]
    first_grid = full["img_position_ids"][0, first_video_row:first_video_row + frame_rows]
    if first_grid.shape[0] != frame_rows:
        raise ValueError("first video frame grid is incomplete")

    device = full["img_position_ids"].device
    remap = torch.arange(seq_len, device=device, dtype=torch.long)
    remap[insert_at:] += frame_rows
    out = dict(full)
    old_grid = full["img_position_ids"]
    out["img_position_ids"] = torch.cat(
        (old_grid[:, :insert_at], first_grid.unsqueeze(0), old_grid[:, insert_at:]), dim=1
    )
    old_tags = full["token_tags"]
    out["token_tags"] = torch.cat(
        (old_tags[:insert_at], torch.zeros(frame_rows, device=device, dtype=old_tags.dtype),
         old_tags[insert_at:]), dim=0
    )
    old_cond = img_pos[:anchor_rows]
    old_video = img_pos[anchor_rows:]
    inserted = torch.arange(insert_at, insert_at + frame_rows, device=device, dtype=torch.long)
    out["img_pos"] = torch.cat((remap[old_cond], inserted, remap[old_video]), dim=0)
    out["audio_pos"] = remap[full["audio_pos"]]
    out["text_pos"] = remap[full["text_pos"]]
    out["img_pos"] = out["img_pos"].contiguous()
    out["action_video_start"] = int(full["action_video_start"]) + frame_rows
    out["seq_len"] = seq_len + frame_rows
    out["cu_seqlens"] = torch.tensor(
        [0, int(full["cu_seqlens"][1]) + frame_rows, seq_len + frame_rows],
        device=device, dtype=torch.int32,
    )
    out["action_real_used"] = int(full.get("action_real_used", seq_len)) + frame_rows
    out["anchor_count"] = 2
    return out


@dataclass
class RawEntry:
    index: int
    key: torch.Tensor
    value: torch.Tensor
    rope: torch.Tensor


@dataclass
class CleanGraphCapture:
    entries: dict = field(default_factory=dict)
    sealed: bool = False


@dataclass
class H3ChunkCache:
    max_history: int = 5
    storage_device: str | None = None
    layers: dict = field(default_factory=dict)
    peak_bytes: int = 0
    commits: int = 0

    def __post_init__(self):
        if self.max_history < 1:
            raise ValueError('max_history must be positive')

    def history(self, layer, index):
        entries = self.layers.get(layer, [])
        expected = list(range(max(0, index-self.max_history), index))
        if [e.index for e in entries] != expected:
            raise RuntimeError(f'cache history mismatch: layer={layer}, chunk={index}')
        return entries

    def commit(self, layer, index, key, value, rope):
        if torch.is_grad_enabled():
            raise RuntimeError('cache commit requires no_grad')
        if key.shape != value.shape or key.shape[0] != rope.shape[0]:
            raise ValueError('K/V/RoPE shapes differ')
        entries = self.history(layer, index)
        device = self.storage_device or key.device
        entry = RawEntry(index, *(x.detach().to(device=device, copy=True) for x in (key, value, rope)))
        self.layers[layer] = [*entries, entry][-self.max_history:]
        self.commits += 1
        self.peak_bytes = max(self.peak_bytes, self.nbytes)

    @property
    def nbytes(self):
        return sum(x.numel()*x.element_size() for entries in self.layers.values()
                   for e in entries for x in (e.key, e.value, e.rope))

    def clear(self):
        self.layers.clear()

    def snapshot(self, layers):
        """Immutable entry lists for teacher-forced tail replay.

        Entries own detached tensors and commit always allocates new entries,
        so copying lists freezes this version without duplicating large K/V.
        """
        return H3ChunkCache(self.max_history, self.storage_device,
                            {i: list(self.layers.get(i, [])) for i in layers})


@dataclass
class ChunkAttention:
    cache: H3ChunkCache
    index: int
    prefix: int
    frame_rows: int
    frame_start: int
    action_rows: torch.Tensor | None
    action_prefix_mode: str = 'own'
    action_feedback: bool = False
    action_cond: torch.Tensor | None = None
    action_adapter: object | None = None
    action_prefix_adapter: object | None = None
    commit: bool = False
    allow_grad_read: bool = False
    clean_graph_entries: CleanGraphCapture | None = None
    _masks: dict = field(default_factory=dict)

    def apply_hidden(self, hidden, layer):
        """Apply an optional action FiLM to current video rows.

        The DiT block calls this immediately after its AdaLN modulation and
        before attention.  Prefix rows remain untouched, so the residual cannot
        alter the static image/text condition or historical cache directly.
        """
        if self.action_prefix_adapter is not None and hasattr(self.action_prefix_adapter, 'apply_hidden'):
            hidden = self.action_prefix_adapter.apply_hidden(
                layer, hidden, prefix=self.prefix, frame_rows=self.frame_rows,
                action_cond=self.action_cond, action_rows=self.action_rows,
                action_frame_start=self.frame_start)
        if self.action_adapter is None or not hasattr(self.action_adapter, 'apply_hidden'):
            return hidden
        return self.action_adapter.apply_hidden(
            layer, hidden, prefix=self.prefix, frame_rows=self.frame_rows,
            action_cond=self.action_cond)

    def masks(self, current_rows, history_rows, device):
        signature = (current_rows, history_rows, str(device))
        if signature not in self._masks:
            ann = torch.full((self.prefix,), -1, device=device, dtype=torch.long)
            if self.action_rows is not None:
                for frame, (lo, hi) in enumerate(self.action_rows.tolist()):
                    ann[lo:hi] = frame
            # Prefix-to-prefix visibility: action rows read their own
            # annotation and common conditions. Optional action-to-video
            # feedback is added separately by attend().
            prefix_mask = (ann[None, :] < 0) | (ann[:, None] == ann[None, :])
            current_frame = self.frame_start + torch.arange(current_rows, device=device)//self.frame_rows
            if self.action_prefix_mode == 'own':
                # Original prototype: each video query reads only its own
                # per-latent action annotation.
                video_prefix = (ann[None, :] < 0) | (ann[None, :] == current_frame[:, None])
            elif self.action_prefix_mode == 'causal':
                # Expose known past/current controls. This is an additional
                # DIRECT path beyond released H3's own-action predicate; it
                # does not imply preservation of H3's action representation.
                video_prefix = (ann[None, :] < 0) | (ann[None, :] <= current_frame[:, None])
            elif self.action_prefix_mode == 'all':
                # Diagnostic upper bound: all action rows are known before
                # sampling, although future schedule rows are visible.
                video_prefix = torch.ones(
                    current_rows, self.prefix, device=device, dtype=torch.bool)
            else:
                raise ValueError(f'unknown action_prefix_mode={self.action_prefix_mode!r}')
            video_mask = torch.cat((video_prefix, torch.ones(
                current_rows, history_rows+current_rows, device=device, dtype=torch.bool)), dim=1)
            self._masks[signature] = (prefix_mask, video_mask)
        return self._masks[signature]

    def attend(self, q, k, v, *, rope_freqs, layer, apply_rope, scale):
        if torch.is_grad_enabled() and (self.commit or not self.allow_grad_read):
            raise RuntimeError('cached H3 attention is inference-only; train with clean-history loss')
        p = self.prefix
        if self.action_adapter is not None:
            q, k, v = self.action_adapter(
                layer, q, k, v, prefix=p, frame_rows=self.frame_rows,
                action_cond=self.action_cond)
        history = self.cache.history(layer, self.index)
        keys = [e.key.to(k.device) for e in history]
        values = [e.value.to(v.device) for e in history]
        ropes = [e.rope.to(rope_freqs.device) for e in history]
        kv = torch.cat([k[:p], *keys, k[p:]], dim=0)
        vv = torch.cat([v[:p], *values, v[p:]], dim=0)
        kr = torch.cat([rope_freqs[:p], *ropes, rope_freqs[p:]], dim=0)
        prefix_mask, video_mask = self.masks(q.shape[0]-p, sum(x.shape[0] for x in keys), q.device)
        qr = apply_rope(q, rope_freqs)
        kk = apply_rope(kv, kr)

        def attention(query, key, value, mask):
            return F.scaled_dot_product_attention(
                query.transpose(0, 1).unsqueeze(0), key.transpose(0, 1).unsqueeze(0),
                value.transpose(0, 1).unsqueeze(0), attn_mask=mask[None, None],
                scale=scale).squeeze(0).transpose(0, 1)

        if self.action_feedback and self.action_rows is not None:
            # In the released H3 directed action mask, an action sentence can
            # read the video latent it controls.  The original causal split
            # computed prefix queries against prefix keys only, silently
            # removing that feedback path.  Restore only the causal-safe
            # current-chunk edge: an action row may read video rows of its own
            # latent frame, never a future frame or the historical cache.
            prefix_feedback_mask = torch.zeros(
                (p, kk.shape[0]), device=q.device, dtype=torch.bool)
            prefix_feedback_mask[:, :p] = prefix_mask
            ann = torch.full((p,), -1, device=q.device, dtype=torch.long)
            for frame, (lo, hi) in enumerate(self.action_rows.tolist()):
                lo, hi = max(0, int(lo)), min(p, int(hi))
                if hi > lo:
                    ann[lo:hi] = frame
            current_frame = self.frame_start + torch.arange(
                q.shape[0] - p, device=q.device) // self.frame_rows
            current_video_offset = p + sum(x.shape[0] for x in keys)
            for frame in torch.unique(ann[ann >= 0]).tolist():
                q_rows = ann == int(frame)
                k_rows = current_frame == int(frame)
                if bool(q_rows.any()) and bool(k_rows.any()):
                    prefix_feedback_mask[q_rows, current_video_offset:] = k_rows
            prefix_output = attention(qr[:p], kk, vv, prefix_feedback_mask)
        else:
            prefix_output = attention(qr[:p], kk[:p], vv[:p], prefix_mask)
        output = torch.cat((prefix_output,
                            attention(qr[p:], kk, vv, video_mask)), dim=0)
        if self.clean_graph_entries is not None and not self.clean_graph_entries.sealed:
            # Capture into a separate sink. The read cache remains immutable,
            # including during activation-checkpoint recomputation. Each
            # assignment owns a new entry. Seal/clear the temporary collector
            # after prefill so backward does not duplicate CPU K/V or retain
            # graph tensors through its own checkpoint closure.
            device = self.cache.storage_device or k.device
            self.clean_graph_entries.entries[layer] = RawEntry(self.index,
                k[p:].to(device=device, copy=True),
                v[p:].to(device=device, copy=True),
                rope_freqs[p:].to(device=device, copy=True))
        if self.commit:
            self.cache.commit(layer, self.index, k[p:], v[p:], rope_freqs[p:])
        return output


def slice_packed(full, start, stop, frame_rows):
    """Keep the complete prefix, select global-position video rows, drop pad.

    All action sentences stay independently refined in the text refiner.
    ChunkAttention determines which rows video queries may read, according to
    action_prefix_mode; retaining a row does not itself make it visible.
    """
    v0 = int(full['action_video_start'])
    lo, hi = v0+start*frame_rows, v0+stop*frame_rows
    device = full['img_pos'].device
    select = torch.cat((torch.arange(v0, device=device),
                        torch.arange(lo, hi, device=device)))
    remap = torch.full((int(full['seq_len']),), -1, dtype=torch.long, device=device)
    remap[select] = torch.arange(select.numel(), device=device)
    result = dict(full)
    result['img_position_ids'] = full['img_position_ids'][:, select]
    result['token_tags'] = full['token_tags'][select]
    mapped_img = remap[full['img_pos']]
    result['img_pos'] = mapped_img[mapped_img >= 0]
    result['audio_pos'] = remap[full['audio_pos']]
    result['text_pos'] = remap[full['text_pos']]
    # Keep the full per-latent action layout for cached and recompute paths.
    # Visibility depends on action_prefix_mode: own hides older action rows,
    # causal exposes them. Historical video KV retains the state constructed
    # during its clean commit; retaining text rows does not recompute that KV.
    result['action_text_rows'] = full.get('action_text_rows')
    result['action_text_spans_local'] = full.get('action_text_spans_local')
    result['action_video_start'] = v0
    result['seq_len'] = select.numel()
    result['cu_seqlens'] = torch.tensor([0, select.numel()], device=select.device, dtype=torch.int32)
    result['action_real_used'] = select.numel()
    return result


def chunk_forward(dit, current, *, full_packed, prompt, anchor, audio, sigma,
                  index, chunk_frames, cache, commit=False, anchor_frame_index=None,
                  anchor_slot=0, fixed_boundary=None, action_prefix_mode='own',
                  action_feedback=False, action_cond=None, action_adapter=None,
                  action_prefix_adapter=None,
                  allow_grad_read=False, use_gradient_checkpointing=False,
                  use_gradient_checkpointing_offload=False, target_sigma=None,
                  clean_graph_entries=None):
    from diffsynth.pipelines.minimax_h3_audio_video import model_fn_minimax_h3
    if commit and sigma != 0:
        raise ValueError('commit must recompute final CLEAN latents at sigma=0')
    if clean_graph_entries is not None and (commit or sigma != 0 or not allow_grad_read):
        raise ValueError('Clean graph capture requires sigma=0, gradient reads and no cache mutation')
    if hasattr(dit, 'anyflow_conditioner'):
        if commit and target_sigma is None:
            target_sigma = 0.0
        if target_sigma is None:
            raise ValueError('AnyFlow chunk forward requires target_sigma')
        if not 0 <= float(target_sigma) <= float(sigma) <= 1:
            raise ValueError('AnyFlow requires 0 <= target_sigma <= sigma <= 1')
        # The finite-map trajectory accumulates in FP32, but packed H3
        # backbone inputs must share the conditioning/weight compute dtype.
        # Cast only the evaluation inputs, preserving the caller's FP32 state
        # and differentiability through this cast during training/replay.
    elif target_sigma is not None:
        raise ValueError('target_sigma requires an AnyFlow conditioner')
    if hasattr(dit, 'anyflow_conditioner') or hasattr(dit, '_h3_input_dtype'):
        input_dtype = getattr(dit, '_h3_input_dtype', prompt.dtype)
        current = current.to(dtype=input_dtype)
        anchor = anchor.to(dtype=input_dtype)
        audio = audio.to(dtype=input_dtype)
    rows = (current.shape[-2]//2)*(current.shape[-1]//2)
    start = index*chunk_frames
    packed = slice_packed(full_packed, start, start+current.shape[2], rows)
    if anchor_frame_index is not None:
        packed = retime_anchor_position(
            packed, full_packed, anchor_rows=int(anchor.shape[0]),
            frame_index=int(anchor_frame_index), frame_rows=rows,
            anchor_slot=int(anchor_slot))
    input_latents = None
    denoise_mask = None
    if fixed_boundary is not None:
        if fixed_boundary.shape != current[:, :, :1].shape:
            raise ValueError(
                f"fixed boundary must be [B,C,1,H,W], got {tuple(fixed_boundary.shape)} "
                f"for current {tuple(current.shape)}")
        input_latents = current.clone()
        input_latents[:, :, :1] = fixed_boundary
        denoise_mask = torch.ones_like(current)
        denoise_mask[:, :, :1] = 0
    control = ChunkAttention(cache, index, int(packed['action_video_start']), rows,
                             start, full_packed.get('action_text_rows'), action_prefix_mode,
                             action_feedback, action_cond, action_adapter,
                             action_prefix_adapter, commit,
                             allow_grad_read, clean_graph_entries=clean_graph_entries)
    return model_fn_minimax_h3(
        dit, current, audio, packed, prompt,
        timestep_video=torch.tensor(float(sigma)*1000, device=current.device),
        timestep_audio=torch.tensor(1000., device=current.device),
        keyframe_cond_anchor=anchor, fixed_prefix_timesteps=True,
        input_latents_video=input_latents, denoise_mask_video=denoise_mask,
        use_gradient_checkpointing=use_gradient_checkpointing,
        use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
        causal_control=control,
        target_timestep_video=(None if target_sigma is None else
                               torch.tensor(float(target_sigma) * 1000, device=current.device)),
    )[0]


def recompute_forward(dit, current, *, history, full_packed, prompt, anchor, audio,
                      sigma, chunk_frames, window_chunks, anchor_frame_index=None,
                      anchor_slot=0, history_start=0, fixed_boundary=None,
                      action_prefix_mode='own', action_cond=None, action_adapter=None):
    """Reference: recompute ALL causal ancestors, query current chunk only.

    Dropping old ancestors at the input would alter historical hidden states
    in a multilayer transformer; sliding-window attention alone does not make
    that shortcut equivalent to a persistent cache.
    """
    from diffsynth.pipelines.minimax_h3_audio_video import model_fn_minimax_h3
    start = history.shape[2]
    if history_start != 0:
        raise ValueError("recompute requires ALL ancestors; truncation changes historical hidden states")
    whole = torch.cat((history, current), dim=2)
    update = torch.zeros_like(whole)
    update[:, :, start:] = 1
    input_latents = whole
    if fixed_boundary is not None:
        if fixed_boundary.shape != current[:, :, :1].shape:
            raise ValueError(
                f"fixed boundary must be [B,C,1,H,W], got {tuple(fixed_boundary.shape)} "
                f"for current {tuple(current.shape)}")
        input_latents = whole.clone()
        input_latents[:, :, start:start+1] = fixed_boundary
        update[:, :, start:start+1] = 0
    rows = (current.shape[-2]//2)*(current.shape[-1]//2)
    packed = slice_packed(full_packed, history_start, history_start + whole.shape[2], rows)
    if anchor_frame_index is not None:
        packed = retime_anchor_position(
            packed, full_packed, anchor_rows=int(anchor.shape[0]),
            frame_index=int(anchor_frame_index), frame_rows=rows,
            anchor_slot=int(anchor_slot))
    velocity = model_fn_minimax_h3(
        dit, whole, audio, packed, prompt,
        timestep_video=torch.tensor(float(sigma)*1000, device=current.device),
        timestep_audio=torch.tensor(1000., device=current.device),
        keyframe_cond_anchor=anchor, input_latents_video=input_latents, denoise_mask_video=update,
        fixed_prefix_timesteps=True, causal_chunk_size=chunk_frames,
        causal_window_chunks=window_chunks,
    )[0]
    return velocity[:, :, start:]
