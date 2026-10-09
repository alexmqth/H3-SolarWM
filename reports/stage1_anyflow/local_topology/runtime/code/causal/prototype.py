"""Minimal, testable causal chunk prototype for H3-World.

There are two intentionally separate pieces here:

* :func:`build_causal_visibility` describes the information boundary used by
  the H3 packed sequence.  A video query in chunk ``c`` can read static
  condition rows and video chunks in the local history window ending at
  ``c``.  It cannot read a future chunk.  The returned matrix is indexed as
  ``[query, key]`` and ``True`` means visible.
* :class:`LayeredRawKVCache` mirrors SolarWM's raw per-layer cache contract.
  Entries are detached before committing, indices must be consecutive, and a
  bounded history is retained.

The tiny flow-map model is a CPU-friendly training/inference smoke target.
It is not a replacement for H3's DiT, nor does it implement SolarWM Stage2
SGF/DMD.  Its purpose is to make the proposed causal training interface
executable without downloading a training corpus or a 33B model.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class CausalChunkLayout:
    """Packed H3 sequence layout for chunk-level causal visibility.

    The layout follows H3's FL2VA order ``[text | condition | audio |
    video | padding]``.  ``real_seq_len`` excludes right padding; when it is
    omitted, the real sequence ends before ``pad_rows``. ``condition_rows`` includes the
    first-frame anchor rows and remains visible to every video chunk.
    """

    text_rows: int
    condition_rows: int
    audio_rows: int
    latent_frames: int
    frame_rows: int
    chunk_frames: int = 5
    window_chunks: int = 5
    real_seq_len: int | None = None
    pad_rows: int = 0

    def __post_init__(self) -> None:
        ints = (
            self.text_rows,
            self.condition_rows,
            self.audio_rows,
            self.latent_frames,
            self.frame_rows,
            self.chunk_frames,
            self.window_chunks,
            self.pad_rows,
        )
        if any(int(x) < 0 for x in ints):
            raise ValueError("layout sizes must be non-negative")
        if self.latent_frames <= 0 or self.frame_rows <= 0:
            raise ValueError("latent_frames and frame_rows must be positive")
        if self.chunk_frames <= 0 or self.window_chunks <= 0:
            raise ValueError("chunk_frames and window_chunks must be positive")
        if self.real_seq_len is not None and not 0 <= self.real_seq_len <= self.seq_len:
            raise ValueError("real_seq_len must lie within the packed sequence")

    @property
    def video_start(self) -> int:
        return self.text_rows + self.condition_rows + self.audio_rows

    @property
    def video_rows(self) -> int:
        return self.latent_frames * self.frame_rows

    @property
    def seq_len(self) -> int:
        return self.video_start + self.video_rows + self.pad_rows

    @property
    def video_end(self) -> int:
        return self.video_start + self.video_rows

    @property
    def num_chunks(self) -> int:
        return (self.latent_frames + self.chunk_frames - 1) // self.chunk_frames

    @property
    def effective_real_seq_len(self) -> int:
        return self.video_end if self.real_seq_len is None else self.real_seq_len

    def chunk_for_row(self, row: int) -> int:
        if not self.video_start <= row < self.video_end:
            raise ValueError(f"row {row} is not a video row")
        frame = (row - self.video_start) // self.frame_rows
        return frame // self.chunk_frames

    def validate(self) -> None:
        """Raise a useful error when an H3 packed layout is inconsistent."""

        if self.video_start < 0 or self.video_end > self.seq_len:
            raise ValueError("video span falls outside the packed sequence")
        if self.effective_real_seq_len < self.video_end:
            raise ValueError("real_seq_len truncates the video rows")
        if self.effective_real_seq_len != self.video_end:
            raise ValueError("real_seq_len must end at the video boundary in FL2VA")


def build_causal_visibility(
    layout: CausalChunkLayout,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Build a chunk-causal ``[seq, seq]`` visibility matrix.

    Static rows (text, first-frame condition, and audio) can attend to static
    rows.  A video query in chunk ``c`` can attend to static rows and video
    chunks ``max(0, c-window_chunks+1) .. c``.  A static query never reads a
    video row, preventing a future video sample from being fed back into the
    condition stream.  Padding rows are self-visible only, matching the H3
    FlexAttention padding convention.
    """

    layout.validate()
    n = layout.seq_len
    real = layout.effective_real_seq_len
    visible = torch.zeros((n, n), dtype=torch.bool, device=device)

    # Static prefix is available as a condition but does not consume target
    # video information.  The diagonal is added for all real rows below.
    static_end = layout.video_start
    if static_end:
        visible[:static_end, :static_end] = True

    for chunk in range(layout.num_chunks):
        frame_lo = chunk * layout.chunk_frames
        frame_hi = min(layout.latent_frames, (chunk + 1) * layout.chunk_frames)
        q_lo = layout.video_start + frame_lo * layout.frame_rows
        q_hi = layout.video_start + frame_hi * layout.frame_rows
        if q_lo >= q_hi:
            continue
        first_chunk = max(0, chunk - layout.window_chunks + 1)
        k_lo = layout.video_start + first_chunk * layout.chunk_frames * layout.frame_rows
        k_hi = q_hi
        visible[q_lo:q_hi, :static_end] = True
        visible[q_lo:q_hi, k_lo:k_hi] = True

    # Never expose padding.  Self visibility avoids an all-masked row if a
    # caller passes this matrix to a softmax implementation.
    visible[real:, :] = False
    visible[:, real:] = False
    if real:
        idx = torch.arange(real, device=visible.device)
        visible[idx, idx] = True
    if real < n:
        idx = torch.arange(real, n, device=visible.device)
        visible[idx, idx] = True
    return visible


@dataclass
class _RawKVEntry:
    chunk_index: int
    key: torch.Tensor
    value: torch.Tensor


@dataclass
class LayeredRawKVCache:
    """Per-layer raw K/V cache with SolarWM-compatible lifecycle checks."""

    max_history: int = 5
    layers: dict[int, list[_RawKVEntry]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.max_history <= 0:
            raise ValueError("max_history must be positive")

    def history(self, layer: int, chunk_index: int) -> tuple[_RawKVEntry, ...]:
        if chunk_index < 0:
            raise ValueError("chunk_index must be non-negative")
        entries = self.layers.get(int(layer), [])
        expected = list(range(max(0, chunk_index - self.max_history), chunk_index))
        actual = [entry.chunk_index for entry in entries]
        if actual != expected:
            raise RuntimeError(
                f"raw KV history differs at layer={layer}, chunk={chunk_index}: "
                f"expected {expected}, got {actual}"
            )
        return tuple(entries)

    def commit(self, layer: int, chunk_index: int, key: torch.Tensor, value: torch.Tensor) -> None:
        """Commit detached raw K/V after a no-grad chunk forward."""

        if torch.is_grad_enabled():
            raise RuntimeError("raw KV commits must happen in a no-grad forward")
        if not torch.is_tensor(key) or not torch.is_tensor(value):
            raise TypeError("key and value must be tensors")
        if key.shape != value.shape:
            raise ValueError(f"key/value shapes differ: {key.shape} vs {value.shape}")
        entries = list(self.history(layer, chunk_index))
        entries.append(
            _RawKVEntry(
                int(chunk_index),
                key.detach().clone(),
                value.detach().clone(),
            )
        )
        self.layers[int(layer)] = entries[-self.max_history :]

    def clear(self) -> None:
        self.layers.clear()

    def __len__(self) -> int:
        return sum(len(entries) for entries in self.layers.values())


def scaled_dot_product_chunk_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    scale: float | None = None,
) -> torch.Tensor:
    """Attention helper accepting ``[B, heads, query, dim]`` tensors."""

    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("query/key/value must be [batch, heads, tokens, dim]")
    if key.shape != value.shape or query.shape[:2] != key.shape[:2] or query.shape[-1] != key.shape[-1]:
        raise ValueError("incompatible attention tensor shapes")
    return F.scaled_dot_product_attention(query, key, value, scale=scale)


class TinyCausalFlowMap(nn.Module):
    """Small teacher-forced flow-map model for a forward/backward smoke test.

    Inputs are arbitrary latent feature vectors, so a caller can use a tiny
    synthetic tensor or a saved H3 latent chunk.  The model performs current
    chunk queries against clean history plus the current noisy chunk.  At
    inference, the same projection can use :class:`LayeredRawKVCache` to
    avoid recomputing history K/V.  This is a causal mechanism demo, not a
    trained H3 checkpoint or Stage2 student.
    """

    def __init__(self, feature_dim: int, hidden_dim: int = 64, heads: int = 4, max_timestep: float = 1.0):
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("hidden_dim must be divisible by heads")
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.heads = int(heads)
        self.head_dim = hidden_dim // heads
        self.max_timestep = float(max_timestep)
        self.input_proj = nn.Linear(feature_dim, hidden_dim)
        self.condition_proj = nn.Linear(feature_dim, hidden_dim)
        self.time_proj = nn.Sequential(nn.Linear(1, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, feature_dim)

    def _heads(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        return x.view(b, t, self.heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        noisy_chunk: torch.Tensor,
        *,
        clean_history: torch.Tensor | None = None,
        condition: torch.Tensor | None = None,
        timestep: torch.Tensor | float = 0.5,
        cache: LayeredRawKVCache | None = None,
        layer: int = 0,
        chunk_index: int = 0,
        commit_cache: bool = False,
    ) -> torch.Tensor:
        if noisy_chunk.ndim != 3 or noisy_chunk.shape[-1] != self.feature_dim:
            raise ValueError("noisy_chunk must be [batch, tokens, feature_dim]")
        b, tokens, _ = noisy_chunk.shape
        if clean_history is not None and (clean_history.ndim != 3 or clean_history.shape[0] != b
                                          or clean_history.shape[-1] != self.feature_dim):
            raise ValueError("clean_history must be [batch, history_tokens, feature_dim]")
        if cache is not None and clean_history is not None:
            raise ValueError("pass either explicit clean_history or a cache, not both")
        if condition is None:
            condition = torch.zeros((b, self.feature_dim), device=noisy_chunk.device, dtype=noisy_chunk.dtype)
        if condition.shape != (b, self.feature_dim):
            raise ValueError("condition must be [batch, feature_dim]")
        t = torch.as_tensor(timestep, device=noisy_chunk.device, dtype=noisy_chunk.dtype)
        t = t.reshape(-1, 1)
        if t.shape[0] == 1:
            t = t.expand(b, -1)
        if t.shape != (b, 1):
            raise ValueError("timestep must be scalar or [batch]")
        if commit_cache and (torch.is_grad_enabled() or bool((t != 0).any())):
            raise ValueError("commit requires a no-grad clean chunk forward at noise timestep 0")

        current = self.input_proj(noisy_chunk)
        current = current + self.condition_proj(condition).unsqueeze(1)
        current = current + self.time_proj(t / self.max_timestep).unsqueeze(1)
        q = self._heads(self.q_proj(current))
        current_k = self._heads(self.k_proj(current))
        current_v = self._heads(self.v_proj(current))

        cached = ()
        if cache is not None:
            cached = cache.history(layer, chunk_index)
        if cached:
            history_k = torch.cat([entry.key.to(current_k.device, current_k.dtype) for entry in cached], dim=2)
            history_v = torch.cat([entry.value.to(current_v.device, current_v.dtype) for entry in cached], dim=2)
        elif clean_history is not None and clean_history.shape[1]:
            history = self.input_proj(clean_history)
            history = history + self.condition_proj(condition).unsqueeze(1)
            # Cache commits and teacher forcing must encode clean history at
            # the same timestep, including the time MLP's nonzero biases.
            history = history + self.time_proj(torch.zeros_like(t)).unsqueeze(1)
            history_k = self._heads(self.k_proj(history))
            history_v = self._heads(self.v_proj(history))
        else:
            history_k = current_k[:, :, :0]
            history_v = current_v[:, :, :0]

        key = torch.cat([history_k, current_k], dim=2)
        value = torch.cat([history_v, current_v], dim=2)
        attended = scaled_dot_product_chunk_attention(q, key, value)
        attended = attended.transpose(1, 2).reshape(b, tokens, self.hidden_dim)
        result = self.out_proj(attended + current)
        if commit_cache:
            if cache is None:
                raise ValueError("commit_cache requires a cache")
            cache.commit(layer, chunk_index, current_k, current_v)
        return result


def teacher_forcing_flow_loss(
    model: TinyCausalFlowMap,
    clean_video: torch.Tensor,
    *,
    condition: torch.Tensor | None = None,
    chunk_frames: int = 5,
    max_history: int = 5,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Compute a minimal clean-history teacher-forcing flow-matching loss.

    ``clean_video`` is ``[batch, latent_frames, feature_dim]``.  Each chunk
    receives a noisy interpolation ``x_t=(1-t)x_0+t eps`` and learns the
    velocity ``eps-x_0``.  History is clean and detached, matching SolarWM
    Stage1's teacher-forced conditioning at the interface level.
    """

    if clean_video.ndim != 3 or clean_video.shape[-1] != model.feature_dim:
        raise ValueError("clean_video must be [batch, latent_frames, feature_dim]")
    if chunk_frames <= 0 or max_history <= 0 or clean_video.shape[1] == 0:
        raise ValueError("chunk_frames, max_history and frame count must be positive")
    b, frames, _ = clean_video.shape
    if condition is None:
        condition = torch.zeros((b, model.feature_dim), device=clean_video.device, dtype=clean_video.dtype)
    losses = []
    for start in range(0, frames, chunk_frames):
        stop = min(frames, start + chunk_frames)
        clean = clean_video[:, start:stop]
        noise = torch.randn(clean.shape, device=clean.device, dtype=clean.dtype, generator=generator)
        timestep = torch.rand((b, 1), device=clean.device, dtype=clean.dtype, generator=generator).clamp_(0.02, 0.98)
        noisy = (1.0 - timestep[:, None]) * clean + timestep[:, None] * noise
        history = clean_video[:, max(0, start - max_history * chunk_frames):start].detach()
        velocity = model(noisy, clean_history=history, condition=condition, timestep=timestep)
        losses.append(F.mse_loss(velocity, noise - clean))
    return torch.stack(losses).mean()
