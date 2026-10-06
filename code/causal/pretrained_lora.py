"""Additional causal QKV adapter, separate from the released H3 action LoRA."""
import torch
from torch import nn


class CausalActionResidual(nn.Module):
    """Zero-initialized action residual for causal attention.

    H3's released action LoRA was trained with full-sequence bidirectional
    attention.  In the causal cache path, the per-latent action text still
    reaches the current video query, but it no longer has the same
    action/video feedback path.  This small adapter adds an action-dependent
    Q/K/V residual only to the current chunk.  It is zero at initialization,
    so installing it cannot change the pretrained causal output before
    training.  Historical K/V are affected only when the clean commit forward
    is run with the trained adapter, which is the intended streaming state.

    ``block_indices`` keeps the capacity in the same tail blocks as the causal
    QKV adapter.  One projection maps the per-latent one-hot action vector to
    Q/K/V residuals; the residual is broadcast over the spatial rows of that
    latent frame.
    """

    def __init__(self, block_indices, num_buttons, num_heads, head_dim,
                 device="cpu", dtype=None, mode="qkv", hidden_size=None,
                 gain_compute_dtype="float32"):
        super().__init__()
        if not block_indices or num_buttons < 1 or num_heads < 1 or head_dim < 1:
            raise ValueError("invalid causal action residual dimensions")
        self.block_indices = [int(i) for i in block_indices]
        self.num_buttons = int(num_buttons)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.mode = str(mode)
        if gain_compute_dtype not in ("float32", "projection"):
            raise ValueError("gain_compute_dtype must be float32 or projection")
        self.gain_compute_dtype = gain_compute_dtype
        # Teacher replay temporarily disables the student-only residual while
        # keeping the same packed action condition and cache protocol.  This
        # flag is deliberately runtime-only and is not serialized as a weight.
        self.enabled = True
        if self.mode not in ("qkv", "hidden"):
            raise ValueError("action residual mode must be qkv or hidden")
        self.token_dim = self.num_heads * self.head_dim
        if self.mode == "hidden":
            if hidden_size is None or int(hidden_size) < 1:
                raise ValueError("hidden action residual requires hidden_size")
            self.hidden_size = int(hidden_size)
            out_dim = 2 * self.hidden_size
        else:
            self.hidden_size = None
            out_dim = 3 * self.token_dim
        self.projections = nn.ModuleDict({
            str(i): nn.Linear(self.num_buttons, out_dim, bias=False)
            for i in self.block_indices
        })
        # Action-conditioned gain is initialized to one for backward
        # compatibility with all existing residual checkpoints.  It lets a
        # student learn that different controls can need different residual
        # amplitudes under the causal topology (A and D were empirically
        # asymmetric), without changing the shared projection capacity.
        gain_dtype = dtype or torch.float32
        self.action_gain = nn.Parameter(
            torch.ones(self.num_buttons, device=device, dtype=gain_dtype))
        for proj in self.projections.values():
            nn.init.zeros_(proj.weight)
        if dtype is not None:
            self.to(device=device, dtype=dtype)
        else:
            self.to(device=device)
        # Keep gains in FP32 so a small optimizer update is not rounded away
        # when the DiT itself runs in BF16. The small action projection also
        # computes in FP32; casting the gain before projection would erase
        # sub-BF16 updates (e.g. 8.0092 becomes 8.0).
        self.action_gain.data = self.action_gain.data.float()

    def project_action(self, key, action_cond):
        weight = self.projections[key].weight
        if self.gain_compute_dtype == "projection":
            # Preserve the exact computation of already archived checkpoints.
            scaled = action_cond.to(weight) * self.action_gain.to(weight.dtype)
            return self.projections[key](scaled)
        # Only a [chunk_frames, num_buttons] input and the small residual
        # projection use FP32, never the 33B backbone. Disable autocast here
        # so AMP cannot silently reintroduce the early gain quantization.
        with torch.autocast(device_type=weight.device.type, enabled=False):
            scaled = action_cond.to(device=weight.device, dtype=torch.float32)
            scaled = scaled * self.action_gain.float()
            return torch.nn.functional.linear(scaled, weight.float())

    def forward(self, layer_index, q, k, v, *, prefix, frame_rows, action_cond):
        if not self.enabled:
            return q, k, v
        key = str(int(layer_index))
        if key not in self.projections:
            return q, k, v
        if action_cond is None:
            return q, k, v
        if action_cond.ndim != 2 or action_cond.shape[1] != self.num_buttons:
            raise ValueError(
                f"action_cond must be [latent,{self.num_buttons}], got {tuple(action_cond.shape)}")
        current_rows = q.shape[0] - int(prefix)
        expected_rows = int(action_cond.shape[0]) * int(frame_rows)
        if current_rows != expected_rows:
            raise ValueError(
                f"action residual rows mismatch: current={current_rows}, expected={expected_rows}")
        if self.mode != "qkv":
            return q, k, v
        # Cast only the completed residual to the attention tensor dtype.
        delta = self.project_action(key, action_cond)
        delta = delta.view(action_cond.shape[0], 3, self.num_heads, self.head_dim)
        delta = delta.repeat_interleave(int(frame_rows), dim=0).to(q.dtype)
        dq, dk, dv = delta.unbind(dim=1)
        q_current = q[int(prefix):] + dq
        k_current = k[int(prefix):] + dk
        v_current = v[int(prefix):] + dv
        return (torch.cat((q[:int(prefix)], q_current), dim=0),
                torch.cat((k[:int(prefix)], k_current), dim=0),
                torch.cat((v[:int(prefix)], v_current), dim=0))

    def apply_hidden(self, layer_index, hidden, *, prefix, frame_rows, action_cond):
        """Apply a zero-initialized FiLM residual to current video rows."""
        if not self.enabled:
            return hidden
        if self.mode != "hidden":
            return hidden
        key = str(int(layer_index))
        if key not in self.projections or action_cond is None:
            return hidden
        current_rows = hidden.shape[0] - int(prefix)
        expected_rows = int(action_cond.shape[0]) * int(frame_rows)
        if current_rows != expected_rows:
            raise ValueError(
                f"hidden action rows mismatch: current={current_rows}, expected={expected_rows}")
        delta = self.project_action(key, action_cond)
        delta = delta.view(action_cond.shape[0], 2, self.hidden_size)
        delta = delta.repeat_interleave(int(frame_rows), dim=0).to(hidden.dtype)
        scale, shift = delta.unbind(dim=1)
        current = hidden[int(prefix):]
        current = current * (1 + scale) + shift
        return torch.cat((hidden[:int(prefix)], current), dim=0)


class CausalActionPrefixResidual(nn.Module):
    """Zero-initialized residual for the per-latent action text rows.

    Causalization changes the topology seen by H3's action sentences: the
    action rows are recomputed in every chunk, but their original full-horizon
    feedback path is gone.  This small adapter can retune only those prefix
    rows from the one-hot action condition while leaving image, audio and video
    rows untouched.  ``action_frame_start`` maps the current chunk's local
    action tensor to the global action-row spans in the packed prefix.
    """

    def __init__(self, block_indices, num_buttons, hidden_size, device="cpu", dtype=None):
        super().__init__()
        if not block_indices or num_buttons < 1 or hidden_size < 1:
            raise ValueError("invalid action-prefix residual dimensions")
        self.block_indices = [int(i) for i in block_indices]
        self.num_buttons = int(num_buttons)
        self.hidden_size = int(hidden_size)
        self.enabled = True
        self.projections = nn.ModuleDict({
            str(i): nn.Linear(self.num_buttons, self.hidden_size, bias=False)
            for i in self.block_indices
        })
        for proj in self.projections.values():
            nn.init.zeros_(proj.weight)
        self.to(device=device, dtype=dtype)

    def apply_hidden(self, layer_index, hidden, *, prefix, frame_rows,
                     action_cond, action_rows=None, action_frame_start=0):
        if not self.enabled or action_cond is None:
            return hidden
        key = str(int(layer_index))
        if key not in self.projections or action_rows is None:
            return hidden
        if action_cond.ndim != 2 or action_cond.shape[1] != self.num_buttons:
            raise ValueError(
                f"action_cond must be [latent,{self.num_buttons}], got {tuple(action_cond.shape)}")
        if hidden.shape[0] < int(prefix):
            raise ValueError("hidden sequence is shorter than causal prefix")
        delta = self.projections[key](action_cond.to(self.projections[key].weight))
        out = hidden.clone()
        # Prefix action rows include all latent controls. Only the current
        # chunk rows receive a residual; historical/future action sentences
        # keep their frozen H3 representation in this forward.
        for local_frame, span in enumerate(action_rows.tolist()):
            global_frame = local_frame
            if global_frame < int(action_frame_start) or global_frame >= int(action_frame_start) + action_cond.shape[0]:
                continue
            lo, hi = max(0, int(span[0])), min(int(prefix), int(span[1]))
            if hi <= lo:
                continue
            out[lo:hi] = out[lo:hi] + delta[global_frame - int(action_frame_start)]
        return out


def install_action_residual(dit, block_indices, num_buttons=9, device="cpu", mode="qkv"):
    """Create a zero-initialized causal action residual matching ``dit``."""
    indices = [int(i) for i in block_indices]
    if not indices or any(i < 0 or i >= len(dit.blocks) for i in indices):
        raise ValueError("action residual block index is out of range")
    ref = next(dit.parameters())
    return CausalActionResidual(
        indices, num_buttons, dit.num_attention_heads,
        dit.blocks[indices[0]].attn.head_dim, device=device, dtype=ref.dtype,
        mode=mode, hidden_size=dit.hidden_size)


def save_action_residual(path, adapter, metadata):
    if not isinstance(adapter, CausalActionResidual):
        raise TypeError("expected CausalActionResidual")
    torch.save(dict(
        format="h3_causal_action_residual_v1",
        block_indices=list(adapter.block_indices),
        num_buttons=adapter.num_buttons,
        num_heads=adapter.num_heads,
        head_dim=adapter.head_dim,
        mode=adapter.mode,
        hidden_size=adapter.hidden_size,
        action_gain=adapter.action_gain.detach().cpu(),
        gain_compute_dtype=adapter.gain_compute_dtype,
        weights={k: v.weight.detach().cpu() for k, v in adapter.projections.items()},
        metadata=metadata,
    ), path)


def save_action_prefix_residual(path, adapter, metadata):
    if not isinstance(adapter, CausalActionPrefixResidual):
        raise TypeError("expected CausalActionPrefixResidual")
    torch.save(dict(
        format="h3_causal_action_prefix_residual_v1",
        block_indices=list(adapter.block_indices),
        num_buttons=adapter.num_buttons,
        hidden_size=adapter.hidden_size,
        weights={k: v.weight.detach().cpu() for k, v in adapter.projections.items()},
        metadata=metadata,
    ), path)


def load_action_residual(dit, path, device):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("format") != "h3_causal_action_residual_v1":
        raise ValueError("unsupported causal action residual format")
    adapter = CausalActionResidual(
        state["block_indices"], state["num_buttons"], state["num_heads"],
        state["head_dim"], device=device, dtype=next(dit.parameters()).dtype,
        mode=state.get("mode", "qkv"), hidden_size=state.get("hidden_size"),
        gain_compute_dtype=state.get("gain_compute_dtype", "projection"))
    # Reconstruct the same injection space before loading the projection.
    if adapter.num_heads != dit.num_attention_heads:
        raise ValueError("action residual head count does not match DiT")
    with torch.no_grad():
        if state.get("action_gain") is not None:
            adapter.action_gain.copy_(state["action_gain"].to(adapter.action_gain))
        for key, weight in state["weights"].items():
            if key not in adapter.projections:
                raise ValueError(f"action residual contains unknown block {key}")
            adapter.projections[key].weight.copy_(weight.to(adapter.projections[key].weight))
    adapter.requires_grad_(False)
    return dict(adapter=adapter, block_indices=list(adapter.block_indices),
                metadata=state.get("metadata", {}))


def load_action_prefix_residual(dit, path, device):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("format") != "h3_causal_action_prefix_residual_v1":
        raise ValueError("unsupported causal action prefix residual format")
    adapter = CausalActionPrefixResidual(
        state["block_indices"], state["num_buttons"], state["hidden_size"],
        device=device, dtype=next(dit.parameters()).dtype)
    if adapter.hidden_size != dit.hidden_size:
        raise ValueError("action prefix hidden size does not match DiT")
    with torch.no_grad():
        for key, weight in state["weights"].items():
            if key not in adapter.projections:
                raise ValueError(f"action prefix residual contains unknown block {key}")
            adapter.projections[key].weight.copy_(weight.to(adapter.projections[key].weight))
    adapter.requires_grad_(False)
    return dict(adapter=adapter, block_indices=list(adapter.block_indices),
                metadata=state.get("metadata", {}))


def save_h3_lora_adapter(path, entries, metadata):
    """Save an online update to a selected subset of the released H3 LoRA."""
    torch.save(dict(
        format="h3_online_lora_v1",
        weights={name: {"lora_A": a.detach().cpu(),
                        "lora_B": b.detach().cpu()}
                 for name, a, b in entries},
        metadata=metadata,
    ), path)


def load_h3_lora_adapter(dit, path, device):
    """Load a student LoRA update after the released H3 LoRA is installed."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("format") != "h3_online_lora_v1":
        raise ValueError("unsupported H3 online LoRA adapter format")
    modules = {getattr(module, "name", ""): module
               for module in dit.modules() if hasattr(module, "lora_A_weights")}
    loaded = []
    for name, values in state["weights"].items():
        module = modules.get(name)
        if module is None or not module.lora_A_weights or not module.lora_B_weights:
            raise ValueError(f"H3 LoRA module is not loaded: {name}")
        module.lora_A_weights[0] = values["lora_A"].to(device=device)
        module.lora_B_weights[0] = values["lora_B"].to(device=device)
        loaded.append(name)
    return {"metadata": state.get("metadata", {}), "module_names": loaded}


def replay_tail(blocks, block_indices, hidden, block_kwargs):
    """Replay a captured tail using each block's own historical cache layer."""
    if len(blocks) != len(block_indices):
        raise ValueError("tail blocks and original layer indices must align")
    for block, index in zip(blocks, block_indices):
        hidden = block(hidden, **dict(block_kwargs, layer_index=index))
    return hidden


class CausalQKVLoRA(nn.Module):
    def __init__(self, base, rank=8, device="cpu"):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.enabled = True
        self.rank = rank
        self.scale = 1.0 / rank
        # FP32 optimizer states; base H3 continues to run in its native dtype.
        self.lora_A = nn.Parameter(torch.randn(rank, base.in_features, device=device) * .02)
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, device=device))

    def forward(self, x):
        if not self.enabled:
            return self.base(x)
        residual = ((x.float() @ self.lora_A.T) @ self.lora_B.T) * self.scale
        return self.base(x) + residual.to(x.dtype)


def install_adapter(dit, rank=8, block_index=-1, device="cpu"):
    if rank < 1 or not -len(dit.blocks) <= block_index < len(dit.blocks):
        raise ValueError("rank or block index is out of range")
    index = block_index % len(dit.blocks)
    attention = dit.blocks[index].attn
    if isinstance(attention.qkv_proj, CausalQKVLoRA):
        raise ValueError("causal adapter already installed")
    adapter = CausalQKVLoRA(attention.qkv_proj, rank, device)
    attention.qkv_proj = adapter
    return adapter, index


def install_adapters(dit, rank=8, block_indices=(-4, -3, -2, -1), device="cpu"):
    """Install independent causal QKV adapters on a tail of DiT blocks."""
    if not block_indices:
        raise ValueError("at least one tail block is required")
    adapters, indices = [], []
    for block_index in block_indices:
        adapter, index = install_adapter(dit, rank, block_index, device)
        adapters.append(adapter)
        indices.append(index)
    return adapters, indices


def save_adapter(path, adapter, block_index, metadata):
    torch.save(dict(format="h3_causal_tail_qkv_v1", rank=adapter.rank,
                    block_index=block_index, scale=adapter.scale,
                    lora_A=adapter.lora_A.detach().cpu(),
                    lora_B=adapter.lora_B.detach().cpu(), metadata=metadata), path)


def save_adapters(path, adapters, block_indices, metadata):
    if not adapters or len(adapters) != len(block_indices):
        raise ValueError("adapters and block_indices must be non-empty and aligned")
    first = adapters[0]
    if any(a.rank != first.rank or a.scale != first.scale for a in adapters):
        raise ValueError("all tail adapters must share rank and scale")
    torch.save(dict(format="h3_causal_tail_qkv_v2", rank=first.rank,
                    block_indices=list(block_indices), scale=first.scale,
                    lora_A=[a.lora_A.detach().cpu() for a in adapters],
                    lora_B=[a.lora_B.detach().cpu() for a in adapters],
                    metadata=metadata), path)


def load_adapter(dit, path, device):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state["format"] == "h3_causal_tail_qkv_v1":
        adapter, index = install_adapter(dit, state["rank"], state["block_index"], device)
        if adapter.scale != state["scale"]:
            raise ValueError("adapter scaling mismatch")
        with torch.no_grad():
            adapter.lora_A.copy_(state["lora_A"])
            adapter.lora_B.copy_(state["lora_B"])
        adapter.requires_grad_(False)
        return dict(block_index=index, block_indices=[index], rank=adapter.rank, metadata=state["metadata"])
    if state["format"] != "h3_causal_tail_qkv_v2":
        raise ValueError("unsupported causal adapter format")
    adapters, indices = install_adapters(dit, state["rank"], state["block_indices"], device)
    if any(a.scale != state["scale"] for a in adapters):
        raise ValueError("adapter scaling mismatch")
    with torch.no_grad():
        for adapter, a, b in zip(adapters, state["lora_A"], state["lora_B"]):
            adapter.lora_A.copy_(a)
            adapter.lora_B.copy_(b)
    for adapter in adapters:
        adapter.requires_grad_(False)
    return dict(block_indices=indices, block_index=indices[-1], rank=adapters[0].rank,
                metadata=state["metadata"])
