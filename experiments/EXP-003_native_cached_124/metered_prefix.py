"""Isolated current-video prefix feedback candidate; not a production default.

Requires own-action visibility and current-action feedback. The only change
inside attention is common-prefix queries reading current video keys. Current
video still reads detached past KV; prefix does not read historical video KV.
No future video/action access is added. Multi-chunk behavior needs evaluation.
"""
from contextlib import contextmanager
import ast
import inspect
import textwrap
import time

import torch
from torch.nn import functional as F
from causal.h3_cached import ChunkAttention, RawEntry
from current_prefix import verify_one_edge_change as verify_accepted_one_edge_change


class KVTransferMeter:
    """Timing-only instrumentation around the unchanged historical K/V reads."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.read_seconds = 0.0
        self.enqueue_seconds = 0.0
        self.bytes = 0
        self.events = []

    def finish(self):
        if self.events:
            torch.cuda.synchronize()
        gpu_seconds = sum(a.elapsed_time(b) for a,b in self.events)/1000
        result = dict(cache_read_cpu_seconds=self.read_seconds,
                      cache_transfer_enqueue_seconds=self.enqueue_seconds,
                      cache_transfer_gpu_seconds=gpu_seconds,
                      cache_transfer_bytes=self.bytes)
        self.reset()
        return result


METER = KVTransferMeter()

def current_prefix_attend(self, q, k, v, *, rope_freqs, layer, apply_rope, scale):
    if torch.is_grad_enabled() and (self.commit or not self.allow_grad_read):
        raise RuntimeError('cached H3 attention is inference-only; train with clean-history loss')
    p = self.prefix
    if self.action_adapter is not None:
        q, k, v = self.action_adapter(
            layer, q, k, v, prefix=p, frame_rows=self.frame_rows,
            action_cond=self.action_cond)
    read_t0 = time.perf_counter()
    history = self.cache.history(layer, self.index)
    METER.read_seconds += time.perf_counter() - read_t0
    transfer_t0 = time.perf_counter()
    start_event = end_event = None
    if history and q.device.type == 'cuda':
        start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start_event.record()
    keys = [e.key.to(k.device) for e in history]
    values = [e.value.to(v.device) for e in history]
    ropes = [e.rope.to(rope_freqs.device) for e in history]
    if start_event is not None:
        end_event.record()
        METER.events.append((start_event,end_event))
    METER.enqueue_seconds += time.perf_counter() - transfer_t0
    METER.bytes += sum(x.numel()*x.element_size() for e in history
                       for x in (e.key,e.value,e.rope))
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
        # Experimental: non-action prefix reads only current video.
        prefix_feedback_mask[ann < 0, current_video_offset:] = True
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


def verify_one_edge_change(original):
    old = ast.parse(textwrap.dedent(inspect.getsource(original))).body[0]
    new = ast.parse(textwrap.dedent(inspect.getsource(current_prefix_attend))).body[0]
    new.name = old.name
    expected = ast.dump(ast.parse('prefix_feedback_mask[ann < 0, current_video_offset:] = True').body[0])
    class RemoveAddedEdge(ast.NodeTransformer):
        removed = 0
        def visit_Assign(self, node):
            if ast.dump(node) == expected:
                self.removed += 1
                return None
            return self.generic_visit(node)
    edit = RemoveAddedEdge(); new = edit.visit(new)
    if edit.removed != 1 or ast.dump(old) != ast.dump(new):
        raise RuntimeError('Copied attention differs beyond the declared prefix edge')


@contextmanager
def current_prefix_feedback():
    original = ChunkAttention.attend
    verify_accepted_one_edge_change(original)
    def checked(self, *args, **kwargs):
        if self.action_prefix_mode != 'own' or not self.action_feedback or self.action_rows is None:
            raise ValueError('Candidate requires own-action rows and action_feedback=True')
        return current_prefix_attend(self, *args, **kwargs)
    ChunkAttention.attend = checked
    try:
        yield
    finally:
        ChunkAttention.attend = original
