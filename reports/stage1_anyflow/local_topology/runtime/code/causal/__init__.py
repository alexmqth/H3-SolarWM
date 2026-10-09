"""Small causal-chunk building blocks used by the H3-World prototype.

The package deliberately does not import the 33B H3 model.  It gives the
experiment a CPU-testable contract for chunk visibility, raw-KV cache
lifetime, and teacher-forced flow-map training before the same mask is wired
into the patched H3 DiT.
"""

from .prototype import (
    CausalChunkLayout,
    LayeredRawKVCache,
    TinyCausalFlowMap,
    build_causal_visibility,
    scaled_dot_product_chunk_attention,
    teacher_forcing_flow_loss,
)
from .h3_cached import last_frame_anchor

__all__ = [
    "CausalChunkLayout",
    "LayeredRawKVCache",
    "TinyCausalFlowMap",
    "build_causal_visibility",
    "scaled_dot_product_chunk_attention",
    "teacher_forcing_flow_loss",
    "last_frame_anchor",
]
