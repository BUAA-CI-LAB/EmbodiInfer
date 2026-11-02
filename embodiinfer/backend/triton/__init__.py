"""EmbodiInfer-owned Triton kernels."""

from embodiinfer.backend.triton.activation import gated_gelu, supports_swiglu, swiglu
from embodiinfer.backend.triton.attention import gqa_decode_graph, graph_gqa_available
from embodiinfer.backend.triton.capability import (
    TritonCapability,
    require_triton,
    triton_capability,
)
from embodiinfer.backend.triton.norm import (
    ada_rms_norm,
    add_rms_norm,
    gated_residual,
    rms_norm,
    supports_rms_norm,
)
from embodiinfer.backend.triton.packed_rope import (
    rotate_half_rope,
    warmup_rotate_half_rope,
)
from embodiinfer.backend.triton.rotary import (
    fused_rope_cache,
    fused_rope_cache_graph,
    rotate_qk,
    supports_fused_rope_cache,
)
from embodiinfer.backend.triton.sampling import (
    GreedyWorkspace,
    fused_greedy,
    fused_lm_head_greedy,
    mark_penalized,
    supports_fused_greedy,
    supports_fused_lm_head,
)
from embodiinfer.backend.triton.segmented_attention import (
    SegmentedTritonAttention,
    segmented_attention,
    warmup_segmented_attention,
)

__all__ = [
    "GreedyWorkspace",
    "SegmentedTritonAttention",
    "TritonCapability",
    "ada_rms_norm",
    "add_rms_norm",
    "fused_greedy",
    "fused_lm_head_greedy",
    "fused_rope_cache",
    "fused_rope_cache_graph",
    "gated_gelu",
    "gated_residual",
    "gqa_decode_graph",
    "graph_gqa_available",
    "mark_penalized",
    "require_triton",
    "rms_norm",
    "rotate_half_rope",
    "rotate_qk",
    "segmented_attention",
    "supports_fused_greedy",
    "supports_fused_lm_head",
    "supports_fused_rope_cache",
    "supports_rms_norm",
    "supports_swiglu",
    "swiglu",
    "triton_capability",
    "warmup_rotate_half_rope",
    "warmup_segmented_attention",
]
