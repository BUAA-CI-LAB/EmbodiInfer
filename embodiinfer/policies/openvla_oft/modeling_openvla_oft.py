"""OpenVLA-OFT policy: a single causal forward producing discrete action tokens.

embodiinfer owns the forward. The pieces are loaded from the RLinf ``openvla_oft`` checkpoint:

  * vision: vendored ``PrismaticVisionBackbone`` (DINOv2+SigLIP) + ``PrismaticProjector``
    — a leaf run once in ``encode_prefix``;
  * LLM: the Llama-2 decoder's *modules* (embed / per-layer q,k,v,o + gate,up,down + norms /
    final norm / lm_head) are held by a stock ``LlamaForCausalLM``, but the transformer
    *forward* — RMSNorm, RoPE, attention (via the pluggable ``AttentionBackend``) and SwiGLU
    — is re-run here, exactly as pi0.5 self-hosts its Gemma stack. No model-level black box.

The forward is split into a compute-bound prefill and a small decode so the decode is
CUDA-graph-capturable (the same encode/decode split the flow policies use):

  * ``encode_prefix``  — run ``[BOS, patches, prompt..space]`` through the Llama stack,
    caching per-layer K/V + the final space-position hidden (which predicts action 0 and
    feeds the value head);
  * ``decode_action_logits`` — append the 56 zeroed action-query tokens, decode them
    against the cached prefix K/V, and read out the 56 action-token logits. Causal
    attention makes this bit-identical to one full forward (verified on the box).

Aligned bit-exact against RLinf's ``OpenVLAOFTForRLActionPrediction`` (``docs/proposals/0003``).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

from ...layers.attention import get_attention_backend
from ..base import VLAPolicy
from ..config import VLAPolicyConfig
from ..decoder import ParallelDecoder
from ..factory import register_policy
from .head import CategoricalActionHead

if TYPE_CHECKING:
    from transformers import LlamaForCausalLM

    from .processor_openvla_oft import OpenVLAOFTBatch, OpenVLAOFTProcessor
    from .vision_prismatic import PrismaticProjector, PrismaticVisionBackbone

# Heavy deps (timm / transformers / torchvision / safetensors) are imported inside the
# builder so ``from embodiinfer.policies import ...`` (and the registry trigger) stays light.

SPACE_TOKEN = 29871
STOP_INDEX = 2
PLACEHOLDER_TOKEN = 1


# ---- Llama attention math (embodiinfer-owned forward over the loaded Llama modules) --
def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(q, k, cos, sin):
    cos = cos.unsqueeze(1)  # [B, 1, S, hd] broadcast over heads
    sin = sin.unsqueeze(1)
    return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)


def _llama_rmsnorm(norm, x: torch.Tensor) -> torch.Tensor:
    """HF ``LlamaRMSNorm``: fp32 variance, ``weight * normed`` (no 1+weight; that is Gemma)."""
    input_dtype = x.dtype
    xf = x.float()
    var = xf.pow(2).mean(-1, keepdim=True)
    xf = xf * torch.rsqrt(var + norm.variance_epsilon)
    return norm.weight * xf.to(input_dtype)


def _mlp(mlp, x: torch.Tensor) -> torch.Tensor:
    return mlp.down_proj(mlp.act_fn(mlp.gate_proj(x)) * mlp.up_proj(x))


@dataclass
class OFTPrefix:
    """Cached prefill state consumed by the decode.

    ``kv`` is the per-Llama-layer ``(key, value)`` of ``[BOS, patches, prompt..space]``
    (post-RoPE, ``[B, n_kv, prefix_len, head_dim]``). ``space_hidden`` is the post-norm
    hidden at the final (space) prefill position — it predicts action 0 and feeds the
    value head. ``prefix_pad_mask`` is the ``[B, prefix_len]`` key-padding mask and
    ``decode_start`` the ``[B]`` position id of the first action query.
    """

    kv: list[tuple[torch.Tensor, torch.Tensor]]
    space_hidden: torch.Tensor
    prefix_pad_mask: torch.Tensor
    decode_start: torch.Tensor
    batch_size: int

    def to(self, device: torch.device | str) -> OFTPrefix:
        return OFTPrefix(
            [(k.to(device), v.to(device)) for k, v in self.kv],
            self.space_hidden.to(device),
            self.prefix_pad_mask.to(device),
            self.decode_start.to(device),
            self.batch_size,
        )

    def expand(self, num_samples: int) -> OFTPrefix:
        if num_samples == 1:
            return self
        kv = [
            (k.repeat_interleave(num_samples, dim=0), v.repeat_interleave(num_samples, dim=0))
            for k, v in self.kv
        ]
        return OFTPrefix(
            kv,
            self.space_hidden.repeat_interleave(num_samples, dim=0),
            self.prefix_pad_mask.repeat_interleave(num_samples, dim=0),
            self.decode_start.repeat_interleave(num_samples, dim=0),
            self.batch_size * num_samples,
        )


class OpenVLAOFTPolicy(VLAPolicy):
    def __init__(
        self,
        config: VLAPolicyConfig,
        vision_backbone: PrismaticVisionBackbone,
        projector: PrismaticProjector,
        language_model: LlamaForCausalLM,
        head: CategoricalActionHead,
        processor: OpenVLAOFTProcessor,
        n_action_bins: int,
        vocab_size: int,
        num_action_chunks: int,
        action_dim: int,
        attention: str = "sdpa",
        sample_temperature: float = 1.6,
        sample_top_k: int = -1,
    ):
        super().__init__(config)
        self.vision_backbone = vision_backbone
        self.projector = projector
        self.language_model = language_model
        self._model = language_model.model  # embed_tokens, layers, norm, rotary_emb
        self._lm_head = language_model.lm_head
        self.head = head
        self._processor = processor
        self.attention = attention
        self._attn = get_attention_backend(attention)
        self.n_action_bins = n_action_bins
        self.vocab_size = vocab_size  # detok vocab (= text vocab - pad_to_multiple_of)
        self.num_action_chunks = num_action_chunks
        self.action_dim = action_dim
        self.n_tokens = num_action_chunks * action_dim
        self.sample_temperature = sample_temperature
        self.sample_top_k = sample_top_k
        self._hidden_size = language_model.config.hidden_size
        self._prefix_len: int | None = None  # set by encode_prefix; sizes the static graph buffer
        self._decoder = ParallelDecoder(self)

    @property
    def decoder(self) -> ParallelDecoder:
        return self._decoder

    def collate(self, observations, request_ids) -> OpenVLAOFTBatch:
        return self._processor.collate(observations, request_ids)

    def pad(self, batch: OpenVLAOFTBatch, target_batch_size: int) -> OpenVLAOFTBatch:
        return batch.pad(target_batch_size)

    # ---- the self-hosted Llama forward (prefill collects K/V; decode consumes it) ----
    def _attn_sublayer(self, attn, h, cos, sin, mask, prefix_kv=None, collected=None):
        B, S = h.shape[0], h.shape[1]
        hd = attn.head_dim
        q = attn.q_proj(h).view(B, S, -1, hd).transpose(1, 2)  # [B, n_head, S, hd]
        k = attn.k_proj(h).view(B, S, -1, hd).transpose(1, 2)  # [B, n_kv, S, hd]
        v = attn.v_proj(h).view(B, S, -1, hd).transpose(1, 2)
        q, k = _apply_rope(q, k, cos, sin)
        if collected is not None:  # prefill pass: cache this layer's K/V
            collected.append((k, v))
        if prefix_kv is not None:  # decode pass: attend [prefix ++ this step]
            pk, pv = prefix_kv
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)
        out = self._attn.attend(q, k, v, attn_mask=mask, scaling=attn.scaling)
        return attn.o_proj(out.transpose(1, 2).reshape(B, S, -1))

    def _llama_forward(self, inputs_embeds, attn_mask, position_ids, prefix_kv=None, collect=False):
        """Run the Llama-2 decoder stack; returns (post-norm hidden, collected per-layer K/V)."""
        model = self._model
        hidden = inputs_embeds.to(model.layers[0].self_attn.q_proj.weight.dtype)
        cos, sin = model.rotary_emb(hidden, position_ids)
        collected: list | None = [] if collect else None
        for i, layer in enumerate(model.layers):
            residual = hidden
            h = _llama_rmsnorm(layer.input_layernorm, hidden)
            pk = None if prefix_kv is None else prefix_kv[i]
            hidden = residual + self._attn_sublayer(layer.self_attn, h, cos, sin, attn_mask, pk, collected)
            residual = hidden
            h = _llama_rmsnorm(layer.post_attention_layernorm, hidden)
            hidden = residual + _mlp(layer.mlp, h)
        return _llama_rmsnorm(model.norm, hidden), collected

    def _causal_mask(self, attention_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """Additive causal + left-padding mask ``[B, 1, S, S]`` (0 attend, min masked)."""
        B, S = attention_mask.shape
        min_val = torch.finfo(dtype).min
        causal = torch.triu(
            torch.full((S, S), min_val, dtype=dtype, device=attention_mask.device), diagonal=1
        )
        mask = causal[None, None].expand(B, 1, S, S).clone()
        pad = (1 - attention_mask[:, None, None, :].to(dtype)) * min_val
        return mask + pad

    def _decode_mask(self, prefix_pad_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """Additive mask ``[B, 1, n_tokens, prefix_len + n_tokens]``: action queries attend all
        non-pad prefix keys + causally among themselves."""
        B, pl = prefix_pad_mask.shape
        nt = self.n_tokens
        min_val = torch.finfo(dtype).min
        prefix_part = (1 - prefix_pad_mask[:, None, None, :].to(dtype)) * min_val  # [B,1,1,pl]
        prefix_part = prefix_part.expand(B, 1, nt, pl)
        causal = torch.triu(
            torch.full((nt, nt), min_val, dtype=dtype, device=prefix_pad_mask.device), diagonal=1
        )
        action_part = causal[None, None].expand(B, 1, nt, nt)
        return torch.cat([prefix_part, action_part], dim=-1)

    @torch.no_grad()
    def encode_prefix(self, batch: OpenVLAOFTBatch) -> OFTPrefix:
        input_ids = batch.input_ids  # [B, L], ends in the space token
        attention_mask = batch.attention_mask
        pixel_values = batch.pixel_values
        B = input_ids.shape[0]

        embed = self._model.embed_tokens(input_ids)
        patches = self.projector(self.vision_backbone(pixel_values))  # [B, 256*n_img, D]
        mm_embed = torch.cat([embed[:, :1], patches, embed[:, 1:]], dim=1)  # [BOS] ‖ patches ‖ text
        patch_mask = torch.ones(patches.shape[:2], dtype=attention_mask.dtype, device=attention_mask.device)
        mm_mask = torch.cat([attention_mask[:, :1], patch_mask, attention_mask[:, 1:]], dim=1)
        position_ids = mm_mask.long().cumsum(dim=1) - 1

        attn_add = self._causal_mask(mm_mask, mm_embed.dtype)
        hidden, kv = self._llama_forward(mm_embed, attn_add, position_ids, collect=True)
        self._prefix_len = mm_mask.shape[1]
        space_hidden = hidden[:, -1]  # the space position predicts action 0 + feeds the value head
        decode_start = position_ids[:, -1] + 1  # [B] first action-query position id
        return OFTPrefix(kv, space_hidden, mm_mask, decode_start, B)

    def _decode(self, prefix: OFTPrefix) -> torch.Tensor:
        """Decode the 56 action-query tokens against the cached prefix -> action logits."""
        B = prefix.batch_size
        dtype = prefix.space_hidden.dtype
        device = prefix.space_hidden.device
        nt = self.n_tokens
        query = torch.zeros(B, nt, self._hidden_size, device=device, dtype=dtype)  # zeroed action queries
        pos = prefix.decode_start[:, None] + torch.arange(nt, device=device)  # [B, nt]
        mask = self._decode_mask(prefix.prefix_pad_mask, dtype)
        hidden, _ = self._llama_forward(query, mask, pos, prefix_kv=prefix.kv)  # [B, nt, D]
        # causal read: action0 from the (prefill) space hidden, action_{i+1} from query i.
        action_hidden = torch.cat([prefix.space_hidden[:, None], hidden[:, : nt - 1]], dim=1)
        return self._lm_head(action_hidden)  # [B, nt, vocab(+pad)]

    def decode_action_logits(self, prefix: OFTPrefix, graphs=None, bucket=None) -> torch.Tensor:
        if graphs is not None:
            g = graphs.get(bucket, 1)
            g.set_prefix(prefix)
            return g.run()
        return self._decode(prefix)

    @torch.no_grad()
    def action_logits_full(self, batch: OpenVLAOFTBatch) -> tuple[torch.Tensor, torch.Tensor]:
        """Single full forward (no prefill/decode split) -> (action_logits ``[B, n_tokens,
        vocab]``, space_hidden ``[B, D]``).

        Assembles the complete ``[BOS ‖ patches ‖ prompt ‖ n_tokens action]`` sequence
        exactly as the native OFT forward (``_prepare_input_for_action_prediction`` appends
        the action placeholders + STOP; ``_build_embedding`` drops the trailing STOP, zeros
        the action embeds, and ``_build_multimodal_attention`` inserts patches after <BOS>),
        runs the Llama stack once, and reads the logits at the space+action positions.
        Bit-exact against the native generator's full forward, so a PPO ratio at
        ``theta == theta_behavior`` is exactly 1. The ``encode_prefix``/``_decode`` split
        (above) trades this exactness for a CUDA-graph-able decode — kept as the fast path;
        this full path is what the RL rollout uses when bit-exact behaviour is required."""
        input_ids = batch.input_ids  # [B, L], ends in the space token
        attention_mask = batch.attention_mask
        pixel_values = batch.pixel_values
        B = input_ids.shape[0]
        nt = self.n_tokens
        device = input_ids.device

        # append nt action placeholders (STOP is appended-then-dropped by native, so omit)
        ph = torch.full((B, nt), PLACEHOLDER_TOKEN, dtype=input_ids.dtype, device=device)
        ids = torch.cat([input_ids, ph], dim=1)
        amask = torch.cat(
            [attention_mask, torch.ones(B, nt, dtype=attention_mask.dtype, device=device)], dim=1
        )
        embed = self._model.embed_tokens(ids)
        action_positions = torch.zeros(B, ids.shape[1], dtype=torch.bool, device=device)
        action_positions[:, -nt:] = True
        embed = embed * (~action_positions.unsqueeze(-1))  # action queries carry no content
        patches = self.projector(self.vision_backbone(pixel_values))
        mm_embed = torch.cat([embed[:, :1], patches, embed[:, 1:]], dim=1)  # [BOS ‖ patches ‖ rest]
        patch_mask = torch.ones(patches.shape[:2], dtype=amask.dtype, device=device)
        mm_mask = torch.cat([amask[:, :1], patch_mask, amask[:, 1:]], dim=1)
        position_ids = mm_mask.long().cumsum(dim=1) - 1
        attn_add = self._causal_mask(mm_mask, mm_embed.dtype)
        hidden, _ = self._llama_forward(mm_embed, attn_add, position_ids, collect=False)
        logits = self._lm_head(hidden)  # [B, seq, vocab(+pad)]
        # causal read: [space, action0..action_{nt-2}] positions predict [action0..action_{nt-1}]
        return logits[:, -nt - 1 : -1], hidden[:, -nt - 1]

    # ---- CUDA-graph capability (static-shape decode) ------------------------
    @property
    def supports_cuda_graph(self) -> bool:
        return True

    @property
    def cuda_graph_kind(self) -> str:
        return "single_forward"  # one fixed-shape decode over the cached prefix (no denoise loop)

    def allocate_static_prefix(self, batch_size, device, dtype) -> OFTPrefix:
        pl = self._prefix_len
        if pl is None:
            raise RuntimeError("call encode_prefix once before allocate_static_prefix (sets prefix_len)")
        lc = self.language_model.config
        n_kv = lc.num_key_value_heads
        hd = getattr(lc, "head_dim", None) or (lc.hidden_size // lc.num_attention_heads)
        kv = [
            (
                torch.zeros(batch_size, n_kv, pl, hd, device=device, dtype=dtype),
                torch.zeros(batch_size, n_kv, pl, hd, device=device, dtype=dtype),
            )
            for _ in range(len(self._model.layers))
        ]
        return OFTPrefix(
            kv,
            torch.zeros(batch_size, self._hidden_size, device=device, dtype=dtype),
            torch.ones(batch_size, pl, device=device, dtype=torch.long),
            torch.zeros(batch_size, device=device, dtype=torch.long),
            batch_size,
        )

    def copy_prefix_into(self, dst: OFTPrefix, src: OFTPrefix) -> None:
        for (dk, dv), (sk, sv) in zip(dst.kv, src.kv):
            dk.copy_(sk)
            dv.copy_(sv)
        dst.space_hidden.copy_(src.space_hidden)
        dst.prefix_pad_mask.copy_(src.prefix_pad_mask)
        dst.decode_start.copy_(src.decode_start)


# ---- checkpoint loading + builder ------------------------------------------
def _load_openvla_oft(policy: OpenVLAOFTPolicy, checkpoint: str) -> None:
    """Route the checkpoint's ``vision_backbone.* / projector.* / language_model.*`` weights
    into embodiinfer's modules (value_head.* is ignored until the RL value path lands)."""
    from safetensors.torch import load_file

    with open(os.path.join(checkpoint, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]
    shards = sorted(set(weight_map.values()))
    sd: dict[str, torch.Tensor] = {}
    for shard in shards:
        sd.update(load_file(os.path.join(checkpoint, shard)))

    vb, proj, lm = {}, {}, {}
    for k, v in sd.items():
        if k.startswith("vision_backbone."):
            vb[k[len("vision_backbone.") :]] = v
        elif k.startswith("projector."):
            proj[k[len("projector.") :]] = v
        elif k.startswith("language_model."):
            lm[k[len("language_model.") :]] = v  # -> LlamaForCausalLM (model.* + lm_head.weight)

    miss_vb = policy.vision_backbone.load_state_dict(vb, strict=False)
    miss_proj = policy.projector.load_state_dict(proj, strict=False)
    miss_lm = policy.language_model.load_state_dict(lm, strict=False)
    for name, m in [("vision", miss_vb), ("projector", miss_proj), ("llama", miss_lm)]:
        if m.missing_keys:
            raise RuntimeError(f"OpenVLA-OFT load: {name} missing keys {m.missing_keys[:6]} ...")


@register_policy("openvla_oft")
def _build_openvla_oft(checkpoint: str | None = None, attention: str = "sdpa", **overrides) -> VLAPolicy:
    if checkpoint is None:
        raise ValueError("openvla_oft needs checkpoint=<RLinf OpenVLAOFT ckpt dir>")

    from transformers import LlamaConfig, LlamaForCausalLM

    from .processor_openvla_oft import OpenVLAOFTProcessor
    from .vision_prismatic import PrismaticProjector, PrismaticVisionBackbone

    with open(os.path.join(checkpoint, "config.json")) as f:
        cfg = json.load(f)
    tcfg = cfg["text_config"]
    pad_mult = cfg.get("pad_to_multiple_of", 64)
    n_action_bins = cfg.get("n_action_bins", 256)
    num_images = cfg.get("num_images_in_input", 1)
    llm_dim = tcfg.get("hidden_size") or 4096
    detok_vocab = tcfg["vocab_size"] - pad_mult

    action_dim, num_chunks = 7, 8  # LIBERO OFT (7 dims x chunk 8)

    vb = PrismaticVisionBackbone(
        use_fused_vision_backbone=cfg.get("use_fused_vision_backbone", True),
        image_sizes=cfg.get("image_sizes", [224, 224]),
        timm_model_ids=cfg["timm_model_ids"],
        timm_override_act_layers=cfg.get("timm_override_act_layers", [None, None]),
    )
    vb.set_num_images_in_input(num_images)
    projector = PrismaticProjector(True, vision_dim=vb.embed_dim, llm_dim=llm_dim)
    language_model = LlamaForCausalLM(LlamaConfig(**tcfg))

    with open(os.path.join(checkpoint, "dataset_statistics.json")) as f:
        ds = json.load(f)
    unnorm_key = cfg.get("unnorm_key") or next(iter(ds))
    if unnorm_key not in ds:
        unnorm_key = next(iter(ds))
    stats = ds[unnorm_key]["action"]
    mask = stats.get("mask", [True] * (action_dim - 1) + [False])
    head = CategoricalActionHead(
        detok_vocab,
        n_action_bins,
        action_dim,
        num_chunks,
        q01=np.asarray(stats["q01"]),
        q99=np.asarray(stats["q99"]),
        mask=np.asarray(mask),
    )

    processor = OpenVLAOFTProcessor.from_checkpoint(checkpoint)
    vla_cfg = VLAPolicyConfig(
        name="openvla_oft", action_dim=action_dim, action_horizon=num_chunks, default_num_steps=1, **overrides
    )
    policy = OpenVLAOFTPolicy(
        vla_cfg,
        vb,
        projector,
        language_model,
        head,
        processor,
        n_action_bins=n_action_bins,
        vocab_size=detok_vocab,
        num_action_chunks=num_chunks,
        action_dim=action_dim,
        attention=attention,
        sample_temperature=cfg.get("temperature", 1.6),
        sample_top_k=cfg.get("top_k", -1),
    )
    _load_openvla_oft(policy, checkpoint)
    return policy
