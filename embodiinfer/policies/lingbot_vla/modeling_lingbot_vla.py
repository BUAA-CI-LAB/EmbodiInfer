"""LingBot-VLA adapter: Robbyant ``lingbot-vla-4b`` weights, **embodiinfer-owned** forward.

LingBot-VLA (arXiv 2601.18692) is a pi0-flavored flow-matching VLA: a
Qwen2.5-VL-3B VLM backbone plus a narrow Qwen2 action expert, coupled by a
Mixture-of-Transformers (MoT) *shared attention* per layer. Structurally it is
the same two-stage shape the engine schedules and the same decomposition pi0.5
uses:

  * ``encode_prefix`` runs the VL stream once over ``[image patches, language]``
    (bidirectional prefix) and caches its per-layer post-RoPE K/V;
  * ``denoise_step`` runs the action-expert stream over ``[state, action]`` tokens
    attending ``[cached VL K/V ++ suffix K/V]``, N times.

The **whole transformer forward is self-hosted** (no
model-level black box, unlike GR00T's backbone): the loaded ``lingbotvla`` modules
are weight holders; the decoder stacks (RMSNorm/AdaRMSNorm, 1-D RoPE, attention via
embodiinfer's ``AttentionBackend``, SwiGLU) are re-run here, so the KV cache is ours and the
denoise loop is CUDA-graph-capturable. The vision ViT (Qwen2.5-VL ``visual``) stays a
leaf (runs once in ``encode_prefix``), as with pi0.5's SigLIP and OpenVLA-OFT's Prismatic.

Blueprint: source-verified against the upstream implementation on 2026-07-14.
NOTE: exact module attribute paths + the fp32-attention-core / mask-fill / AdaRMSNorm
numerics are pinned to a box parity pass against RLinf ``LingbotvlaActionModel`` (see
``docs/proposals/0005``); marked ``# box-parity`` below.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ...layers import get_attention_backend
from ...types import Observation
from ..base import FlowVLAPolicy, VLAPolicy
from ..config import VLAPolicyConfig
from ..factory import register_policy

# native promotes the attention core to fp32 and fills masked logits with this exact
# value (not -inf); reproduced for bit-exactness (blueprint §F.2, U:158).
_MASK_FILL = -2.3819763e38
_ROPE_THETA = 10000.0  # 1-D RoPE, NOT Qwen mRoPE / theta=1e6 (blueprint §F.1)


# ---- attention math (1-D RoPE theta=10000, fp32 core) -----------------------
def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _rope_cos_sin(
    position_ids: torch.Tensor, head_dim: int, device, dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """pi0-style 1-D rotary tables in fp32 (``max_wavelength=10000``, split-half layout).

    Returns ``(cos, sin)`` of shape ``[B, seq, head_dim]``. Built in fp32 (the native
    ``apply_rope`` computes rope in fp32 regardless of the model dtype)."""
    half = head_dim // 2
    fraction = torch.arange(half, device=device, dtype=torch.float32) / half
    timescale = _ROPE_THETA**fraction  # [half]
    pos = position_ids.float()[..., None]  # [B, seq, 1]
    sinusoid = pos / timescale  # [B, seq, half]
    sin = torch.sin(sinusoid)
    cos = torch.cos(sinusoid)
    # split-half layout: duplicate across the two halves
    cos = torch.cat([cos, cos], dim=-1)
    sin = torch.cat([sin, sin], dim=-1)
    return cos, sin


def _apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    # q,k: [B, heads, seq, hd]; cos,sin: [B, seq, hd] -> broadcast over heads
    cos = cos[:, None]
    sin = sin[:, None]
    return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)


def _rmsnorm(weight: torch.Tensor, x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Plain Qwen2 RMSNorm (``weight * x_normed``, fp32 variance) — VL stream + final norms."""
    input_dtype = x.dtype
    xf = x.float()
    var = xf.pow(2).mean(-1, keepdim=True)
    xf = xf * torch.rsqrt(var + eps)
    return (weight.float() * xf).to(input_dtype)


def _ada_rmsnorm(norm, x: torch.Tensor, cond: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Expert AdaRMSNorm: ``x_hat = weight * RMSNorm(x)``; ``(1 + gamma(cond)) * x_hat + beta(cond)``.

    ``gamma``/``beta`` are ``Linear(768->768)`` on the raw sinusoidal time embedding; the
    modulation is computed and applied in fp32 (blueprint §C/§F.3)."""
    input_dtype = x.dtype
    xf = x.float()
    var = xf.pow(2).mean(-1, keepdim=True)
    x_hat = xf * torch.rsqrt(var + eps) * norm.weight.float()
    gamma = norm.gamma(cond).float()
    beta = norm.beta(cond).float()
    if x.ndim == 3 and gamma.ndim == 2:
        gamma = gamma[:, None]
        beta = beta[:, None]
    return ((1.0 + gamma) * x_hat + beta).to(input_dtype)


def _mlp(mlp, x: torch.Tensor) -> torch.Tensor:
    return mlp.down_proj(F.silu(mlp.gate_proj(x)) * mlp.up_proj(x))


@dataclass
class LingBotPrefix:
    """LingBot-VLA prefix state: our per-layer VL (K, V) cache + pad mask + state token.

    ``kv[i]`` = the VL stream's ``(key, value)`` at layer ``i``, ``[B, num_kv, prefix_len,
    head_dim]`` (post-RoPE). ``state_emb`` is the ``state_proj``-embedded proprio token
    ``[B, 1, 768]`` (constant across denoise steps, so precomputed here). Read-only across
    steps; the expert concatenates ``kv`` with its own suffix K/V each step.
    """

    kv: list[tuple[torch.Tensor, torch.Tensor]]
    prefix_pad_masks: torch.Tensor  # [B, prefix_len]
    state_emb: torch.Tensor  # [B, 1, 768]
    last_hidden: torch.Tensor | None = None  # VL final hidden, for a value head

    @property
    def batch_size(self) -> int:
        return self.prefix_pad_masks.shape[0]

    def to(self, device: torch.device | str) -> LingBotPrefix:
        return LingBotPrefix(
            [(k.to(device), v.to(device)) for k, v in self.kv],
            self.prefix_pad_masks.to(device),
            self.state_emb.to(device),
            self.last_hidden.to(device) if self.last_hidden is not None else None,
        )

    def expand(self, num_samples: int) -> LingBotPrefix:
        if num_samples == 1:
            return self
        kv = [
            (k.repeat_interleave(num_samples, dim=0), v.repeat_interleave(num_samples, dim=0))
            for k, v in self.kv
        ]
        return LingBotPrefix(
            kv,
            self.prefix_pad_masks.repeat_interleave(num_samples, dim=0),
            self.state_emb.repeat_interleave(num_samples, dim=0),
        )


class LingBotVLAPolicy(FlowVLAPolicy):
    """First-class LingBot-VLA policy: lingbotvla weights, embodiinfer-owned MoT forward."""

    def __init__(
        self,
        config: VLAPolicyConfig,
        checkpoint,
        attention: str = "eager",
        backbone_path: str | None = None,
    ):
        super().__init__(config)
        self.attention = attention
        self._attn = get_attention_backend(attention)
        # Option B (package-free, runs in embodiinfer_env): stock Qwen2.5-VL VL
        # backbone (weight holder) + vendored action expert, weights loaded directly from the
        # checkpoint. No lingbotvla runtime dependency (its LeRobot-v3/torch-2.8 stack conflicts
        # with embodiinfer_env) — the GR00T pattern. ``checkpoint`` is a dir path (or a pre-built tuple
        # for tests). box-parity: Qwen2.5-VL class/config + ViT interface.
        vl, visual, expert, proj = (
            checkpoint if isinstance(checkpoint, tuple) else _build_and_load(checkpoint, backbone_path)
        )
        self._vl = vl  # Qwen2.5-VL text model: embed_tokens, layers (plain RMSNorm), norm
        self._visual = visual  # Qwen2.5-VL ViT leaf
        self._expert = expert  # vendored ActionExpert: layers (AdaRMSNorm) + plain final norm
        self._proj = proj  # ActionProjections: state / action / action_time heads
        self._chunk = config.action_horizon
        self._processor = None  # Qwen2.5-VL processor for collate; set by the builder
        self._backbone_path = backbone_path or "Qwen/Qwen2.5-VL-3B-Instruct"
        self._cached_prefix_meta: tuple[int, torch.dtype] | None = None

    # ---- one attention sub-layer (shared math; fp32 core) -------------------
    def _attn_sublayer(self, attn, h, cos, sin, mask, prefix_kv, collected):
        B, S = h.shape[0], h.shape[1]
        hd = _head_dim(attn)
        q = attn.q_proj(h).view(B, S, -1, hd).transpose(1, 2)  # [B, 16, S, hd]
        k = attn.k_proj(h).view(B, S, -1, hd).transpose(1, 2)  # [B, 2, S, hd]
        v = attn.v_proj(h).view(B, S, -1, hd).transpose(1, 2)
        # native promotes the attention core to fp32 (blueprint §F.2)
        q, k, v = q.float(), k.float(), v.float()
        q, k = _apply_rope(q, k, cos, sin)
        if collected is not None:  # prefill: cache post-RoPE K/V
            collected.append((k, v))
        if prefix_kv is not None:  # decode: attend [prefix ++ suffix]
            pk, pv = prefix_kv
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)
        out = self._attn.attend(q, k, v, attn_mask=mask, scaling=hd**-0.5)
        out = out.transpose(1, 2).reshape(B, S, -1).to(h.dtype)
        return attn.o_proj(out)

    # ---- VL stream (prefill, collect K/V; plain RMSNorm) --------------------
    def _vl_forward(self, hidden, position_ids, mask):
        cos, sin = _rope_cos_sin(
            position_ids, _head_dim(self._vl.layers[0].self_attn), hidden.device, hidden.dtype
        )
        collected: list = []
        for layer in self._vl.layers:
            residual = hidden
            h = _rmsnorm(layer.input_layernorm.weight, hidden)
            hidden = residual + self._attn_sublayer(layer.self_attn, h, cos, sin, mask, None, collected)
            residual = hidden
            h = _rmsnorm(layer.post_attention_layernorm.weight, hidden)
            hidden = residual + _mlp(layer.mlp, h)
        hidden = _rmsnorm(self._vl.norm.weight, hidden)
        return hidden, collected

    # ---- action-expert stream (decode, attend prefix K/V; AdaRMSNorm) -------
    def _expert_forward(self, hidden, position_ids, mask, prefix_kv, ada_cond):
        cos, sin = _rope_cos_sin(
            position_ids, _head_dim(self._expert.layers[0].self_attn), hidden.device, hidden.dtype
        )
        for i, layer in enumerate(self._expert.layers):
            residual = hidden
            h = _ada_rmsnorm(layer.input_layernorm, hidden, ada_cond)
            hidden = residual + self._attn_sublayer(layer.self_attn, h, cos, sin, mask, prefix_kv[i], None)
            residual = hidden
            h = _ada_rmsnorm(layer.post_attention_layernorm, hidden, ada_cond)
            hidden = residual + _mlp(layer.mlp, h)
        hidden = _rmsnorm(self._expert.norm.weight, hidden)  # final_norm_adanorm=False -> plain
        return hidden

    # ---- LingBot integrates t: 1 -> 0 with dt = -1/N (like pi0.5) -----------
    def flow_schedule(self, num_steps: int) -> list[tuple[float, float]]:
        dt = -1.0 / num_steps
        return [(1.0 + i * dt, dt) for i in range(num_steps)]

    # ---- stage 1: encode prefix (once) --------------------------------------
    def _embed_image(
        self, pixel_values: torch.Tensor, grid_thw: torch.Tensor, batch_size: int
    ) -> torch.Tensor:
        """Run the Qwen2.5-VL ViT (leaf) -> per-observation image tokens ``[B, n_cam*tok, 2048]``.

        Qwen packs every image's merged tokens along dim 0, so ``grid_thw`` has one row per
        *image* (``B * num_cam`` rows). We reshape by the real observation count ``batch_size``
        so a multi-camera observation's tokens land on one batch row."""
        out = self._visual(pixel_values, grid_thw=grid_thw)
        # tf5.x Qwen2.5-VL visual returns ``pooler_output`` = the post-merger merged image
        # tokens ``[sum_merged, 2048]`` (``last_hidden_state`` is the pre-merger 1280-dim patches).
        feats = getattr(out, "pooler_output", None)
        if feats is None:
            feats = out[0] if isinstance(out, (tuple, list)) else out
        return feats.reshape(batch_size, -1, feats.shape[-1])

    @torch.no_grad()
    def encode_prefix(self, batch, return_hidden: bool = False) -> LingBotPrefix:
        # VL embeddings: ViT image features (leaf) ++ language token embeddings
        img_emb = self._embed_image(batch.pixel_values, batch.image_grid_thw, batch.lang_tokens.shape[0])
        lang_emb = self._vl.embed_tokens(batch.lang_tokens)  # [B, 72, 2048]
        embs = torch.cat([img_emb, lang_emb], dim=1)
        pad_masks = torch.cat([batch.img_pad_masks, batch.lang_pad_masks], dim=1)  # [B, prefix_len]
        # prefix is bidirectional (att_masks all-zero): attend all non-pad prefix keys
        mask = _prefix_mask(pad_masks, embs.dtype)
        position_ids = torch.cumsum(pad_masks.long(), dim=1) - 1
        hidden, kv = self._vl_forward(embs, position_ids, mask)
        state_emb = self._proj.state_proj(batch.state.to(embs.dtype))[:, None]  # [B, 1, 768]
        self._cached_prefix_meta = (pad_masks.shape[1], pad_masks.dtype)
        return LingBotPrefix(kv, pad_masks, state_emb, hidden if return_hidden else None)

    # ---- suffix embedding (state token + action-time tokens) ----------------
    def _embed_suffix(self, x_t: torch.Tensor, t: torch.Tensor, prefix: LingBotPrefix):
        p = self._proj
        B = x_t.shape[0]
        # sinusoidal time embedding; the raw one is the AdaRMSNorm conditioning
        time_emb = _sinusoidal_time(t, 768, min_period=4e-3, max_period=4.0).to(x_t.dtype)  # [B, 768]
        action_emb = p.action_in_proj(x_t)  # [B, 50, 768]
        # separate_time_proj=False: fuse action + broadcast time via action_time_mlp
        te = time_emb[:, None].expand(B, self._chunk, 768)
        h = p.action_time_mlp_in(torch.cat([action_emb, te], dim=-1))
        h = F.silu(h)
        action_time_emb = p.action_time_mlp_out(h)  # [B, 50, 768]
        embs = torch.cat([prefix.state_emb, action_time_emb], dim=1)  # [B, 51, 768]
        return embs, time_emb  # time_emb is ada_cond

    # ---- stage 2: one denoising step (N times) ------------------------------
    def denoise_step(self, x_t: torch.Tensor, t: torch.Tensor, prefix: LingBotPrefix) -> torch.Tensor:
        suffix_embs, ada_cond = self._embed_suffix(x_t, t, prefix)
        B, suffix_len = suffix_embs.shape[:2]
        prefix_pad = prefix.prefix_pad_masks
        # suffix attends [prefix (state blind to actions, actions bidirectional)]; see blueprint §B mask
        mask = _decode_mask(prefix_pad, suffix_len, suffix_embs.dtype)
        prefix_offset = prefix_pad.long().sum(dim=-1, keepdim=True)
        suffix_pos = torch.arange(suffix_len, device=x_t.device)[None, :]
        position_ids = prefix_offset + suffix_pos  # suffix positions continue after prefix
        hidden = self._expert_forward(suffix_embs, position_ids, mask, prefix.kv, ada_cond)
        v_out = hidden[:, -self._chunk :]  # last 50 positions = action tokens
        return self._proj.action_out_proj(v_out)  # [B, 50, 75]

    # ---- CUDA-graph capability: static-shape denoise loop -------------------
    @property
    def supports_cuda_graph(self) -> bool:
        return True

    def allocate_static_prefix(self, batch_size, device, dtype) -> LingBotPrefix:
        if self._cached_prefix_meta is None:
            raise RuntimeError("allocate_static_prefix needs a prior encode_prefix (sets prefix_len)")
        prefix_len, pad_dtype = self._cached_prefix_meta
        attn0 = self._vl.layers[0].self_attn
        hd = _head_dim(attn0)
        num_kv = attn0.k_proj.out_features // hd
        kv = [
            (
                torch.zeros(batch_size, num_kv, prefix_len, hd, device=device, dtype=torch.float32),
                torch.zeros(batch_size, num_kv, prefix_len, hd, device=device, dtype=torch.float32),
            )
            for _ in range(len(self._vl.layers))
        ]
        pad = torch.ones(batch_size, prefix_len, device=device, dtype=pad_dtype)
        state = torch.zeros(batch_size, 1, 768, device=device, dtype=dtype)
        return LingBotPrefix(kv, pad, state)

    def copy_prefix_into(self, dst: LingBotPrefix, src: LingBotPrefix) -> None:
        for (dk, dv), (sk, sv) in zip(dst.kv, src.kv):
            dk.copy_(sk)
            dv.copy_(sv)
        dst.prefix_pad_masks.copy_(src.prefix_pad_masks)
        dst.state_emb.copy_(src.state_emb)

    # ---- batch construction -------------------------------------------------
    def collate(self, observations: list[Observation], request_ids: list[str]):
        from .processor_lingbot_vla import LingBotVLABatch

        if self._processor is None:
            from transformers import AutoProcessor

            self._processor = AutoProcessor.from_pretrained(self._backbone_path)
        return LingBotVLABatch.from_observations(
            observations, request_ids, self._processor, self.config.action_dim
        )

    def pad(self, batch, target_batch_size: int):
        return batch.pad(target_batch_size)


# ---- small helpers ----------------------------------------------------------
def _head_dim(attn) -> int:
    return getattr(attn, "head_dim", None) or 128


def _sinusoidal_time(t: torch.Tensor, dim: int, min_period: float, max_period: float) -> torch.Tensor:
    """``create_sinusoidal_pos_embedding``: period = min*(max/min)^linspace(0,1,dim/2);
    emb = cat([sin(2*pi/period * t), cos(...)]) (blueprint §C, U:33)."""
    half = dim // 2
    frac = torch.linspace(0, 1, half, device=t.device, dtype=torch.float32)
    period = min_period * (max_period / min_period) ** frac
    ang = (2 * torch.pi / period)[None, :] * t.float()[:, None]  # [B, half]
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)  # [B, dim]


def _prefix_mask(pad_masks: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Additive bidirectional prefix mask ``[B, 1, L, L]``: attend all non-pad keys."""
    B, L = pad_masks.shape
    key_ok = pad_masks[:, None, None, :].to(torch.bool).expand(B, 1, L, L)
    return torch.where(key_ok, 0.0, _MASK_FILL).to(dtype)


def _decode_mask(prefix_pad: torch.Tensor, suffix_len: int, dtype: torch.dtype) -> torch.Tensor:
    """Additive decode mask ``[B, 1, suffix_len, prefix_len + suffix_len]``.

    Suffix layout = ``[state, action*50]``. state (idx 0) attends prefix + itself;
    actions attend prefix + state + all actions (bidirectional). Prefix keys gated by
    ``prefix_pad`` (blueprint §B/§F.4)."""
    B, prefix_len = prefix_pad.shape
    device = prefix_pad.device
    # prefix part: all suffix queries attend all non-pad prefix keys
    prefix_ok = prefix_pad[:, None, None, :].to(torch.bool).expand(B, 1, suffix_len, prefix_len)
    # suffix part: att_masks = [1, 1, 0, 0, ...] over [state, action...] -> cumsum trick.
    # state(0) sees only <=state; actions see state + all actions.
    att = torch.zeros(suffix_len, device=device)
    att[:2] = 1.0  # first two entries open a bidirectional block (native att_masks[:, :2]=1)
    cs = torch.cumsum(att, dim=0)
    suffix_ok = (cs[None, :] <= cs[:, None])[None, None].expand(B, 1, suffix_len, suffix_len)
    ok = torch.cat([prefix_ok, suffix_ok], dim=-1)
    return torch.where(ok, 0.0, _MASK_FILL).to(dtype)


# ---- checkpoint build + load (option B: package-free, embodiinfer_env-compatible) -----
def _build_and_load(checkpoint: str, backbone_path: str | None = None):
    """Build stock Qwen2.5-VL + vendored action expert, load ``lingbot-vla-4b`` weights directly.

    Returns ``(vl_text_model, visual, expert, projections)``. No lingbotvla import — the
    checkpoint's ``qwenvl.{model,visual}.*`` route into a stock ``Qwen2_5_VLForConditionalGeneration``,
    ``qwen_expert.model.*`` into the vendored :class:`ActionExpert`, and ``model.{state,action}*``
    into :class:`ActionProjections`. box-parity: the exact Qwen2.5-VL config source + key prefixes
    are confirmed against the real checkpoint on the first box build-smoke.
    """
    import json
    import os

    from safetensors.torch import load_file
    from transformers import AutoConfig, Qwen2_5_VLForConditionalGeneration

    from .modules_lingbot_vla import ActionExpert, ActionProjections

    # VL backbone: stock Qwen2.5-VL-3B (weight holder). Prefer the checkpoint's own config;
    # fall back to the canonical id. box-parity: the lingbot config.json nests the qwenvl config.
    if backbone_path is not None:
        # An explicit local source pins architecture assets for offline benchmarks.
        # Fail on an invalid source instead of silently fetching another revision.
        vl_cfg = AutoConfig.from_pretrained(backbone_path)
    else:
        try:
            vl_cfg = AutoConfig.from_pretrained(checkpoint)
        except Exception:
            vl_cfg = AutoConfig.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
    # build in fp32 so the F32 checkpoint master loads exactly (the config's torch_dtype may be
    # bf16; EngineCore casts the whole policy to the execution dtype afterwards).
    vl_full = Qwen2_5_VLForConditionalGeneration(vl_cfg).float()
    # The Qwen2.5-VL container layout differs across transformers versions, but the text
    # decoder always exposes ``layers``/``embed_tokens``/``norm`` and the ViT the same keys,
    # so the checkpoint's ``qwenvl.model.*`` / ``qwenvl.visual.*`` route in unchanged either way:
    #   * tf>=4.53 / 5.x: nested — ``.model.language_model`` (text) + ``.model.visual`` (ViT);
    #   * tf<=4.51:       flat  — ``.model`` is the text decoder + ``.visual`` is top-level.
    _m = vl_full.model
    if hasattr(_m, "language_model"):
        vl_text, visual = _m.language_model, _m.visual
    else:
        vl_text, visual = _m, vl_full.visual
    expert = ActionExpert()
    proj = ActionProjections()

    # gather the checkpoint tensors (sharded or single-file)
    sd: dict[str, torch.Tensor] = {}
    idx = os.path.join(checkpoint, "model.safetensors.index.json")
    if os.path.exists(idx):
        with open(idx) as f:
            shards = sorted(set(json.load(f)["weight_map"].values()))
        for shard in shards:
            sd.update(load_file(os.path.join(checkpoint, shard)))
    else:
        sd.update(load_file(os.path.join(checkpoint, "model.safetensors")))

    # route keys (drop the top-level ``model.`` container prefix). box-parity: confirm prefixes.
    cw = "model.qwenvl_with_expert."
    vl_text_sd, visual_sd, expert_sd, proj_sd = {}, {}, {}, {}
    for k, v in sd.items():
        if k.startswith(cw + "qwenvl.model."):
            vl_text_sd[k[len(cw + "qwenvl.model.") :]] = v
        elif k.startswith(cw + "qwenvl.visual."):
            visual_sd[k[len(cw + "qwenvl.visual.") :]] = v
        elif k.startswith(cw + "qwen_expert.model."):
            expert_sd[k[len(cw + "qwen_expert.model.") :]] = v
        elif k.startswith("model.") and "." not in k[len("model.") :].split(".weight")[0].split(".bias")[0]:
            proj_sd[k[len("model.") :]] = v  # model.{state_proj,action_in_proj,...}.{weight,bias}

    for name, mod, part in [
        ("vl_text", vl_text, vl_text_sd),
        ("visual", visual, visual_sd),
        ("expert", expert, expert_sd),
        ("proj", proj, proj_sd),
    ]:
        miss = mod.load_state_dict(part, strict=False)
        # Only a missing *parameter* is fatal; missing buffers (e.g. rotary ``inv_freq``,
        # persistent across some transformers versions but init-computed) are tolerated.
        param_names = {n for n, _ in mod.named_parameters()}
        missing_params = [k for k in miss.missing_keys if k in param_names]
        if missing_params:
            raise RuntimeError(f"LingBot-VLA load: {name} missing param keys {missing_params[:6]} ...")
    return vl_text.eval(), visual.eval(), expert.eval(), proj.eval()


@register_policy("lingbot_vla")
def _build_lingbot_vla(
    checkpoint: str | None = None,
    attention: str = "eager",
    backbone_path: str | None = None,
    **overrides,
) -> VLAPolicy:
    if checkpoint is None:
        raise ValueError(
            "lingbot_vla needs a checkpoint, e.g. make_policy('lingbot_vla', checkpoint='robbyant/lingbot-vla-4b')"
        )
    # engine-contract fields: chunk_size=50, model action/state dim=75 (env-sliced downstream),
    # num_steps=10. Architecture is owned by the loaded checkpoint.
    cfg = VLAPolicyConfig(
        name="lingbot_vla", action_dim=75, action_horizon=50, default_num_steps=10, **overrides
    )
    return LingBotVLAPolicy(cfg, checkpoint=checkpoint, attention=attention, backbone_path=backbone_path)
