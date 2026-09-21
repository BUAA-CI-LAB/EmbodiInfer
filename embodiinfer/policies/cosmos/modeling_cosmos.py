"""Cosmos Policy (WAM) -> embodiinfer ``VLAPolicy``: self-hosted video-diffusion world-action model.

Cosmos Policy is a generative DiT **world-action model** (predict-then-act): images,
proprio, the action chunk, the predicted future state, and a scalar value are all packed
as latent *frames* of a latent video, and a Cosmos-Predict2 2B DiT denoises them (EDM-sigma
Karras sampler + rectified-flow preconditioning) under a frame-replace conditioning mask.
The same DiT + mask gives the policy (denoise the action frame), the world model (future
state), and the **value** (denoise the value frame -> scalar) — so the best-of-N planning
scorer *is* the value frame, produced jointly with the action in one diffusion pass.

Mapping onto the engine:

    encode_prefix  <-  VAE-encode the conditioning images into latent frames + inject
                       proprio + build the frame-replace mask + carry the T5 cross-attn
                       context. Constant across denoising steps AND across best-of-N
                       candidates (the compute-bound VAE / text encode runs once, §4.3).
    DiffusionDecoder  <- the EDM/rectified-flow sampler loop (Karras + 2ab + sample_clean),
                       each step a self-hosted DiT x0-prediction with frame replacement.
    best_of_n      <-  sample N action candidates from one prefix, read each candidate's
                       value frame, keep the argmax — the WAM planning surface (§7.2).

The DiT forward is self-hosted (:mod:`embodiinfer.models.video_dit.cosmos_predict2`) and the Wan2.1 VAE is a vendored leaf
(:mod:`embodiinfer.models.video_vae.wan`) — same "modules load, forward is self-hosted, perception is a leaf"
contract as pi0.5 / OpenVLA-OFT / LingBot-VLA. Diffusion is not linear flow, so Cosmos gets
its own :class:`CosmosDiffusionDecoder` (a fourth ``ActionDecoder`` alongside Flow / Parallel /
Autoregressive); the slim model-agnostic engine is unchanged.

Parity vs cosmos-policy's own inference (LIBERO Predict2-2B, bf16, fixed noise): self-hosted
DiT ``max|Δ|≈0.03``, VAE encode ``max|Δ|≈0.02``, end-to-end latent ``max|Δ|≈0.02`` / action
frame ``≈1.8e-3`` / value frame exact — the bf16 cross-kernel noise floor.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ...models.schedulers import rectified_flow_scaling, sample_2ab
from ...models.video_dit import CosmosPredict2DiT, CosmosPredict2DiTConfig, load_cosmos_predict2_dit
from ..base import VLAPolicy
from ..config import VLAPolicyConfig
from ..decoder import ActionDecoder
from ..factory import register_policy
from .processor_cosmos import CosmosBatch, CosmosProcessor

_INSTALL = (
    "Cosmos Policy needs the LIBERO Predict2-2B checkpoint (DiT .pt) + the Wan2.1 VAE "
    "(tokenizer.pth from Cosmos-Predict2-2B-Video2World). Pass checkpoint=<dir> with both, "
    "or dit_ckpt=/vae_ckpt= explicitly."
)


# ---- latent injection / readout (mirrors cosmos-policy's tile-fill) ----------
def _inject_vector(latent: torch.Tensor, vec: torch.Tensor, frame_idx: int) -> torch.Tensor:
    """Tile-fill ``vec`` ``[B, d]`` into ``latent[:, :, frame_idx]`` ``[B, C, H, W]`` (flatten + repeat)."""
    B, C, T, H, W = latent.shape
    n_elem = C * H * W
    flat = vec.reshape(B, -1)
    reps = (n_elem + flat.shape[1] - 1) // flat.shape[1]
    filled = flat.repeat(1, reps)[:, :n_elem].reshape(B, C, H, W)
    out = latent.clone()
    out[:, :, frame_idx, :, :] = filled.to(latent.dtype)
    return out


def read_action_frame(latent: torch.Tensor, frame_idx: int, horizon: int, action_dim: int) -> torch.Tensor:
    """Inverse of the tile-fill: average all tiled copies -> action chunk ``[B, horizon, action_dim]``."""
    frame = latent[:, :, frame_idx, :, :]  # [B, C, H, W]
    flat = frame.reshape(frame.shape[0], -1)
    n_elem = horizon * action_dim
    n_chunks = flat.shape[1] // n_elem
    chunks = flat[:, : n_chunks * n_elem].reshape(frame.shape[0], n_chunks, horizon, action_dim)
    return chunks.mean(dim=1)


def read_value_frame(latent: torch.Tensor, frame_idx: int) -> torch.Tensor:
    """Average the entire value frame -> scalar value ``[B]`` (raw; ``(v+1)/2`` clamp applied by callers)."""
    return latent[:, :, frame_idx, :, :].reshape(latent.shape[0], -1).mean(dim=1)


@dataclass
class CosmosPrefix:
    """The reusable Cosmos conditioning: clean latent frames + frame-replace mask + text context.

    ``gt_frames`` ``[B, 16, T', H', W']`` are the VAE-encoded conditioning frames (proprio
    injected); ``mask`` ``[B, 1, T', H', W']`` marks the leading conditioning frames (1) vs the
    to-generate frames (0). ``crossattn`` ``[B, N, 1024]`` is the T5 embedding. All constant
    across denoising steps and across best-of-N candidates.
    """

    gt_frames: torch.Tensor
    mask: torch.Tensor
    crossattn: torch.Tensor
    padding_mask: torch.Tensor
    batch_size: int

    def to(self, device) -> CosmosPrefix:
        return CosmosPrefix(
            gt_frames=self.gt_frames.to(device),
            mask=self.mask.to(device),
            crossattn=self.crossattn.to(device),
            padding_mask=self.padding_mask.to(device),
            batch_size=self.batch_size,
        )

    def expand(self, num_samples: int) -> CosmosPrefix:
        if num_samples == 1:
            return self
        rep = lambda t: t.repeat_interleave(num_samples, dim=0)  # noqa: E731
        return CosmosPrefix(
            gt_frames=rep(self.gt_frames),
            mask=rep(self.mask),
            crossattn=rep(self.crossattn),
            padding_mask=rep(self.padding_mask),
            batch_size=self.batch_size * num_samples,
        )


class CosmosDiffusionDecoder(ActionDecoder):
    """EDM/rectified-flow sampler loop for Cosmos Policy (a fourth ``ActionDecoder``).

    ``init_state`` seeds ``x_sigma_max`` (noise x sigma_max); ``produce_chunk`` runs the
    Karras + 2ab + sample_clean sampler with the policy's frame-replace ``denoise`` as the
    x0 function, then reads the (dataset-scale) action frame. No flow schedule. The per-step
    ``denoise`` (the DiT forward) is CUDA-graphed by the engine's ``GraphManager`` (kind
    ``diffusion_step`` -> :class:`~embodiinfer.engine.graph.CosmosDenoiseGraph`: static prefix copied
    in once, replayed per step) when the engine has graphs enabled — bit-exact vs eager,
    ~1.14x on the 5-step sampler at small batch."""

    def __init__(self, policy: CosmosPolicy):
        self.policy = policy

    def _device_dtype(self):
        p = next(self.policy.parameters())
        return p.device, p.dtype

    def init_state(self, batch_size: int, generator: torch.Generator | None = None) -> torch.Tensor:
        c = self.policy
        device, _ = self._device_dtype()
        shape = (batch_size, c.latent_ch, c.state_t, c.latent_hw, c.latent_hw)
        return torch.randn(shape, device=device, dtype=torch.float32, generator=generator) * c.sigma_max

    def produce_chunk(self, state, prefix: CosmosPrefix, num_steps: int, bucket: int, graphs) -> torch.Tensor:
        graph = None
        if graphs is not None:
            graph = graphs.get(bucket, num_steps)  # CosmosDenoiseGraph (per-step DiT), engine-owned
            graph.set_prefix(prefix)  # stage this generation's prefix into the static buffers once
        latent = self.policy.sample_latent(prefix, x_sigma_max=state, num_steps=num_steps, graph=graph)
        return self.policy.unnormalize_action(self.policy.read_action(latent))  # dataset-scale action

    # Not an ``RLDecoder``: Cosmos is a predict-then-act WAM (RLinf has no cosmos generator).
    # Its planning surface is the jointly-produced value frame via ``CosmosPolicy.best_of_n_plan``,
    # not a flow-SDE behavior/recompute log-prob — so ``sample_with_logprob`` / ``recompute_logprob``
    # are intentionally absent.


class CosmosPolicy(VLAPolicy):
    """Cosmos Policy world-action model as an embodiinfer ``VLAPolicy``.

    Holds the self-hosted DiT (:class:`CosmosPredict2DiT`) and the vendored Wan2.1 VAE
    (:class:`~embodiinfer.models.video_vae.wan.WanVAE`). ``encode_prefix`` VAE-encodes the
    conditioning images and builds the frame-replace prefix; ``sample_latent`` runs the
    diffusion sampler; ``read_action`` / ``read_value`` decode the action and value frames;
    ``best_of_n_plan`` is the WAM planning surface.
    """

    def __init__(
        self,
        config: VLAPolicyConfig,
        dit: CosmosPredict2DiT,
        vae,
        *,
        text_embedder=None,
        dataset_stats=None,
        latent_idx=None,
        state_t=9,
    ):
        super().__init__(config)
        self.dit = dit
        self.vae = vae
        self.text_embedder = text_embedder  # CosmosTextEmbedder (pkl fast-path + T5 fallback)
        self.dataset_stats = dataset_stats  # {actions_min/max, proprio_min/max}
        self._decoder: CosmosDiffusionDecoder | None = None
        self.processor = CosmosProcessor()
        # latent layout + diffusion constants (LIBERO Predict2-2B)
        from .processor_cosmos import LIBERO_LATENT_IDX, LIBERO_NUM_CONDITIONAL_FRAMES

        self.latent_idx = latent_idx or dict(LIBERO_LATENT_IDX)
        self.state_t = state_t
        self.num_conditional_frames = LIBERO_NUM_CONDITIONAL_FRAMES
        self.latent_ch = 16
        self.latent_hw = 28
        self.sigma_data = 1.0
        self.sigma_min = 4.0
        self.sigma_max = 80.0
        self.rho = 7.0
        self.sigma_conditional = 0.0

    # ---- CUDA-graph capability (per-step DiT, captured by the engine's GraphManager) ----
    @property
    def supports_cuda_graph(self) -> bool:
        return True

    @property
    def cuda_graph_kind(self) -> str:
        return "diffusion_step"

    # ---- prefix encode (VAE + inject + mask), run once ----------------------
    @torch.no_grad()
    def encode_prefix(self, batch: CosmosBatch) -> CosmosPrefix:
        device = next(self.parameters()).device
        batch = batch.to(device, next(self.parameters()).dtype)
        clean = self.vae.encode(batch.pixel_video)  # [B, 16, 9, 28, 28]
        clean = clean.float()
        # inject current proprio into its latent frame
        clean = _inject_vector(clean, batch.proprio.float().to(device), self.latent_idx["current_proprio"])
        # frame-replace mask: leading conditioning frames = 1
        B = clean.shape[0]
        mask = torch.zeros((B, 1, self.state_t, self.latent_hw, self.latent_hw), device=device)
        mask[:, :, : self.num_conditional_frames, :, :] = 1.0
        return CosmosPrefix(
            gt_frames=clean,
            mask=mask,
            crossattn=batch.crossattn.to(device),
            padding_mask=batch.padding_mask.to(device),
            batch_size=B,
        )

    # ---- one preconditioned DiT x0-prediction with frame replacement --------
    def denoise(self, x_f32: torch.Tensor, sigma_f32: torch.Tensor, prefix: CosmosPrefix) -> torch.Tensor:
        B, C, T, _, _ = x_f32.shape
        gt = prefix.gt_frames
        mask_C = prefix.mask.repeat(1, C, 1, 1, 1)
        s = sigma_f32.reshape(B, 1, 1, 1, 1)
        c_skip, c_out, c_in, c_noise = rectified_flow_scaling(s)
        net_in = x_f32 * c_in
        net_in = gt * mask_C + net_in * (1 - mask_C)
        _, _, _, c_noise_cond = rectified_flow_scaling(torch.full_like(s, self.sigma_conditional))
        m = mask_C.mean(dim=[1, 3, 4], keepdim=True)  # [B,1,T,1,1]
        c_noise_f = c_noise_cond * m + c_noise * (1 - m)
        timesteps = c_noise_f.reshape(B, T)
        dtype = next(self.parameters()).dtype
        net_out = self.dit(
            net_in.to(dtype),
            timesteps.to(dtype),
            prefix.crossattn.to(dtype),
            prefix.mask.to(dtype),
            prefix.padding_mask.to(dtype),
        ).float()
        x0 = c_skip * x_f32 + c_out * net_out
        return gt * mask_C + x0 * (1 - mask_C)  # denoise_replace_gt_frames

    @torch.no_grad()
    def sample_latent(
        self, prefix: CosmosPrefix, x_sigma_max: torch.Tensor, num_steps: int, graph=None
    ) -> torch.Tensor:
        """Run the Karras + 2ab + sample_clean sampler.

        ``graph`` (a :class:`~embodiinfer.engine.graph.CosmosDenoiseGraph` from the engine's
        ``GraphManager``, its prefix already staged) replaces the per-step DiT forward with a
        replay when given; otherwise the DiT runs eagerly. The 2ab solver arithmetic is
        host-side float64 either way."""
        if graph is not None:

            def x0_fn(x64, s64):
                return graph.run(x64.to(torch.float32), s64.to(torch.float32)).to(torch.float64)
        else:

            def x0_fn(x64, s64):
                return self.denoise(x64.to(torch.float32), s64.to(torch.float32), prefix).to(torch.float64)

        return sample_2ab(
            x0_fn, x_sigma_max, num_steps, sigma_min=self.sigma_min, sigma_max=self.sigma_max, rho=self.rho
        ).float()

    # ---- CUDA-graph of the per-step denoise (static prefix, replayed each step) ----
    def allocate_static_prefix(self, batch_size: int, device, dtype) -> CosmosPrefix:
        z = lambda *s, dt: torch.zeros(*s, device=device, dtype=dt)  # noqa: E731
        return CosmosPrefix(
            gt_frames=z(
                batch_size, self.latent_ch, self.state_t, self.latent_hw, self.latent_hw, dt=torch.float32
            ),
            mask=z(batch_size, 1, self.state_t, self.latent_hw, self.latent_hw, dt=torch.float32),
            crossattn=z(batch_size, 512, 1024, dt=dtype),
            padding_mask=z(batch_size, 1, 224, 224, dt=dtype),
            batch_size=batch_size,
        )

    def copy_prefix_into(self, dst: CosmosPrefix, src: CosmosPrefix) -> None:
        dst.gt_frames.copy_(src.gt_frames)
        dst.mask.copy_(src.mask)
        dst.crossattn.copy_(src.crossattn)
        dst.padding_mask.copy_(src.padding_mask)

    def read_action(self, latent: torch.Tensor) -> torch.Tensor:
        return read_action_frame(
            latent, self.latent_idx["action"], self.config.action_horizon, self.config.action_dim
        )

    def read_value(self, latent: torch.Tensor) -> torch.Tensor:
        return read_value_frame(latent, self.latent_idx["value"])

    # ---- best-of-N planning (the WAM surface) -------------------------------
    @torch.no_grad()
    def best_of_n_plan(
        self,
        batch: CosmosBatch,
        num_samples: int,
        num_steps: int | None = None,
        generator: torch.Generator | None = None,
    ):
        """Sample ``num_samples`` action candidates from one prefix, keep the highest-value one.

        The prefix (VAE + text encode) runs once and is broadcast across the N candidates
        (``expand``); each candidate is a diffusion trajectory that jointly denoises the action
        and value frames. Returns ``(best_actions [B, H, A], best_value [B], all_actions
        [B, N, H, A], all_values [B, N])``.
        """
        num_steps = num_steps or self.config.default_num_steps
        prefix = self.encode_prefix(batch)  # once
        B = prefix.batch_size
        exp = prefix.expand(num_samples)
        x = self.decoder.init_state(B * num_samples, generator)
        latent = self.sample_latent(exp, x, num_steps)
        actions = self.read_action(latent).reshape(
            B, num_samples, self.config.action_horizon, self.config.action_dim
        )
        values_raw = self.read_value(latent).reshape(B, num_samples)  # [B, N]
        values = ((values_raw + 1) / 2).clamp(0, 1)
        best = values.argmax(dim=1)
        idx = best.view(B, 1, 1, 1).expand(-1, 1, self.config.action_horizon, self.config.action_dim)
        best_actions = torch.gather(actions, 1, idx).squeeze(1)
        best_value = torch.gather(values, 1, best.view(B, 1)).squeeze(1)
        return best_actions, best_value, actions, values

    @property
    def decoder(self) -> CosmosDiffusionDecoder:
        if self._decoder is None:
            self._decoder = CosmosDiffusionDecoder(self)
        return self._decoder

    def unnormalize_action(self, actions: torch.Tensor) -> torch.Tensor:
        """Un-normalize an action chunk ``[-1,1]`` -> dataset scale (if ``dataset_stats`` is set)."""
        if self.dataset_stats is None:
            return actions
        from .processor_cosmos import unnormalize_actions

        return unnormalize_actions(actions, self.dataset_stats)

    # ---- engine batch construction ------------------------------------------
    def collate(self, observations, request_ids):
        """Build a :class:`CosmosBatch` from ``Observation``\\ s so the engine can run Cosmos.

        Each ``Observation`` carries 2 cameras (``images[0]``=wrist, ``images[1]``=primary,
        ``[3,224,224]`` in ``[0,1]``), the proprio vector (``state``), and the task string
        (``instruction``). Images are mapped to ``[-1,1]`` (blank frames -> -1, matching the
        native ``uint8/127.5-1``); proprio is rescaled via ``dataset_stats``; the instruction
        becomes a T5 cross-attn embedding via ``text_embedder`` (pkl fast-path). NOTE: LIBERO's
        exact image preprocessing (flip / JPEG-q95 / center-crop 212->224) is a serving-frontend
        step assumed applied upstream of the ``Observation`` (same posture as pi0.5's tokenize)."""
        from .processor_cosmos import IMAGE_SIZE, build_libero_pixel_video, rescale_proprio

        if self.text_embedder is None:
            raise ValueError(
                "Cosmos Policy needs a text_embedder to collate Observations; build via make_policy('cosmos', checkpoint=<dir>)."
            )
        videos, proprios, instructions = [], [], []
        for obs in observations:
            imgs = obs.images  # [num_cam, 3, 224, 224] in [0, 1]
            vid = build_libero_pixel_video(imgs[1], imgs[0])  # (primary, wrist) in [0,1], blanks = 0
            videos.append(vid * 2 - 1)  # [0,1] -> [-1,1]; blank 0 -> -1
            pr = obs.state.float()
            proprios.append(rescale_proprio(pr, self.dataset_stats) if self.dataset_stats is not None else pr)
            instructions.append(obs.instruction or "")
        pixel_video = torch.stack(videos, dim=0)
        crossattn = self.text_embedder.embed(instructions).to(pixel_video.dtype)
        padding_mask = torch.zeros((len(observations), 1, IMAGE_SIZE, IMAGE_SIZE))
        return CosmosBatch(
            pixel_video=pixel_video,
            proprio=torch.stack(proprios, dim=0),
            crossattn=crossattn,
            padding_mask=padding_mask,
            request_ids=list(request_ids),
        )

    def pad(self, batch: CosmosBatch, target_batch_size: int) -> CosmosBatch:
        """Pad up to ``target_batch_size`` (bucket reuse) by repeating the last observation."""
        b = batch.batch_size
        if b >= target_batch_size:
            return batch
        n = target_batch_size - b
        rep = lambda t: torch.cat([t, t[-1:].expand(n, *t.shape[1:])], dim=0)  # noqa: E731
        return CosmosBatch(
            pixel_video=rep(batch.pixel_video),
            proprio=rep(batch.proprio),
            crossattn=rep(batch.crossattn),
            padding_mask=rep(batch.padding_mask),
            request_ids=list(batch.request_ids) + [batch.request_ids[-1]] * n,
        )


@register_policy("cosmos")
def _build_cosmos(
    checkpoint: str | None = None,
    dit_ckpt: str | None = None,
    vae_ckpt: str | None = None,
    attention: str = "sdpa",
    device: str = "cuda",
    dtype=torch.bfloat16,
    **overrides,
) -> VLAPolicy:
    import json
    import os

    stats_path = pkl_path = None
    if checkpoint is not None:
        # a directory holding the DiT .pt (+ optional dataset stats + T5 embedding pkl)
        for f in os.listdir(checkpoint):
            if f.endswith(".pt") and "tokenizer" not in f and dit_ckpt is None:
                dit_ckpt = os.path.join(checkpoint, f)
            elif f.endswith("dataset_statistics.json"):
                stats_path = os.path.join(checkpoint, f)
            elif f.endswith("t5_embeddings.pkl"):
                pkl_path = os.path.join(checkpoint, f)
    if dit_ckpt is None:
        raise ValueError(_INSTALL)
    cfg = VLAPolicyConfig(
        name="cosmos_policy_wam", action_dim=7, action_horizon=16, default_num_steps=5, **overrides
    )
    sd = torch.load(dit_ckpt, map_location="cpu", weights_only=False)
    dit = (
        load_cosmos_predict2_dit(sd, CosmosPredict2DiTConfig(), attention=attention)
        .to(device)
        .to(dtype)
        .eval()
    )
    if vae_ckpt is None:
        raise ValueError("Cosmos Policy needs the Wan2.1 VAE: pass vae_ckpt=<tokenizer.pth>. " + _INSTALL)
    from ...models.video_vae import load_wan_vae  # deferred (pulls einops)

    vae = load_wan_vae(vae_ckpt, device=device, dtype=dtype)
    dataset_stats = None
    if stats_path:
        with open(stats_path) as f:
            dataset_stats = json.load(f)
    text_embedder = None
    if pkl_path:
        from .processor_cosmos import CosmosTextEmbedder

        text_embedder = CosmosTextEmbedder(pkl_path, device=device, dtype=dtype)
    return CosmosPolicy(cfg, dit, vae, text_embedder=text_embedder, dataset_stats=dataset_stats)
