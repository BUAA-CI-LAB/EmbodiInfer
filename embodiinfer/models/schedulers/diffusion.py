"""Cosmos Policy diffusion sampler — self-hosted EDM/rectified-flow denoise loop.

Cosmos Policy denoises latent frames with an EDM-sigma-space Karras schedule and a
2nd-order Adams-Bashforth (RES exponential) multistep solver, while the network is
preconditioned in rectified-flow form (``sigma_data=1``). This module reimplements
that loop op-for-op, so embodiinfer owns the denoise loop rather
than delegating to cosmos-policy's ``CosmosPolicySampler`` / ``res_sampler``.

The solver arithmetic runs in float64 (as in the reference); the ``x0_fn`` it calls
runs the DiT forward in the model dtype and casts back to float64.
"""

from __future__ import annotations

from collections.abc import Callable

import torch


# ---- rectified-flow preconditioning (sigma_data = 1.0) ----------------------
def rectified_flow_scaling(sigma: torch.Tensor):
    """Return ``(c_skip, c_out, c_in, c_noise)`` for RectifiedFlowScaling, sigma_data=1.

    ``t = sigma/(sigma+1)``; ``c_skip = c_in = 1-t``; ``c_out = -t``; ``c_noise = t``.
    The DiT's per-frame ``timesteps`` input is ``c_noise``.
    """
    t = sigma / (sigma + 1.0)
    return 1.0 - t, -t, 1.0 - t, t


# ---- Karras timestep schedule ----------------------------------------------
def karras_sigmas(t_min: float, t_max: float, num_steps: int, rho: float, device) -> torch.Tensor:
    """``get_rev_ts``: reverse Karras sigmas, ``num_steps+1`` values from ``t_max`` to ``t_min``."""
    step = torch.arange(num_steps + 1, dtype=torch.float64, device=device)
    sig = (t_max ** (1.0 / rho) + step / num_steps * (t_min ** (1.0 / rho) - t_max ** (1.0 / rho))) ** rho
    return sig


def _batch_mul(a: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return a.reshape(a.shape[0], *([1] * (x.ndim - 1))) * x


def _phi1(u: torch.Tensor) -> torch.Tensor:
    return torch.expm1(u) / u


def _phi2(u: torch.Tensor) -> torch.Tensor:
    return (_phi1(u) - 1.0) / u


def _reg_x0_euler_step(x_s, s, t, x0_s):
    coef_x0 = (s - t) / s
    coef_xs = t / s
    return _batch_mul(coef_x0, x0_s) + _batch_mul(coef_xs, x_s)


def _res_x0_rk2_step(x_s, t, s, x0_s, s1, x0_s1):
    s_l = -torch.log(s)
    t_l = -torch.log(t)
    m_l = -torch.log(s1)
    dt = t_l - s_l
    c2 = (m_l - s_l) / dt
    phi1v, phi2v = _phi1(-dt), _phi2(-dt)
    b1 = torch.nan_to_num(phi1v - phi2v / c2, nan=0.0)
    b2 = torch.nan_to_num(phi2v / c2, nan=0.0)
    return _batch_mul(torch.exp(-dt), x_s) + _batch_mul(dt, _batch_mul(b1, x0_s) + _batch_mul(b2, x0_s1))


def sample_2ab(
    x0_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    x_sigma_max: torch.Tensor,
    num_denoising_steps: int,
    sigma_min: float,
    sigma_max: float,
    rho: float = 7.0,
) -> torch.Tensor:
    """Cosmos ``CosmosPolicySampler``: 2ab multistep + a final ``sample_clean`` step.

    ``num_denoising_steps`` counts the total NFE (default 5). With ``sample_clean`` on,
    the solver runs ``num_denoising_steps-1`` multistep iterations over Karras sigmas,
    then one clean denoise at ``sigma_min`` — matching the shipped LIBERO recipe.
    ``x0_fn(x, sigma_scalar_broadcast)`` returns the x0 prediction.
    """
    device = x_sigma_max.device
    n = num_denoising_steps - 1 if num_denoising_steps > 1 else num_denoising_steps
    sigmas = karras_sigmas(sigma_min, sigma_max, n, rho, device)  # [n+1], float64
    x = x_sigma_max.to(torch.float64)
    B = x.shape[0]
    ones = torch.ones(B, device=device, dtype=torch.float64)

    if num_denoising_steps > 1:
        mem = None  # (x0_prev, sigma_prev)
        for i in range(n):
            s0, s1 = sigmas[i], sigmas[i + 1]
            x0 = x0_fn(x, s0 * ones)
            if mem is None:
                x = _reg_x0_euler_step(x, s0 * ones, s1 * ones, x0)
            else:
                x0_prev, s_prev = mem
                x = _res_x0_rk2_step(x, s1 * ones, s0 * ones, x0, s_prev * ones, x0_prev)
            mem = (x0, s0)
        x = x0_fn(x, sigmas[-1] * ones)  # sample_clean
    else:
        x = x0_fn(x, sigmas[0] * ones)
    return x
