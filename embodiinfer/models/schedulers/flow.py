"""Weightless flow-matching sampling math — the sibling of :mod:`.diffusion`.

The numeric transitions a flow ``ActionDecoder`` follows, factored out of the policies and
the engine graph so there is one source of truth. Before, the plain Euler step ``x + v*dt``
was inlined in four places (the reference sampler, the flow decoder, the loop graph, the
rollout log-prob) and the flow-SDE coefficients lived in the engine graph module:

  * :func:`euler_step` — the deterministic flow-matching ODE step ``x + v*dt``.
  * :func:`sde_coefficients` — per-step flow-SDE noise scale + score-correction
    coefficients (the pi_RL / RLinf trainer family), from the schedule's time direction.
  * :func:`sde_transition_mean` — the score-corrected stochastic transition mean.

No weights, no ``policies`` / ``engine`` imports — a leaf driven by the decoder / graph /
rollout log-prob (the flow analogue of ``diffusion.sample_2ab``).
"""

from __future__ import annotations

import torch


def euler_step(x: torch.Tensor, v: torch.Tensor, dt: float) -> torch.Tensor:
    """Deterministic flow-matching Euler step: ``x + v*dt``."""
    return x + v * dt


def sde_coefficients(
    schedule: list[tuple[float, float]], sigmas: list[float]
) -> tuple[list[float], list[float], list[float]]:
    """Per-step flow-SDE coefficients (host math; used by the SDE loop graph and rollout).

    The score-corrected flow-SDE transition of the pi_RL / RLinf trainer family subtracts a
    correction proportional to the NOISE-side prediction. With the noise end of the flow at
    ``t_noise`` and the data end at ``t_data`` (both derived from the schedule's integration
    direction: ``dt < 0`` integrates 1 -> 0, so noise sits at t = 1; ``dt > 0`` integrates
    0 -> 1, noise at t = 0), each step ``k`` gets:

        c_noise[k] = sigma_k * sqrt(|dt|)                        (transition std)
        gamma[k]   = t_noise - t_k                               (noise-side pred: x + gamma*v)
        c_corr[k]  = sigma_k^2 * |dt| / (2 * |t_k - t_data|)     (score correction)

    yielding the transition (see :func:`sde_transition_mean`)

        x_{k+1} = x_k + v*dt - c_corr[k] * (x_k + gamma[k]*v) + c_noise[k] * eps_k

    For the pi0.5/openpi convention (t: 1 -> 0) this reduces to the correction on
    ``x1_pred = x + v*(1-t)`` with denominator ``2t``; for the GR00T convention (t: 0 -> 1)
    to the correction on ``x0_pred = x - v*t`` with denominator ``2(1-t)`` — verified
    algebraically identical to each trainer's x0/x1-weight form.

    ``sigma_k == 0`` gives ``c_noise == c_corr == 0`` — a plain deterministic Euler (ODE)
    step — so a mixed ODE/SDE pattern (noise on one selected step only) is just a coefficient
    vector, not a different control flow. Pure host math (CPU-testable); the graph reads these
    from device buffers per replay.
    """
    if len(sigmas) != len(schedule):
        raise ValueError(f"got {len(sigmas)} sigmas for {len(schedule)} steps")
    t_noise, t_data = (1.0, 0.0) if schedule[0][1] < 0 else (0.0, 1.0)
    c_noise: list[float] = []
    c_corr: list[float] = []
    gamma: list[float] = []
    for (t_val, dt), sig in zip(schedule, sigmas):
        adt = abs(dt)
        c_noise.append(sig * adt**0.5)
        gamma.append(t_noise - t_val)
        denom = abs(t_val - t_data)
        if sig == 0.0:
            c_corr.append(0.0)  # ODE step; also avoids 0/0 at the data-end grid point
        elif denom == 0.0:
            raise ValueError(f"flow-SDE drift correction is singular at t == {t_data} with sigma > 0")
        else:
            c_corr.append(sig * sig * adt / (2.0 * denom))
    return c_noise, c_corr, gamma


def sde_transition_mean(x: torch.Tensor, v: torch.Tensor, dt: float, c_corr, gamma) -> torch.Tensor:
    """Score-corrected flow-SDE transition mean ``x + v*dt - c_corr*(x + v*gamma)``.

    ``c_corr`` / ``gamma`` are the per-step coefficients from :func:`sde_coefficients`
    (Python floats or 0-dim device tensors). ``c_corr == 0`` reduces this to
    :func:`euler_step` (a deterministic ODE step)."""
    return x + v * dt - c_corr * (x + v * gamma)
