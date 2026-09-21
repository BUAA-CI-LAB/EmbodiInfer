"""Flow-matching action log-probability for policy-gradient rollout.

Flow-matching samplers are deterministic ODE integrators and do not, by
themselves, expose an action log-probability — the quantity PPO/GRPO need. The
standard fix (cf. pi_RL, arXiv 2510.25889, "probability path integration") is to
run the sampler as a stochastic (SDE) process and accumulate the per-step
Gaussian transition log-densities.

We implement the Euler-Maruyama form:

    x_{k+1} = x_k + v(x_k, t_k) * dt + sigma(t_k) * sqrt(dt) * eps_k,   eps_k ~ N(0, I)
    log p  += log N( x_{k+1} ; x_k + v*dt , sigma(t_k)^2 * dt * I )
            = -0.5 * ( eps_k^2 + D * log(2*pi*sigma(t_k)^2*dt) )   summed over dims

``sigma`` is either a constant (temperature-style, the original behavior) or a
schedule ``sigma(t)`` evaluated at each step's ``t`` — e.g. the flow-SDE
schedule ``sigma(t) = noise_level * sqrt(t / (1 - t))`` used by the pi_RL /
RLinf family of flow-VLA trainers. Passing a callable keeps this module
model-agnostic: the step grid (and its time convention) comes from
``policy.flow_schedule``, the noise profile from the caller.

The returned quantity per sample is a *surrogate* trajectory log-prob, adequate
for prototype policy-gradient wiring; it is labelled as such and not claimed to
be the exact marginal action likelihood. With ``per_step=True`` the un-summed
``[B, num_steps]`` per-transition log-densities are returned instead, letting a
trainer apply its own step weighting or selection (e.g. scoring one random
denoise step per sample, as RLinf's PPO does).

Two entry points serve the two halves of an on-policy update:

  * :func:`flow_sample_with_logprob` — the *behavior* pass. Runs under
    ``no_grad`` at rollout time and returns the actions, the behavior log-prob,
    and (optionally) the visited state trajectory ``[x_0, ..., x_N]``.
  * :func:`flow_logprob_recompute` — the *policy-gradient* pass. Re-scores the
    stored trajectory under the current parameters with autograd enabled,
    yielding ``log pi_theta(a | s)`` whose gradient flows back into the policy.

Holding the same realized trajectory across both passes is what makes the two
log-probs comparable: at ``theta == theta_behavior`` the recomputed value is
identical to the behavior value (the residual reduces to the sampled ``eps_k``),
so the importance ratio starts at exactly 1 and carries the correct gradient.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch

from ...models.schedulers.flow import euler_step
from ...policies.base import PrefixState, VLAPolicy

SigmaLike = float | Callable[[float], float]

# Optional transition-mean hook: ``mean_fn(x, v, t, dt, sigma_t) -> mean``.
# Default (None) is the Euler-Maruyama drift ``x + v * dt``. Score-based flow-SDE
# variants (pi_RL / RLinf family) add a correction term to the mean, e.g.
# ``x + v*dt - (sigma(t)^2*|dt|/(2t)) * (x + v*(1-t))``; passing it here keeps
# the sampler generic while matching such trainers' transition kernels exactly.
MeanFn = Callable[[torch.Tensor, torch.Tensor, float, float, float], torch.Tensor]


def _sigma_at(sigma: SigmaLike, t_val: float) -> float:
    """Resolve a constant or scheduled ``sigma`` at time ``t_val``."""
    return float(sigma(t_val)) if callable(sigma) else float(sigma)


def _transition_const(dim: int, sigma_t: float, adt: float) -> tuple[float, float]:
    """Return ``(variance, D*log(2*pi*variance))`` for one Gaussian SDE step."""
    var = sigma_t * sigma_t * adt
    return var, dim * math.log(2 * math.pi * var)


def _likelihood_mask(mask: torch.Tensor | None, state: torch.Tensor) -> tuple[torch.Tensor | None, int]:
    """Validate a selection mask once and count scored elements per sample."""
    if mask is None:
        return None, state.shape[1] * state.shape[2]
    try:
        mask = torch.broadcast_to(mask.to(device=state.device, dtype=torch.bool), state.shape)
    except RuntimeError as error:
        raise ValueError("Flow likelihood mask does not match its state") from error
    dimensions = mask.flatten(1).sum(dim=1)
    if not torch.equal(dimensions, dimensions[:1].expand_as(dimensions)):
        raise ValueError("Flow likelihood dimensions must be equal across the batch")
    count = int(dimensions[0].item())
    if count <= 0:
        raise ValueError("Flow likelihood mask must select at least one element")
    return mask, count


@torch.no_grad()
def flow_sample_with_logprob(
    policy: VLAPolicy,
    prefix: PrefixState,
    x0: torch.Tensor,
    num_steps: int,
    sigma: SigmaLike = 0.1,
    return_trajectory: bool = False,
    per_step: bool = False,
    generator: torch.Generator | None = None,
    mean_fn: MeanFn | None = None,
    return_velocities: bool = False,
    *,
    logprob_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, ...]:
    """Stochastic flow sampler returning ``(actions, surrogate_logprob)``.

    Args:
        prefix: cached prefix KV for the batch (already expanded for best-of-N).
        x0: [B, H, A] initial noise.
        num_steps: solver steps.
        sigma: SDE noise level — a constant, or a schedule ``sigma(t)``
            evaluated at each step's ``t`` from ``policy.flow_schedule`` (in the
            policy's own time convention). ``sigma(t) == 0`` makes that step a
            deterministic ODE step (no noise, per-step logprob recorded as 0),
            so mixed ODE/SDE samplers — e.g. noise on one selected step only —
            need no epsilon floor.
        return_trajectory: if True, also return the visited states stacked as
            ``[B, num_steps + 1, H, A]`` (``x_0`` through ``x_N``). This is the
            record a later :func:`flow_logprob_recompute` re-scores under the
            current policy for the policy-gradient update.
        per_step: if True, return the logprob un-summed as ``[B, num_steps]``
            (one Gaussian transition log-density per solver step) instead of
            the ``[B]`` total.
        generator: optional RNG for the per-step noise draws.
        mean_fn: optional transition-mean override (see :data:`MeanFn`).
        return_velocities: if True, also return the per-step model velocities
            stacked as ``[B, num_steps, H, A]`` (``v(x_k, t_k)`` for each solver
            step). Trainers that score a transition in their own elementwise
            convention (e.g. openpi/RLinf's per-dim Gaussian at one selected
            denoise step) need ``v`` at that step; returning it here avoids an
            extra ``denoise_step`` forward after sampling.
        logprob_mask: optional broadcastable selection mask; nonzero elements
            select scored dimensions, with the same count in every batch row.
            Full states are still sampled and stored. Masked scores use realized
            transition residuals, matching recompute even at reduced precision;
            None preserves the original unmasked sampled-noise score.
    Returns:
        actions: [B, H, A]
        logprob: [B] summed transition log-density (behavior policy), or
            ``[B, num_steps]`` when ``per_step``.
        trajectory: [B, num_steps + 1, H, A] — only when ``return_trajectory``.
        velocities: [B, num_steps, H, A] — only when ``return_velocities``
            (always last in the returned tuple).
    """
    B, _, _ = x0.shape
    device = x0.device
    x = x0
    mask, D = _likelihood_mask(logprob_mask, x0)
    steps: list[torch.Tensor] = []
    xs = [x] if return_trajectory else None
    vs: list[torch.Tensor] | None = [] if return_velocities else None
    for t_val, dt in policy.flow_schedule(num_steps):
        adt = abs(dt)
        sig = _sigma_at(sigma, t_val)
        t = torch.full((B,), t_val, device=device, dtype=x.dtype)
        v = policy.denoise_step(x, t, prefix)
        if vs is not None:
            vs.append(v)
        mean = euler_step(x, v, dt) if mean_fn is None else mean_fn(x, v, t_val, dt, sig)
        if sig > 0.0:
            std = sig * math.sqrt(adt)
            _, const = _transition_const(D, sig, adt)
            eps = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)
            x = mean + std * eps
            if mask is None:
                steps.append(-0.5 * (eps.float().pow(2).flatten(1).sum(dim=1) + const))
            else:
                energy = (x - mean).float().pow(2) / std**2
                const = D * math.log(2 * math.pi * sig**2 * adt)
                steps.append(-0.5 * ((energy * mask).flatten(1).sum(dim=1) + const))
        else:
            x = mean
            steps.append(torch.zeros(B, device=device, dtype=torch.float32))
        if xs is not None:
            xs.append(x)
    logprob_steps = torch.stack(steps, dim=1)  # [B, num_steps]
    logprob = logprob_steps if per_step else logprob_steps.sum(dim=1)
    out: tuple[torch.Tensor, ...] = (x, logprob)
    if xs is not None:
        out = out + (torch.stack(xs, dim=1),)
    if vs is not None:
        out = out + (torch.stack(vs, dim=1),)
    return out


def flow_logprob_recompute(
    policy: VLAPolicy,
    prefix: PrefixState,
    trajectory: torch.Tensor,
    num_steps: int,
    sigma: SigmaLike = 0.1,
    per_step: bool = False,
    mean_fn: MeanFn | None = None,
    *,
    logprob_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Re-score a stored trajectory under the *current* policy, with grad enabled.

    This is the differentiable counterpart of :func:`flow_sample_with_logprob`.
    Given the states ``[x_0, ..., x_N]`` visited during behavior sampling, it
    recomputes the drift ``v_theta(x_k, t_k)`` with the live parameters and
    accumulates the same Euler-Maruyama transition log-density, holding the
    realized next-states fixed:

        residual_k = x_{k+1} - x_k - v_theta(x_k, t_k) * dt
        log p     += -0.5 * ( ||residual_k||^2 / (sigma(t_k)^2 * dt) + D * log(2*pi*sigma(t_k)^2*dt) )

    The gradient flows through ``denoise_step`` (and, if ``prefix`` was encoded
    with grad, through the backbone) into the policy — this is the quantity
    ``log pi_theta(a | s)`` that GRPO/PPO differentiate. Deliberately *not*
    wrapped in ``no_grad``.

    Args:
        prefix: prefix KV encoded under the current policy, already expanded to
            match ``trajectory``'s leading dimension (``B * group_size``).
        trajectory: ``[B, num_steps + 1, H, A]`` states from behavior sampling.
        num_steps: solver steps (must match the behavior pass).
        sigma: SDE noise level (must match the behavior pass) — constant or
            schedule ``sigma(t)``. Steps with ``sigma(t) == 0`` were
            deterministic during behavior sampling; their term is 0 and no
            gradient flows through them.
        per_step: if True, return ``[B, num_steps]`` per-transition terms
            instead of the ``[B]`` total.
        mean_fn: optional transition-mean override (must match the behavior
            pass; see :data:`MeanFn`).
        logprob_mask: the same broadcastable selection mask used when sampling;
            only selected elements contribute to the score and its gradient.
    Returns:
        logprob: ``[B]`` (or ``[B, num_steps]``) recomputed transition
            log-density, differentiable.
    """
    B, num_states, _, _ = trajectory.shape
    expected = num_steps + 1
    if num_states != expected:
        raise ValueError(f"trajectory has {num_states} states but num_steps={num_steps} needs {expected}")
    device = trajectory.device
    mask, D = _likelihood_mask(logprob_mask, trajectory[:, 0])
    steps: list[torch.Tensor] = []
    for k, (t_val, dt) in enumerate(policy.flow_schedule(num_steps)):
        adt = abs(dt)
        sig = _sigma_at(sigma, t_val)
        x_k = trajectory[:, k]
        x_next = trajectory[:, k + 1]
        if sig <= 0.0:
            steps.append(torch.zeros(B, device=device, dtype=torch.float32))
            continue
        var, const = _transition_const(D, sig, adt)
        t = torch.full((B,), t_val, device=device, dtype=x_k.dtype)
        v = policy.denoise_step(x_k, t, prefix)
        mean = euler_step(x_k, v, dt) if mean_fn is None else mean_fn(x_k, v, t_val, dt, sig)
        residual = x_next - mean
        if mask is None:
            steps.append(-0.5 * (residual.float().pow(2).flatten(1).sum(dim=1) / var + const))
        else:
            energy = residual.float().pow(2) / var
            steps.append(-0.5 * ((energy * mask).flatten(1).sum(dim=1) + const))
    logprob_steps = torch.stack(steps, dim=1)
    return logprob_steps if per_step else logprob_steps.sum(dim=1)
