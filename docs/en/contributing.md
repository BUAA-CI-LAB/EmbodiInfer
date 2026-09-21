# Contributing

EmbodiInfer welcomes contributions: bug fixes, new model adapters, decoders, engine
mechanisms, tests, and documentation.

The full contribution guide is
[`CONTRIBUTING.md`](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/CONTRIBUTING.md)
in the repository root. This page covers development setup, checks, and model
onboarding.

## Development setup

Use uv 0.12.x from the repository root:

```bash
uv sync --frozen                              # core + development tools
uv sync --frozen --extra serve                # + websocket server
uv sync --python 3.12 --frozen --no-dev --group pi05   # pi0.5 runtime
uv sync --frozen --no-dev --group activevln   # ActiveVLN runtime
```

The checked-in `uv.lock` is the reproducible source of resolved versions. Model
profiles are dependency groups named `pi05`, `openvla-oft`, `lingbot-vla`,
`activevln`, `streamvln`, `qwen25-vln`, and `cosmos`; they are mutually exclusive, so
uv rejects combinations with incompatible Torch or Transformers requirements. Declare
new dependencies in the right group and regenerate the lock against the public index.

## Checks to run

```bash
ruff check embodiinfer tests
ruff format --check embodiinfer tests
pytest tests/ -q               # CPU unit tests
pytest tests/ -q -m gpu        # CUDA tests, on a GPU host
pytest tests/ -q -m pi05       # pi0.5 weights + lerobot, needs VVLA_PI05_CKPT
```

Tests that need CUDA or a checkpoint skip when those resources are unavailable.
Include the checks you ran, any skips, and the hardware used in your pull request.

## Engine invariants

1. Keep the engine model-agnostic. Policies declare serving, RL, CUDA-graph, and
   recurrent capabilities through the shared contracts (`ActionDecoder` /
   `RLDecoder`); do not add model-name branches to the engine.
2. Preserve numerical parity. Changes under `embodiinfer/layers/` or `embodiinfer/backend/` must
   keep the existing parity tests passing, and any intentional difference must be
   documented with the reference, versions, command, and observed numbers. An
   optimisation is lossless by default: bit-exact for engine-layer work, or
   numerically identical to `ε ~ 1e-6` with the source of the difference stated.
3. Keep the public API backward compatible, and introduce an optimisation behind an
   `EngineConfig` switch that behaves exactly as before when it is off.
4. Include a recorded run when adding a policy or backend to the support list.

## Change flow

A capability change goes through a proposal, design, implementation, parity
verification, CPU tests, a benchmark, and documentation, in that order. Add the design
document under `docs/proposals/` using
[`docs/proposals/TEMPLATE.md`](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/docs/proposals/TEMPLATE.md),
and get it reviewed before implementing. Small fixes and documentation-only changes
stay scoped to the concrete problem.

For numerical validation, fix the seed, run the same input through the
optimised and unoptimised paths, and report `max|Δaction|` against the criterion the
design chose. Record the conditions along with the result. Keep scratch scripts and
logs out of the tracked tree.

## Adding a model

A new model follows the same flow: a design document under `docs/proposals/`, then
the adapter and contract changes, then parity tooling, unit tests, a benchmark, and
documentation. Declare capabilities through the shared contracts:

- implement `VLAPolicy.encode_prefix`;
- implement the serving `ActionDecoder`;
- inherit `RLDecoder` only when the generic policy-gradient contract is met;
- declare `supports_cuda_graph` and `cuda_graph_kind` only when static shapes are
  capturable;
- declare `is_recurrent` when state crosses calls, and let the engine's
  `SessionStore` manage the transaction lifecycle.

[`CONTRIBUTING.md`](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/CONTRIBUTING.md)
describes the end-to-end verification an adapter must pass — action parity,
log-probability parity at `θ₀`, and training-curve parity against the framework's
native rollout — and the responsibility split behind it: EmbodiInfer owns rollout,
the training framework keeps the actor and learner, and `refit` carries the weights.

## Reporting issues

Use [GitHub Issues](https://github.com/BUAA-CI-LAB/EmbodiInfer/issues). Include the
dependency profile or group, the Python version, the GPU and driver with the CUDA
version, the checkpoint, the exact command, and the observed result. State whether
real weights were involved and whether the run was repeatable. Redact tokens,
addresses, private paths, and personal data.

**Do not open a public issue for a security problem.** Follow
[`SECURITY.md`](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/SECURITY.md)
instead.

## License of contributions

EmbodiInfer is licensed under Apache-2.0. See `LICENSE`, `NOTICE`, and
`THIRD_PARTY_NOTICES.md`. By submitting a contribution you agree that it is provided
under the same license. Model weights, checkpoints, datasets, and upstream model code
keep their own licenses; do not add them to this repository.
