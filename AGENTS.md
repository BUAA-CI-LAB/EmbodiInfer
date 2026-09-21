# EmbodiInfer

## Repository purpose

This repository provides EmbodiInfer (Python package `embodiinfer`), the inference and
RL-rollout runtime for embodied vision-language-action and world-action models.
It owns checkpoint loading, policy
adapters, tensor/model execution, recurrent model state, batching, parallelism,
rollout/refit, and model-neutral serving.

It does not own robot or simulator communication, environment adapters, episode
lifecycle, task metrics, physical safety, or deployment orchestration. Those belong
in the downstream EmbodiRun repository and communicate with EmbodiInfer through stable
contracts. Do not reintroduce Habitat/evaluation loops or trainer-specific adapters
into this package.

## Source layout and ownership

- `embodiinfer/engine/`: model-neutral execution and runtime capabilities. Keep small,
  central mechanisms as focused modules, but give a substantial, cohesive feature
  domain its own subpackage when it has multiple cooperating modules, contracts,
  or lifecycle concerns. `parallel/`, `rollout/`, and `serve/` illustrate the
  intended scale; they are examples rather than a closed list, and comparable
  future runtime capabilities may add sibling subpackages. Do not create a
  subpackage for a single helper or force new functionality into today's folders
  merely because they already exist.
- `embodiinfer/policies/`: public policy/decoder contracts, registry, checkpoint-facing
  adapters, processors, and model-specific inference behavior.
- `embodiinfer/models/`: reusable model components that are independent of engine
  scheduling and environment semantics.
- `embodiinfer/layers/`: operator protocols, registries, and backend routing only.
- `embodiinfer/backend/`: concrete Torch/Triton implementations, capability probes,
  warmup, and graph-safe execution. Backend code must not depend on `models`,
  `policies`, `engine`, or environment semantics.
- `tests/`: CPU unit tests plus explicitly marked CUDA/checkpoint parity tests.
- `docs/`: maintained architecture, development, model, and proposal documents.
  Treat this as a curated set of canonical references, not a place for one-off
  plans, progress notes, debugging records, validation logs, or duplicate guides.
- `examples/`: small, stable demos of supported public workflows. Prefer updating
  an existing example; add one only when a genuinely distinct user-facing workflow
  cannot be demonstrated clearly by the current set. Examples are not formal
  benchmarks or developer setup automation.
- `scripts/`: developer automation for setup and artifact preparation, including
  dependency/checkpoint preparation, source-pin verification, and reference-data
  production that may need model-specific isolated environments. Scripts are not
  public APIs and must not be used to publish performance claims.
- `benchmarks/`: formal, reproducible performance or accuracy measurement programs
  and their reports. Put benchmark-specific dataset preparation here, record all
  experimental conditions, and keep measurement logic independent from demos and
  environment-installation helpers.

## Architectural invariants

- Keep the engine model-agnostic. Core scheduling must depend on public policy
  protocols and capability declarations, never model-name branches or private
  policy fields.
- Put model-specific loading, preprocessing, decoding, and fused input-layout
  behavior under the owning policy. Keep reusable neural components under
  `models/`.
- Add replaceable capabilities through the existing Protocol and registry
  patterns. Do not make callers depend directly on a concrete backend when a
  registry contract already exists.
- A decoder must advertise only capabilities it actually implements. In
  particular, do not expose `RLDecoder` behavior for planning-only or recurrent
  policies that cannot provide the generic policy-gradient contract.
- Preserve recurrent-session transactions: explicit `SessionKey`, checkout,
  commit on success, rollback on failure, and reset/cancel invalidation. Do not
  update weights while committed or in-flight recurrent state belongs to an old
  policy version.
- Preserve episode affinity in data parallelism. Recurrent state is replica-local
  and must not migrate or retry on a different replica after failure.
- Keep public imports coherent with the current layout under `embodiinfer.engine.*`,
  `embodiinfer.policies.*`, and `embodiinfer.backend.*`. When changing a public path, either
  retain a compatibility export or document the migration explicitly.

## Correctness and numerical discipline

- Treat inference optimizations as lossless unless the task explicitly approves
  another accuracy contract. Bit-exact action parity is the default target when
  the mathematical operation and accumulation order are unchanged.
- For numerically equivalent changes, state the reference path, dtype, attention
  semantics, seed/noise, comparison quantity, tolerance, and reason for the
  expected floating-point difference. Do not loosen an assertion merely to make
  a test pass.
- Compare optimized and reference paths with the same random generator,
  `torch.set_float32_matmul_precision` setting, batch shape, schedule, and number
  of decode/denoise steps.
- Validate failure behavior as well as successful output: malformed observations,
  unsupported capabilities, cancelled sessions, stale leases, replica failure,
  and incompatible graph shapes should fail explicitly.
- Performance claims must identify hardware, dtype, batch size, warmup, measured
  iterations, model/checkpoint, runtime switches, and baseline. Never generalize
  a synthetic or batch-one result beyond the measured conditions.

## Development workflow

- For a non-trivial new capability, optimization, policy adapter, or public API
  change, add or update a proposal in `docs/proposals/` following
  [`docs/proposals/TEMPLATE.md`](docs/proposals/TEMPLATE.md) and the change flow in
  `CONTRIBUTING.md`. Small fixes and documentation-only changes should stay
  scoped to the concrete problem.
- Before creating any documentation file, search `README.md` and the existing
  files under `docs/`, then update the narrowest canonical document that already
  owns the subject. Add a new document only for a distinct, durable contract or
  design artifact that the task explicitly requires and that no existing document
  can absorb cleanly. Prefer updating an
  existing related proposal over creating another one. Keep temporary analysis,
  implementation diaries, and raw validation output out of the tracked tree.
- Place the change in the narrowest owning layer and avoid unrelated refactors.
  Engine changes must explain why they are model-independent; policy changes must
  keep model assumptions local.
- Public functions, classes, dataclasses, and Protocol methods require useful type
  annotations and docstrings that explain the contract rather than restating the
  implementation.
- Keep new dependencies optional unless every supported policy needs them. Update
  `pyproject.toml` and installation documentation together when dependency
  behavior changes.
- Route new executable tooling by purpose: user-facing demonstrations belong in
  `examples/`, environment or artifact preparation belongs in `scripts/`, and
  formal measurement belongs in `benchmarks/`. Do not duplicate one workflow
  across these directories.

## Environments and dependencies

- Support Python 3.10 and newer.
- Core development setup: `uv sync --frozen`.
- Model dependency groups are intentionally isolated. In particular, pi0.5 and Qwen2.5-VL
  profiles require incompatible Transformers versions; do not install conflicting
  groups into one environment or silently relax their pins.
- Official GR00T, LingBot-VLA, Cosmos, and other reference packages may be used in
  separate reference-generation environments. Do not add them as EmbodiInfer runtime
  imports when the adapter is designed to be self-hosted.
- Do not download large checkpoints or run GPU-heavy benchmarks unless the task
  requires it and the environment is prepared. Do not claim real-weight or GPU
  validation when only mock/CPU tests ran.

## Verification

Run the narrowest relevant checks first, then the broader applicable suite:

```bash
python -m pytest tests/<relevant_test_file>.py -q
python -m pytest tests/ -q
ruff check embodiinfer tests
ruff format --check embodiinfer tests
```

- Prefer deterministic CPU tests with mock policies or fake replicas for engine
  logic and error paths.
- Mark CUDA-only tests with `gpu` and checkpoint-specific tests with the existing
  policy marker. Keep optional tests skipped cleanly when their hardware,
  checkpoint, or dependency is absent.
- Run real-weight parity only in the matching isolated environment and record the
  exact checkpoint/revision and numerical threshold.
- Documentation-only changes do not require the model test suite, but still run
  `git diff --check` and verify commands, paths, and public examples against the
  current tree.

## Code review rules

Review changes first for:

1. Violations of inference/deploy, engine/policy, layers/backend, or
   runtime/trainer boundaries.
2. Hidden model assumptions in generic scheduling or incorrect decoder capability
   declarations.
3. Numerical drift, changed sampling/log-probability semantics, graph replay using
   stale buffers, or refit/session version mismatches.
4. Concurrency bugs in async batching, session transactions, replica affinity, and
   cancellation/error cleanup.
5. Missing regression tests or performance claims without reproducible conditions.

Report concrete behavioral risks and cite exact files and lines. Do not block on
style preferences already enforced by Ruff.

## Git conventions

- Do not create a commit without explicit user approval; always ask before running
  `git commit`.
- Preserve unrelated user changes and do not rewrite history unless explicitly
  requested.
- Keep commits focused. Commit messages use the repository's conventional prefixes
  such as `feat:`, `fix:`, `refactor:`, `perf:`, `test:`, and `docs:`.
- Commit messages contain only the human author's attribution: do not add AI/tool
  signatures or collaborator trailers, and do not include `#<number>` issue
  references in the commit message.
