# Contributing to EmbodiInfer

Thanks for your interest in EmbodiInfer. This document covers the development setup,
the engineering rules a change must respect, the flow a change follows, and how to
report problems. It is kept in English so that there is a single authoritative version.

## Code of Conduct

This project follows the
[Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating,
you are expected to uphold it. Report unacceptable behavior to
**cclonelycc@outlook.com**; reports are handled privately. For security
problems, follow [`SECURITY.md`](SECURITY.md) instead of opening a public issue.

## License of contributions

EmbodiInfer is licensed under Apache-2.0. See [`LICENSE`](LICENSE),
[`NOTICE`](NOTICE), and [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). By
submitting a pull request you agree that your contribution is provided under
the same license, as described in section 5 of the license.

Model weights, checkpoints, datasets, and upstream model code keep their own
licenses. Do not add them to this repository, and do not paste code from a model
repository without its license and attribution. The relicensing history and the
contributor-consent record are in [`docs/en/license.md`](docs/en/license.md).

## Where EmbodiInfer ends

EmbodiInfer is the inference and RL-rollout engine: it owns model execution,
batching and scheduling, the serving contract, and the RL capabilities a policy
declares. It does not own robot or simulator adapters, episode loops, task
metrics, or deployment configuration — those belong to the deployment runtime.
Integrations call the versioned HTTP or WirelessComm API instead of importing
runtime code across the boundary. See [`docs/en/architecture.md`](docs/en/architecture.md)
for the ownership boundary and the package layout.

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
`activevln`, `streamvln`, `qwen25-vln`, and `cosmos`; they are mutually
exclusive, so uv rejects combinations with incompatible Torch or Transformers
requirements. Declare new dependencies in the right group and regenerate the
lock against the public index.

## Checks to run

```bash
ruff check embodiinfer tests
ruff format --check embodiinfer tests
pytest tests/ -q               # CPU unit tests
pytest tests/ -q -m gpu        # CUDA tests, on a GPU host
pytest tests/ -q -m pi05       # pi0.5 weights + lerobot, needs VVLA_PI05_CKPT
```

Tests that need CUDA or a checkpoint skip themselves with an explicit reason.
**A skip is not a pass**, and it never counts as model verification. State in
the pull request which of the CPU, CUDA, and weight-dependent checks you
actually ran, and on what hardware.

## Engineering principles

Five rules apply to every change, not to one stage of it.

**Readability.** Match the surrounding module: comment density, naming, and idiom.
Give every public function, class, and Protocol method a type annotation and a
docstring that explains what it does and why, rather than restating the
implementation. `ruff` enforces the mechanical part — the configuration is in
`pyproject.toml`, with `line-length = 110` and the rule set `E/F/W/I/UP/B/C4/SIM` —
and both `ruff check` and `ruff format` must be clean before you commit.

**Extensibility.** New capabilities enter through the existing Protocol-plus-registry
pattern rather than a hard-coded branch. The extensible points are `AttentionBackend`
(`embodiinfer/layers/attention.py`, `register_attention`), `VLAPolicy`
(`embodiinfer/policies/factory.py`, `register_policy`), `Replica` / `Dispatcher` /
`ReplicaExecutor` (`embodiinfer/engine/parallel/data_parallel.py`), and `PrefixState`
(`embodiinfer/policies/base.py`). When you add a class of replaceable component, define the
Protocol first, then provide and register a default, and let callers ask for it by
name instead of depending on a concrete implementation.

**Maintainability.** A behavioural change comes with unit tests, and CI stays green.
An optimisation is introduced behind a switch (`EngineConfig` boolean flags) and must
behave exactly as before when the switch is off; making it the default requires
precision and benchmark evidence first. Keep the public API backward compatible:
retain the old path or document the migration. Removing a public symbol is a breaking
change and must be declared in the proposal.

**Model-agnosticism.** The engine's value is that a model is a forward unit plugged
into it while the engine owns all scheduling and optimisation. An optimisation placed
in `embodiinfer/engine/` or `embodiinfer/layers/` may depend only on the public policy protocols —
`VLAPolicy`, `PrefixState`, `flow_schedule`, `encode_prefix`, `denoise_step`,
`supports_cuda_graph`, `cuda_graph_kind`, `allocate_static_prefix`,
`copy_prefix_into` — and must never reference a specific policy's internals or branch
on a model name. Such an optimisation then applies to every flow policy
automatically. A model-specific optimisation belongs in `policies/<name>/`, must be
optional, and must have a default fallback that the engine does not depend on. Every
proposal has to say whether the feature is engine-layer (model-agnostic) or
policy-layer (model-specific); an engine-layer feature that hides an assumption about
one model is a design defect.

**Numerical discipline.** VLA models are precision-sensitive, so unless a change is
explicitly approved otherwise, an optimisation is lossless and carries a reproducible
criterion:

- **bit-exact** — `max_i ||Δa_i||_inf = 0`. For rewrites with the same mathematics and
  the same floating-point accumulation order, such as full-loop graph capture against
  step-by-step eager execution. This is the default target for engine-layer work.
- **numerically identical** — `max_i ||Δa_i||_inf <= ε`, typically `ε ~ 1e-6`, with the
  difference attributable to floating-point reordering. Using this level means stating
  the source of the difference and why it is acceptable.
- **Same precision conditions** — any comparison must use the same dtype, the same
  `torch.set_float32_matmul_precision` setting, and the same attention semantics.
  Changing the precision conditions is not a lossless optimisation; it is a separate
  proposal that needs explicit approval.

## Change flow

**0 — Proposal and scope.** Write one sentence stating the problem and the boundary:
which modules change, whether the public API is touched, and whether the change is
engine-layer. Split anything larger into several proposals. You are done when two or
three sentences cover what is being solved, where it lands, and whether it is
model-agnostic.

**1 — Design.** Add a design document under `docs/proposals/NNNN-<slug>.md` following
[`docs/proposals/TEMPLATE.md`](docs/proposals/TEMPLATE.md). It must cover the
motivation and the gap, goals and non-goals, the approach with at least one alternative
and its trade-offs, public interface changes, the model-agnosticism verdict, the
losslessness criterion, and risks and limitations. Review the design before
implementing. You are done when the design is complete, the key trade-offs are written
down, and the losslessness criterion is actionable — which script, comparing what, with
which threshold.

**2 — Implementation.** Follow the layer rules below, and rule 4 of the engineering
principles. Introduce optimisations behind a switch, off by default or on once
verified. Keep type annotations and docstrings complete and `ruff` clean.

**3 — Correctness and numerical parity.** This is what separates EmbodiInfer from a
plain inference library. The reference is a path of the same precision and semantics:
the unoptimised path, or the existing implementation. Write a reproducible parity
script that fixes the random seed (`torch.Generator`, or a fixed noise tensor), runs
the same input through both the optimised and unoptimised paths, and reports
`max|Δaction|`. Meet the criterion you chose in the principles above. Whatever can be
checked on CPU — shapes, masks, schedules, mock-policy values — is checked there;
anything needing real weights or CUDA is checked on a GPU host. You are done when the
parity result is reproducible and the criterion, the conditions (batch, dtype, steps,
precision setting), and the result are recorded together.

**4 — Unit tests.** Prefer fakes and mocks on CPU, as `tests/test_data_parallel.py`
does with its fake replica and `tests/test_grpo.py` does. Mark CUDA-dependent tests
`@pytest.mark.gpu`, and pi0.5 weight and lerobot tests `@pytest.mark.pi05` (gated by
`VVLA_PI05_CKPT`). Unmarked tests run in CI on CPU. Cover the normal path, the
boundaries (batch of one, padding, empty input), equivalence with the switch off, and
the error paths through the typed errors in `embodiinfer/exceptions.py`.

**5 — Benchmark.** Report the full conditions: batch, dtype, precision setting, step
count, scenario, and baseline. State the range where the gain holds and where it
disappears — for example "gains in the small-batch launch-bound region, converging to
about 1× when compute-bound" — without extrapolating or padding with unmeasured
numbers.

**6 — Documentation.** Update the affected docstrings and `README.md` when public usage
changes, and comment a new switch in `EngineConfig` with the design claim it serves.
Set the design document's status to `Accepted` and add an implementation note.

**7 — Review and merge.** Commit freely during development, then squash into one clean
commit. Commit messages are signed by the author alone: add no co-author and no
tool-generated attribution, and do not reference issues with `#NNN` — issue references
belong in the pull request description. Before merging, walk the completion criteria
above and the pull-request checklist.

Keep scratch material out of the tree. Parity scripts and logs live in a local
working directory, not in git.

## Documentation

User documentation has parallel English and Simplified Chinese trees: `docs/en/`
and `docs/zh/`. `mkdocs.yml` builds `docs/en/` and `mkdocs.zh.yml` builds
`docs/zh/`; the shared `docs/assets/`, `docs/stylesheets/`, and
`docs/requirements.txt` stay at the `docs/` root.

Read the Docs serves the Chinese site as `embodiinfer-zh`, a **translation** of
`embodiinfer`, from this same repository and branch. Both projects build the
root `.readthedocs.yaml`, which selects `mkdocs.zh.yml` from
`READTHEDOCS_LANGUAGE`, so no custom build configuration path is needed. Create
the project once from the dashboard, set its language to **Simplified Chinese**
(`zh-cn`), and add it as a translation of `embodiinfer`.

- **Keep both navigation trees in step.** When you add or rename a page, update
  `mkdocs.yml` and `mkdocs.zh.yml` together and translate the page in the same
  change.
- **Keep relative links relative.** The Chinese tree mirrors the English tree
  path for path, so a link such as `serving.md` resolves in both. Only external
  links stay absolute.
- **Keep cross-page anchors stable.** When a translated heading is the target of
  a `page.md#anchor` link, keep the English slug with an attr_list id, e.g.
  `## CUDA graph 捕获 {#cuda-graph-capture}`.
- **Governance and legal pages stay English.** `contributing.md`,
  `code-of-conduct.md`, and `license.md` exist only under `docs/en/`. The Chinese
  navigation links to the English pages; do not add translated copies.
- `docs/en/assets`, `docs/en/stylesheets`, `docs/zh/assets`, and
  `docs/zh/stylesheets` are symlinks to the shared `docs/assets/` and
  `docs/stylesheets/`, so both builds share one logo, favicon, and stylesheet.
  Check out with symlinks enabled.

## Releases

A consumer pins a release tag, never `main`: `main` also moves for documentation
and internal cleanup, so a pin that follows it drifts for reasons unrelated to
the code the consumer runs.

A release is a semver tag on `main` whose number matches `version` in
`pyproject.toml` and `__version__` in `embodiinfer/__init__.py`. The number is written
in both places today, so keep them equal.

```bash
git tag -a vX.Y.Z -m "vX.Y.Z"
git push origin vX.Y.Z
```

Two repositories embed this one as a submodule and install from it: EmbodiRun at
`third_party/embodiinfer`, and the organization fork of RLinf at
`third_party/embodiinfer`. Each records a commit, so a release is also the moment to
repoint both. Where a consumer only needs the package, prefer a tag dependency
over a submodule:

```bash
uv pip install "embodiinfer @ git+https://github.com/BUAA-CI-LAB/EmbodiInfer@vX.Y.Z"
```

Depending on `embodiinfer==X.Y.Z` from a package index is the end state; it would
let both consumers drop their submodule entirely.

## Where code goes

| Layer | Directory | Owns | Constraint |
|---|---|---|---|
| policy | `embodiinfer/policies/<name>/` | weight loading, the forward, collate/pad, `flow_schedule` | model-specific logic lives only here; implements the base protocols |
| engine | `embodiinfer/engine/` | scheduling, execution, CUDA graphs; `parallel/` carries both data and tensor parallelism | depends only on the public policy protocols |
| operator routing | `embodiinfer/layers/` | attention and other Protocols, registry, backend selection | no concrete operator implementations; callers depend on protocols and registered names |
| compute backend | `embodiinfer/backend/` | Torch and Triton implementations, capabilities, warmup, graph-safe execution | must not depend on models, policies, engine, or environment semantics; numbers must be comparable against a reference implementation |
| engine config | `embodiinfer/engine/config.py` | `EngineConfig` | execution switches only, each commented |
| policy config contract | `embodiinfer/policies/config.py` | `VLAPolicyConfig` | only the fields the engine reads; architecture config stays with each policy |
| rollout | `embodiinfer/engine/rollout/` (with `demo/`) | the RL rollout surface, log-probability, weight sync; `demo/` is a toy GRPO trainer and env | depends on the engine, never on a concrete policy |
| serve | `embodiinfer/engine/serve/` | the inference API and HTTP/websocket frontends | model-neutral communication and engine calls only; simulator and robot protocols belong to the deployment runtime |
| errors | `embodiinfer/exceptions.py` | typed errors under a `VvlaError` base | user-facing errors go here |

Environment connectors, the episode execution loop, and metrics such as SR/SPL belong
to the downstream deployment runtime and must not enter the inference package.

## Adding a model

A new model follows the same flow as any other capability: a design document under
`docs/proposals/`, then the adapter and contract changes, then parity tooling, unit
tests, a benchmark, and documentation. Declare capabilities through the shared
contracts rather than by asking the engine to special-case anything:

- implement `VLAPolicy.encode_prefix`;
- implement the serving `ActionDecoder`;
- inherit `RLDecoder` only when the generic policy-gradient contract is met;
- declare `supports_cuda_graph` and `cuda_graph_kind` only when static shapes are
  capturable;
- declare `is_recurrent` when state crosses calls, and let the engine's
  `SessionStore` manage the transaction lifecycle.

### Verification through an RL framework

Verification happens end to end inside the training pipeline (LIBERO, ManiSkill, or
RoboTwin), compared against the framework's native rollout. The responsibility split is
fixed:

| Responsibility | Owner | Reason |
|---|---|---|
| rollout (action generation) | **EmbodiInfer** | replaces the naive HuggingFace forward with optimised inference: denoising CUDA graphs, prefix K/V reuse, cross-stream parallelism, data parallelism |
| actor and learner (gradients, optimizer, PPO/GRPO loss) | **the training framework** | EmbodiInfer contains no trainer |
| weight transport | `refit` | after each optimizer step the actor's weights move to the rollout replica in place; framework-specific name mapping stays in the framework adapter |

The mapping onto the public interface is:

| Framework policy method | EmbodiInfer surface | Purpose |
|---|---|---|
| deterministic `predict_action_batch` | `generate` → `EngineCore.execute` | serving path with denoising graphs and prefix K/V reuse |
| sampling `predict_action_batch` | `sample_group` / `generate_with_logprob` | group sampling: one prefix `expand(G)` broadcast to the group, plus behavior log-probability and the trajectory |
| weight sync (actor to rollout) | `refit`, or `refit_state_dict` + `commit_refit` for zero-copy | in-place update after each optimizer step |
| differentiable policy-gradient log-probability | `flow_logprob_recompute` (optional) | leave the framework's native path by default; use this only if EmbodiInfer takes over training-side re-scoring |
| value (PPO critic) | — | provided by the training framework; GRPO, being group-relative, needs no critic |
| training forward | — | stays with the framework; the adapter covers rollout only |

Three criteria decide whether the integration is correct:

1. **Action parity.** With the same weights, observation, and initial noise or seed,
   `max|Δa|` between the EmbodiInfer rollout and the native rollout. With the same
   environment, dtype, and attention semantics, the target is bit-exactness or `<= ε`.
   A cross-implementation residual — the framework's HuggingFace forward against
   EmbodiInfer's own forward — stays inside the noise floor, at the order of the bf16
   arithmetic difference; state the threshold and its precision setting in the proposal.
2. **Log-probability parity.** The behavior log-probability and the recomputed
   log-probability at `θ = θ_behavior` agree, so the importance ratio is approximately
   1. This protects on-policy correctness and is not satisfied by matching the action
   tensor alone. Check that both sides agree on the SDE noise scale, the step count, and
   the schedule definition.
3. **Training-curve parity.** Reward and success rate track the native baseline within
   run-to-run variance. RL is stochastic, so this is the final "within noise" verdict.

Report performance the same way as any other benchmark: rollout throughput or wall
clock per iteration, the same environment, model, and hardware, and the full conditions
including where the gain disappears.

## Before opening a pull request

1. Run the CPU suite and `ruff`; both must be clean.
2. Keep the engine model-agnostic. Policies declare serving, RL, CUDA-graph, and
   recurrent capabilities through the shared contracts (`ActionDecoder` /
   `RLDecoder`); do not add model-name branches to the engine.
3. Preserve numerical parity. Changes under `embodiinfer/layers/` or `embodiinfer/backend/`
   must keep the existing parity tests passing, and any intentional difference
   must be documented with the reference, versions, command, and observed
   numbers.
4. Do not mark a policy or backend verified, supported, or end-to-end without a
   recorded run. Follow the change flow above: a design document under
   `docs/proposals/`, then the adapter and contract changes, then parity
   tooling, unit tests, a benchmark, and documentation.
5. Do not commit weights, checkpoints, datasets, benchmark outputs, recordings,
   credentials, addresses, private paths, or machine-specific operational
   scripts.

## Reporting issues

Use [GitHub Issues](https://github.com/BUAA-CI-LAB/EmbodiInfer/issues). Include
the revision, the dependency profile or group, the Python version, the GPU and
driver with the CUDA version, the checkpoint, the exact command, and the
observed result. State whether real weights were involved and whether the run
was repeatable. Redact tokens, addresses, private paths, and personal data.

**Do not open a public issue for a security problem.** Follow
[`SECURITY.md`](SECURITY.md) instead.
