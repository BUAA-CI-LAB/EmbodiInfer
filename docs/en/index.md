<div class="hero" markdown>

<h1 class="hero-title">
  <img src="assets/logo.svg" alt="EmbodiInfer" class="hero-logo">
</h1>

**An inference and RL-rollout engine for embodied models.**

Run a policy locally, serve predictions, or integrate with an RL trainer.

[Installation](installation.md){ .md-button .md-button--primary }
[Quick start](quickstart.md){ .md-button }
[GitHub](https://github.com/BUAA-CI-LAB/EmbodiInfer){ .md-button }

</div>

EmbodiInfer runs vision-language-action policies, world-action models, and
recurrent navigation policies. It provides model loading, batched execution,
CUDA graph acceleration, and recurrent state management through Python and
network interfaces. Available optimizations and rollout features depend on the
selected policy.

It is the sibling of [EmbodiRun](https://github.com/BUAA-CI-LAB/EmbodiRun), the
deployment and execution runtime, and can also be used on its own.

## Choose a path

<div class="grid cards" markdown>

-   :material-download:{ .lg .middle } __Install and run a model__

    ---

    Install with uv, try the Python API, and load your first checkpoint.

    [:octicons-arrow-right-24: Installation](installation.md)

    [:octicons-arrow-right-24: Quick start](quickstart.md)

-   :material-server-network:{ .lg .middle } __Serve or train__

    ---

    Launch an inference service, distribute requests across GPUs, or connect
    an RL trainer.

    [:octicons-arrow-right-24: Serving](serving.md)

    [:octicons-arrow-right-24: Parallelism](parallelism.md)

-   :material-chart-line:{ .lg .middle } __Read the measurements__

    ---

    Compare model latency, engine optimizations, and RL rollout timings.

    [:octicons-arrow-right-24: Benchmark](benchmark.md)

    [:octicons-arrow-right-24: Supported models](models.md)

-   :material-sitemap:{ .lg .middle } __Understand the design__

    ---

    Follow a request through preprocessing, model execution, and decoding.

    [:octicons-arrow-right-24: Architecture](architecture.md)

    [:octicons-arrow-right-24: Python API](api.md)

</div>

## Choose a model and runtime

The [model reference](models.md) lists checkpoints, dependency profiles, and
available optimizations. π0.5 can batch requests from multiple clients through
either HTTP or WirelessComm; see [shared-service setup](serving.md#share-one-service-across-clients).
Recurrent navigation policies keep state across steps in an episode.

For robot and simulator deployments, pair the inference service with
[EmbodiRun](https://embodirun.readthedocs.io/).

## Community

- [Contributing](contributing.md) — development setup, tests, and pull requests.
- [Code of Conduct](code-of-conduct.md) — the Contributor Covenant 2.1 adopted
  by this project.
- [Security policy](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/SECURITY.md)
  — how to report a vulnerability privately.
- [License](license.md) — Apache-2.0 and third-party notices.

The repository README is available in
[English](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/README.md) and
[简体中文](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/README.zh-CN.md).
