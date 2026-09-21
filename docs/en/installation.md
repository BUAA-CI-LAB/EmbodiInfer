# Installation

Install EmbodiInfer from source, then select a dependency profile for your model.
The distribution is named `embodiinfer`; Python imports use `embodiinfer`.

## Prerequisites

- uv 0.12.x, run from the repository root.
- The core package supports Python 3.10 and newer. The `pi05` group requires
  Python 3.12 or newer; select the interpreter for a profile with `--python`.
- The checked-in `uv.lock` is the reproducible source of resolved versions. On
  Linux, Torch and torchvision are locked to the official CUDA 13.0 (`cu130`)
  index; other platforms fall back to PyPI builds.

## Install with uv

Use uv 0.12.x from the repository root:

```bash
uv sync --frozen                              # core + development tools
uv sync --frozen --extra serve                # + websocket server
uv sync --frozen --extra wireless             # + WirelessComm policy server
uv sync --python 3.12 --frozen --no-dev --group pi05  # pi0.5 runtime
uv sync --frozen --no-dev --group activevln   # ActiveVLN runtime
uv sync --frozen --no-dev --group qwen25-vln  # Qwen2.5-VL navigation runtime
```

## Capability groups

Runtime profiles are dependency groups named `pi05`, `openvla-oft`, `lingbot-vla`,
`activevln`, `streamvln`, `qwen25-vln`, and `cosmos`; add `--extra serve` when a
profile also needs the websocket server.

- `pi05` — pi0.5 runtime. Requires Python 3.12 or newer.
- `pi05-openpi` — preparation and tokenizer helpers for OpenPI PyTorch or Orbax
  checkpoints; add it on top of `pi05`.
- `openvla-oft`, `lingbot-vla`, `activevln`, `streamvln`, `qwen25-vln`, `cosmos` —
  runtime profiles for the matching model families.
- `dm05` — an empty capability selector, not an OpenDM installer.

The profiles are mutually exclusive, so uv rejects attempts to combine model stacks
with incompatible Torch or Transformers requirements. The core package continues to
support Python 3.10 and newer.

## Extras

- `serve` — the websocket server.
- `wireless` — the WirelessComm policy server.

HTTP serving is included in the core package. For either transport, install
the selected model's dependencies and prepare its checkpoint.

## CUDA 13 lock

The uv resolver keeps `torch<2.11` for the CUDA 13.0 lock. The four offline benchmark
directories instead retain their verified Torch 2.13 CUDA 13.2 (Thor) / CUDA 12.9
(4090) environments, prepared with their own `setup_env.py` and requirements files.
`--group benchmark` adds the public-data reader dependencies to a uv environment.

## OpenPI and Orbax checkpoint helper

For OpenPI PyTorch or Orbax checkpoints, add the preparation/tokenizer helpers to
the PI0.5 runtime:

```bash
uv sync --python 3.12 --frozen --no-dev --group pi05 --group pi05-openpi
```

## GR00T reference

Generate GR00T parity references in the upstream NVIDIA environment. See the
[GR00T benchmark guide](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/benchmarks/gr00t-benchmark/README.md)
for setup and reference generation.

## DM0.5 and OpenDM

DM0.5 requires a dedicated OpenDM environment because its Torch and Transformers
stack conflicts with other adapters. In an already prepared OpenDM environment, install
this project without replacing that environment's dependencies:

```bash
uv pip install --python /path/to/opendm-env/bin/python -e . --no-deps
```

Hosts under the deployment runtime (EmbodiRun) must provision OpenDM through the
model's `environment_packages` overlay (a control-node-independent installable OpenDM
checkout/package): the host performs an exact core sync before installing that
overlay. Put extra dependencies in that overlay so they survive environment syncs.

## Verify the install

The launchers `embodiinfer-http-serve` (or `embodiinfer-serve`) and
`embodiinfer-wireless-serve` are installed with the project. Model-specific serving
needs the matching capability group and a checkpoint.

The synthetic `mock_flow_vla` policy ships as a CPU test fixture (no weights), so the
engine, CUDA-graph, data-parallel, and rollout paths can be exercised without a
checkpoint. The generic mock benchmark runs that fixture:

```bash
python benchmarks/benchmark.py --preset small --sweep   # hf vs embodiinfer-eager vs embodiinfer-graph
```

The benchmark compares batch-one eager, batched eager, and batched CUDA-graph
execution. It selects CUDA when available and otherwise runs on CPU.
