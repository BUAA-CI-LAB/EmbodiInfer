# Quick start

Start with a synthetic policy to check the engine API, then choose a real model
and its dependency profile. See [Installation](installation.md) for platform
requirements and isolated environments.

## Install and check the engine

From a new checkout:

```bash
git clone https://github.com/BUAA-CI-LAB/EmbodiInfer.git
cd EmbodiInfer
uv sync --frozen
uv run python examples/quickstart.py
```

The example uses `mock_flow_vla`, a synthetic policy with no checkpoint.
It prints one action-chunk shape, followed by eight chunks from a batched
request. It selects CUDA when available and otherwise runs on CPU.

## Understand the Python interface

`Vvla.act(observation)` returns an action chunk. Passing a list of observations
returns a list of chunks. Each `Observation` carries images, state, and language
inputs; the selected policy defines their layout and preprocessing.

The [complete example](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/examples/quickstart.py)
constructs inputs from the synthetic policy's configuration. For a real policy,
use its checkpoint's camera mapping, normalization, and tokenizer.

For multiple clients, the π0.5 HTTP and WirelessComm servers can collect
individual requests into a batch. See [Serving](serving.md#share-one-service-across-clients)
for the batch size and collection-window settings.

## Load a real checkpoint

For π0.5, use Python 3.12+, a CUDA GPU, and the dedicated dependency profile:

```bash
uv sync --python 3.12 --frozen --no-dev --group pi05
uv run --no-sync python examples/pi05_inference.py \
  --ckpt lerobot/pi05_base --envs 1
```

The checkpoint must be accessible locally or through its model hub.
The example loads the weights, runs inference on synthetic observations, and
prints the resulting action chunks.

To use real observations, either integrate the checkpoint's preprocessing in
your Python application or configure a model service using the
[serving guide](serving.md). Existing LeRobot applications can use the
[π0.5 adapter](api.md#lerobot-pi05-adapter).

## Next steps

- [Models](models.md): choose a policy and read its input and session requirements.
- [Serving](serving.md): launch HTTP or WirelessComm inference.
- [Parallelism](parallelism.md): configure replicas and supported multi-GPU paths.
- [RL rollout](api.md#rl-rollout): generate log probabilities and update weights.
- [Recurrent sessions](api.md#activevln-sessions): manage episode-specific state.
