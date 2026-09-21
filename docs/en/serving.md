# Serving

EmbodiInfer exposes model-agnostic inference over HTTP and, optionally, over
WirelessComm. The HTTP and WirelessComm policy servers share the same session,
ordering and idempotency service.

## HTTP server

Install the selected [model profile](installation.md) and make its checkpoint
available first. For π0.5, create an adapter JSON matching the checkpoint's
state vector and camera features; see [Adapter JSON](#adapter-json).

From that environment, start `embodiinfer-http-serve` (or `embodiinfer-serve`):

```bash
uv run --no-sync embodiinfer-http-serve \
  --policy pi05 \
  --checkpoint /models/pi05-checkpoint \
  --adapter-config ./adapter-config.json \
  --host 127.0.0.1 --port 8000
```

Replace both paths with your checkpoint and adapter file. In another terminal:

```bash
curl --fail http://127.0.0.1:8000/healthz
curl --fail http://127.0.0.1:8000/v1/capabilities
```

Health reports the service status; capabilities describe the loaded adapter.
Send an observation next to check your camera and state mappings.
The command above listens locally. For a remote Control node, select a reachable
bind address and configure access appropriate to your network.

## Send your first observation

Keep the HTTP server running. Prepare real JPEG/PNG camera frames and a
`state.json` file containing the named state fields expected by its adapter.
For the two-camera mapping in [Adapter JSON](#adapter-json), the state object
has the form `{"observation.state": [...]}`: replace the array with your
checkpoint's full state vector in the correct order and units.

```bash
uv run --no-sync python examples/http_client.py \
  --endpoint http://127.0.0.1:8000 \
  --state ./state.json \
  --image observation.images.front=./front.jpg \
  --image observation.images.wrist=./wrist.jpg \
  --instruction "Pick up the cube and place it in the bowl."
```

The [client example](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/examples/http_client.py)
reads capabilities, opens a session, sends step zero as multipart data with
its idempotency header, prints the JSON response, and closes the session.
Response fields include `schema: vvla.policy.step.result.v1`, `step_id: 0`,
`session_id`, `action_space`, `actions`, and `timing`; action dimensions depend
on the checkpoint.

For an authenticated server, add `--token-env EMBODIINFER_TOKEN` after setting
that variable locally. Use the same token on the server. If an HTTP request
fails, inspect the server log and response status: 401 indicates authentication,
400 a malformed request, 409 session ordering or idempotency, and 422 an adapter
input error. The example does not automatically retry requests.

This example is a single-step client. A recurrent application must retain its
session and increment `step_id` across observations instead of launching the
example anew for every frame.

## Server options

Model and engine flags shared by both transports:

| Flag | Meaning |
|---|---|
| `--policy` | Policy name; defaults to `pi05`. |
| `--checkpoint` | Checkpoint path or hub id. |
| `--adapter-config` | JSON file containing the adapter config. |
| `--device` | Execution device; defaults to `cuda`. |
| `--dtype` | Execution dtype; defaults to `auto`. |
| `--max-batch` | Maximum requests per batch; defaults to `1`. π0.5 supports cross-session batching. |
| `--max-wait-ms` | Batch collection window in milliseconds; defaults to `5`. |
| `--num-steps` | Decode-step override. |
| `--no-cuda-graph` | Disable CUDA-graph capture. |
| `--capture-full-loop` | Capture the full decode loop instead of per-step. |

HTTP-only flags include `--host` (default `0.0.0.0`), `--port` (default `8000`),
`--token` (optional Bearer token), `--max-body-bytes`, `--max-image-bytes`,
`--max-images`, `--max-sessions`, and `--idempotency-cache-size`.

## Share one service across clients

For three independent π0.5 sessions, add `--max-batch 3 --max-wait-ms 5` to
either server command. Requests ready within the collection window execute in
one tensor batch; a partially filled window runs without waiting for missing
clients. The window bounds collection time, not time waiting behind model work.

Each session still has ordered steps and its own idempotency cache. Results
return to their originating session, and checkpoint action restoration uses
each request's state. A malformed observation fails its row without discarding
valid peers. The pending queue is bounded to 1,024 requests and returns
`429 queue_full` when full.

Batching is opt-in; `--max-batch 1` retains direct single-request execution.
Adapters without the batch protocol reject larger values at startup. Recurrent
adapters retain their existing single-session execution limits. CUDA graphs
are cached by batch bucket up to the configured maximum; `--max-batch 3`
uses a three-row graph for a full batch.

## WirelessComm server

To serve over a statically configured WirelessComm node:

```bash
embodiinfer-wireless-serve \
  --policy pi05 \
  --checkpoint <pi05-checkpoint> \
  --adapter-config <adapter-config.json> \
  --comm-config configs/wireless.example.yaml \
  --token <shared-token>
```

The node YAML follows WirelessComm's `local`/`peers`/`comm` schema. Pacing rates are
deployment-specific and are not enabled implicitly. The current transport is intended
for trusted isolated networks; its application token is authorization, not link
encryption or peer authentication.

WirelessComm-only flags include `--comm-config` (required), `--token`, and
`--max-in-flight`; `--max-image-bytes`, `--max-images`, `--max-sessions`, and
`--idempotency-cache-size` are shared limits.

## Sessions, ordering, and idempotency

Both transports share one transport-neutral session and step service.

- A session is opened with the session schema `vvla.policy.session.v1`; the server
  assigns a `session_id` and starts `session_revision` at 0. Past `--max-sessions`
  it returns `429 too_many_sessions`.
- Each step carries a `request_id` and a `step_id`. Steps within one session are
  serialized.
- `step_id` must equal the next expected value. A value below it returns
  `409 step_id_too_old`; a gap or future value returns `409 out_of_order_step`.
- Repeating a `request_id` with identical content returns the response while it
  remains cached. The response cache is bounded; an evicted, already committed
  step is rejected rather than recomputed.
  Reusing it with different content returns `409 idempotency_conflict`. A
  `request_id` that belongs to an earlier session revision returns
  `409 stale_idempotency_key`.
- `reset` requires the `Idempotency-Key` header to equal `request_id`. It clears
  cached step responses, sets `next_step_id` back to 0, and increments
  `session_revision`.
- An unknown or closed `session_id` returns `404 session_not_found`.
- A step response uses the schema `vvla.policy.step.result.v1` and reports
  `session_revision`, `action_space`, `actions`, `timing`, and `policy_revision`.

## Adapter JSON

`--adapter-config` points at a JSON object. Here is a π0.5 example with one
state vector and two cameras:

```json
{
  "state_fields": ["observation.state"],
  "image_fields": ["observation.images.front", "observation.images.wrist"],
  "image_keys": {
    "observation.images.front": "observation.images.image",
    "observation.images.wrist": "observation.images.image2"
  }
}
```

Adjust the feature names and state ordering to match your checkpoint.
EmbodiRun normally generates this file
from its binding and deployment configuration.

- `state_fields` — names of fields in the request's `state` object. For example,
  `"observation.state"` reads `state["observation.state"]`.
- `image_fields` — lists incoming camera names in the desired input order.
- `image_keys` — optional object mapping those names to checkpoint image features;
  for example, `"observation.images.wrist_image": "observation.images.image2"`.
  Unmapped names retain their original names. Mapping sources must appear in
  `image_fields`, and targets must exist in the checkpoint. Duplicate incoming camera
  names and multiple cameras mapped to the same feature are rejected.
- `policy_kwargs` — model construction options passed to the policy factory. HTTP
  serving of DM0.5 requires `policy_kwargs.is_history: true`.

## Registering EmbodiInfer as a rollout backend

The fork [BUAA-CI-LAB/RLinf](https://github.com/BUAA-CI-LAB/RLinf/tree/vvla-rollout-backend)
registers EmbodiInfer as a rollout backend on its `vvla-rollout-backend` branch
(`rollout.model.model_type` = `embodiinfer` / `vvla_gr00t` /
`vvla_openvla_oft` / `vvla_lingbotvla`): EmbodiInfer serves rollout inside PPO/GRPO,
the actor uses the native model. The adapters use EmbodiInfer's policy factory
and refit API for model construction and weight updates. The integrations cover
π0.5, GR00T, and OpenVLA-OFT on LIBERO, and LingBot-VLA on RoboTwin.

See [refit integration patterns](api.md#zero-copy-refit) for the weight-update lifecycle.
