# V1 inference-engine scope and follow-up work

EmbodiInfer owns model execution, batching, scheduling, and the serving contract. EmbodiRun owns devices, tasks, action execution, and deployment lifecycle. The two projects connect through the public serving API; robot control does not belong in the inference engine.

## Delivered

- RTC and VLASH asynchronous inference interfaces and serving integration.
- π0.5 cross-session batching documentation: batch defaults to 1 and can be enabled with `--max-batch`.
- Model, serving, performance, and bilingual documentation entry points.
- Explicit CUDA and checkpoint gates; a skipped test is not model verification.

## Pending

- Reproducible raw benchmarks across Jetson AGX, Jetson Thor, and 4090.
- Operator-optimization results for π0.5, ActiveVLN, and other supported models.
- A tool that collects hardware facts, searches model/device candidates, proposes operators, and emits a report for a new target.
- Deployment-cost, resource-usage, and multi-node scaling comparisons with EmbodiRun.

These are roadmap items. No performance claim should be invented without the checkpoint, target hardware, and raw measurements.
