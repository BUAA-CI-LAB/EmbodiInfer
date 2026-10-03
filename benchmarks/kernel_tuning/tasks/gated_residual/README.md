# Gated residual BF16 task

This self-contained example covers the multiply-then-add semantics used by
`embodiinfer/backend/triton/norm.py::gated_residual`. Inputs are contiguous BF16
`residual, update: [B,T,H]` and `gate: [B,1,H]` or `[B,T,H]`. Output has the same
shape and dtype as residual. Round the product to BF16 before adding; contracting
the expression into an FMA can violate the bit-exact contract. Inputs are immutable.

The mathematical reference is embedded in `definition.json`. `baseline.py` is a
separate eager Torch implementation, written for this task under the repository's
Apache-2.0 license. It is an executable template baseline, **not a claim to beat
EmbodiInfer's existing Triton kernel**. Replace it with the best-known standalone
implementation before using this task to claim a production improvement. Include
all local helpers inside the task package and record their provenance here.

The three synthetic workloads exercise broadcast gates, per-token gates, and a
non-power-of-two hidden dimension. They do not represent measured model traffic.
Use workload shapes and weights from the intended deployment for real tuning.

On the target machine, prepare its compatible Torch/CUDA/Triton environment and
set `evaluator_python` in `tuning.yaml` to that environment's absolute Python path.
The default evaluator also needs `benchmarks/kernel_tuning/requirements.txt`.
Select `language: cuda` for a CUDA C++ search; the agent then supplies a CUDA
Solution with a supported Torch or TVM-FFI binding. The CUDA toolkit/compiler must
already be available. `target_hardware` is a Solution label, not a capability check;
the evaluator probes the actual device (including Thor) at run time.

From the repository root, in the optional Humanize2 tooling environment:

```bash
python -m scripts.kernel_tuning check benchmarks/kernel_tuning/tasks/gated_residual
python -m scripts.kernel_tuning check benchmarks/kernel_tuning/tasks/gated_residual --environment
python -m scripts.kernel_tuning run benchmarks/kernel_tuning/tasks/gated_residual --agent 'HARNESS/MODEL:EFFORT'
python -m scripts.kernel_tuning status results/kernel_tuning/RUN_ID
python -m scripts.kernel_tuning resume results/kernel_tuning/RUN_ID
python -m scripts.kernel_tuning export results/kernel_tuning/RUN_ID results/exported_gated_residual
```

Replace the agent placeholder with a configured Humanize2 agent specification.
`check` performs static validation. `--environment` additionally imports the
evaluator and probes its hardware/dependencies. `run` and `resume` execute real
coding-agent turns and measurements. Each candidate's `evaluation/command.json`
records the exact reproducible worker command and timeout. Fixed conditions,
paired measurements, and acceptance limits are in `tuning.yaml`.

For the optional custom `benchmark.py` interface and artifact contract, see
[`0008-kernel-tuning.md`](../../../../docs/proposals/0008-kernel-tuning.md).
