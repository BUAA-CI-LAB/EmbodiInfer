<!--
Thanks for contributing. Keep this description short and concrete.
-->

## Summary

<!-- What changes, and why. Link the issue if there is one. -->

## Type of change

- [ ] Bug fix
- [ ] New model adapter or policy-contract change
- [ ] Engine, scheduler, session, or rollout change
- [ ] Kernel, layer, or numerical change
- [ ] Documentation
- [ ] Build, dependency group, or lock-file change

## Verification

Paste the exact commands you ran and their result. State which checks you did
**not** run (CUDA, real weights) rather than implying they pass.

```text
$ ruff check embodiinfer tests
$ ruff format --check embodiinfer tests
$ pytest tests/ -q
```

- Hardware: <!-- CPU only, or GPU model and count, driver and CUDA version -->
- Weights: <!-- none, or which checkpoint and profile group -->
- Parity: <!-- reference, command, and observed numbers, or "not applicable" -->

## Engine invariants

- [ ] No model-name branch was added to the engine; capabilities are declared
      through `ActionDecoder` / `RLDecoder` and the shared metadata.
- [ ] Parity tests under the affected layers or backend still pass, or the
      intentional difference is documented with numbers.
- [ ] New dependencies are declared in the correct profile group, and
      `uv.lock` is consistent.
- [ ] No deployment-runtime code was imported across the boundary; integrations
      use the versioned API.

## Checklist

- [ ] New models follow the flow in
      [`CONTRIBUTING.md`](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/CONTRIBUTING.md)
      and have a design document under `docs/proposals/`.
- [ ] Documentation reflects the change, and parity or runtime numbers in
      [`docs/en/benchmark.md`](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/docs/en/benchmark.md)
      are updated when they change.
- [ ] No checkpoints, datasets, benchmark outputs, recordings, credentials,
      addresses, private paths, or machine-specific scripts are included.
- [ ] I agree that this contribution is licensed under Apache-2.0, per
      [`CONTRIBUTING.md`](https://github.com/BUAA-CI-LAB/EmbodiInfer/blob/main/CONTRIBUTING.md).
