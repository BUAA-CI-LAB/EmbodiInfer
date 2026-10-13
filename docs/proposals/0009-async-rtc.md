# 0009 — Real-Time Chunking inference

- Status: Implemented
- Author: EmbodiInfer contributors
- Date: 2026-10-14

## 1. Summary

Add opt-in Real-Time Chunking (RTC) guidance to the model-neutral decode path and
the π0.5 serving adapter. RTC overlaps inference with action execution while
conditioning each new chunk on the unexecuted portion of the previous chunk.

## 2. Motivation and current gap

Synchronous action-chunk serving pauses policy progress at every replan. Ordinary
asynchronous serving removes that pause but can introduce discontinuities when a
new chunk replaces actions that the robot has already begun to execute. RTC uses
the execution-side delay and the previous model-space chunk to preserve continuity.

## 3. Goals and non-goals

The change defines a versioned RTC request payload, model-neutral guidance math,
decoder capability checks, and session-cached π0.5 prefixes. Robot scheduling and
latency measurement remain in EmbodiRun. Checkpoint conversion and training are
outside this change.

## 4. Design

`embodiinfer.engine.async_inference` validates the optional `metadata.async.rtc`
block and builds hard-prefix or soft-guidance tensors. Flow decoders advertise RTC
support explicitly. The π0.5 serving adapter caches the full model-space chunk per
session and resolves `cache_start_index` before invoking the decoder. Requests
without RTC metadata follow the existing path.

Sending robot-space actions as the prefix was rejected because normalization and
padding differ by policy. The serving adapter instead owns the model-space cache.

## 5. Model-agnosticism verdict

The guidance contract is engine-layer and depends on the public flow-decoder
velocity interface. π0.5-specific cache ownership stays in its serving adapter.

## 6. Losslessness and precision criterion

With RTC disabled, decoder output is bit-exact with the existing path. CPU tests
compare guided velocity and prefix schedules against deterministic tensor
references. Real-weight validation compares the same checkpoint, input, seed,
noise, dtype, and denoising schedule.

## 7. Implementation plan

Add `engine/async_inference/{contracts,rtc,serving}.py`, extend decoder capability
declarations, thread optional guidance through `EngineCore`, and add the π0.5
session cache. The transport schema remains backward compatible because RTC is an
optional metadata block.

## 8. Test plan

CPU tests cover parsing, validation, prefix weights, hard-prefix behavior,
soft-guided velocity, batching, session reset, and missing-cache failures. Existing
synchronous engine and serving tests guard compatibility.

## 9. Benchmark plan

Measure synchronous, naive asynchronous, and RTC execution with the same π0.5
checkpoint, action horizon, control rate, artificial delay, hardware, and task
seeds. Report success rate, policy latency, action interval, and control overruns.

## 10. Risks and limitations

Soft RTC guidance needs autograd during denoising and cannot reuse the ordinary
captured CUDA graph. A lost session cache must fail explicitly once the execution
side reports a positive cache offset.
