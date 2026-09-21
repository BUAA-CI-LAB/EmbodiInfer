# NNNN — Title

Copy this file to `docs/proposals/NNNN-<slug>.md` and fill it in. The design is
reviewed before implementation starts; see the change flow in
[`CONTRIBUTING.md`](../../CONTRIBUTING.md).

- Status: Draft | Accepted | Implemented
- Author:
- Date:

## 1. Summary

One paragraph: the feature, its layer (engine or policy), and the direction of the
expected gain.

## 2. Motivation and current gap

What the current implementation does, where the gap is, and why the change is worth
making. Cite code locations as `file:line`.

## 3. Goals and non-goals

State the boundary explicitly, including adjacent work this proposal does not do.

## 4. Design

The approach, the interfaces, and how it coexists with the existing components.
Include at least one alternative and the trade-off that rejected it.

## 5. Model-agnosticism verdict

Is this engine-layer or policy-layer? Which public protocols does it depend on? Does it
harbour any implicit assumption about one model?

## 6. Losslessness and precision criterion

bit-exact or numerically identical? Name the reference, the compared quantity, the
threshold, and the script that reproduces it.

## 7. Implementation plan

Files changed, switches introduced, and the backward-compatibility strategy.

## 8. Test plan

What CPU, GPU, and weight-dependent tests cover, and which cases are new.

## 9. Benchmark plan

Which conditions are measured, against which baseline, and the expected range and the
point where the gain disappears.

## 10. Risks and limitations
