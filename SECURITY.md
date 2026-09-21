# Security Policy

EmbodiInfer loads model weights, executes tensor code on accelerators, and
serves inference frontends. A vulnerability here can run arbitrary code on a
shared GPU host, or expose the weights and data it serves.

## Reporting a vulnerability

**Do not open a public issue, pull request, or discussion for a security
problem.**

Report it privately through either channel:

1. GitHub private vulnerability reporting, if it is enabled for this
   repository: **Security** tab → **Report a vulnerability**.
2. Email **cclonelycc@outlook.com** with `[EmbodiInfer security]` in the
   subject.

Include, as far as you can:

- the affected revision or release, and the component (`embodiinfer/policies`,
  `embodiinfer/models`, `embodiinfer/layers`, `embodiinfer/backend`, `embodiinfer/engine`,
  `embodiinfer/engine/parallel`, `embodiinfer/engine/rollout`, `embodiinfer/engine/serve`, or the
  build and dependency setup);
- the impact, and the conditions required to trigger it;
- a minimal reproduction, or the exact command, profile, and observed result;
- whether the issue is already public anywhere.

Redact tokens, credentials, addresses, private paths, and personal data from
anything you send. Never test against systems you do not own or have explicit
permission to test.

## What to expect

This is a research project maintained on a best-effort basis, so these are
targets rather than guarantees, measured from the first private report:

| Step | Target |
|---|---|
| Acknowledgement of the report | within 5 working days |
| Initial assessment and severity triage | within 10 working days |
| Fix or documented mitigation for confirmed issues | agreed with the reporter, based on severity |
| Public disclosure | coordinated with the reporter after a fix or mitigation is available |

We will credit reporters in the advisory unless you ask us not to.

## In scope

- Weight and checkpoint loading: a path that executes code from an untrusted
  checkpoint, config, or profile, or that resolves a download over an
  unauthenticated or unverified transport.
- Remote exposure: an inference frontend that binds beyond its documented
  interface, accepts unauthenticated requests that change engine or session
  state, or silently weakens transport protection.
- Session and rollout integrity: a checkout, commit, rollback, reset, or cancel
  that reports success while leaving state inconsistent between requests or
  replicas.
- Secret and data handling: leakage of credentials, tokens, private paths, or
  prompt and observation data through logs, error responses, or saved
  artifacts.
- Dependency and build integrity: executing untrusted code through the
  documented install path, or a pinned dependency that no longer resolves to
  the reviewed revision.

## Out of scope

- Model accuracy, the quality or safety of generated actions, and the licensing
  of checkpoints, datasets, and upstream model code. These are not distributed
  here; report them upstream.
- The correctness of a third-party framework's integration with EmbodiInfer.
- Vulnerabilities in third-party dependencies with no EmbodiInfer-specific
  impact, though we welcome a heads-up.
- Denial of service through resource exhaustion on a host you control, and
  reports produced only by a scanner without a demonstrated impact.
- Numerical differences from a reference implementation that stay within the
  documented parity thresholds.

## Supported versions

Security fixes are applied to the latest release on `main`. There are no
long-term support branches, so please confirm an issue reproduces on the
current `main` before reporting it.
