# Phase B acceptance ledger

**Tracking:** [hataori-rs#14](https://github.com/tensor4all/hataori-rs/issues/14)

**Classification:** PASS for Phase B only. Phase C-E remains unimplemented.

| Requirement | Durable evidence | Verification |
|---|---|---|
| Long-lived checked lifecycle | `runtime.rs`: builder/start, running progress, draining, stopped/failed states, explicit shutdown report | local/remote execution, client rejection after shutdown, two runtime lifecycles: PASS |
| Backend-independent runtime | boxed Phase A transport traits; runtime source contains no MPI/TCP concrete type | backend leak scanner; TCP and MPI smokes: PASS |
| Domain registry and bounded execution | immutable domain configs, fixed workers, bounded synchronous job/result queues | queue saturation rejects the third job without retention: PASS |
| Typed actions and registry compatibility | `Action`, `WireValue`, stable IDs/schema IDs, sorted fingerprint, panic-safe erased handler | order-independent fingerprint, collision, codec, user failure, panic tests: PASS |
| Pending promises and `RemoteFuture` | bounded pending table, source/action validation, must-use typed future | local/remote completion and pending-table tests: PASS |
| Structured scopes | higher-ranked scope lifetime prevents future escape; scope exit cancels tracked requests | joined local scoped action and zero pending state: PASS |
| Deadlines and cancellation | nonzero local deadlines, immediate waiter removal, control-channel cancel, pre-execution token | loss/deadline and delayed cancel-before-action tests: PASS |
| Bounded deduplication | running/completed/cancelled states, metadata validation, TTL, entry/byte limits, completion marker | duplicate transport fault executes once; byte exhaustion and tombstone tests: PASS |
| Response backpressure | bounded response outbox; completion cached only after transport accepts response | injected response saturation retries and returns the value exactly once: PASS |
| Typed failure boundaries | state/resource/deadline/cancel/user/protocol/transport/peer/shutdown variants | user failure, panic, saturation, malformed wire, disconnect tests: PASS |
| Observability without payload dumps | state, calls, channels, domains, dedup, response queue, ticket, transport counts; compact `Debug` | stats assertions and clippy scanner: PASS |
| Clean reusable shutdown | atomic submission close, pending cancellation, in-flight worker/completion drain, dedup clear, response/transport flush, retained-resource gate | concurrent submit/shutdown race, memory/TCP/MPI reuse, zero retained bytes: PASS |
| Production transport parity | same runtime implementation over Phase A TCP and MPI | TCP two-locality smoke; MPI n=1/2/4, two runs: PASS |
| Existing behavior preserved | new unpublished workspace crate; root default member and P0/P1 source unchanged | existing core and Phase A lanes remain required by CI: PASS |
| Documentation and examples | Phase B design, worklog, README/status, TCP/MPI binaries | rustdoc and rendered Quarto site: PASS |

## Bounded local command evidence

- `scripts/check-phase-b-runtime.sh` — PASS;
- `scripts/check-runtime-foundation.sh` — PASS;
- core acceptance matrix, immutable legacy runner, and performance manifest —
  unchanged and retained in CI;
- `scripts/build_docs_site.sh` — PASS;
- `git diff --check` — PASS.

Environment: Linux x86_64, Open MPI/OpenRTE 4.1.6, default rustc 1.97.1,
explicit MSRV rustc 1.85.0. Every build/test/backend process used the short
process-group watchdogs recorded in `AGENTS.md`.
