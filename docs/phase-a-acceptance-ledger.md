# Phase A acceptance ledger

**Tracking:** [hataori-rs#14](https://github.com/tensor4all/hataori-rs/issues/14)

**Classification:** PASS for Phase A. This ledger does not accept Phase B-E or
replace the final repository-scale audit required after all phases.

## Requirement-to-evidence map

| Phase A requirement | Durable implementation evidence | Fresh verification evidence |
|---|---|---|
| Immutable P0/P1 reference | `benchmarks/legacy-runner/` pins full commit `34cb1b1371c8b2f8ef750e2d49d10f9ef8f0782e` in manifest and lockfile | `scripts/check-legacy-runner.sh`: PASS |
| Cases, metrics, host gates, statistics, and thresholds frozen before candidate data | `benchmarks/performance/manifest.toml`; fail-closed `scripts/check-performance-manifest.py` | 11 required case families; SHA-256 `52cd8b93bc4eee0f5b6f18cbcb39ba1a9b0697cf490869b11ecf1075ea5ec6bd`: PASS |
| Stable transport-independent IDs | `protocol.rs`: `RunId`, `LocalityId`, `DomainId`, `MessageId`, `TaskId`, `ActionId`, `ObjectTypeId`, `RequestId`, `TraceId` | checked-construction and round-trip tests: PASS |
| Version/run/registry/capability handshake | fixed-size checked `Hello`, negotiated capabilities and limits | mismatch and malformed-field tests plus TCP/MPI bootstrap: PASS |
| Channels, framing, segmentation, and checked limits | checked parcel codec; control/action/bulk channels; segmented payload; trace identity; count/byte/control reservations | malformed discriminants/lengths/limits and 1 MiB segmented codec tests: PASS |
| Bounded deterministic memory backend | `memory.rs` with explicit queue/byte ownership and scripted faults | pass, delay, duplicate, loss, reorder, saturation, disconnect, priority, wrong-run, and cleanup tests: PASS |
| TCP rendezvous and fixed membership | bounded run-scoped rendezvous plus three-channel full mesh | wrong-run rendezvous/hello, four-locality membership/ring, process reuse: PASS |
| TCP progress, backpressure, peer failure, flush, and shutdown | nonblocking partial read/write state, per-channel queues, shared per-peer byte bound with control reserve, typed failure cleanup | count/byte saturation, control progress, oversized prefix, EOF failure, pending flush, shutdown rejection, zero retention: PASS |
| MPI parity without leaked backend types or unsafe wrappers | duplicated-communicator driver, backend-private ranks/tags, 4 KiB chunks, main-thread check | shared conformance assertions; source scanners; no unsafe `Send`/`Sync`; `mpiexec` n=1/2/4: PASS |
| Large segmented production transfer | TCP segmented parcel and MPI multi-chunk transfer | checksum/value-verified payload ≥1 MiB on TCP and MPI: PASS |
| Clean reusable shutdown | atomic submission close, explicit flush, pending-event preservation, partial-frame detection, retained-resource stats | repeated TCP construction and repeated MPI drivers on one communicator; zero retained resources: PASS |
| Existing P0/P1 behavior preserved | foundation is an unpublished workspace member; root default member and existing source paths unchanged | full `scripts/check-core.sh` acceptance matrix: PASS |
| Docs and runnable examples | README status, durable design/worklog, `tcp_transport_smoke`, `mpi_transport_smoke` | snippet checker and rendered Quarto/rustdoc site: PASS |

## Fresh command evidence

The final local audit used watchdogs around all backend processes and ran:

- `scripts/check-runtime-foundation.sh` — PASS, including 29 MPI-feature unit
  tests, clippy `-D warnings`, rustdoc, Rust 1.85, TCP smoke, and MPI world sizes
  1/2/4;
- `scripts/check-legacy-runner.sh` — PASS;
- the `scripts/check-core.sh` acceptance sequence — PASS under bounded stage
  watchdogs: six feature combinations for tests/clippy/rustdoc/MSRV, dependency
  and call-boundary gates, `check-rsmpi-rt.sh`, `check-hybrid.sh`,
  `check-placement.sh`, and `check-faults.sh`;
- `scripts/build_docs_site.sh` — PASS;
- `git diff --check` — PASS.

Environment: Linux x86_64, Open MPI/OpenRTE 4.1.6, default rustc 1.97.1,
explicit MSRV rustc 1.85.0, MPIwrapper fixture commit
`966f4231c96153a08295fc7d0bcbd65e916a73fd`.

## Explicit boundary

Phase A supplies protocol and transport infrastructure only. Runtime lifecycle,
scopes, actions/futures, pending promises, deduplication, remote objects,
leases, placement, migration, rebuilt algorithms, and the sugar facade remain
unimplemented Phase B-E behavior and are not presented as current API.
