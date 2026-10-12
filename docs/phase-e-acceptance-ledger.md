# Phase E acceptance ledger

**Tracking:** [Issue #14](https://github.com/tensor4all/hataori-rs/issues/14)

| Requirement | Evidence | Status |
|---|---|---|
| Canonical batch `pmap` | `hataori-algorithms::pmap`; local/remote ordered integration tests | PASS |
| Batch accounting and bounded prefetch | `ControllerStats`; one/two slots per target; batch-size tests | PASS |
| Same-locality zero codec/transport | typed `ActionValue` path and runtime/algorithm counter assertions | PASS |
| Exact and object-aware placement | `pmap_at`, `pmap_colocated`, `pmap_preferred_colocated` lower to existing runtime placement | PASS |
| Broadcast/scatter/gather | registered echo action plus ordered `GatherFuture` | PASS |
| Direct async/blocking facade | top-level re-exports and `hataori::blocking::pmap` | PASS |
| Checked segmented batch codec | checked counts/conversions/lengths and malformed tests | PASS |
| TCP/MPI/runtime regressions | `scripts/check-phase-e-algorithms.sh` | PASS |
| Frozen legacy continuity and sugar transparency | valid-host paired report required by manifest | **INCONCLUSIVE** on current host (`schedutil`; declarations absent) |
| Automatic migration policy | deliberately deferred until measured workload | NOT REQUIRED |

Phase E implementation correctness is complete, but Issue #14's Phase E
performance acceptance is not complete until every frozen required case has a
valid-host `PASS`. P0/P1 therefore remains present.
