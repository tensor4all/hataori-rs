# Phase D acceptance ledger

**Tracking:** [Issue #14](https://github.com/tensor4all/hataori-rs/issues/14)

**Contract:** [`design/phase-d-migration.md`](design/phase-d-migration.md)

## Requirement-to-evidence map

| Requirement | Implementation evidence | Executable evidence |
|---|---|---|
| Mobility contract | `Mobility::{Pinned,Reconstructible,Migratable}`, `MobileObject`, mobile builder registration and handshake fingerprint | pinned rejection and registry tests |
| Versioned segmented snapshot | `MigrationPacket` preserves `Segments`; per-snapshot version/schema and byte/segment bounds | malformed packet, limit, and 1 MiB+ snapshot tests |
| Freeze and admission close | freeze is a write admitted behind prior object work; lifecycle rejects later admission without blocking a Rayon worker | migration, rollback, and repeated race tests |
| Prepare/commit/activate | inactive prepared state, checked epoch increment, authority commit, destination activation | local/remote and repeated migration tests |
| Authority and lifetime | stable creator authority; migration future retains a locality lease; roots/leases remain at authority | transfer/import, root, repeated migration, collection tests |
| Redirect/retry | structured `Moved` wire response; resolver refresh; same `RequestId` retry; redirect release from dedup | stale-handle retry and request-counter test |
| Pre-commit rollback | source admission reopens and prepared destination is discarded | destination restore-failure test |
| Post-commit boundary | activation/retirement failure marks the runtime failed with `MigrationCommittedFailure` | state-machine error branches and shutdown tests |
| Forwarding/retirement | bounded forwarding state; authority queues remote retirement after final lifetime release | migration collection reaches zero on both localities |
| Reconstructible resources | `RestoreContext` supplies destination `Place`; only logical snapshot is transmitted | destination-provider reconstruction test |
| Same-locality fast path | domain-to-domain migration shares the runtime object service and avoids transport | same-locality domain migration test |
| Observability/bounds | migration/prepared/forwarding/snapshot limits and active/completed/rollback/failure/byte counters | saturation, stats, and clean-shutdown assertions |
| Production backends | migration uses runtime actions and logical places only | TCP smoke and MPI world sizes 1, 2, and 4 |

## Validation command

```bash
scripts/check-phase-d-migration.sh
```

The lane runs formatting, default/MPI tests, clippy `-D warnings`, rustdoc,
Rust 1.85 checks, TCP migration/reuse, and MPI migration/reuse at world sizes 1,
2, and 4. Every backend process uses the watchdog policy in `AGENTS.md`.

## Boundary

Phase D implements explicit caller-driven migration. Automatic migration,
replication, process-failure recovery, direct transport-to-transport snapshot
streaming, GPU/provider adapters, object-aware `pmap`, collectives, and the
public Phase E facade remain unimplemented.
