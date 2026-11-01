# Phase C acceptance ledger

**Tracking:** [Issue #14](https://github.com/tensor4all/hataori-rs/issues/14)  
**Contract:** [`design/phase-c-objects.md`](design/phase-c-objects.md)

## Requirement-to-evidence map

| Requirement | Implementation evidence | Executable evidence |
|---|---|---|
| Stable location-independent identity | foundation `ObjectId`, `ObjectTypeId`, checked `ObjectLocation` | protocol identity tests |
| Registry compatibility | sorted object type/action schemas and access mode in Phase A hello object hash | builder/handshake and registry tests |
| Exact local/remote creation | `RuntimeClient::create_at`; direct typed local move; one encoded remote factory action | memory integration, TCP/MPI smoke |
| Typed object calls | registered read/write traits and object envelope with ID/type/epoch | local/remote call tests |
| Same-locality fast path | typed state/action moves directly; no codec or transport call | deliberately failing codecs plus channel counters: PASS |
| Resolver/cache | bounded FIFO cache, clear/stats, handle location fallback, epoch validation | resolver and stale-epoch tests |
| Exclusive/ReadWrite admission | bounded FIFO per object; parallel front readers; writer fairness; admission before Rayon domain submit | pure admission ordering, parallel-reader, saturated-writer tests |
| Roots and collection | RAII `ObjectRoot`; roots, leases, calls, mailbox and placement counts gate collection | root/last-handle/collection tests |
| Locality leases | clone-shared `LocalLease`, renewal, release, suspect/grace expiry | clone traffic, release and expiry tests |
| Weak handles | `WeakRemote` plus owner-validated lease acquisition | remote upgrade integration |
| Controlled transfer | destination-specific non-serializable token; destination validation; TTL-backed release fallback | cross-locality transfer/import and local fast-call test |
| Hard/soft colocation | hard resolver placement; one bounded preferred attempt with explicit Any/Place/Reject fallback | colocated action integration and bounded submission tests |
| Bounded resources | validated object/mailbox/cache/lease/root/ticket/transfer limits; checked slots and counters | limit, saturation and clean-shutdown tests |
| Large segmented data | object state uses `Segments`; resident state is never sent by calls | 1 MiB+ memory object test; TCP/MPI object smokes |
| Backend independence | object layer stores only logical IDs and object-safe runtime transport | default/MPI clippy boundary scans; TCP/MPI n=1/2/4 |
| Clean shutdown/reuse | local leases invalidated, roots/objects/cache cleared, transport retained-resource gate unchanged | memory reuse and production smoke rounds |

## Validation command

```bash
scripts/check-phase-c-objects.sh
```

The lane runs formatting, default/MPI unit tests, clippy `-D warnings`, rustdoc,
Rust 1.85 checks, TCP smoke, and MPI smoke at world sizes 1, 2, and 4. Every
backend command uses the watchdog policy in `AGENTS.md`.

## Boundary

Phase C objects are pinned. Snapshot/restore, directory epoch commit,
forwarding, migration rollback, reconstructible resources, object-aware
algorithms, and the public `pmap` facade remain Phase D-E work.
