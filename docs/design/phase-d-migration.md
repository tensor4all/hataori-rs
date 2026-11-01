# Phase D explicit object migration

**Tracking:** [hataori-rs#14](https://github.com/tensor4all/hataori-rs/issues/14)

**Status:** implementation contract for Phase D. Phases A-C remain unchanged.
Phase E algorithms, automatic placement policy, process-failure recovery,
replication, and durable persistence remain unimplemented.

## 1. Scope and ownership

Phase D extends the unpublished `hataori-runtime` object service with manual
migration. The object service owns mobility registration, migration state,
prepared destination slots, forwarding records, resolver updates, and migration
statistics. The Phase A transport continues to move transport-independent
segmented parcels; MPI/TCP types do not enter the object API.

Migration preserves `ObjectId`, locality leases, roots, and logical ownership.
It changes only the authoritative `ObjectLocation` and increments its epoch.
The creator locality remains the bounded directory authority for the object's
run lifetime. This stable authority is internal handle metadata, not part of
`ObjectId`, and avoids consensus or a second distributed directory protocol.
Resident state may move repeatedly between other localities.

## 2. Public mobility contract

Pinned registration remains the Phase C default. Mobile state implements:

```rust
pub trait MobileObject: DistributedObject {
    const MOBILITY: Mobility;
    const SNAPSHOT_VERSION: u32;
    type Snapshot: WireValue;

    fn freeze(&mut self) -> Result<Self::Snapshot, ActionError>;
    fn restore(
        snapshot: Self::Snapshot,
        context: RestoreContext,
    ) -> Result<Self, ActionError>;
}
```

`Mobility` has `Pinned`, `Migratable`, and `Reconstructible` variants.
`register_mobile_exclusive` and `register_mobile_read_write` reject
`Mobility::Pinned`, zero snapshot versions/schema IDs, and duplicate types.
Mobility, snapshot version/schema, concurrency, and object/action schemas are
included in the object-registry handshake fingerprint.

For `Migratable`, the snapshot contains all transferable logical state. For
`Reconstructible`, it contains only logical state; `restore` receives a
`RestoreContext` containing the destination `Place` and rebuilds local
provider/device/executor-related resources. Communicators, sockets, runtime
pools, contexts, caches, raw addresses, object locks, and admission guards are
never supplied by the runtime snapshot codec.

`Runtime::migrate` and `Remote::migrate_to` return a must-use
`MigrationFuture<T>`. Success returns `MigrationReport { object, from, to,
bytes }`. Migrating to the current place is an idempotent zero-transfer success.
A pinned object, unknown place, stale handle, saturated migration table,
oversized snapshot, incompatible registration, or shutdown returns a typed
error.

## 3. Bounded protocol and state machine

Each migration has one checked nonzero migration ID, one deadline, and one
migration pin. `ObjectLimits` bounds active migrations, prepared slots,
forwarders, snapshot bytes, segment count, and forwarding TTL. Statistics expose
active/completed/rolled-back/failed migrations, prepared slots, forwarding
entries/bytes, and transferred snapshot bytes. No migration history or payload
is retained after its TTL/owner expires.

The coordinator lowers one manual migration to these typed internal steps:

```text
Resident(epoch N)
  -> freeze at current resident
  -> transfer versioned Segments through the coordinator
  -> prepare inactive destination slot
  -> commit directory at stable authority (epoch N + 1)
  -> activate destination
  -> install source forwarder
  -> retire source state
```

The coordinator may be any locality. The initial implementation relays the
bounded segmented snapshot through that coordinator instead of adding a second
bulk-transfer subsystem. Each segment remains segmented; the runtime does not
coalesce the complete snapshot into one buffer. This is a deliberate minimal
base protocol, not a claim of zero-copy or direct source-to-destination DMA.

Freeze closes admission before taking exclusive state access. Calls admitted
before freeze drain first. New calls receive `Migrating` or a committed `Moved`
location; no Rayon worker waits on an object lock. Snapshot code executes in the
owning runtime domain, catches panics as typed action failures, and holds no
object guard across transport or future polling.

The destination validates run/object/type, mobility, snapshot version/schema,
segment/byte limits, destination domain, generation, and proposed epoch before
allocating a prepared slot. Prepared state is invisible to calls until commit
and activation. Duplicate protocol steps with the same migration ID are
idempotent; conflicting metadata is a protocol error.

## 4. Commit, routing, and retry

Directory commit is the authority boundary. It atomically changes the location
to the prepared destination at epoch `N + 1`. Epoch overflow fails before
freeze. Before commit, failure rolls back by discarding prepared state and
reopening source admission. After commit, rollback is forbidden; failure to
activate or establish forwarding fails the runtime rather than claiming
transparent recovery.

The old resident retains one bounded forwarding entry after source state is
retired. A stale object call receives a structured `Moved` response containing
the committed location. The origin validates the object/type/newer epoch,
updates its bounded resolver and shared locality lease hint, and retries exactly
once per distinct newer epoch with the same `RequestId` and original deadline.
The destination deduplication table prevents duplicate execution in the
supported live-run failure model. Redirect cycles, non-increasing epochs,
forwarding exhaustion, or deadline expiry are typed failures.

Hard object calls and hard colocation use the refreshed location. Migration
waits for in-flight calls and placement tickets admitted before freeze. Soft
colocation keeps its Phase C single-attempt fallback contract and does not wait
indefinitely for migration.

## 5. Lifetime and shutdown

The stable authority retains leases, roots, and migration metadata while the
resident state moves. Strong/weak handles and transfer tokens carry the
internal authority hint; it is validated against the run and never used as an
object identity. Lease renewal, release, root release, and weak upgrade target
the authority, while object calls target the resolver location.

An object is collectible only when its authority has no roots, leases,
in-flight work, mailbox work, placement tickets, transfer pins, or migration
pin. Collection retires any remote resident before removing the directory
entry. Prepared slots and pre-commit snapshots are migration-owned and are
discarded on cancellation, deadline, peer failure, or shutdown. Explicit
shutdown drains or fails active migrations, releases pins and prepared slots,
expires forwarding entries, and requires all non-durable migration counters to
be zero. `Drop` remains bounded best effort.

## 6. Acceptance evidence

Phase D is accepted when short-watchdog checks demonstrate:

- pure freeze/prepare/commit/activate/forward/retire and rollback state-machine
  transitions, including duplicate and conflicting steps;
- pinned rejection and both migratable and reconstructible registration;
- identity, roots, leases, weak upgrade, repeated migration, and epoch
  preservation across same-process and remote moves;
- queued/in-flight call and placement-ticket drain, stale-call redirect with
  same-request retry, writer/readers versus freeze, and no worker lock wait;
- segmented snapshots above 1 MiB, configured byte/segment saturation, panic,
  codec/schema/version failure, cancellation, deadline, pre-commit rollback,
  post-commit run failure, disconnect, and clean shutdown;
- same-locality migration/call fast paths and bounded migration/forwarding
  statistics;
- memory, TCP, and MPI world sizes 1, 2, and 4; default/MPI clippy, rustdoc,
  Rust 1.85, rendered docs, and unchanged P0/P1/Phase A-C lanes.

Crash recovery after directory-authority loss, consensus, replication,
automatic migration, direct transport streaming, GPU migration adapters, and
Phase E algorithms are explicitly outside Phase D.
