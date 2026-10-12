# Phase C fixed-placement remote objects

**Tracking:** [hataori-rs#14](https://github.com/tensor4all/hataori-rs/issues/14)

**Status:** implementation contract for Phase C. Phase A transport and Phase B
structured execution remain unchanged. Phase D migration is implemented by the
separate [`phase-d-migration.md`](phase-d-migration.md) contract; Phase E
algorithms remain unimplemented.

## 1. Ownership and scope

Phase C extends the unpublished `hataori-runtime` crate with a distinct object
service. The service owns the object-type/action registry, authoritative
fixed-placement directory entries, resident object store, bounded resolver
cache, locality leases, roots, placement admission, and object statistics.
MPI/TCP types remain below the runtime boundary.

Objects are pinned in Phase C. Public identity and routing are nevertheless
migration-ready: `ObjectId` contains only `(RunId, opaque unique value)`, while
`ObjectLocation` separately contains `(LocalityId, DomainId, slot, generation,
epoch)`. Calls always resolve the ID and carry the observed epoch. No call
routes from bits in an object ID.

Deferred to Phase D: snapshots, forwarding, redirects after directory commit,
manual migration, reconstructible resources, and automatic migration. Deferred
to Phase E: object-aware `pmap`, collectives, and facade performance promotion.

## 2. Public typed contract

An object and its actions are registered before handshake:

```rust
pub trait DistributedObject: WireValue {
    const TYPE_ID: u128;
}

pub trait ObjectReadAction<T: DistributedObject>: WireValue {
    const ID: u128;
    type Output: WireValue;
    fn execute(self, state: &T) -> Result<Self::Output, ActionError>;
}

pub trait ObjectWriteAction<T: DistributedObject>: WireValue {
    const ID: u128;
    type Output: WireValue;
    fn execute(self, state: &mut T) -> Result<Self::Output, ActionError>;
}
```

`RuntimeBuilder::register_object` records state schema, concurrency mode, and
factory. Read and write registrations are stable IDs included in the Phase A
object-registry handshake hash. Duplicate IDs/types, invalid IDs, read actions
on an exclusive type, and schema mismatches fail before transport start.

`Runtime::create_at` returns a must-use `CreateFuture<T>`. A successful future
produces `Remote<T>`; `create_rooted_at` also returns `ObjectRoot<T>`. Remote
creation encodes state exactly once. Same-locality creation moves `T` directly
into the store and reports zero codec/transport operations.

`Remote<T>` contains stable identity plus shared locality-lease and resolver
state. Local clones share one lease and generate no protocol traffic.
`Remote::call_read` and `Remote::call_write` return must-use typed futures.
Same-locality calls move the typed action into a domain job and return the typed
result without codec or transport framing. Remote calls encode only action
input/output, never resident state.

`WeakRemote<T>` contains identity/type plus a location hint but no strong lease.
Upgrade acquires a live lease through the authoritative owner. Strong handles
do not implement unrestricted `WireValue`; controlled transfer uses a
single-destination, expiring transfer token and a transfer pin until import or
expiry.

## 3. Directory, resolver, and placement

The resident locality owns the authoritative Phase C directory entry. Because
objects are pinned, that authority is also the current owner; this is not
derived from `ObjectId`. The creator and imported-handle localities retain a
bounded resolver entry. Resolver eviction is safe because each live local
lease retains a location hint; calls repopulate the cache through the same
validated resolver path.

Creation uses an exact `Place`. Unknown membership/domain, a full object store,
slot exhaustion, registry mismatch, codec failure, or saturated destination is
a typed error and leaves no object, lease, root, pin, or pending waiter.

Hard object calls and `spawn_colocated` validate object type and epoch and take
a short placement ticket atomically with admission. The ticket is released on
submit failure, completion, cancellation-before-start, panic, or shutdown.
Pinned objects do not change epoch in Phase C, but stale-epoch tests exercise
the same typed `Moved` response required by Phase D.

Soft colocation tries the resolved place once under ordinary bounded admission,
then applies exactly one declared fallback (`Any`, exact `Place`, or reject).
It does not raise priority, bypass queues, or retain an independent affinity
queue. `Any` deterministically chooses the first registered eligible place;
there is no topology solver or global load balancer.

## 4. Object admission and execution

Every resident object has a bounded FIFO mailbox and an admission state machine:

- `Exclusive`: at most one admitted write; the state requires `Send` only.
- `ReadWrite`: consecutive front readers may run concurrently; a queued writer
  prevents later readers from bypassing it; writes are exclusive; state
  requires `Send + Sync`.

Admission happens on the owner/progress thread before submission to the owning
runtime domain. A domain worker therefore never waits for object admission.
The internal `Mutex`/`RwLock` is only a memory-safety backstop and is immediately
available by construction. No guard crosses a future poll, remote call,
transport operation, or user asynchronous boundary. Synchronous user panics
are converted to bounded action errors and admission/pins are released.

Cancellation marks queued work and prevents user code from starting. Running
Rust code remains cooperative. Deadlines remove origin waiters and eventually
release queued work through the existing Phase B cancellation/dedup path.

## 5. Lifetime protocol

One locality lease represents all strong handles for `(locality, ObjectId)`.
The owner tracks lease state with a TTL, renewal interval, and grace interval.
Normal last-handle drop or explicit release queues a bounded control release.
Periodic renewal is owner-thread progress. Missed renewal enters `Suspect`;
collection is permitted only after grace and only when every root, lease,
in-flight call, mailbox item, placement ticket, and transfer pin is zero.

A transfer reserves one bounded transfer pin, names one destination, and has a
deadline. Import creates/reuses the destination locality lease and acknowledges
the transfer before releasing the pin. Cancellation, timeout, destination
mismatch, duplicate import, and peer failure leave a bounded tombstone until
expiry and cannot collect a reference in transit.

`ObjectRoot<T>` is a runtime-owned RAII root. Explicit release/drop decrements
the authoritative root count. Shutdown invalidates and releases local handles
and roots, progresses releases/expiry, then requires zero non-durable object
resources. `Drop` remains bounded best effort and never waits for user code or
MPI.

## 6. Bounds, errors, and observability

`RuntimeLimits` adds nonzero limits for live objects, object mailbox entries,
resolver entries/bytes, local leases, roots, placement tickets, transfers, and
object progress events, plus checked lease renewal/TTL/grace ordering. All
derived capacities use checked arithmetic.

Typed errors distinguish unknown/collected object, type/action mismatch, stale
epoch, exact placement failure, saturated admission, lease expiry, transfer
failure, pinned migration request, resource exhaustion, peer failure, and
runtime shutdown.

`RuntimeStats` reports live objects, roots, leases, suspect leases, queued
object calls, in-flight object calls, placement tickets, transfers, resolver
entries/bytes, and same-locality versus remote codec/transport counts. Cache
clear and stats operations are public and never dump object data.

## 7. Acceptance evidence

Phase C is accepted when short-watchdog checks demonstrate:

- pure ID/location, directory, resolver, admission, lease, transfer, root, and
  collection state machines;
- exact local/remote creation and typed calls, including same-locality zero
  codec/transport counters and at least 1 MiB segmented remote state/action;
- exclusive non-overlap, parallel readers, writer fairness, and no worker
  blocked on admission;
- hard colocation, soft fallback, stale epoch, saturated placement, dropped
  handles/futures, action error/panic, deadline, cancellation, lease expiry,
  duplicate parcels, disconnect, and failed placement;
- zero-resource explicit shutdown and reuse over memory, TCP, and MPI world
  sizes 1, 2, and 4;
- default/MPI clippy `-D warnings`, rustdoc, Rust 1.85, rendered docs, runnable
  examples, and unchanged P0/P1/Phase A/Phase B lanes.

Every local validation process uses the watchdogs in `AGENTS.md`.
