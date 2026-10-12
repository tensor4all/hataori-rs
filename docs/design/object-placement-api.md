# Object placement and locality-aware task API

**Status:** Phase C object creation, typed calls, leases, roots, weak upgrades,
transfer, resolver routing, and colocation implemented; Phase D manual
migration implemented; Phase E object-aware algorithms remain future work

**Tracking:** [hataori-rs#14](https://github.com/tensor4all/hataori-rs/issues/14)

**Parent design:** [`distributed-runtime.md`](distributed-runtime.md)

## 1. Purpose

Hataori must support keeping a remote object in memory on a selected locality
and sending work to that object instead of repeatedly transferring its state.
It must also support using an object's current location as a task-placement
preference.

This document specifies the high-level API and its lowering contract. It uses
logical `LocalityId` and `DomainId` values; MPI ranks and TCP endpoints remain
transport-backend details.

“Long-lived” means retained for part or all of one running Hataori runtime. It
does not mean durable storage across process or runtime restart. Disk
persistence, replication, and recovery from loss of the owning process are not
part of the initial API.

## 2. Required semantics

The API distinguishes three operations:

1. create an object at a required initial `Place`;
2. invoke an object action at the object's authoritative current location;
3. schedule an independent task using an object location as a hard or soft
   placement constraint.

Placement preference is affinity, not execution priority. A preferred task is
offered to the selected locality/domain before its fallback is considered. It
does not jump ahead of already-admitted work or bypass queue/admission limits.

## 3. Places and placement policies

A place is transport-independent:

```rust
pub struct Place {
    locality: LocalityId,
    domain: DomainId,
}
```

The target policy is conceptually:

```rust
pub enum PlacementTarget {
    Place(Place),
    Colocated(ObjectId),
}

pub enum PlacementFallback {
    Any,
    Place(Place),
    Reject,
}

pub enum TaskPlacement {
    Any,
    Require(PlacementTarget),
    Prefer {
        target: PlacementTarget,
        fallback: PlacementFallback,
    },
}
```

The exact enum/builder spelling remains an implementation decision. The
following semantics are fixed:

- `Any` lets the bounded scheduler select an eligible domain.
- `Require(Place)` must execute at that exact locality/domain or return a typed
  placement error.
- `Require(Colocated(object))` follows the object's authoritative location and
  executes there. Migration races redirect and retry within the call deadline.
- `Prefer` tries the target under a bounded preference/admission budget, then
  applies its explicit fallback. It never waits indefinitely for affinity.
- `Reject` turns inability to satisfy the preference into a typed error.

There is no initial multi-object constraint solver, topology cost model, data
replication planner, or automatic global rebalance. When a task depends on
several objects, the caller selects one primary affinity or expresses the work
as an action on the object that owns the primary mutable state.

## 4. Creating and retaining an object

Illustrative target syntax follows. It is design syntax, not a currently
compilable example:

```rust
let locality = runtime.localities().get(LocalityId::new(2))?;
let place = locality.place(DomainId::DEFAULT)?;

let cache: Remote<ShardCache> = runtime
    .objects()
    .create(ShardCache::new(shard))
    .at(place)
    .mobility(Mobility::Pinned)
    .await?;
```

Creation at a remote place is one registered object-factory action. The initial
logical state is encoded once and restored into a prepared object slot at the
destination. Same-locality creation moves the state directly and performs no
serialization or transport operation.

The returned `Remote<T>` owns a locality lease. Local clones share that lease.
As long as a strong handle, runtime-owned root, in-flight call, transfer pin, or
migration pin exists, the object is not collectible.

An explicit runtime-owned root retains an object even when no client handle is
kept:

```rust
let (cache, root): (Remote<ShardCache>, ObjectRoot<ShardCache>) = runtime
    .objects()
    .create(ShardCache::new(shard))
    .at(place)
    .mobility(Mobility::Pinned)
    .rooted()
    .await?;
```

`ObjectRoot<T>` is an RAII token backed by directory durable-root accounting.
Dropping or explicitly releasing it removes that root. It is not freely
serializable and is not an immortal global registry entry. Runtime shutdown
releases runtime-owned roots and reports any unexpected retained object.

`Mobility::Pinned` means the object remains at the required place for its
lifetime. `Mobility::Migratable` and `Mobility::Reconstructible` treat `.at()`
as the initial location; later explicit migration may move the object without
changing its `ObjectId` or lifetime accounting.

## 5. Object actions

The most direct way to execute where an object lives is an object action:

```rust
let value = cache.call(CacheLookup { key }).await?;
```

The action is registered with stable object/action IDs and is executed against
the resident object state. The initial object concurrency contract is one
bounded serial mailbox per object: an admitted action receives exclusive
mutable access to its logical state. Read-parallel or application-defined
concurrency may be added only after a concrete need; it is not implicit in the
first API.

`Remote<T>::call` has hard colocation semantics:

1. resolve `ObjectId` to `ObjectLocation { locality, slot, generation, epoch }`;
2. send the typed object call with the observed epoch;
3. validate epoch and acquire an in-flight object pin before execution;
4. execute in the object's domain and release the pin on every completion path;
5. on `Moved`, refresh the resolver cache and retry with the same `RequestId`
   within the deadline and deduplication contract.

Object calls do not serialize or transfer the stored object state. A
same-locality call uses the object store and domain scheduler directly, without
transport framing.

### 5.1 Sharing across Rayon workers

An object is process-local memory owned by one `Place`, so Rayon workers in its
owning domain may share it. It is not shared memory across localities. Calls
from another node are typed messages routed to the owning locality.

The high-level concurrency policy is:

```rust
pub enum ObjectConcurrency {
    Exclusive,
    ReadWrite,
}
```

`Exclusive` is the default. The bounded object mailbox admits one action at a
time and supplies exclusive mutable access:

```rust
#[hataori::object_action(access = "write")]
fn insert(state: &mut ShardCache, input: Insert) -> Result<(), CacheError> {
    state.insert(input.key, input.value)
}
```

The object state needs to be `Send`, but it need not be `Sync`, because no two
workers access it concurrently. Consecutive calls may execute on different
Rayon workers; the runtime-owned object cell and admission handoff preserve
exclusive access.

`ReadWrite` is opt-in. Registered actions declare read or write access:

```rust
let table: Remote<Table> = runtime
    .objects()
    .create(Table::new())
    .at(place)
    .concurrency(ObjectConcurrency::ReadWrite)
    .await?;

let value = table.read(TABLE_GET, key).await?;
table.write(TABLE_INSERT, (key, value)).await?;
```

Read actions receive `&T` and may run concurrently on several Rayon workers in
the owning domain. A write action receives `&mut T` and runs only after prior
readers and the prior writer have completed. `ReadWrite` therefore requires
the stored state and read-action captures to satisfy the appropriate `Send +
Sync` bounds.

The runtime uses a bounded reader/writer admission gate before submitting work
to Rayon. A Rayon worker must not block while waiting on an object mutex or
read/write lock. The exact internal primitive is private; it may use a lock as
a memory-safety backstop, but queueing, fairness, cancellation, and deadlines
belong to the runtime admission state machine.

An object action is synchronous while it holds `&T` or `&mut T`. It cannot keep
an object guard across `.await`, a remote call, or detached work. An operation
that needs asynchronous work copies or snapshots the required data, releases
object access, performs the asynchronous task, then submits a separate
version-checked write action if it must commit a result.

Reentrant write access to the same object is rejected rather than blocking.
Migration freeze closes admission, drains admitted readers/the writer, obtains
exclusive access, and snapshots only logical state. Mailboxes, admission gates,
mutexes, read/write locks, and Rayon worker identity are runtime resources and
are never serialized as migratable state.

### 5.2 User-owned synchronization

An object type may contain its own `Mutex`, `RwLock`, atomics, or sharded locks
for application-specific fine-grained synchronization. This is most useful for
pinned or reconstructible objects. Such a type must satisfy the concurrency
mode's `Send`/`Sync` bounds and must define a snapshot that serializes logical
state rather than lock implementation state.

User locks do not replace Hataori admission or lifetime pins. Lock ordering is
the object's responsibility, and no guard may cross an `.await` or remote call.
The initial public API does not expose raw `Arc<Mutex<T>>` storage pointers or
allow another locality to lock process memory directly.

## 6. Colocated independent tasks

An independent task that benefits from data locality but is not an object
method can use explicit placement:

```rust
let exact = runtime
    .spawn_colocated(&cache, BuildIndex { range })
    .await?;

let preferred = runtime
    .task(PrefetchNeighbors { range })
    .prefer_colocated(&cache)
    .fallback_any()
    .spawn()
    .await?;
```

`spawn_colocated` is hard affinity. Before admission, the runtime resolves the
object and acquires a short placement pin/ticket tying the admission decision
to the observed location epoch. Migration waits for that ticket to be released
or the task fails its admission deadline. Once admitted, the task stays on the
selected domain; it is not stolen to another locality.

`prefer_colocated` is soft affinity. It tries the current object locality first
under a bounded preference budget. A stale epoch causes at most the bounded
resolver/redirect path; saturation then applies the explicit fallback. Soft
affinity does not pin an object for the entire task, because the task does not
own direct object access.

If a task must read or mutate the object consistently, it uses `Remote::call`
rather than relying on physical colocation. General colocated tasks never
receive raw pointers or unchecked references into the object store.

Convenience entry points lower to the same canonical operation:

```rust
runtime.spawn_at(place, action)
runtime.spawn_colocated(&object, action)
runtime.task(action).placement(policy).spawn()
```

The convenience layer adds no second scheduler or queue.

## 7. `pmap` placement

The current-style sugar facade can apply one affinity policy to a complete
`pmap` operation:

```rust
let output = hataori::pmap(
    &runtime,
    domain,
    PmapOptions::default()
        .batch_size(16)?
        .placement(TaskPlacement::prefer_colocated(&cache).fallback_any()),
    inputs,
    PROCESS_BATCH,
)
.await?;
```

The policy is applied per batch, not per item. This preserves one pending
promise, placement decision, and transport sequence per in-flight batch. It
does not create a `Remote<T>`, directory lookup, or lease for every input.

Per-item affinity is deferred until a real workload needs it. If introduced,
the canonical representation will group items by compact `ObjectId`/`Place`
keys before scheduling so it cannot accidentally turn one batch into one
allocation, hash, resolver lookup, or parcel per item.

## 8. Placement and migration races

Hard object calls and hard colocation follow the authoritative directory epoch.
The initial protocol uses these rules:

- a stale target returns `Moved { location, epoch }` rather than executing;
- a placement ticket acquired before migration either completes admission
  before freeze or blocks the migration transition within a bounded deadline;
- directory commit invalidates older tickets and cached locations;
- redirects and forwarding entries are bounded by TTL and count;
- retry reuses the same `RequestId` and bounded deduplication state;
- deadline expiry releases waiters, placement tickets, and pins mechanically.

Soft preference may fall back instead of following migration indefinitely.
Migration does not change task semantics after a task has been admitted to a
domain unless that task performs a separate object call.

## 9. Failure and resource behavior

Placement failures are typed and distinguish at least:

- unknown locality/domain;
- domain unavailable or saturated;
- object collected or unavailable;
- stale location/redirect exhaustion;
- pinned object migration request;
- placement/admission deadline;
- runtime or peer failure.

Creation, rooting, calls, and placement use the existing runtime limits. They
do not create unbounded affinity queues. Object mailbox capacity, per-domain
action capacity, pending calls, placement tickets, resolver entries, redirects,
and retained bytes are observable and bounded.

Loss of a process owning an unreplicated object makes that object unavailable
and may fail the run under the initial transport contract. The API does not
claim transparent recovery.

## 10. Performance contract

Placement lookup is O(1) expected time through the bounded resolver cache and
does not scan all localities or objects. Same-locality creation, calls, and
colocated dispatch avoid codec and transport work. Remote object calls encode
only the action input/output, not resident object state.

The high-level API must add no extra scheduler hop over the canonical placement
operation. Instrumentation and benchmarks cover:

- exact versus preferred/fallback dispatch;
- cache-hit and redirect resolver paths;
- same-locality zero serialization/messages;
- repeated calls to a large resident object compared with retransmitting its
  state;
- placement-ticket and migration contention;
- bounded memory under saturated affinity queues.

These cases join the legacy-continuity and sugar-transparency gates in the
parent design before Phase E is accepted.

## 11. Deployment assumptions

The initial runtime serves a fixed set of trusted participants launched for one
configured run. It does not add TLS, authentication, authorization, ACLs,
capability security, tenant isolation, sandboxing, or hostile-peer defenses.

Protocol versions, run IDs, registry hashes, lengths, counts, and checked
allocation limits remain mandatory for correctness, memory safety, accidental
cross-run detection, and bounded resource use. They are not presented as a
security subsystem.

## 12. Acceptance criteria

- [ ] A typed object can be created at an exact `(LocalityId, DomainId)` and
  its observed location matches that place.
- [ ] A pinned object remains there; a migratable object can later move while
  preserving `ObjectId`, leases, roots, and calls.
- [ ] `Remote::call` executes at the authoritative object location and follows
  a concurrent migration without duplicate action execution in the supported
  failure model.
- [ ] `Exclusive` objects may execute consecutive actions on different Rayon
  workers but never expose concurrent access and require only `T: Send`.
- [ ] `ReadWrite` objects run read actions concurrently, serialize write access,
  enforce the required `Send + Sync` bounds, and never block a Rayon worker
  while waiting for admission.
- [ ] Object guards, runtime admission gates, and user lock guards never cross
  `.await`, remote calls, migration snapshots, or transport boundaries.
- [ ] Hard colocated tasks never execute on a fallback locality.
- [ ] Soft colocated tasks prefer the current object location and apply only
  their declared bounded fallback.
- [ ] Placement affinity does not bypass admission, queue limits, or already
  admitted work.
- [ ] Dropping handles, roots, futures, deadlines, failed placement, and
  migration races release every lease, pin, ticket, and pending entry.
- [ ] Same-locality paths report zero serialization and transport messages.
- [ ] Batch `pmap` placement performs accounting and resolution per batch, not
  per item.
- [ ] No hidden security framework or dependency is introduced for the trusted
  fixed-membership initial runtime.

## 13. Provenance

The identity/location split and object-colocated action model are conceptually
informed by HPX components, actions, localities, and AGAS as described in HPX's
public documentation. Hataori's API and implementation are independently
designed in Rust; no HPX implementation source is copied or translated by this
design.
