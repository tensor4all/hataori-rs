# Hataori distributed runtime architecture

**Status:** Accepted long-term direction; phases A-D and Phase E algorithm/facade correctness implemented; Phase E performance acceptance pending a valid-host PASS report

**Tracking:** [hataori-rs#14](https://github.com/tensor4all/hataori-rs/issues/14)

**Compatibility:** The future runtime may replace the current P0/P1 public API
without a compatibility layer

**Facade decision:** A current-style `map`/`map_in`/`pmap` sugar layer is an
accepted required part of Phase E

**Performance decision:** Phase E is not accepted until the new facade passes
the fixed P0/P1 non-regression and sugar-transparency gates in Section 15.3

**Placement decision:** Exact object placement, object-colocated tasks, and
Rayon-thread sharing follow
[`object-placement-api.md`](object-placement-api.md)

**Governance:** The repository is expected to move to the tensor4all GitHub
organization. Cross-repository development policy comes from
[`tensor4all-agent-rules`](https://github.com/tensor4all/tensor4all-agent-rules);
Hataori-specific invariants live in `REPOSITORY_RULES.md`.

## 1. Purpose and positioning

The implemented Hataori P0/P1 system is a synchronous collective data-parallel
engine over MPI and rank-local Rayon domains. Its root coordinator dynamically
distributes fixed input batches and returns ordered results. That design remains
documented in [`../design.md`](../design.md) and
[`bounded-prefetch.md`](bounded-prefetch.md).

The unpublished `hataori-runtime-foundation` workspace crate implements the
Phase A protocol, deterministic memory backend, TCP rendezvous/transport, and
MPI transport described below. The unpublished `hataori-runtime` crate implements the Phase B owner-thread
lifecycle and structured execution described in
[`phase-b-runtime.md`](phase-b-runtime.md), the Phase C remote-object boundary
in [`phase-c-objects.md`](phase-c-objects.md), and Phase D explicit migration in
[`phase-d-migration.md`](phase-d-migration.md), and the Phase E algorithm/facade
boundary in [`phase-e-algorithms.md`](phase-e-algorithms.md). The frozen Phase E
performance gate remains pending.

The long-term system is a long-lived Rust distributed runtime. It should support:

- MPI and TCP as interchangeable production transports;
- transport-independent typed parcels and stable locality identities;
- structured local and remote asynchronous execution;
- typed remote actions and futures;
- long-lived remote objects whose identity is independent of their location;
- mechanical resource cleanup using RAII, bounded leases, deadlines, quotas,
  and observable shutdown;
- explicit object migration, with future room for placement policy and automatic
  migration;
- exact object creation at a logical place, hard/soft object-colocated task
  placement, and synchronized access from the owning domain's Rayon workers;
- `pmap`, collectives, and dataflow as algorithms above the runtime rather than
  as runtime lifecycle primitives;
- a supported current-style sugar facade that keeps simple local and
  distributed data-parallel calls concise without preserving the P0/P1
  implementation;
- optional tensor adapters without tenferro or tensor4all types entering the
  runtime core.

The architecture is informed by HPX's separation of schedulers/executors,
parcels, futures/actions, and AGAS/components. Hataori does not aim to copy HPX
source or reproduce every HPX facility. It adopts the useful layering while
retaining Rust ownership, explicit typed errors, bounded resource contracts,
and a smaller initial scope.

## 2. Immediate scope and non-goals

The first production transports are MPI and TCP. A private in-memory transport
exists only for deterministic tests and fault injection.

The initial runtime assumes fixed membership for the lifetime of one run. MPI
membership comes from the supplied/owned MPI world. TCP membership comes from a
bounded rendezvous phase. Dynamic node join, elastic membership, rank recovery,
and consensus-backed control state are deferred.

The initial remote object implementation uses fixed placement behind a
location resolver. Public IDs and calls must nevertheless be migration-ready
from the first implementation.

The following are not required initially:

- automatic object migration or global load balancing;
- distributed cycle collection;
- exactly-once execution across process crashes;
- simultaneous use of multiple production transports;
- native transport collectives as a base runtime requirement;
- RDMA, GPU-direct, or transport-specific zero-copy APIs;
- dynamic plugin loading;
- node-level work stealing;
- tolerance of arbitrary network partitions.
- TLS, authentication, authorization, ACLs, sandboxing, multi-tenant isolation,
  or hostile-peer defenses for the initial trusted fixed-membership runtime.

## 3. Design principles

### 3.1 Stable logical identity

Logical identifiers do not expose a transport address or current placement.
`LocalityId`, `TaskId`, `RequestId`, `ActionId`, and `ObjectId` are runtime
protocol values. MPI ranks and TCP endpoints belong to backend membership
tables.

### 3.2 One owner for every resource

Transport drivers own transport handles. Runtime objects own schedulers,
registries, directories, tables, queues, and caches. RAII guards own temporary
admission, calls, migrations, transfers, and leases. Global or thread-local
hidden ownership is not part of the design.

### 3.3 Bounded-by-construction operation

Every queue and retained table has a capacity, deadline, lease, or documented
durable root. Backpressure is a public operational outcome rather than an
implicit request to allocate more memory.

### 3.4 Structured concurrency by default

Ordinary work belongs to a scope or a must-use handle. Detached work requires
an explicit permit and quota. Dropping a future removes its local waiter and
initiates best-effort cancellation.

### 3.5 Transport-independent runtime semantics

Action completion, object lifetime, migration, redirects, deadlines, and error
classification are expressed in the Hataori protocol. MPI and TCP only move
parcels and report transport progress/failure.

### 3.6 Same semantics before backend-specific optimization

MPI and TCP pass the same conformance suite. MPI collectives, TCP sendfile-like
paths, and future zero-copy facilities may optimize an accepted operation but
must not define its only semantics.

## 4. Layered architecture

```text
Application
              |
hataori public facade and tensor adapters
              |
Algorithms: pmap, collectives, dataflow, distributed containers
              |
Remote objects: handles, directory, leases, migration
              |
Actions and futures: typed calls, scopes, completion routing
              |
Runtime services: membership, scheduling, progress, shutdown
              |
Transport contract: handle, driver, events, backpressure
              |
MPI transport                    TCP transport
              |
Protocol: IDs, parcels, framing, versions, errors
              |
Executors: domains, pools, affinity, admission
```

The intended crate decomposition is:

```text
hataori                    (existing; public facade and current-style sugar)
hataori-protocol
hataori-executor
hataori-transport
hataori-transport-mpi
hataori-transport-tcp
hataori-runtime
hataori-objects
hataori-algorithms
hataori-tenferro           (existing; adapters/tenferro)
```

The exact number of crates may be introduced incrementally. The ownership and
dependency boundaries are required even if several layers temporarily share a
crate. The facade depends downward on algorithms/runtime and contains no
independent scheduler, protocol state machine, transport, or lifetime owner.

## 5. Identity model

Representative logical identifiers are newtypes with checked construction and
useful `Debug` summaries:

```rust
struct RunId(u128);
struct LocalityId(u64);
struct TaskId(u128);
struct MessageId(u128);
struct ActionId(u128);
struct ObjectTypeId(u128);

struct RequestId {
    origin: LocalityId,
    sequence: u64,
}

struct ObjectId {
    run: RunId,
    unique: u128,
}
```

`ObjectId` does not include a home locality, local slot, generation, epoch, or
address. IDs are not reused within a run. Location is separate:

```rust
struct ObjectLocation {
    locality: LocalityId,
    slot: u64,
    generation: u32,
    epoch: u64,
}
```

The directory owns the authoritative `ObjectId -> ObjectLocation` mapping.
Local caches may retain mappings, but every object call carries its observed
epoch and the destination validates it.

## 6. Runtime lifecycle

The runtime is long-lived rather than reconstructed for each collective call:

```text
Created -> Bootstrapping -> Running -> Draining -> Stopped
                            |
                            +--------> Failed
```

A representative entry point is:

```rust
let runtime = Runtime::builder()
    .transport(TransportConfig::Mpi(mpi_config))
    .limits(limits)
    .start()?;

runtime.block_on(async {
    // Actions, remote objects, migration, and algorithms.
})?;

runtime.shutdown()?;
```

The final API may offer explicit `mpi` and `tcp` builder methods. The public
`Runtime` type is not generic over the concrete transport backend. Internally,
it owns transport parts behind an object-safe interface so adding a backend
does not infect actions, objects, or user types with another generic parameter.

Shutdown proceeds in phases:

1. reject new user work;
2. cancel or drain scoped work according to its contract;
3. finish or fail active object calls and migrations;
4. flush control and release messages;
5. release locality leases and durable runtime-owned roots;
6. flush and stop the transport driver;
7. report retained resources and fail tests if non-durable resources remain.

`Drop` performs only bounded best-effort cleanup because asynchronous shutdown
cannot be guaranteed from a destructor. Correct applications call explicit
shutdown; debug/test builds make incomplete shutdown visible.

## 7. Bootstrap and membership

Bootstrap discovers the fixed membership and constructs backend endpoint
tables. Runtime layers use only `LocalityId` afterward.

### 7.1 MPI bootstrap

The MPI backend maps localities to ranks and owns the communicator/request
lifecycle needed for parcel traffic. The future runtime may own MPI
initialization or accept an explicitly transferred MPI runtime resource; it
must not rely on unsafe `Send`/`Sync` implementations for communicators.
The current crate already splits the MPI binding into the mutually exclusive
`mpi` (link-time rsmpi) and `rsmpi-rt` (runtime-loaded MPIABI) features; the
MPI transport crate must support both.

The transport driver is the only owner allowed to call MPI. The preferred
implementation uses one progress owner even when the available MPI thread level
would permit more concurrent calls.

### 7.2 TCP bootstrap

One configured endpoint initially acts as a bounded rendezvous service:

```text
TCP process starts listener
          |
process joins rendezvous with RunId and endpoint
          |
rendezvous assigns LocalityId and publishes membership
          |
peers establish run-scoped connections
          |
all peers complete protocol handshake
          |
runtime enters Running
```

The initial TCP membership is static after bootstrap. Reconnection may repair a
connection within the same run but does not create a new locality or imply
process-failure recovery.

### 7.3 In-memory test bootstrap

Tests construct several virtual localities in one process with bounded queues.
The backend can deterministically inject delay, loss, duplication, reordering,
saturation, and disconnect. It is not a supported production transport.

## 8. Transport contract

The transport separates a thread-safe submission handle from the driver that
owns progress resources:

```rust
trait TransportHandle: Send + Sync {
    fn try_send(
        &self,
        parcel: OutboundParcel,
    ) -> Result<SendTicket, TransportError>;
}

trait TransportDriver: Send {
    fn local_id(&self) -> LocalityId;
    fn members(&self) -> &[LocalityId];
    fn progress(
        &mut self,
        events: &mut Vec<TransportEvent>,
    ) -> Result<Progress, TransportError>;
    fn flush(&mut self) -> Result<(), TransportError>;
    fn shutdown(&mut self) -> Result<(), TransportError>;
}
```

The exact Rust signatures remain implementation decisions. The semantic
contract requires:

- reliable point-to-point delivery or a visible peer/transport failure;
- peer-specific ordering, or protocol sequence numbers that restore it;
- bounded outbound and inbound backpressure;
- local-send completion when buffers may be reused;
- incoming parcel events;
- explicit flush and shutdown;
- maximum frame/payload limits;
- progress that cannot be indefinitely starved by user computation.

The base contract does not require collective operations, zero-copy, one-sided
operations, RDMA, or backend-specific buffer registration. Those may appear as
optional extension capabilities after a measured need.

## 9. Protocol and parcels

Every connection begins with a bounded handshake:

```rust
struct Hello {
    run_id: RunId,
    protocol_version: u32,
    runtime_version: RuntimeVersion,
    action_registry_hash: [u8; 32],
    object_registry_hash: [u8; 32],
    capabilities: Capabilities,
    limits: AdvertisedLimits,
}
```

The hashes detect accidental binary/schema disagreement; they are not an
anti-tamper mechanism. Hataori initially assumes trusted job participants.

Parcels are transport-independent:

```rust
struct ParcelHeader {
    protocol_version: u32,
    run_id: RunId,
    message_id: MessageId,
    channel: Channel,
    kind: MessageKind,
    source: LocalityId,
    destination: LocalityId,
    payload_len: u64,
    trace_id: Option<TraceId>,
}

enum Channel {
    Control,
    Action,
    Bulk,
}
```

Logical channel separation ensures lease renewal, redirects, cancellation,
directory traffic, and shutdown are not trapped behind large tensor payloads.
An implementation may initially multiplex channels over one MPI communicator
or one TCP connection, but it preserves independent bounded queues and control
priority.

Payloads permit segmentation and streaming:

```rust
enum Payload {
    Inline(Bytes),
    Segmented(Vec<BufferSegment>),
    Stream(StreamDescriptor),
}
```

The initial codecs may copy segments into backend frames. The public/runtime
model must not require one unbounded contiguous allocation, because tensor
transfer and object migration will need chunking.

Completion has three distinct meanings:

- `LocalSendComplete`: source buffers may be reused;
- `RemoteReceived`: the destination runtime accepted the parcel;
- `ActionComplete`: user action execution produced a result or error.

Transport completion never completes a `RemoteFuture` by itself.

## 10. Executors and domains

Each locality owns a `DomainRegistry`. A domain is a bounded local execution
resource with an executor/pool, CPU placement, worker budget, ownership mode,
and admission policy. The first runtime may still configure one domain per
locality, but `DomainId` is no longer restricted to zero in the model.

The scheduler hierarchy is:

```text
Global placement policy
          |
Locality selection or object colocation
          |
Locality scheduler
          |
Domain scheduler
          |
Worker execution
```

Local scoped work may borrow stack data and must complete before its scope
returns. Remote or detached work owns serializable `Send + 'static` state.
This separation preserves useful Rust scoped execution without pretending a
borrowed stack frame can outlive or cross a process boundary.

## 11. Typed actions and futures

Hataori does not transmit arbitrary Rust closures or executable code. Every
remote action has an explicit stable ID, schema, input type, output type, and
registered executor:

```rust
trait Action: Serialize + Send + 'static {
    const ID: ActionId;
    type Output: Serialize + DeserializeOwned + Send + 'static;
}
```

The runtime maps a request to a pending promise:

```text
RequestId -> PendingPromise -> result parcel -> wake waiter
```

Representative usage is:

```rust
let result: RemoteFuture<Output> = runtime.spawn_on(place, action)?;
let output = result.await?;
```

All public remote operations support deadlines. Dropping a future removes its
local waiter and sends best-effort cancellation. A result arriving after local
cancellation is validated and discarded; it must not recreate a waiter or
retain an object pin.

Structured scopes own ordinary child tasks. Detached tasks require an explicit
quota-controlled permit and appear in runtime statistics.

## 12. Remote object model

A typed remote handle contains stable identity and local lease state, not a
location:

```rust
#[must_use]
struct Remote<T> {
    lease: Arc<LocalLease>,
    object_id: ObjectId,
    object_type: ObjectTypeId,
    marker: PhantomData<T>,
}
```

Local clones share one locality-level lease. They do not create a network
reference count update per clone. `Remote<T>` is not `Copy` and does not have an
unrestricted `Serialize` implementation.

Object actions include:

```rust
struct ObjectCallHeader {
    request_id: RequestId,
    object_id: ObjectId,
    expected_epoch: u64,
    action_id: ActionId,
}
```

The destination validates the location epoch:

- matching epoch: admit or queue the call;
- stale epoch: return `Moved` with a newer location/epoch;
- migrating object: apply the bounded migration-mailbox policy;
- collected object: return `ObjectCollected`;
- unknown object or type/action mismatch: return a typed protocol/application
  error.

The caller updates its location cache after `Moved` and may retry with the same
`RequestId`. The receiver's bounded deduplication table suppresses duplicate
execution within the current runtime session. Hataori does not initially claim
exactly-once execution across process crashes.

The accepted high-level creation, rooting, `Remote::call`, exact/soft
colocation, and Rayon reader/writer access API is specified in
[`object-placement-api.md`](object-placement-api.md). An object belongs to one
logical place. Workers in its owning domain may access it through runtime
admission; another locality always uses typed actions rather than shared memory.

## 13. Mechanical leak prevention

The directory owns object lifetime metadata independently from location:

```rust
struct DirectoryEntry {
    location: ObjectLocation,
    leases: LocalityLeaseSet,
    durable_roots: u32,
    in_flight_calls: u32,
    migration_pins: u32,
    transfer_pins: u32,
    lifecycle: ObjectLifecycle,
}
```

An object becomes collectible only when all of the following are zero or empty:

```text
durable roots
active locality leases
in-flight calls
migration pins
lease-transfer pins
retained mailbox work
```

Normal cleanup uses `Drop` and explicit release. Each locality lease also has a
bounded expiry and periodic renewal so a lost release, forgotten handle, or
failed locality cannot retain an object forever. Lease timing is conservative:
the TTL is much larger than the renewal interval and normal progress delay, and
expiry may enter a grace/suspect state before collection.

Moving a `Remote<T>` between localities uses a runtime lease-transfer protocol.
The source or directory holds a transfer pin until the destination imports and
acknowledges its lease. A reference in transit therefore cannot disappear from
lifetime accounting.

Reference cycles are constrained by type/API design. Persistent object state
uses `WeakRemote<T>` by default. Upgrading a weak reference obtains a live
lease through the directory. Strong remote handles do not serialize into object
state accidentally; an explicitly supported persistent strong edge must name
its ownership and cycle policy.

All retained runtime structures are bounded:

- pending promises have deadlines;
- object mailboxes and action queues have capacities;
- forwarding entries, tombstones, and deduplication results have TTLs;
- location caches have entry/byte limits and clear/stats APIs;
- outbound/inbound queues track parcel count and bytes;
- detached tasks require permits;
- buffer pools and caches have explicit runtime owners and retained-byte stats.

Tests assert a clean non-durable shutdown state, including zero live objects,
leases, pending calls, queued parcels, detached tasks, and migration pins.

## 14. Object mobility

Object implementations declare their mobility:

```rust
enum Mobility {
    Pinned,
    Reconstructible,
    Migratable,
}
```

- `Pinned`: cannot leave its current locality.
- `Reconstructible`: logical state moves, while locality-owned runtime/provider
  resources are rebuilt at the destination.
- `Migratable`: a registered snapshot fully captures transferable logical
  state.

Communicators, sockets, executor pools, admission guards, provider handles,
device contexts, caches, and raw addresses are never migratable state. Tensor
adapters move logical tensor data and metadata, then reconstruct destination
resources in the target domain/context.

A migration follows this state machine:

```text
Resident(epoch N)
       |
Freezing
       |
Transferring
       |
Prepared at destination
       |
Directory commit(epoch N -> N+1)
       |
Destination resident(epoch N+1)
       |
Source forwarding
       |
Source retired
```

Detailed protocol:

1. acquire a migration pin and close new source admission;
2. finish or cancel already-admitted calls according to their contract;
3. freeze the object into a versioned bounded/streamed snapshot;
4. allocate a prepared destination slot and transfer/restore the snapshot;
5. receive destination `Prepared` acknowledgement;
6. atomically update the authoritative directory location and epoch;
7. activate the destination object;
8. retain a bounded source forwarder for stale cached locations;
9. retire the source slot after the forwarding/tombstone policy permits it;
10. release the migration pin.

Before directory commit, failure may roll back to the source. After commit, the
new directory epoch is authoritative. A runtime without crash-tolerant
consensus treats loss of the directory authority or an ambiguous post-commit
process failure as a run-level failure rather than claiming transparent
recovery.

Migration changes location and epoch only. `ObjectId`, leases, durable roots,
and logical ownership remain unchanged.

## 15. Algorithms above the runtime

`pmap` is rebuilt from runtime facilities:

```text
bounded work controller
       |
spawn_on / object-colocated call
       |
RemoteFuture results
       |
when_all / ordered collection
```

The controller may retain the existing useful properties: stable task IDs,
bounded resident work, exactly-once evaluation in a successful no-retry run,
ordered results, deterministic error selection for the algorithm, and explicit
quiescence. These are `pmap` semantics, not requirements imposed on every
runtime action.

Broadcast, scatter, gather, barriers, and reductions are initially implemented
over the base parcel contract. An MPI-specific extension may later accelerate
them with native collectives after parity and reuse are demonstrated. TCP keeps
the reference semantics.

Object-aware scheduling uses the directory as a placement oracle without
making every task a remote object. `Remote::call` has hard colocation semantics;
general tasks may require or prefer an object's current place. Soft preference
has an explicit bounded fallback and does not mean queue priority. The full
contract is in [`object-placement-api.md`](object-placement-api.md).

### 15.1 Current-style sugar facade

The top-level `hataori` crate provides a supported facade resembling the useful
parts of the current API. It is a canonical user-facing layer, not a deprecated
compatibility shim. Similarity means familiar concepts and a short call path;
it does not require the old collective participation model, `Option` return
shape, communicator argument, public-field layout, or exact trait bounds.

The target mapping is:

| Implemented P0/P1 concept | Future facade |
|---|---|
| `map(items, closure)` | Same local borrowed-closure operation |
| `map_in(domain, mode, items, closure)` | Same explicit-domain local operation |
| `Domain::sequential/managed/external` | Facade constructors registered into a runtime `DomainRegistry` |
| `LocalMode::{Sequential, Outer, Inner}` | Same fan-out vocabulary, with one declared fan-out owner |
| `PmapOptions { batch_size, local_mode, prefetch, .. }` | Same scheduling vocabulary through a canonical builder/value |
| collective `pmap(world, domain, options, root_items, closure)` | initiator-only async `pmap(runtime, domain, options, items, action)` |
| `broadcast/scatter/gather(world, ...)` | initiator/runtime collectives over logical localities |
| `(rank, domain_id)` place | transport-independent `(LocalityId, DomainId)` place |

Illustrative target usage follows. It is design syntax, not a currently
compilable example:

```rust
#[hataori::action]
fn count_for_a(input: PiInput) -> Result<i64, PiError> {
    // Registered on every participating locality.
    todo!()
}

let domain = runtime.domains().sequential()?;
let values = hataori::pmap(
    &runtime,
    domain,
    PmapOptions::default().batch_size(10)?,
    inputs,
    COUNT_FOR_A,
)
.await?;
```

The action macro or equivalent registration API produces a typed, stable,
zero-sized action descriptor. Distributed sugar does not accept an arbitrary
closure because closures and executable code are not transmitted. Local
`map`/`map_in` retain borrowed closures and avoid serialization or `'static`
bounds where execution remains scoped.

`pmap` is invoked only by the initiating locality. It owns `Vec<T>` and receives
`Vec<U>` directly; service localities need not enter the same call or pass
`None`. A separate `hataori::blocking` facade may provide the same shape for a
non-runtime thread by driving the returned future through the existing runtime.
Blocking entry from a runtime worker is rejected rather than creating a nested
executor or risking deadlock.

### 15.2 Sugar lowering and fast path

The facade constructs a canonical `PmapPlan` and calls the same
`hataori-algorithms` controller available to explicit runtime users:

```text
hataori::pmap sugar
       |
validate options and build PmapPlan
       |
canonical bounded batch controller
       |
one typed action/promise per in-flight batch
       |
domain execution or one transport parcel sequence
```

The facade must not add another task graph, scheduler, mailbox, retry layer, or
result reorder pass. Input indexing and the ordered result table are allocated
once by the canonical controller. A batch is encoded at most once for one
remote dispatch and decoded at most once at its destination; returned values
have the corresponding single encode/decode boundary. Same-locality batches
are owned moves and bypass the action codec, transport, directory, leases, and
remote-object resolver.

`pmap` accounting is batch-granular. There is one pending promise and bounded
admission unit per in-flight batch, not one `RemoteFuture`, allocation, or
directory lookup per item unless `batch_size == 1`. Remote objects are not used
to represent ordinary `pmap` inputs or results.

The default options preserve the low-overhead current behavior: bounded dynamic
scheduling, ordered results, no automatic migration, no tracing payload when
tracing is disabled, and no mandatory placement-policy computation beyond the
selected domain/locality set. TCP and MPI share the semantics; an accepted MPI
fast path may use backend capabilities without changing the facade contract.

### 15.3 Legacy performance acceptance gate

The performance reference for “current Hataori” is commit
`34cb1b1371c8b2f8ef750e2d49d10f9ef8f0782e` (2026-08-23). Before runtime work
changes or removes the P0/P1 path, Phase A freezes a buildable benchmark runner
for that exact commit. `benchmarks/performance/manifest.toml` freezes the case
families, host gates, statistics, thresholds, and TCP targets, with
`scripts/check-performance-manifest.py` enforcing its hard requirements. The
baseline and manifest are immutable after candidate measurements begin.
Existing `mpi_pi_pmap`, `mpi_mandelbrot_pmap`, and
`mpi_mandelbrot_hybrid` workloads seed the suite but do not by themselves
constitute the full gate.

Two comparisons are required:

1. **Legacy continuity:** future facade versus the fixed P0/P1 implementation
   for the same successful logical workload on MPI and local/Rayon paths.
2. **Sugar transparency:** future facade versus direct use of the canonical new
   algorithm API at the same candidate commit.

The predeclared case manifest covers at least:

- serial `map` and explicit-domain `map_in` at small, medium, and large item
  counts;
- world size one, latency-dominated small actions, compute-dominated actions,
  and payload-dominated batches;
- MPI at 2, 4, and, where the declared verification host supports it, 8
  localities;
- hybrid rank/thread combinations with `Sequential`, `Outer`, and `Inner`, and
  both accepted prefetch settings;
- batch sizes 1, the current measured π value 10, and a coarse batch;
- ordered result collection and payload sizes spanning inline data through at
  least 1 MiB segmented transfer;
- broadcast, scatter, and gather for small control-like and large payloads;
- repeated calls on one running runtime so startup is not hidden inside the
  operation metric.

Runtime bootstrap and shutdown have their own reported latency and retained-
resource measurements. They are not mixed into the steady-state operation
metric because the new runtime is intentionally long-lived, but they may not be
silently omitted from the benchmark report.

Before any candidate run, the worklog records the baseline/candidate commits,
benchmark source revision, optimized build profile, hardware and CPU affinity,
MPI implementation, network path, rank/thread/provider settings, complete case
manifest, warm-up and repetition counts, comparison statistic, host-noise
observables, and validity thresholds. Baseline and candidate execute as one
paired, interleaved experiment. Selective retries or post-hoc case removal are
not permitted.

For duration, the primary statistic is the paired candidate/baseline ratio with
a one-sided 95% confidence interval. The measurement tolerance is derived from
a baseline-versus-baseline noise calibration before the candidate is run and
is capped at 2%. If the host cannot resolve a 2% regression, the experiment is
`INCONCLUSIVE`, not a pass. Acceptance requires all of the following:

- the upper confidence bound of every required latency ratio is at most
  `1 + tolerance`;
- the geometric mean point estimate across the predeclared primary cases is at
  most `1.00`;
- no required throughput, scaling-efficiency, peak incremental memory, or
  correctness gate regresses beyond its predeclared measurement tolerance;
- sugar versus canonical lowering shows no additional payload copy,
  serialization, parcel sequence, scheduler/queue hop, or per-item heap
  allocation, and its timing passes the same capped transparency test;
- same-locality instrumentation reports zero serialization and zero transport
  messages (today this exists only as a test-only codec counter in
  `src/wire.rs` exercised by the `HATAORI_CODEC_TEST`-gated test; the runtime
  must add equivalent instrumentation);
- all semantic, bounded-memory, shutdown, and leak counters pass in the same
  build being measured.

The tolerance describes measurement resolution, not an accepted slowdown
budget. Any required case outside the bound is `FAIL`; invalid host/noise data
is `INCONCLUSIVE`. Both classifications block Phase E acceptance and block
removal or replacement of the P0/P1 implementation. TCP has no legacy
performance comparator, so it must pass sugar transparency plus absolute
latency/throughput/scaling targets declared before its candidate is measured.

## 16. Errors, cancellation, and failure boundaries

Errors are typed by ownership layer:

- configuration/bootstrap errors;
- transport/peer errors;
- protocol/version/schema errors;
- resource exhaustion and deadline errors;
- remote action/user errors;
- object lookup, stale location, lifecycle, and migration errors;
- runtime shutdown or failed-state errors.

Cancellation is best effort. A cancelled queued action may never start. A
running action may finish unless its implementation observes a cancellation
token. Assigned bulk transfer and migration frames are drained or the peer/run
is failed according to the protocol; cancellation does not abandon announced
buffers or lifetime pins.

MPI process loss commonly remains a run-level failure. TCP disconnect is a
visible peer failure and may reconnect within the same fixed membership, but
initial Hataori does not claim application continuity after process loss.

## 17. Backpressure and limits

Runtime configuration includes conservative bounded defaults:

```rust
struct RuntimeLimits {
    max_frame_bytes: usize,
    max_inflight_bytes_per_peer: usize,
    max_queued_parcels_per_peer: usize,
    max_pending_calls: usize,
    max_action_queue_per_domain: usize,
    max_object_mailbox: usize,
    max_live_objects: usize,
    max_detached_tasks: usize,
    max_location_cache_entries: usize,
    max_location_cache_bytes: usize,
}
```

Counts and byte lengths use checked arithmetic. Limit exhaustion returns a
typed error or waits through an explicit asynchronous capacity permit; it does
not silently allocate an unbounded fallback queue.

Control-plane reservations ensure that saturation by action/bulk traffic does
not prevent lease renewal, cancellation, redirects, failure notification, or
shutdown.

## 18. Observability

Backend-independent statistics include:

- parcels and bytes sent/received by logical channel;
- queue lengths and in-flight bytes per peer;
- pending, completed, cancelled, and timed-out requests;
- action execution and queue latency;
- live objects, leases, durable roots, and retained object bytes;
- location-cache entries/bytes, redirects, and stale-epoch retries;
- active/completed/failed migrations and transferred bytes;
- detached tasks and outstanding shutdown work;
- transport progress stalls and peer failures.

`TraceId` in the parcel header permits optional distributed tracing without
making a logging framework part of the base protocol. Debug formatting must
summarize runtime state without dumping payloads or entire object/cache tables.

## 19. Deployment assumptions

Hataori initially treats all fixed-membership participants in one configured
run as trusted. It does not add TLS, authentication, authorization, ACLs,
sandboxing, tenant isolation, capability security, or hostile-peer defenses.
TCP listens only on explicitly configured interfaces/endpoints.

The handshake detects accidental cross-run, version, registry, and limit
mismatch; it is not a cryptographic identity protocol. All network-derived
sizes, counts, enum codes, IDs, versions, and allocation requests are still
validated for correctness, memory safety, and bounded resource use. This
validation is not presented as a security subsystem.

## 20. Implementation phases

### Phase A: protocol and transport foundation

The implemented boundary and acceptance evidence are detailed in
[`phase-a-transport-foundation.md`](phase-a-transport-foundation.md).

- freeze the exact P0/P1 performance baseline runner and predeclare its case
  manifest before changing the measured implementation path;
- introduce stable logical IDs, `RunId`, handshake, parcel channels, and limits;
- build the deterministic private in-memory transport and conformance suite;
- implement TCP rendezvous, framing, progress, backpressure, and shutdown;
- implement the same contract over MPI without leaking MPI types upward;
- prove clean reusable shutdown and large segmented payload transfer.

### Phase B: long-lived runtime and structured execution

Implemented in the unpublished `hataori-runtime` crate and detailed in
[`phase-b-runtime.md`](phase-b-runtime.md):

- runtime lifecycle, domain registry, scopes, pending promise table, deadlines,
  cancellation, and observability;
- typed action registration and `spawn_on`/`RemoteFuture`;
- bounded deduplication, retryable response backpressure, and control-plane
  progress.

### Phase C: fixed-placement remote objects

- add stable `ObjectId`, directory, location resolver/cache, object store,
  typed object actions, locality leases, weak handles, and clean collection;
- add exact `create_at`, runtime-owned roots, hard/soft object colocation, and
  `Exclusive`/`ReadWrite` Rayon access from the owning domain;
- initially mark all objects pinned while preserving migration-ready routing;
- verify lease transfer, expiry, dropped futures, and zero-resource shutdown.

### Phase D: explicit migration

Implemented in `hataori-runtime` and detailed in
[`phase-d-migration.md`](phase-d-migration.md):

- mobility capabilities, bounded segmented freeze/restore snapshots, epoch
  commit, redirects, forwarding, rollback-before-commit, and manual `migrate`;
- reconstructible object behavior that rebuilds destination-local resources
  without transferring executor or context identity.

### Phase E: algorithms and policy

Implemented algorithm/facade correctness is detailed in
[`phase-e-algorithms.md`](phase-e-algorithms.md); performance promotion remains
blocked until every frozen case passes on a valid host.

- rebuild `pmap` and collectives over actions/futures;
- implement the current-style facade and blocking entry, both lowering directly
  to the canonical batch controller;
- expose object-aware placement in `spawn` and batch-oriented `pmap` without
  adding per-item directory lookups or promises;
- pass the legacy-continuity and sugar-transparency performance gates before
  replacing or removing the P0/P1 path;
- add object-colocated placement and explicit placement policies;
- consider automatic migration only after measured workloads define a stable
  benefit, bounded policy state, and failure behavior.

## 21. Acceptance gates

Every phase requires:

- protocol/state-machine tests independent of I/O;
- the shared transport contract on MPI and TCP where applicable;
- deterministic fault-injection coverage in the in-memory test backend;
- typed public errors and runnable public examples;
- bounded queue/table/cache defaults and retained-byte statistics;
- cleanup coverage for success, error, deadline, cancellation, disconnect,
  migration rollback, and shutdown;
- documentation that distinguishes implemented behavior from future design;
- focused performance measurements before promoting an optimization that adds
  semantic or maintenance complexity.

Phase E is accepted only when every required case in Section 15.3 is `PASS`.
`FAIL` and `INCONCLUSIVE` both block promotion. The report must include the
fixed legacy commit, candidate commit, complete paired results and confidence
intervals, host-validity observations, allocation/message instrumentation, and
the sugar-versus-canonical comparison. Passing correctness tests alone is not
sufficient to replace the current implementation.

The final repository-scale implementation program also requires the shared
tensor4all cross-phase audits for specification/architecture, safety/resource
lifecycle, performance/parallelism, public API/documentation, and relevant
backend/hardware lanes.

## 22. Open decisions

The following decisions should be made at the phase that first needs them:

- exact async executor implementation and whether it reuses Rayon internally;
- MPI initialization ownership and minimum accepted MPI thread level;
- TCP event-loop library and reconnection policy;
- directory implementation after the initial single authority;
- lease TTL, renewal, and grace defaults;
- snapshot schema evolution and compatibility policy;
- whether strong persistent remote edges are prohibited or supported through an
  explicitly tracked ownership graph;
- when native MPI collectives or zero-copy paths are justified by measurement;
- exact action-registration syntax and the final builder spelling for the
  current-style facade, while preserving the accepted lowering contract;
- exact private reader/writer admission primitive for shared local objects;
- reconciling Section 11's value-carrying `Action` trait with Section 15.1's
  zero-sized action descriptor generated by `#[hataori::action]`.

Deferring these choices is intentional. The invariants above keep the choices
open without adding unused abstractions.

## 23. References and provenance

This document uses HPX as a conceptual architecture reference, not as a source
for translated implementation code:

- [HPX documentation](https://docs.hpx.dev/branches/master/html/)
- [HPX distributed applications: AGAS, actions, and components](https://docs.hpx.dev/latest/html/manual/writing_distributed_hpx_applications.html)
- [HPX runtime resources and schedulers](https://docs.hpx.dev/latest/html/manual/hpx_runtime_and_resources.html)
- [HPX LCI parcelport](https://docs.hpx.dev/latest/html/manual/using_the_lci_parcelport.html)

If future implementation reads or adapts specific third-party source files,
the affected Hataori source must record that reference and its derivation or
independent-validation status at implementation time.
