# Hataori P0 Design

**Status:** Implemented (P0 core); pre-implementation review verdict `Correct-to-merge`. See [`core-acceptance-ledger.md`](core-acceptance-ledger.md) for the delivered matrix.

**Scope:** A generic synchronous collective data-parallel engine for MPI ranks and rank-local Rayon execution domains

**Tracking:** [hataori-rs#1](https://github.com/shinaoka/hataori-rs/issues/1), [tenferro-rs#1716](https://github.com/tensor4all/tenferro-rs/issues/1716), [tensor4all-rs#663](https://github.com/tensor4all/tensor4all-rs/issues/663)

## 1. Positioning

### Name

**Hataori** comes from the Japanese word **機織り** (*hataori*), meaning weaving on a loom. The name reflects the engine's job: weave independent strands of work across MPI ranks and Rayon threads into one ordered result.

### Scope

Hataori makes common local, process-parallel, and hybrid data-parallel execution easy:

| Cargo features | Execution mode |
|---|---|
| none | single process, single thread |
| `rayon` | single process, multiple threads |
| `mpi` | multiple processes through upstream rsmpi, one compute lane per rank |
| `mpi,rayon` | upstream rsmpi plus one multithreaded execution domain per rank |
| `rsmpi-rt` | runtime-loaded MPI, one compute lane per rank |
| `rsmpi-rt,rayon` | runtime-loaded MPI plus one multithreaded execution domain per rank |

Hataori is inspired by Distributed.jl's dynamically scheduled `pmap`, but P0 is not a Distributed.jl-equivalent distributed object runtime. It does not provide arbitrary remote calls, futures, remote object handles, distributed garbage collection, or dynamic process creation.

The default cross-rank path uses serialization. Applications needing zero-copy communication, MPI derived datatypes, GPU-aware MPI, custom collectives, or specialized communication should use MPI directly.

### Dependency policy

The default build is standard-library-only. P0 keeps one crate with optional features rather than splitting every backend into a separate abstraction crate:

| Feature | Direct dependency added |
|---|---|
| `rayon` | `rayon` |
| `mpi` | upstream `mpi` (rsmpi), `serde`, and `bincode` |
| `rsmpi-rt` | [`tensor4all/rsmpi-rt`](https://github.com/tensor4all/rsmpi-rt) as package `mpi`, plus `serde` and `bincode` |
| Linux managed affinity | `libc` |
| `tenferro` | `tenferro-tensor` (pinned git revision), used only by the Mandelbrot examples under `examples/`; requires Rust >= 1.96 because the pinned tenferro-rs workspace sets `rust-version = "1.96"`, so it is excluded from the Rust 1.85 feature matrix |

Optional integration stays in separate workspace crates so those dependency
trees never enter ordinary Hataori builds; the default, `rayon`, and MPI trees
contain no tenferro or tensor4all crate (enforced by `scripts/check-core.sh`
and `scripts/check-tenferro.sh`), and the opt-in `tenferro` example feature is
built and run by `scripts/check-tutorial-examples.sh`, which builds and runs
the Mandelbrot examples with the `tenferro` feature. The tenferro-only
`hataori-tenferro` adapter is pinned to tenferro merge `74aa7fac31490f7773978e6fc2b1e80391018b73`
and binds Hataori lanes to a tenferro held session
([adapter design](design/tenferro-adapter.md)); Phase 20b adds tensor4all
contexts and reconstruction after tensor4all-rs#663.

Hataori uses standard-library collections, synchronization, channels, and errors. P0 does not add an async runtime, general channel/locking crate, error-derive crate, logging facade, CLI framework, topology library, or generic transport/executor plugin system.

#### MPI backend features

`mpi` and `rsmpi-rt` provide the same Hataori API and are mutually exclusive. Enabling both is a compile error. Cargo features are additive, so every crate in one final dependency graph must select the same Hataori MPI backend; adapter crates forward these features rather than choosing a backend internally.

- `mpi` pins upstream `mpi = "=0.8.1"` with `default-features = false`; its mandatory build/link-time `mpi-sys` backend remains active while `user-operations`, `derive`, and `complex` stay disabled.
- `rsmpi-rt` uses the API-compatible rsmpi v0.8.1 fork at revision `6db6a2d6f96115b17c9a925e53ce719797c15dbb`, with `default-features = false` and only `mpi-rt-sys-backend`. Moving the pin requires the same compile and multi-rank gates as the initial pin.
- Hataori does not enable rsmpi's optional `user-operations`, `derive`, or `complex` features. P0 needs only built-in MPI operations and manually encoded protocol headers.
- A tiny private module aliases the selected crate as the MPI backend. Hataori does not introduce a public transport trait solely to hide two API-compatible implementations.

The runtime-loaded backend requires no C compiler, MPI headers, system MPI, or libclang at Hataori build time. At runtime it requires an MPIABI-compatible MPIwrapper library selected through `MPI_RT_LIB`; symbol loading is provided by `mpi-rt-sys`/`libloading`.

Hataori accepts a communicator and does not require ownership of MPI initialization or finalization. This lets an `rsmpi-rt` build share the MPI library and communicator initialized by MPI.jl or mpi4py. Both backends retain the same `MPI_THREAD_FUNNELED` and initialization-thread rules.

## 2. Architectural boundary

### 2.1 Core

Hataori core owns only generic execution concerns:

- local `map` and collective `pmap`;
- coordinator-driven dynamic scheduling;
- `Place = (rank, domain_id)`;
- rank-local execution-domain ownership and admission;
- blocking MPI transport and generic serialization envelopes;
- managed or caller-owned affinity;
- lifecycle, ordering, and error convergence.

Core treats input `T`, output `U`, and error `E` as generic values. It has no tensor, backend, cache, or provider semantics.

### 2.2 Optional integration adapter

Integration is split at the upstream contract boundary:

```text
hataori-tenferro ---> hataori
       |
       +-----------> tenferro

future tensor adapter ---> hataori-tenferro
       |
       +-----------------> tensor4all ---> tenferro

tenferro   -X-> hataori
tensor4all -X-> hataori
```

`hataori-tenferro` owns the tenferro held-session handle, phase-lane entry,
two-way cancellation bridging, and typed error mapping. The future tensor
adapter owns tensor4all explicit contexts and domain-owned runtimes/caches,
tensor logical wire representation and target-context reconstruction, and
context-specific errors.

Neither tenferro nor tensor4all may depend on Hataori types. Hataori core must build without either dependency unless the opt-in `tenferro` example feature is requested.

## 3. Public execution model

The public calls have this argument and return shape; feature-specific implementations apply the exact bounds in Section 12:

```rust
pub struct PmapOptions {
    pub root: i32,
    pub batch_size: NonZeroUsize,
    pub local_mode: LocalMode,
    pub prefetch: bool,
}

pub fn map<T, U, E, F>(items: Vec<T>, f: F)
    -> Result<Vec<U>, MapError>;

#[cfg(feature = "rayon")]
pub fn map_in<T, U, E, F>(
    domain: &Domain,
    mode: LocalMode,
    items: Vec<T>,
    f: F,
) -> Result<Vec<U>, MapInError>;

/// Lanes-based local fan-out; `map_in` `Outer` and hybrid `pmap` use it.
pub fn map_in_lanes<L, T, U, E, F>(
    lanes: &mut L,
    items: Vec<T>,
    f: F,
) -> Result<Vec<U>, MapInError>
where
    L: Lanes;

#[cfg(feature = "rayon")]
pub fn pmap_lanes<L, C, T, U, E, F>(
    world: &C,
    lanes: &mut L,
    options: PmapOptions,
    root_items: Option<Vec<T>>,
    f: F, // Fn(T, &mut L::Lane<'_>) -> Result<U, E> + Sync
) -> Result<Option<Vec<U>>, PmapError>
where
    L: Lanes,
    C: Communicator;

pub fn pmap<C, T, U, E, F>(
    world: &C,
    domain: &Domain,
    options: PmapOptions,
    root_items: Option<Vec<T>>,
    f: F,
) -> Result<Option<Vec<U>>, PmapError>
where
    C: Communicator;
```

Here `Communicator` is the existing trait from the one selected rsmpi crate, not a Hataori transport trait. Core uses only communicator duplication, rank/size, tagged blocking send and exact-source receive, `immediate_probe_with_tag`, signed built-in min reductions, and broadcast. The two mutually exclusive dependencies expose that same rsmpi surface through a private crate alias.

`MapError` records the lowest failing input index and bounded `E::to_string()`. Serial `map` stops at the first callback error. Rayon `Sequential` and `Inner` likewise stop after the failing item. Rayon `Outer` evaluates every input exactly once through the lane executor described in Section 6.1, preserves input order, then deterministically reports the lowest failed input index; it does not short-circuit admission. `MapInError` distinguishes `MissingPool`, `ForeignPool`, `DomainBusy`, `Callback(MapError)`, and `Executor(LaneExecutorError)` because only callback failures have an input index. `map_in` checks them in that order: target pool presence, foreign-pool origin, domain admission, then callbacks. `PmapError` is a core struct exposing `kind()`, `key()`, and `message()`; `PmapErrorKind` distinguishes `Reentrant`, `Preflight`, `User`, `Wire`, and `Protocol` (busy-domain and other admission failures are reported as `Preflight`), and the error carries the deterministic error key and bounded message where applicable. Tensor/backend integration errors remain in the adapter.

- `map` is local serial execution and has no MPI, Rayon, or serialization requirements. `map_in` is its explicit-domain Rayon counterpart.
- `pmap` is collective: all ranks call it in the same order on the same communicator. Calling it on different communicators or in a different collective order is a caller contract violation that MPI cannot recover from.
- `pmap` collectively duplicates `world` once per call and uses only that private communicator for preflight, scheduler traffic, drain, and final convergence. This prevents collision with caller traffic; the duplicate is released only after successful convergence or recoverable-error drain.
- Only `options.root` supplies `Some(Vec<T>)`; other ranks supply `None`.
- Only `options.root` receives `Some(Vec<U>)` on success.
- Each input is evaluated exactly once in successful executions.
- Results are returned in input order, independent of completion order.
- `map` is not implemented as world-size-one `pmap`.

Hataori does not transmit closures. All ranks run the same binary and construct `f` at the same SPMD call site. Root-only captured data must be included in input or distributed explicitly.

## 4. Place and execution domain

A place is:

```text
(rank, domain_id)
```

A domain is a multithreaded execution resource, not a Rayon worker thread. P0 supports exactly one domain per rank, so only `domain_id = 0` is valid. The value model leaves room for multiple domains later without implementing a registry or selection policy now.

A core domain contains only Hataori-owned execution facts:

```text
Domain
├─ domain_id
├─ Rayon pool handle, when enabled
├─ declared or verified CPU set
├─ worker budget
├─ pool ownership mode
└─ nonblocking running slot
```

It never contains tenferro or tensor4all types.

At most one coarse operation is admitted per domain. Admission is nonblocking and RAII-managed. Contention returns a typed `DomainBusy` error; a worker never blocks waiting for admission.

## 5. Affinity and lifecycle

### 5.1 Hataori-managed mode

Managed mode creates and owns one Rayon pool from an explicit CPU set and worker count.

On Linux, managed placement is strict and verified:

- the CPU set must be a subset of the process's actual allowed CPUs;
- a worker start hook pins each worker to a distinct declared CPU, records its outcome without panicking, and signals a standard-library channel;
- after `ThreadPoolBuilder::build` returns, construction waits with a bounded timeout until every start hook reports; any pin failure or missing report drops the owned pool and returns a typed construction error;
- pin failure, duplicate assignment, an out-of-range CPU, or too many workers is a construction error;
- placement is reported as `Verified`.

On non-Linux targets, P0 uses a documented degraded managed mode: Hataori still creates, owns, and shuts down the pool, but treats the CPU set as a declaration, performs no pinning, and reports placement as `CallerDeclared`. In particular, macOS affinity tags are not CPU binding and Apple Silicon core quality is outside Hataori's contract. Strict affinity assertions run only on Linux; the scheduling, transport, and bounds suites run on every supported target.

Hataori owns pool shutdown and does not infer a second resource policy from global Rayon state, `RAYON_NUM_THREADS`, `from_env`, or automatic topology discovery.

### 5.2 External-managed mode

External mode accepts a caller-owned pool plus declared CPU set and worker budget.

- Hataori never re-pins workers.
- Hataori never shuts down the pool.
- It validates only observable facts and reports whether placement is verified or caller-declared.
- A pool injected first and pinned afterward by Hataori is not supported.

The core constructor does not accept a tenferro handle. The optional adapter takes a caller-built tenferro backend and never associates a Hataori domain with it: tenferro owns its pool and admission, and Hataori schedules items onto its phase lanes.

## 6. Intra-domain execution modes

`LocalMode` is always defined so `PmapOptions` has one shape across MPI backends. Without Rayon only `Sequential` is accepted; with Rayon an operation selects one generic fan-out policy:

```rust
enum LocalMode {
    Sequential,
    Outer,
    Inner,
}
```

- `Sequential`: no local fan-out; the callback must remain sequential.
- `Outer`: Hataori evaluates every independent item exactly once in the selected domain, preserves input order, and reports the lowest callback-error index after all item results are available; each item must not perform another competing fan-out.
- `Inner`: Hataori presents items sequentially and admits the callback as the sole coarse operation for the domain. The callback may use the domain's full declared `worker_count` through that same pool.

`Outer` and `Inner` are mutually exclusive for one operation. Synchronous entry from a foreign pool is rejected before the callback runs.

P0 intentionally has no partial inner budget. Rayon cannot cap arbitrary nested same-pool work to fewer than the pool's workers without another scheduling mechanism, and tenferro's motivating contract asks for the complete supplied executor. A future partial budget requires a real backend need and a separately tested mechanism.

### 6.1 Execution lanes

Every local fan-out runs through one scheduling abstraction: execution lanes plus a caller-thread coordinator (`src/lanes.rs`).

```rust
trait Lanes {
    type Lane<'l>;
    type Error: std::error::Error + Send + Sync + 'static;
    fn lane_count(&self) -> usize;
    fn run<E: Send>(
        &mut self,
        lane: impl Fn(usize, &mut Self::Lane<'_>, &LaneControl) -> Result<(), E> + Sync,
    ) -> Result<(), LanesError<E, Self::Error>>;
    fn run_with_coordinator<C, E: Send>(
        &mut self,
        coordinator: impl FnOnce(&LaneControl) -> Result<C, E>,
        lane: impl Fn(usize, &mut Self::Lane<'_>, &LaneControl) -> Result<(), E> + Sync,
    ) -> Result<C, LanesError<E, Self::Error>>;
}
```

A lane is one execution resource (for `RayonLanes`, one pool worker) lent to the fan-out; the coordinator is the calling thread and is not a lane. Both methods join every lane before returning and resume a lane panic only after that join, and a lane panic always takes precedence over an ordinary lane error. A lane error or panic cancels the shared control immediately, so lanes that wait on it cooperatively are released before the join. `LaneControl` is one concrete cancelled/cancel capability shared by the coordinator and every lane, so cancellation is visible in both directions; an adapter bridges its own cancellation state into it. When a backend can stop a run without reaching a lane callback, the adapter lends the side that waits a `LaneControl::with_probe` view of the same control, so that wait also observes the backend's own stop signal while `cancel` from either side still reaches both. The trait carries no `Send`, `Sync`, or `'static` supertrait, so an executor, its lane resources, a coordinator capture, and a coordinator result may stay on one thread.

A lane callback receives the lane's own resource, so `map_in_lanes` exposes `f: Fn(T, &mut L::Lane<'_>) -> Result<U, E>` and `pmap_lanes` keeps that shape, which is how a backend adapter hands its per-worker context (for example a tenferro child session) to the callback. The ordinary `pmap` adapts its `Fn(T)` callback to `RayonLanes`' unit lane.

`map_in` `Outer` is `map_in_lanes` over `RayonLanes`: lanes pull the next input index from one shared counter and write into per-index result slots, so there is no Rayon job and no extra heap allocation per item. `Sequential` and `Inner` keep running one callback at a time on a pool worker, so nested same-pool work stays available.

`RayonLanes` borrows a `Domain` and admits it for the executor's lifetime. `run` uses `ThreadPool::broadcast`; `run_with_coordinator` uses `ThreadPool::in_place_scope` plus `Scope::spawn_broadcast` and keeps the calling thread outside the pool, so even a one-worker pool runs a lane concurrently with the coordinator. Entry from a foreign Rayon pool and a busy domain keep their typed rejections. `run_with_coordinator` additionally rejects entry from one of the executor's own pool workers with `LanesError::Reentrant` before spawning, because such a coordinator could not make progress while its lanes are joined; the lane-only `run` keeps supporting a same-pool worker caller.

An executor that cannot overlap a coordinator with its lanes returns the typed `LanesError::SingleLane` without running either callback. The hybrid `pmap` protocol then falls back explicitly to executing each batch on the coordinator thread through `Lanes::run` between MPI progress steps; the fallback is documented, covered by an MPI smoke stage, and never runs work twice. `lane_count()` is never used to infer the fallback. In that inline mode a panicking batch aborts with `MPI_Abort(71)` as soon as its outcome is known, before the prefetched batch is executed.

`Inner` is a generic core resource contract, not a tenferro-specific variant. The callback still has the ordinary `Fn(T) -> Result<U, E>` shape and obtains no core executor handle. A backend adapter supplies its own `Lanes` implementation instead: `hataori-tenferro` opens tenferro's held session and exposes its phase lanes, so `map_in_lanes` and `pmap_lanes` schedule items across tenferro's own workers while tenferro keeps its pool, admission, and per-worker child sessions. `map_in` takes a Hataori `Domain` and therefore never drives a tenferro session; core neither constructs a second pool nor imports backend types.

## 7. Dynamic P0 scheduler

### Collective preflight

A small thread-local RAII guard rejects recursive `pmap` on the MPI initialization thread before any MPI call. Subject to that collective precondition, ranks duplicate the caller communicator and enter one preflight before any `READY`/`TASK` traffic. Preflight checks initialization-thread identity, supported MPI thread level, root and option agreement, `0 <= root < world_size`, exactly one root payload on the designated root, no payload elsewhere, domain identity/mode validity, and local domain admission. Deterministic error keys are not validated in preflight: `ErrorKey::new` checks the encoded key against the reserved `i64::MAX` no-error sentinel at the point an error is reported (`src/pmap.rs` `error_key`, `src/wire.rs`), and a key that cannot be encoded is an unrecoverable failure that aborts the job with code 72 like other unrecoverable protocol failures. Locally acquired RAII admission guards remain held through the call. Signed built-in reductions converge any validation failure so every rank either enters the scheduler or returns the same typed error; no rank returns alone while peers enter protocol traffic.

Collective call order, communicator identity, and non-reentrant participation on every rank are API preconditions rather than recoverable preflight checks: ranks that do not enter the same collective call cannot communicate enough to diagnose that misuse.

### Scheduling protocol

P0 uses coordinator-driven dynamic scheduling rather than static rank chunks. It targets coarse items whose compute time is much larger than one `READY` round trip.

```text
READY(rank, domain=0)
    -> TASK(batch_id, indexed_items, payload)
    -> RESULT(batch_id, indices, status, payload)

READY(rank, domain=0)
    -> STOP
    -> DRAIN
```

- The root owns a FIFO queue of indexed items. In fixed-input P0, the checked original input index is also the stable `task_key`; no separate task-ID field is transmitted.
- Batch size is a fixed positive call parameter; the default is 1. Batch IDs start at zero and increase monotonically with checked overflow.
- Each domain, including the root's, has `running <= 1` and `prefetched = 0` (with the default `prefetch = false`; the opt-in P1 prefetch in Section 18 and [`design/bounded-prefetch.md`](design/bounded-prefetch.md) allows `prefetched <= 1` on remote hybrid domains).
- Every remote `READY` receives exactly one `TASK` or `STOP`.
- `READY` supplies backpressure; P0 has no credit, acknowledgement, heartbeat, retry, cancellation, cost model, or adaptive batch policy.
- The root restores order with an O(N) indexed result table.

The root retains the input queue and result table, so the input plus all returned payloads must fit in root memory. P0 does not stream results to external storage.

The root rank participates in computation through its own domain. The scheduler assigns it an owned batch directly, without a self-addressed MPI message or serialization. Root-local idleness acts like `READY`, and local completion makes the domain eligible for another batch. Calling the root-dispatch transition while its lane is running is a typed protocol error and leaves state unchanged; only an idle root with no assignable work becomes stopped. This keeps all ranks useful, including world size one, while preserving the same capacity and ordering rules as remote domains.

The MPI initialization thread drives all communication and the root scheduler. With Rayon, root and remote hybrid domains both run the protocol loop as the coordinator of `Lanes::run_with_coordinator` (`src/pmap.rs` `hybrid_root_protocol`, `hybrid_worker_protocol`): the loop body stays on the calling MPI initialization thread and keeps scheduler/MPI progress, while local batches execute on the lanes. `RayonLanes` drives those lanes through `ThreadPool::in_place_scope` plus `Scope::spawn_broadcast`, which preserves FUNNELED identity and scoped non-`'static` borrowing and joins every lane before `in_place_scope` returns or propagates an unwind. No borrowed caller frame can be destroyed while a lane remains live, and no lane calls MPI.

One bounded local batch queue connects the coordinator and its lanes (`src/lanes/batch.rs`). The queue holds at most one resident batch, which is the protocol's `running = 1` capacity; a prefetched batch stays in a coordinator slot rather than in a lane. `Outer` batches are claimed item by item so a batch spreads over the lanes, while `Sequential` and `Inner` batches are claimed whole by one lane. Results are indexed per item, so completion order never affects ordering. Idle lanes never park on a condition variable: they poll with `rayon::yield_now()` and a short bounded park, so nested same-pool work still finds a worker. The coordinator never blocks unconditionally either; its waits recheck `LaneControl::cancelled()` and then abort per the rules below.

Callback panics are caught per batch and become a `Panic` batch outcome: the coordinator on the MPI thread calls `MPI_Abort` with 70 (root) or 71 (remote) before anything is joined, exactly as before. A remote rank checks that outcome before it hands the prefetched batch to the lanes or executes it inline, so a hung peer callback can never delay the abort. Callback errors are data: they travel to the coordinator as the batch's lowest failing index plus bounded message and never cancel the executor. A prefetch transfer failure keeps abort code 75, also before any join.

While root-local work is running, the initialization thread checks local completion and services at most one `MPI_Iprobe`-detected message per loop iteration, alternating priority when both remain ready. A detected message is consumed with the ordinary blocking receive path. When neither source is ready, the loop calls `std::thread::yield_now`; P0 adds no progress thread, timer policy, or async runtime. This rule is simple, testable, and prevents either source from starving.

Without Rayon, the existing MPI-only bounds do not require `Send`, so the root executes its local batch synchronously on the initialization thread. MPI progress may pause for at most that batch's callback; after it returns, the root resumes receiving before assigning more work. This preserves scoped non-`Send` execution and eventual rendezvous progress, but does not promise communication progress during an MPI-only root callback.

`MPI_Iprobe` is permitted only for root event multiplexing. P0 data transfer remains blocking: it has no `Isend`, `Irecv`, request lifecycle, prefetch, or background progress (the opt-in P1 prefetch in Section 18 overlaps one blocking transfer with computation without adding nonblocking requests). The coordinator has no dedicated CPU in P0.

### Normative domain state transitions

Remote domains follow exactly these transitions:

```text
Idle -> ReadySent -> Running -> ResultSent -> Idle
Idle -> ReadySent -> StopReceived -> Drained
```

A successful or failed batch produces exactly one complete `RESULT` frame, followed by the normal `READY` path. Completion always validates lane state, batch ID, count, and the exact original-index sequence, including after a prior failure. Validation is a full first pass: protocol errors leave the result table and lane state unchanged. On a valid success after failure, decoded values are discarded after validation; the lane still returns to idle and then follows `READY` to `STOP`. When a corrupt header makes its batch ID untrustworthy, transport marks the running lane failed through a dedicated coordinator transition that uses the coordinator's pinned `RunningMeta.batch_id`, never the received ID; this guarantees the next valid `READY` receives `STOP`. `DRAIN` is sent exactly once, only after `STOP`, callback completion, and completion of every announced send. The root domain follows the same logical states without `READY`, `STOP`, or `DRAIN` MPI messages. A callback panic is not recoverable: under `panic=unwind` the callback boundary catches it only to request `MPI_Abort` from the initialization thread; under `panic=abort` the process terminates directly.

The coordinator is implemented as private pure state transitions over queue, in-flight, result, and rank-state data. Tests drive those transitions directly; P0 adds neither a public nor private generic transport trait. The MPI loop is a thin caller of that state machine.

## 8. Error and drain protocol

After the first recoverable error:

1. the root stops assigning new work;
2. every current or later `READY` receives `STOP`;
3. already assigned batches finish or fail;
4. every announced length and payload is received or drained;
5. each rank sends `DRAIN` once stopped and idle; root-local work is likewise completed and drained before convergence;
6. after all ranks drain, one final collective selects the deterministic winning error;
7. the winning bounded UTF-8 error message is broadcast.

In preflight, a failing rank contributes its rank and a passing rank contributes `world_size`; signed `MPI_INT`/`MPI_MIN` therefore selects the lowest failing rank or `world_size` when all pass. The selected failing rank broadcasts its typed validation error. After scheduling begins, each recoverable error has a canonical tuple `(task_key, class, reporting_rank)`: `task_key` is the stable task ID when one exists and the checked input length for call-level errors, while `class` is a fixed internal ordinal for wire/protocol, user callback, and drain. The key is checked as `((task_key * ERROR_CLASS_COUNT + class) * world_size + reporting_rank)` in the nonnegative range of `i64`; no-error is `i64::MAX`. Final convergence uses `MPI_INT64_T`/`MPI_MIN`, then the encoded reporting rank broadcasts a message capped at 4096 UTF-8 bytes and truncated only at a character boundary. There is no collective after every batch and no custom MPI reduction. P0 deliberately avoids unsigned `MPI_MIN`, which produced incorrect results in [MPICH 4.0.2](https://github.com/pmodels/mpich/issues/6083) and [Open MPI 4.1.4](https://github.com/open-mpi/ompi/issues/10648).

Recoverable errors must not return early and strand a blocking sender. Process death, OOM, callback panic, MPI transport failure, or rank loss requires `MPI_Abort`; these are not converted into `PmapError`.

Only one high-level `pmap` may be active on an MPI initialization thread. A small RAII reentrancy guard is sufficient; P0 has no communicator registry.

## 9. MPI thread model

P0 targets `MPI_THREAD_FUNNELED` behavior.

- MPI-only execution accepts an externally provided `MPI_THREAD_SINGLE` or stronger runtime because no other thread executes during the callback. Hybrid execution rejects `MPI_THREAD_SINGLE` and accepts `MPI_THREAD_FUNNELED`, `MPI_THREAD_SERIALIZED`, or `MPI_THREAD_MULTIPLE`. A stronger provided level does not relax Hataori's rules.
- Every rank must enter hybrid `pmap` on the MPI initialization/main thread and not on a Rayon pool worker. P0’s portable local guard rejects exactly contexts for which `rayon::current_thread_index()` is `Some`; Rayon exposes no general public marker for a scope body that remains on an ordinary calling thread. A `ThreadPool::in_place_scope` body that stays on the MPI main thread and reports `None` is allowed. On the pinned Rayon version, `ThreadPool::install`, `ThreadPool::scope`, and `rayon::scope` bodies report `Some` and are rejected. Hataori does not own MPI initialization and cannot capture a stronger private initialization-thread token without changing the integration contract. The pure helper is available in Rayon-only builds and is the first statement on every hybrid `pmap` entry, including re-entry. A failed guard returns locally before Active/domain admission/thread-level queries/`MPI_Is_thread_main`/communicator duplication or any MPI call; it deliberately does not join collective preflight because doing so from a worker would violate `FUNNELED`. Ranks that disagree about this entry precondition are caller misuse, like missing collective participation, and may strand peers. Once the guard passes, local thread-level and main-thread query results, pool/domain state, options, and root-input shape are folded through collective preflight. The initialization thread remains outside the selected pool, so even a one-worker pool runs its lane concurrently with the coordinator; `RayonLanes` never reports `SingleLane`.
- Only the thread that initialized MPI may call MPI, and externally initialized runtimes are checked with `MPI_Is_thread_main` rather than assumed.
- Communicators never enter Rayon closures.
- A mutex does not permit MPI calls from another thread under FUNNELED.
- Hataori adds no unsafe `Send` or `Sync` implementation for communicators.
- P0 transfers data with blocking MPI and has no background progress thread.
- Root may call `MPI_Iprobe` on that same initialization thread solely to multiplex MPI arrivals with root-local completion.

## 10. Generic wire and locality

Cross-rank values use one fixed bincode 2 serde configuration:

```rust
bincode::config::standard()
    .with_fixed_int_encoding()
    .with_little_endian()
```

Each message uses a fixed header tag followed, when nonempty, by one payload tag from the same source on the private communicator. The fixed-width header contains protocol version, message kind, batch ID, item count, status, and `u64` payload length. Status is a checked scheduler tag; the wire codec treats payload bytes as opaque, while each scheduler/transport message kind defines its payload schema in the task that consumes it. Capacity one means a rank never has two payloads outstanding. Receivers first match the header, then receive exactly its announced payload from that source; no wildcard payload receive is allowed.

`MAX_WIRE_BYTES` is `i32::MAX` for P0, matching rsmpi's MPI `Count`. Length is checked against that limit and local `usize` before allocation or send. Decoding validates the received MPI count, every integer conversion, protocol version, message/state compatibility, expected structure, and full byte consumption. Truncation, overflow, unexpected kind/tag/source, and trailing bytes are protocol errors.

Same-rank movement is an owned move and bypasses serialization and MPI. Distributed `pmap` retains serde bounds because placement is decided at runtime; callers needing non-serializable local work use `map`.

Core transports only opaque serializable `T` and `U` values or bytes. Tensor shape, dtype, layout, allocation, and target-context reconstruction belong to the optional tensor adapter. Closures, communicators, pools, contexts, admission state, cache identity, and addresses never appear on the wire.

## 11. Data placement helpers

P0 provides three synchronous collective, root-coordinated helpers that reuse the same private-communicator, wire, preflight, and deterministic error-convergence rules. All ranks call each helper in the same collective order:

```rust
pub fn broadcast<C, T>(
    world: &C,
    root: i32,
    root_value: Option<T>,
) -> Result<T, PlacementError>;

pub fn scatter<C, T>(
    world: &C,
    root: i32,
    root_shards: Option<Vec<T>>,
) -> Result<T, PlacementError>;

pub fn gather<C, T>(
    world: &C,
    root: i32,
    value: T,
) -> Result<Option<Vec<T>>, PlacementError>;
```

`C` is the selected backend's `Communicator`; `T: Serialize + DeserializeOwned`. `PlacementErrorKind` has only `Preflight`, `Wire`, and `Protocol`. `PlacementError` exposes its kind, the lowest reporting rank when applicable, and a message bounded by the common 4096-byte limit.

Preflight collectively agrees on an in-range root and operation kind. Exactly the root supplies `Some` to `broadcast`/`scatter`; scatter additionally requires exactly `world.size()` shards. Before scheduler traffic, every payload that its source must encode is encoded and checked against wire/MPI limits: broadcast/scatter encode remote root values, while gather encodes only non-root values. Encode failures converge before any point-to-point frame.

Transport uses one private communicator and dedicated checked placement message kinds. Every announced frame is fully received even when header/payload decoding fails, so no sender remains blocked. Broadcast/scatter receivers and the gather root finish their fixed receive schedule before a final signed minimum-rank convergence and bounded winner-message broadcast. Unexpected internal MPI/transport failures that cannot be drained use `MPI_Abort`; ordinary codec/protocol failures return the same converged `PlacementError` on every rank.

Returned values are caller-owned and gather order is rank order. The root's local broadcast output, scatter shard, and gather contribution are owned moves rather than codec round trips; the broadcast source may still be serialized by reference to create remote copies. World size one performs no encode/decode and therefore exercises the pure owned-move path. P0 has no remote registry, cross-call object cache, remote references, or distributed garbage collector.

## 12. Trait bounds

P0 uses scoped synchronous execution and does not add `'static` for hypothetical future overlap.

| Entry point | Bounds |
|---|---|
| serial `map` | `F: FnMut(T) -> Result<U, E>`; `E: Display`; no serde, `Send`, `Sync`, or `'static` |
| Rayon `map_in` | `T/U/E: Send`; `F: Fn(T) -> Result<U, E> + Send + Sync`; `E: Display`; no `'static` |
| MPI-only `pmap` | `T/U: Serialize + DeserializeOwned`; `F: FnMut(T) -> Result<U, E>`; `E: Display`; no `Send`, `Sync`, or `'static` |
| hybrid `pmap` | MPI bounds plus `T/U/E: Send`; `F: Fn(T) -> Result<U, E> + Send + Sync`; no `'static` |
| `broadcast` / `scatter` / `gather` | `T: Serialize + DeserializeOwned`; no Rayon, `Send`, `Sync`, or `'static` bounds |

`E` is moved from a worker and does not require `Sync`. Shared-data APIs add `Sync` only where actual sharing requires it.

## 13. Tensor integration contract

The tensor path has this ownership chain:

```text
tenferro held session (one root admission, tenferro's own pool)
  -> tenferro phase lanes (one worker + one owner-inheriting child session each)
    -> Hataori schedules items onto those lanes
```

Hataori owns scheduling (which item runs on which lane, ordering, error rules,
MPI distribution); tenferro owns the pool, admission, and lane resources.
Neither layer borrows the other's pool. Phase 20b places tensor4all's explicit
context and domain-owned runtimes/caches above the same lane seam; it does not
replace the pool or admission chain.

Required tenferro capabilities:

- a held root CPU session that owns its pool and its admission: the root entry acquires from tenferro's process-global `ResourceArbiter`, so a held session started while another tenferro execution owns its CPUs waits in FIFO order instead of failing;
- a phase lease that lends that session's own workers as lanes with one child session per lane;
- a caller-thread coordinator that runs concurrently with the lanes, can cancel them, and can observe a phase stop that never reaches a lane body;
- no second pool, no second admission for the same work, and no root-session reentry from a lane;
- typed errors for lane, coordinator, session, single-lane, and target-pool-caller failures.

Required tensor4all capabilities:

- explicit context entry points for plain, graph, and eager AD paths;
- no fallback to `DEFAULT_CPU_CONTEXT`, `DEFAULT_BACKEND`, `DEFAULT_GRAPH_RUNTIME`, `DEFAULT_EAGER_RUNTIME`, `from_env`, or a global backend mutex;
- backend/runtime/cache ownership scoped to the domain context;
- typed rejection of cross-context compiled-object or workspace reuse.

Tensor transport contains only logical host data and metadata. The receiving rank reconstructs the tensor in its target domain context. Executor, admission, runtime, and cache identities are never serialized.

## 14. Provider threads

Core does not encode tensor provider policy. The tensor adapter must account for all provider threads within the selected CPU budget.

The default hybrid integration requires single-threaded BLAS/OpenMP providers. Thread counts are configured before execution, not changed while operations are running. An integrated mode is rejected when provider behavior cannot be verified or would introduce an uncontrolled thread source.

## 15. P0 hard failures

Core rejects before work begins:

- more than one domain per rank;
- ambient/global Rayon or hidden pool construction;
- `map_in` on a sequential domain with no pool;
- synchronous foreign-pool entry;
- invalid or unverifiable Linux managed placement;
- invalid external ownership declarations;
- worker or background-thread MPI calls;
- domain admission contention;
- requests for nonblocking data transfer, MPI-only prefetch, background progress, or dedicated-coordinator behavior.

The build rejects simultaneous `mpi` and `rsmpi-rt` features.

The optional adapter separately rejects:

- a held session that another root execution already owns;
- a phase opened from one of its own pool's workers (`TargetPoolCaller`);
- a coordinator run on a one-lane backend (`SingleLane`, forwarded so hybrid `pmap` takes its explicit fallback);
- tensor4all global-default fallback;
- context/cache mismatch;
- uncontrolled provider threads.

Backend-specific errors do not enter the core error enum.

## 16. Acceptance criteria

### Core scheduling and transport

- Empty input, more ranks than inputs, batch size 1, and fixed batch size greater than 1 work.
- Every successful input executes once and root results preserve input order under reverse completion order.
- A skewed workload demonstrates meaningful improvement over static contiguous partitioning.
- Instrumentation proves `running <= 1`, `prefetched = 0` (with `prefetch = false`; see Section 18 for the prefetch bound), bounded resident batches, one complete `RESULT` per assigned batch, and exactly one `STOP`/`DRAIN` transition per remote rank.
- With Rayon, remote result payloads of at least 1 MiB make progress while the root domain computes; the watchdog fixture also records that a remote header was serviced before local completion, rather than inferring rendezvous progress from timing alone. Root event polling does not starve local completion.
- Without Rayon, the root participates synchronously, preserves the MPI-only non-`Send` bounds, and resumes rendezvous progress after each finite local batch.
- World size one executes every input through the root domain without MPI self-messages or serialization.
- Simultaneous user/decode errors converge on one deterministic signed-key winner without deadlock; tests cover unsigned values that failed in affected MPI releases.
- Preflight disagreement or local validation failure causes every rank to return before any scheduler message is sent.
- A caller communicator is reusable after a recoverable failure, and private-communicator traces contain no unmatched header or payload.
- A test-only private MPI-call wrapper asserts `MPI_Is_thread_main` at every call site.
- Hybrid root and remote ranks run their protocol loop as the coordinator of `Lanes::run_with_coordinator`; borrowed callback data remains valid and the coordinator stays on the MPI initialization thread while the lanes are joined before return or unwind.
- A one-lane executor's typed `SingleLane` rejection falls back to executing batches on the coordinator thread through `Lanes::run`; `lane_count()` never selects that path.
- Same-rank serialization count is zero under an instrumented codec.
- Version, kind, tag/source, length, count, overflow, truncation, and trailing-byte errors are typed.

### Domain and affinity

- On Linux, managed workers remain on distinct declared CPUs within the launcher/cgroup allowed set and placement reports `Verified`.
- On non-Linux targets, managed mode performs no pinning and reports `CallerDeclared`; all non-affinity suites still run.
- External pools are neither re-pinned nor shut down.
- Global Rayon is never entered.
- Admission is reacquired after success, error, or unwind.
- Entry from every context where `rayon::current_thread_index()` is `Some` fails locally before the first MPI call. An MPI-free guard matrix in the Rayon-only build records the actual predicate for plain main/ordinary threads, global/custom workers, `ThreadPool::install`, `ThreadPool::scope`, `ThreadPool::in_place_scope`, and `rayon::scope` on the pinned Rayon version; only `Some` cases are rejection cases.

### Compile-time boundaries

- Scoped local and hybrid execution can borrow non-`'static` data.
- MPI-only use is not forced to satisfy Rayon `Send`/`Sync` bounds.
- Local non-serialized use is not forced to implement serde.
- No communicator or backend receives an unsafe thread-safety wrapper.
- Core's Rust 1.85 feature matrix (default, `rayon`, `mpi`, `rsmpi-rt`, and their hybrids) builds without tenferro or tensor4all in its dependency tree; only the opt-in `tenferro` example feature adds `tenferro-tensor`.
- Default and Rayon-only builds contain no MPI, serde, or bincode dependency.
- `mpi` and pinned `rsmpi-rt` revision `6db6a2d6f96115b17c9a925e53ce719797c15dbb` each build the same Hataori MPI API; enabling both fails at compile time.
- An `rsmpi-rt` build succeeds without MPI headers, a C compiler, or libclang, then passes a multi-rank smoke test with `MPI_RT_LIB` set.
- Hataori can use a caller-supplied communicator without owning MPI initialization/finalization.

### Adapter and three-repository integration

- `hataori-tenferro` opens one tenferro held session and creates no pool, domain, or OS thread.
- tenferro's held-session admission is the only tensor-path admission; Hataori adds no second admission for the same work.
- Items scheduled through `map_in_lanes`/`pmap_lanes` run on tenferro's own named workers, and lane kernels stay under tenferro's lane policy.
- A lane error, lane panic, or coordinator exit cancels the shared Hataori control and tenferro's phase control, and every lane is joined before return or unwind.
- A coordinator waiting on `LaneControl` is released by tenferro's own phase stop (including a failure before any lane body runs) and by the executor's cancellation-on-exit, so it never waits forever.
- tensor4all plain, graph, and eager AD paths all use the explicit target context after Phase 20b.
- Global/default tensor runtime sentinels remain untouched.
- Remote tensor reconstruction occurs in the target context and serializes no execution identity.
- Outer and Inner each have exactly one fan-out owner.
- Managed and external modes respect the declared total thread budget.

## 17. Explicit P0 non-goals

- Arbitrary `remotecall`, spawn, futures, task graphs, or task registries
- Controller-generated task insertion or an expanding work queue; this is a separate P1 candidate
- Remote object handles, channels, distributed reference counting, or distributed GC
- Dynamic process creation; ranks are owned by the MPI launcher or scheduler
- Transmitting Rust closures or executable code
- Nonblocking data transfer, prefetch, request pipelines, or background MPI progress in P0; P1 added bounded prefetch, see Section 18
- Using `MPI_THREAD_MULTIPLE` semantics or relaxing Hataori's single-MPI-thread rule
- Multiple concurrently scheduled domains per rank
- A dedicated coordinator CPU
- Automatic topology or NUMA discovery and memory-placement guarantees
- Adaptive batches, cost models, retries, cancellation, heartbeats, or fault recovery
- A generic `Executor`, `Place`, or `Domain` trait hierarchy
- A process-global Hataori CPU broker or tensor cache
- Zero-copy, MPI derived datatypes, GPU-aware MPI, or custom transport/codec abstractions

## 18. P1 bounded prefetch and later gates

The broader, compatibility-breaking direction toward a long-lived MPI/TCP
runtime, typed actions and futures, migration-ready remote objects, and
mechanical resource-lifecycle management is specified separately in
[`design/distributed-runtime.md`](design/distributed-runtime.md). The design in
this file remains the source of truth for the implemented synchronous P0/P1
architecture; the long-term document must not be read as a claim that those
runtime capabilities already exist.

The long-term design includes a supported current-style `map`/`map_in`/`pmap`
sugar facade, but does not require source compatibility with these exact
signatures. Commit `34cb1b1371c8b2f8ef750e2d49d10f9ef8f0782e` is the fixed
legacy performance reference. The P0/P1 execution path must not be replaced
until the paired legacy-continuity and sugar-transparency gates in the
long-term design pass; a failed or inconclusive required case blocks
replacement.

P1 adds the opt-in, default-off bounded-prefetch mode specified in
[`design/bounded-prefetch.md`](design/bounded-prefetch.md). It permits exactly
one running and one prefetched batch on each remote hybrid domain. The MPI
initialization thread overlaps ordinary blocking transfer with borrowed callback
execution in the explicit Rayon pool; P1 adds no MPI request lifecycle,
progress thread, async runtime, or `'static` bound. MPI-only calls reject the
option during collective preflight, and the root domain remains capacity one.

The option remains default-off. Enable it by default only if:

- across at least seven runs, median end-to-end time improves by at least 10%;
- representative compute-heavy and small-payload regressions are at most 5%;
- memory remains bounded by two resident batches per remote domain;
- FUNNELED identity, ordering, draining, and reuse remain correct.

A later owning/detached API that truly requires ownership may add `Send +
'static`; those bounds must not leak into the existing scoped APIs.

### Local-execution replacement and the waived performance gate

`REPOSITORY_RULES.md` requires a paired release-mode comparison against
documented legacy baseline `34cb1b1371c8b2f8ef750e2d49d10f9ef8f0782e` before
the implemented P0/P1 local execution path is replaced. The lanes executor
(6.1) replaces that path in one step with no legacy copy: `map_in` `Outer` and
the hybrid `pmap` local-batch execution now run through `Lanes`, and the former
per-batch spawned-job and capacity-one outcome-channel machinery is deleted.

The maintainer waived that paired gate for this refactor on 2026-10-19
("厳密な測定はいらないよ"). The waiver covers this replacement only; it does not
waive the frozen performance-acceptance gate for any later facade or
runtime-path promotion. The work log records the decision and the structural
review that replaced the measurement.

### Other P1 candidates

- a controller-driven expanding queue, only after a real workload defines bounded pending-memory and determinism requirements;
- multiple disjoint domains per rank, after a second admitted CPU domain per process is proven;
- a dedicated coordinator policy, only if progress latency is measured as a bottleneck;
- NUMA-aware placement or multithreaded providers, each behind separate correctness and performance evidence.

## 19. Completion definition

P0 is complete only when:

1. Hataori core scheduling, transport, affinity, bounds, and error tests pass;
2. tenferro provides and tests held-session admission and phase lanes with a caller-thread coordinator;
3. tensor4all provides and tests explicit plain/graph/eager-AD contexts with no default fallback;
4. the optional adapter passes the joint ownership, admission, context, wire, and thread-budget gate;
5. Hataori pins known-good tenferro and tensor4all revisions.

Core may ship earlier, but integrated tensor compatibility must not be claimed before the joint gate passes.
