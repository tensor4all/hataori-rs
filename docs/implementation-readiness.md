# Hataori P0 Implementation Readiness

**Status:** Core, P1 bounded prefetch, and the tenferro adapter foundation are implemented (see [`core-acceptance-ledger.md`](core-acceptance-ledger.md) and [`tenferro-acceptance-ledger.md`](tenferro-acceptance-ledger.md)); the tensor4all context/reconstruction adapter remains upstream-blocked.

This document records why the core design was judged implementable, what evidence the implementation had to produce, and what must not be built early. The canonical behavior remains in [`design.md`](design.md).

## 1. Scope split

| Scope | Readiness |
|---|---|
| Hataori core | Implemented; independent review `Correct-to-merge`; delivered matrix in `core-acceptance-ledger.md` |
| tenferro adapter | Implemented (`adapters/tenferro`) against tenferro-rs #2206: the caller holds one engine per lane and `map_in_lanes`/`pmap_lanes` run on them; ledger in `tenferro-acceptance-ledger.md` |
| tensor4all context/reconstruction adapter | Blocked on tensor4all-rs #663 |
| P1 bounded prefetch | Implemented (commit `f7050ad`); independent review `Correct-to-merge` recorded in `docs/review-log.md` |
| Expanding queues and multiple domains | Deferred behind separate evidence gates in `design.md` |

Core implementation and release do not wait for the tensor adapter. Integrated tensor compatibility must not be claimed until both upstream contracts and the joint gate pass.

## 2. Findings and resolutions (Blackboard Round 3 + review rounds)

| Finding | Resolution in `design.md` |
|---|---|
| Root compute vs. FUNNELED MPI | Root uses `ThreadPool::in_place_scope`; the scope body and MPI event loop stay on the initialization thread while borrowed callback work runs in the explicit pool. |
| Root event multiplexing fairness | Each loop checks local completion and at most one `MPI_Iprobe` message, alternates priority, and yields only when neither is ready. A capacity-one local channel distinguishes `Success`, `UserError`, and `Panic`; only the MPI thread aborts. |
| MPI-only non-`Send` callback | Root executes one local batch synchronously; communication resumes after the finite callback. |
| Undefined `Inner { budget }` | P0 uses whole-domain `Inner` with no partial budget. The tenferro adapter is entered through `map_in_lanes`/`pmap_lanes` only, so it needs neither an `Inner` entry nor a partial budget. |
| Divergent local validation | A collective preflight converges validation before scheduler traffic. Collective order/communicator identity remain caller preconditions. |
| Message collision and framing | Every call duplicates the communicator and uses fixed header/payload tags with exact-source payload receive and a checked `i32::MAX` wire limit. |
| Drain/error state ambiguity | Normative remote-domain transitions and one-RESULT/one-DRAIN rules are specified. |
| Unsigned `MPI_MIN` portability bug | Error keys use checked nonnegative `i64` and `MPI_INT64_T`/`MPI_MIN`; preflight uses signed rank reduction. |
| Non-Linux strict affinity impossible | Linux placement is verified; non-Linux managed placement is explicitly degraded and `CallerDeclared`. |
| Scheduler testability vs. transport abstraction | Coordinator transitions are private pure state, tested directly. No generic transport trait is added. |
| `rsmpi-rt` API/pin uncertainty | Pin `6db6a2d6f96115b17c9a925e53ce719797c15dbb`; compile and runtime evidence is recorded below. |
| Root memory and task granularity | Root-memory residency and coarse-task assumptions are explicit. |
| Preflight no-error and error-key overflow | Passing ranks contribute `world_size`; largest-key overflow is a typed preflight failure. |
| MPI public trait ambiguity | `Communicator` is explicitly the selected rsmpi dependency trait; core adds no transport trait and lists the used surface. |
| Upstream rsmpi feature drift | Upstream `mpi` is pinned to `=0.8.1` with default/optional features disabled. |
| Rayon local error split | `map_in` returns `MapInError` for missing/foreign/busy preconditions and wraps indexed callback failures as `Callback(MapError)`. |
| Deterministic Rayon Outer errors | `Outer` evaluates every item exactly once, preserves order, and reports the lowest callback-error index without short-circuiting admission. |
| Rayon worker-start race | Managed construction waits with a bounded standard-library channel for every non-panicking start hook to report pin success or failure. |

## 3. Upstream evidence

### Upstream rsmpi

The `mpi` backend pins crates.io `mpi = "=0.8.1"` with `default-features = false`. In that release `mpi-sys` is mandatory, while the default feature only adds `user-operations`; disabling defaults therefore keeps the normal build/link-time backend and removes `libffi`. `derive` and `complex` remain disabled.

Hataori declares Rust 1.85 as its initial MSRV because fixed bincode 2.0.1 requires 1.85; the complete Task #9 feature matrix was checked with Rust 1.85.0. Rayon resolves to 1.12.0 and remains within that bound.

### rsmpi-rt

Pinned revision:

```text
6db6a2d6f96115b17c9a925e53ce719797c15dbb
```

Verified against that revision:

- feature `mpi-rt-sys-backend` exists and selects `mpi-rt-sys`;
- `Communicator::duplicate` calls `MPI_Comm_dup`;
- `Source::immediate_probe[_with_tag]` calls `MPI_Iprobe`;
- blocking probe/receive and tagged exact-source receive APIs exist;
- Rust `i64` maps to `RSMPI_INT64_T_fn`, and built-in `SystemOperation::min()` maps to `RSMPI_MIN_fn` without the optional `user-operations` feature;
- the runtime backend builds with:

```bash
cargo check --no-default-features --features mpi-rt-sys-backend
```

The check passed on 2026-08-21. The no-headers/no-C-compiler environment and `MPI_RT_LIB` multi-rank smoke remain implementation CI gates, not design assumptions.

### Rayon

The selected API is `ThreadPool::in_place_scope`, whose scope closure runs on the calling thread and whose `Scope::spawn` jobs enter the selected pool. This is the required shape for a FUNNELED root coordinator with borrowed callback state.

A temporary proof built and ran with `rayon 1.12.0` / `rayon-core 1.13.0` on 2026-08-21. It verified that:

- non-`Send` `Cell` state remains in the caller-side scope body;
- the scope body stays on the calling thread;
- a spawned callback borrows stack-owned input and runs on a target-pool worker;
- nested parallel iteration observes only the target pool's two worker indices.

The equivalent proof becomes a checked-in regression test before the hybrid scheduler merges. It also pins Rayon’s join-before-return/unwind guarantee for borrowed jobs. Hybrid `pmap` uses a pure Rayon-only helper as its first operation and locally rejects exactly callers for which `rayon::current_thread_index()` is `Some`, before any state change or MPI call. This check cannot join collective preflight under `FUNNELED`; rank disagreement is caller misuse. An MPI-free matrix records the predicate in plain main/ordinary threads and plain/custom/global worker, `ThreadPool::install`, `ThreadPool::scope`, `ThreadPool::in_place_scope`, and `rayon::scope` contexts on the selected Rayon version. Only `ThreadPool::in_place_scope` keeps its body on the plain MPI calling thread and reports `None`; the other listed pool/scope bodies report `Some` and are rejected. Correct participation still requires every rank to enter on the MPI initialization thread.

### MPI error reduction

Unsigned `MPI_MIN` is avoided because released implementations produced wrong answers:

- MPICH 4.0.2: <https://github.com/pmodels/mpich/issues/6083>
- Open MPI 4.1.4: <https://github.com/open-mpi/ompi/issues/10648>

### Tensor prerequisites

- tenferro caller-held engines with no sessions and no tenferro-owned threads: <https://github.com/tensor4all/tenferro-rs/issues/2206> — completed by merged PR #2209, superseding the held-session model of #2201 (itself superseding the caller-managed external domain of #1716, removed by #2004 D9)
- tensor4all explicit plain/graph/eager-AD contexts: <https://github.com/tensor4all/tensor4all-rs/issues/663> — still open

Only the tensor4all context/reconstruction phase remains externally blocked.

## 4. Minimal implementation order

1. **Crate and features** — single crate, dependency-free default, `rayon`, `mpi`, and pinned `rsmpi-rt`; reject both MPI features together.
2. **Serial `map`** — ordered results, bounded callback errors, no serde/Send/Sync/`'static` leakage.
3. **Domain and local Rayon** — managed/external ownership, RAII admission, `Sequential`/deterministic full-evaluation `Outer`/whole-domain `Inner`, typed `MapInError`, Linux affinity and non-Linux declared placement.
4. **Pure protocol state and wire** — FIFO coordinator transitions, zero-based checked batch IDs, original indices as P0 task keys, framing, checked codec, signed deterministic error keys; no transport trait or separate task-ID seam.
5. **MPI-only `pmap`** — private communicator, collective preflight, dynamic scheduler, synchronous root participation, drain and reuse.
6. **Hybrid `pmap`** — `in_place_scope`, local completion channel, fair `MPI_Iprobe` loop, rendezvous progress.
7. **Collective placement helpers** — root-coordinated `broadcast`, `scatter`, and `gather` on the same private wire rules.
8. **tenferro adapter** — pin merged PR #2209, run one Hataori lane per caller-held tenferro engine, and retain core dependency isolation.
9. **tensor4all context/reconstruction adapter** — only after #663 lands at a known-good revision.
10. **Joint integration** — run the three-repository MPI and thread-ownership gate after both adapter layers exist.

Each numbered item is an independently reviewable implementation task. Do not combine scheduler, hybrid execution, affinity, and tensor integration into one initial change.

## 5. Acceptance and feature matrix

Commands become runnable when Step 1 creates `Cargo.toml`; they are merge gates for the corresponding step.

| Area | Required evidence |
|---|---|
| Default | `cargo test --no-default-features`; `cargo tree` contains no MPI, serde, bincode, tenferro, or tensor4all |
| Rayon | `cargo test --no-default-features --features rayon`; borrowed non-`'static` compile/run test; global-pool sentinel remains untouched |
| Upstream MPI | `cargo test --no-default-features --features mpi`; multi-rank watchdog suite under a build/link-time MPI |
| Runtime MPI | `cargo test --no-default-features --features rsmpi-rt`; build in an image without MPI headers/C compiler/libclang; `MPI_RT_LIB` multi-rank smoke |
| Hybrid upstream | `cargo test --no-default-features --features mpi,rayon`; rendezvous progress and root-local fairness test |
| Hybrid runtime | `cargo test --no-default-features --features rsmpi-rt,rayon`; same hybrid and thread-identity suite |
| Feature exclusion | `cargo check --no-default-features --features mpi,rsmpi-rt` must fail with the intended compile error |
| Docs/lints | `cargo fmt --check`; `cargo clippy --all-targets` for each valid feature set; `cargo doc --no-deps` for each public API feature set |

### Protocol tests

- empty input, world size one, ranks greater than items, batch one, and fixed batches greater than one;
- reverse completion preserves input order and every successful index executes once;
- a deterministic skew fixture records each dynamic assignment, sums item costs per lane, and proves the maximum dynamic lane cost is strictly lower than the maximum static-contiguous lane cost for the same items;
- with `prefetch = false`, trace assertions prove `running <= 1`, `prefetched = 0`, one complete result per batch, and one STOP/DRAIN per remote rank; root dispatch while running is a state-preserving typed error;
- preflight mismatch or an agreed root outside `[0, world_size)` emits no scheduler traffic; error-key encoding is validated when an error is reported (`ErrorKey::new` unit tests in `src/wire.rs`), and a key at or above `i64::MAX` aborts the job with code 72 rather than returning a typed error;
- version, kind, source/tag, length/count, overflow, truncation, and trailing-byte failures are typed;
- simultaneous user/wire failures select one signed deterministic key on every rank; after failure, every assigned completion is fully validated, valid values are discarded, and malformed metadata leaves state/results unchanged; a corrupt received batch ID releases the running lane only through its coordinator-pinned batch metadata before the next READY→STOP;
- recoverable failure leaves the caller communicator reusable and no private frame unmatched;
- root-local codec instrumentation remains zero for `pmap`, `broadcast`, `scatter`, and `gather`;
- serial `map` and Rayon `Sequential`/`Inner` stop at the first callback error, while Rayon `Outer` evaluates every input exactly once and reports the lowest failed index after ordered collection;
- callback panic takes the abort path rather than attempting recoverable drain.

### P1 bounded-prefetch tests

- MPI-only preflight collectively rejects `prefetch = true` before any callback or scheduler traffic;
- both hybrid backends preserve the P0 path with `prefetch = false` and pass empty, one-item, ordered/exactly-once, more-ranks-than-items, and world-size-one calls with `prefetch = true`;
- pure coordinator traces prove one running and at most one prefetched batch, transactional current-to-prefetched promotion, STOP-before-final-result ordering, and bounded two-batch residency;
- an n=2 serialization handshake blocks the current root/remote callbacks until task N+1 is deserialized, then blocks callback N+1 until result N is deserialized on root, proving both overlap directions without timing inference;
- current and prefetched callback/decode failures execute or reject exactly the assigned callbacks, preserve the lowest deterministic error, drain, and permit a later successful prefetched call;
- a subprocess watchdog injects result-N serialization failure after callback N+1 starts and requires prompt init-thread `MPI_Abort(75)` without scoped join or drain;
- every P1 MPI operation remains covered by the test-only `MPI_Is_thread_main` assertion and source scanner; borrowed callback data retains the existing hybrid compile-pass contract.

### Domain tests

- Linux managed workers occupy distinct declared allowed CPUs and report `Verified`; every start hook reports through the bounded construction latch, and injected/observable failures return typed construction errors without panic;
- non-Linux managed mode does not claim pinning and reports `CallerDeclared`;
- external pools are not repinned or shut down;
- admission is released after success, error, and unwind;
- `map_in` precondition precedence is missing pool, foreign pool, then busy admission; each fails before callback execution;
- foreign-pool entry and conflicting outer/inner ownership fail before callback execution.

### Compile-time boundary tests

Use compile-pass/compile-fail fixtures to prove:

- serial local code needs no serde, `Send`, `Sync`, or `'static`;
- MPI-only callback/input/output need no Rayon bounds;
- Rayon and hybrid callbacks can borrow non-`'static` state but satisfy the required `Send`/`Sync` bounds;
- communicator/backend types receive no unsafe `Send`/`Sync` wrapper.

## 6. Explicitly deferred work

Do not add placeholders, traits, or partial implementations for:

- `Isend`/`Irecv`, arbitrary request pipelines or prefetch depth, progress threads, or async runtimes;
- expanding/controller-generated queues or parked-READY machinery;
- partial inner worker budgets;
- multiple domains, dedicated coordinator CPUs, NUMA/topology discovery, or multithreaded providers;
- public transport/executor/place/domain trait hierarchies;
- retries, cancellation, heartbeats, rank-failure recovery, remote objects, futures, registries, or distributed GC;
- tensor/provider policy in core.

## 7. Review gate

The blackboard Round 3 is design input, not the required independent reviewer verdict. Before any delegated implementation begins:

1. review the amended `docs/design.md` and this readiness record with the user-selected independent reviewer;
2. record reviewer identity, round, findings, fixes, and final verdict;
3. begin implementation only after `Correct-to-merge`, or after findings are fixed and re-reviewed.

This gate was completed by `reviewer-flash-opencode-go`; the final exact-state verdict is recorded in [`review-log.md`](review-log.md). Each later implementation task still requires its own post-diff different-family review.
