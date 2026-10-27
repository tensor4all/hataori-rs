# tenferro adapter: lanes over caller-held engines

**Status:** Implemented (this change). This document supersedes the held-session
design of hataori-rs#24, which tenferro-rs no longer supports.

**Upstream contract:** tenferro-rs
[#2206](https://github.com/tensor4all/tenferro-rs/issues/2206) (caller-held
engines, no sessions, no tenferro-owned threads) and its implementation
`8fec1ca70852b930c092e7177c30bb8af7594a57`. The cross-repository model is
tensor4all-meta
[`docs/design/core-boundary-and-execution-ownership.md`](https://github.com/tensor4all/tensor4all-meta/blob/main/docs/design/core-boundary-and-execution-ownership.md)
§"Example: nested parallelism in hataori".

**Scope:** the tenferro-only adapter. tensor4all contexts, tensor transport, and
tensor reconstruction stay outside this change.

## 1. Crate and dependency boundary

`hataori-tenferro` is a non-publishable workspace member. It depends on:

- local `hataori` with `rayon` enabled (for the `Lanes` contract and
  `pmap_lanes`);
- `tenferro-cpu` (default features off, **no provider feature**) at exactly
  `062298519`. tenferro-cpu's CPU providers (`native`, `blas`,
  `provider-inject`, ...) are mutually exclusive, so the consumer selects one
  through its own tenferro-cpu dependency and Cargo feature unification; the
  adapter's tests, doctests and examples enable `native` through a
  dev-dependency only;
- `tenferro-tensor` as a **dev-dependency only**: the adapter itself names no
  tensor type, and only its tests and example build real tensors;
- an optional `mpi` feature (`hataori/mpi` plus upstream `mpi`) used only by the
  `pmap_lanes` world-size-1 smoke example.

The adapter declares `rust-version = "1.96"`, matching the pinned tenferro-rs
workspace. It adds no Rayon handle, pool handle, tensor4all crate, or MPI
backend to ordinary Hataori builds; the root package's opt-in `tenferro` example
feature is separate and outside the Rust 1.85 core matrix.

## 2. Ownership

| Layer | Owns | Does not own |
|---|---|---|
| tenferro | an engine's pool and its invariants (worker stack, plan cache, scratch, buffer pool), and the refusal to run an operation from a worker of that same pool | which item runs where, how many engines exist, any thread of its own |
| hataori | scheduling: work queues, item-to-lane assignment, ordering and error rules, MPI distribution | a compute pool, a session, or an admission hierarchy |
| caller | the engines: how many lanes, how many workers each engine's pool has, and any CPU placement | scheduling inside a lane |

The adapter is the seam, and it is now thin: it holds the caller's engines, runs
one lane body per engine on a thread of its own, and maps the two cancellation
and error vocabularies. It creates no pool and holds no session.

A lane is one thread the adapter spawns for one run plus the engine lent to it
for that run. Because a lane thread is not a worker of the engine's pool,
tenferro's "an operation called from a worker of its engine's own pool is
refused" rule holds by construction rather than by a runtime check, and a kernel
issued from a lane may use every worker of that lane's engine.

Two consequences the caller sees:

- **the engine is a value.** It is lent to one lane for one run and handed back;
  prepared plans, scratch and the buffer pool stay warm in it between runs.
  Nothing in the adapter outlives the run except the caller's engines;
- **there is no `close()`.** With sessions gone there is nothing to release
  except the engines themselves, which the caller owns and drops.

## 3. Public API

```rust
pub struct TenferroLanes { /* Vec<tenferro_cpu::CpuEngine> */ }

impl TenferroLanes {
    pub fn new(engines: Vec<tenferro_cpu::CpuEngine>) -> Result<Self, NoEngine>;
    pub fn lane_count(&self) -> usize;
}

impl hataori::Lanes for TenferroLanes {
    type Lane<'l> = tenferro_cpu::CpuEngine;
    type Error = LaneThreadError;
    // run(lane)                 -> one scoped thread per engine
    // run_with_coordinator(..)  -> the coordinator on the calling thread
}
```

`new` keeps the engines in the given order: lane `i` is always lent engine `i`.
It rejects an empty vector with `NoEngine`, because a lane executor without an
engine could only report that no item was ever evaluated.

`Self::Error` is `LaneThreadError`, reported as `LanesError::Executor`: the one
executor-level failure left is a lane thread that could not be started, in which
case the already-started lanes are cancelled and joined like any other failure.
`LanesError::SingleLane` and `LanesError::Reentrant` are unreachable here — the
adapter always has a lane per engine and runs its lanes on its own threads, so
it can always overlap a coordinator, and it is never driven from one of its own
workers.

The caller builds the engines, which is where the machine layout belongs:

```rust
let engines = (0..lanes)
    .map(|_| Pool::builder().threads(threads).build().map(CpuEngine::new))
    .collect::<Result<Vec<_>, _>>()?;
let mut lanes = TenferroLanes::new(engines)?;
```

`tenferro_cpu::Pool::builder()` is tenferro's re-export of `threadteam`'s pool
builder (it presets the 16 MiB worker stack the providers need); the worker
count, and any placement added through `on_start`, are the caller's decision.
`CpuEngine::serial()` is a valid lane engine too and costs no pool at all.

## 4. One run

`map_in_lanes` and `pmap_lanes` drive the same Hataori `Lanes` methods, so the
adapter needs no queue of its own. (`map_in` takes a Hataori `Domain`, not a
`Lanes` executor, so it is not a tenferro entry.)

```text
hataori::map_in_lanes(map) / pmap_lanes
  -> Lanes::run                  -> scope { spawn k lanes }
  -> Lanes::run_with_coordinator -> scope { spawn k lanes; coordinator on this thread }
       lane k = its own thread + engine k
         loop { item = hataori's next index; f(item, &mut engine); write slot }
```

`Lanes::run` spawns one scoped thread per engine and joins them in index order.
A lane thread gets the same 16 MiB stack `threadteam`'s pool default gives
its workers (`LANE_STACK_BYTES` in the adapter), because a lane body runs
provider code on that thread; tenferro no longer exports a stack constant.
`run_with_coordinator` spawns the same lanes and then runs the coordinator on the
calling thread, so the coordinator always overlaps: with a single lane as well.
The `SingleLane` fallback the hybrid `pmap` used for a one-worker backend is
therefore gone; `pmap_lanes` executes its batches through the lanes at every
width.

Every run creates one `LaneControl` and lends it to the lane bodies and to the
coordinator:

- a lane body that returns `Err` cancels the control, and an unwind guard cancels
  it if the body panics, so a peer waiting on the control is released;
- the coordinator runs under a guard that cancels the control when it leaves,
  normally or by panicking, before the lanes are joined. This is the `Lanes`
  contract: "when the coordinator returns, errors, or panics, the control is
  cancelled and every lane is joined". This is the signal a *lane* waiting on the
  control observes; a coordinator that is itself waiting on the control is
  released by a lane error, a lane panic, or an explicit `cancel`, not by its own
  exit.

The adapter owns the join because `std::thread::scope` cannot join elsewhere: all
lanes are joined in index order, the first lane error is reported as
`LanesError::Lane`, and a lane or coordinator panic is resumed on the calling
thread only after every lane was joined and cancelled.

Error mapping is direct — there is no intermediate vocabulary left to translate:

| run outcome | Hataori |
|---|---|
| lane body returned `Err(e)` | `LanesError::Lane(e)` |
| coordinator returned `Err(e)` while every lane ended cleanly | `LanesError::Coordinator(e)` |
| lane body panicked | the panic is resumed after the join |
| coordinator panicked | the panic is resumed after the join |

## 5. Removed chains

tenferro-rs #2206 removed the session layer, so this change removes the
held-session adapter of hataori-rs#24:

- `TenferroLanes::open(&CpuBackend)` and `close()`; `CpuBackend`,
  `CpuHeldSession`, `CpuPhaseError`, `PhaseLane`, `PhaseControl`,
  `PhaseRunError` and `SessionEntryError` no longer exist upstream;
- `LanesError::SingleLane` forwarding and the `with_probe` coordinator view: the
  only stop signal now is the run's own `LaneControl`, because nothing upstream
  can stop a run without reaching a lane callback;
- `TenferroAdapterError` and its four `source()`-preserving variants, replaced by
  the construction-only `NoEngine` and a `LaneThreadError` for a thread that
  could not be started;
- `!Send`: engines move with the lane thread instead of belonging to an opening
  thread, so the executor is an ordinary `Send` value and callers need no
  per-thread executor.

tenferro-rs #2004 (D9) had earlier removed borrowing an external pool, so the
Phase 20a chain (`TenferroDomain`, `with_backend`, `ExternalCpuDomain`,
`CpuBackend::from_external_managed_domains`, the `ConcurrentEntry` gate and the
`OutsideDomainPool` check) was already gone. The engine model makes a
Hataori-owned `Domain` unnecessary as well: a lane pool would have needed a
declared CPU set, and placement belongs to the caller.

## 6. Acceptance evidence

`scripts/check-tenferro.sh` is the single entry point. It proves:

- core default and `rayon` trees still contain no tenferro or tensor4all crate;
- the adapter tree carries exactly the pinned tenferro revision and no
  tensor4all package, and both pinned tenferro crates are checked through
  `cargo metadata` (`tenferro-tensor` is a dev-dependency now, so it is not in
  the normal tree);
- `cargo test -p hataori-tenferro` proves, with two engines of different worker
  counts: known-value tenferro operations through `map_in_lanes(TenferroLanes)`
  (every item once, input order preserved), lane `i` lent engine `i` on a
  distinct thread, never inside that engine's own pool (the pool's
  `current_thread_index()` is `None` inside the lane, which is exactly tenferro's
  precondition for issuing an operation) and computing a real `add_read` there,
  the lowest callback-error index, and a *callback* panic — raised outside an
  engine operation — that is resumed only after the join with the caller's
  engines reusable afterwards;
- direct `Lanes` liveness cases with peers waiting on `LaneControl`: a normal
  coordinator return releases the lanes without an explicit `cancel` (the
  cancellation-on-exit guard is what releases them), and a coordinator error, a
  coordinator panic, a lane error, and a lane unwind each release the waiting
  peers and join every lane; a single lane overlaps a coordinator, which the
  held-session design refused with `SingleLane`;
- `TenferroLanes::new` rejects an empty engine vector with `NoEngine`, and the
  lane count is the number of engines;
- `pmap_lanes(TenferroLanes)` succeeds at world size one with an explicit lane
  count: `--lanes 1` asserts a single lane with the coordinator overlapping it,
  and `--lanes 2` (launcher set not to pin the rank) asserts two lanes;
- `cargo fmt`, all-target clippy with `-D warnings` (with and without the `mpi`
  example feature), rustdoc with `-D warnings`, doctests, the Rust 1.85 core
  checks, and `git diff --check` pass, each stage inside its own process-group
  watchdog (a cold dependency build is retried as a fixed number of bounded
  attempts so cargo resumes from its cache).

Not claimed here: multi-rank hybrid acceptance for the adapter (the smoke runs
at world size one), tensor4all explicit contexts and reconstruction (Phase 20b),
and any three-repository integration gate (Phase 20c). The repository MPI
harness (`examples/mpi_*`) belongs to the root `hataori` package and cannot host
the adapter, because the dependency direction is adapter -> `hataori`; the
adapter therefore carries its own `mpi`-gated example.

Not measured here: the model costs one scoped thread per lane per run (a run is
one wave of items, not one item) and adds no per-item synchronization beyond what
the caller's `Lanes` implementation already had. An engine that panics inside an
operation is marked broken upstream and is not reusable; only a panic in the
callback around it leaves the caller's engines intact.
