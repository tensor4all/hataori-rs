# Lanes executor for local fan-out

**Scope:** hataori-rs #24. This evolves the retained synchronous P0/P1 local
execution engine (`map_in` `Outer` and the hybrid `pmap` local-batch path); it
does not touch the canonical runtime facade (`hataori-runtime` /
`hataori-algorithms`) and does not add a tenferro dependency.

## Why

The tenferro adapter has to become one scheduling seam: tenferro owns its CPU
pool, admission, and per-worker child sessions, while hataori decides which item
runs where. The implemented P0/P1 local path could not express that, because it
spawned one Rayon job per batch and reported through a private outcome channel,
so only a Rayon pool could drive local execution. tensor4all/hataori-rs#24 makes
local fan-out run through an execution-lanes abstraction, and the adapter unit
supplies lanes backed by `tenferro_cpu::CpuPhase` (tenferro-rs#2201); that
rewrite is recorded in `2026-10-19-tenferro-held-session-lanes.md`.

## Chosen design

- `hataori::Lanes` (`src/lanes.rs`) is the one local fan-out contract:
  `run(lane)` and `run_with_coordinator(coordinator, lane)`. Both join every
  lane, resume a lane panic only after the join, and prefer a panic payload over
  an ordinary lane error. A lane error or panic cancels the shared control
  immediately so lanes waiting on cancellation are released before the join.
  The same concrete `LaneControl` (cancelled/cancel) is lent to the coordinator
  and to every lane. `LaneControl<'a>` owns its flag and offers a `with_probe`
  view that borrows the same flag and additionally reports an external stop
  signal, so an adapter whose backend can stop a run without reaching a lane
  callback lends the side that waits a view of the run's control instead of a
  second cancellation vocabulary. The trait has no `Send`/`Sync`/`'static`
  supertrait, so an executor, its lane resources, a coordinator capture, and a
  coordinator result can all stay on one thread.
- A lane callback receives its own lane resource. `map_in_lanes` and
  `pmap_lanes` therefore expose `f: Fn(T, &mut L::Lane<'_>) -> Result<U, E>`;
  this is the adapter seam that will hand a tenferro child session to the
  callback. Ordinary `pmap` adapts its `Fn(T)` shape to `RayonLanes`' unit lane.
- `map_in_lanes` spreads items with one shared atomic index counter and one
  per-index result slot: no Rayon job and no allocation per item, lowest failing
  index after every item finished, input order preserved. `map_in` `Outer` now
  lowers to it through `RayonLanes`; `Sequential`/`Inner` keep running one
  callback at a time on a pool worker.
- `RayonLanes` borrows a `Domain` and holds its admission. `run` uses
  `ThreadPool::broadcast`; `run_with_coordinator` uses `in_place_scope` plus
  `spawn_broadcast` and keeps the calling thread outside the pool, so one worker
  still overlaps a coordinator and its lane (a review requirement: the hybrid
  smoke depends on that overlap, and `lane_count() == 1` is not a reason to stop
  coordinating). `run_with_coordinator` rejects entry from one of the executor's
  own pool workers with the typed `LanesError::Reentrant` before spawning, since
  such a coordinator cannot make progress while its lanes are joined; the
  lane-only `run` keeps supporting a same-pool worker caller.
- Hybrid `pmap` now has one implementation of the root and worker loops, generic
  over `Lanes` and over a `BatchSink`. The coordinator is the existing protocol
  loop on the MPI main thread (MPI progress, scheduler, prefetch) and hands
  batches to lanes through a bounded local batch queue (`src/lanes/batch.rs`).
  The queue holds exactly one resident batch, which is the protocol's
  `running = 1` capacity; a prefetched batch waits in a coordinator slot, so
  `Sequential`/`Inner` still run one batch at a time and `Outer` spreads a batch
  over the lanes.
- Idle lanes poll (`rayon::yield_now()` plus a 200 µs park) instead of parking on
  a condition variable, so an idle lane still services nested same-pool work
  (`pool.broadcast`, `rayon::join`) issued by a busy lane. The coordinator's own
  waits recheck `LaneControl::cancelled()` and then abort, so nothing blocks
  unconditionally.
- Callback failures stay data (lowest index plus bounded message); callback
  panics are caught per batch into a `Panic` outcome that aborts 70 (root) or 71
  (remote) from the coordinator thread before any join; prefetch transfer
  failures keep abort 75 before any join. A remote rank inspects the current
  batch outcome before it submits or executes the prefetched batch, so a
  panicking batch aborts before a hung peer callback can run. Callback
  `Err`/`cancel()` is reserved for infrastructure failure.
- A one-lane executor returns the typed `LanesError::SingleLane` without running
  either callback. `pmap_lanes` then falls back explicitly to executing each
  batch on the coordinator thread through `Lanes::run` between MPI progress
  steps. The fallback is documented, and both MPI backends exercise it through a
  new `HATAORI_SINGLE_LANE_FALLBACK` stage in `scripts/check-hybrid.sh`.
- Deleted with no legacy copy: `spawn_local_batch`, `spawn_worker_batch`,
  `spawn_root_batch`, `NextWorkerBatch::Running` and its per-batch
  spawned-job/receiver machinery. MPI-only (`not(feature = "rayon")`) paths are
  unchanged.

Alternatives rejected: keeping the per-batch spawned-job path behind a second
entry point (two engines to test and document); making the queue capacity
configurable (the protocol already fixes one running plus at most one prefetched
batch); requiring `Send` on lane resources or coordinator results (the tenferro
child session is `!Send`, and the coordinator must be able to return non-`Send`
state on the calling thread); a per-backend mirrored cancellation type instead of
one hataori `LaneControl` that adapters bridge into.

## Waived performance gate

`REPOSITORY_RULES.md` requires a predeclared paired release-mode comparison
against legacy baseline `34cb1b13` before the P0/P1 local execution path is
replaced. The maintainer waived that gate for this refactor on 2026-10-19
("厳密な測定はいらないよ"). The waiver is recorded in `docs/design.md` §18 and
covers this replacement only; it does not waive the frozen gate for any later
facade or runtime-path promotion.

In place of the paired measurement, the changed inner paths were reviewed
structurally: per item the lane side costs one atomic increment, one short
mutex-guarded slot take, and one short result store, with no allocation, no
per-item job, and no mode dispatch inside item execution (the mode is matched
once when a batch is claimed). Compared with the previous `into_par_iter()`
fan-out for `map_in` `Outer`, this trades work stealing for a shared counter;
compared with the previous per-batch `Scope::spawn` plus capacity-one channel for
hybrid `pmap`, it removes the per-batch heap-allocated channel and spawned job.
No numerical kernel changed.

## Verification

- `cargo test --features rayon` (unit + doctests), `cargo test
  --no-default-features --features mpi,rayon`, `clippy --all-targets -D warnings`
  for default/`rayon`/`mpi`/`mpi,rayon`/`rsmpi-rt,rayon`, and `cargo doc` for
  every feature set.
- `scripts/check-hybrid.sh` (both MPI backends, n=1/2/4, panic,
  live-prefetch-transfer, and single-lane-prefetch-panic abort watchdogs, new
  single-lane fallback stage). The abort fixtures require the exact abort code
  from the launcher banner or, because Open MPI intermittently replaces the
  banner with its own `show_help` error, from the launcher exit status.
  `scripts/check-placement.sh`, `scripts/check-faults.sh`,
  `scripts/check-compile-boundaries.sh`, `scripts/check-mpi-call-boundaries.py`,
  `scripts/check-llms-index.py`.
- The hybrid smoke runs the full `LocalMode` x prefetch matrix on one- and
  two-worker domains, with an `Inner` callback asserting that a same-pool
  `broadcast` reaches every worker while the idle lanes wait, and the
  `HATAORI_SINGLE_LANE_PREFETCH_PANIC` stage is a 15 s watchdog regression for
  the inline prefetch abort ordering (it times out without that pre-check).
- `pmap_lanes` applies the same first-statement Rayon-worker guard as `pmap`,
  before the thread-level query and communicator duplication.
- `cargo test -p hataori-tenferro` stayed green at this stage through its
  Phase 20a external-domain tests; the adapter rewrite that follows replaces an
  `Outer` overlap fixture with the held-session lane suite recorded in
  `2026-10-19-tenferro-held-session-lanes.md`.
- Not run on this host: the Rust 1.85 MSRV matrix (toolchain absent) and the
  `x86_64-apple-darwin` non-Linux check (target absent).
