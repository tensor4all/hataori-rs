# tenferro adapter: lanes over a held session

**Scope:** hataori-rs #24, adapter unit. Rewrites `hataori-tenferro` on
tenferro's held CPU session and phase lanes and removes the Phase 20a
external-domain binding. Depends on the core lanes executor recorded in
`2026-10-19-lanes-executor.md`.

## Why

tenferro-rs #2004 (D9) removed borrowing a caller-managed external pool, so the
adapter's `ExternalCpuDomain` / `from_external_managed_domains` construction no
longer exists upstream, and tenferro-rs #2202 added a phase that runs a
caller-thread coordinator concurrently with its lanes. That is exactly the split
hataori needs: tenferro owns the pool, admission, and one child session per
worker; hataori decides which item runs where.

## Chosen design

- The adapter pins tenferro `74aa7fac` (`native`/Faer provider, Rust 1.96) and
  exposes one type, `TenferroLanes<'b>`, which owns a `CpuHeldSession<'b>`.
  `open(backend)` admits the held root session and records
  `backend.num_threads()` as the lane count; `close()` releases it and reports
  affinity-restoration failure. The adapter creates no pool, domain, thread, or
  TLS selector, and it is `!Send` exactly like the held session.
- `impl Lanes for TenferroLanes` maps `run` to `held.phase(|p| p.run(..))` and
  `run_with_coordinator` to `held.phase(|p| p.run_with_coordinator(..))`, with
  `Lane = PhaseLane`. Each lane is one tenferro worker plus a warm child session,
  and `map_in_lanes`/`pmap_lanes` hand that session to the per-item callback.
  Lane kernels keep tenferro's own lane policy; hataori only schedules items, so
  a lane is not promised to be serial (the #2201 review correction).
- One `LaneControl` per run is lent to both sides. tenferro -> hataori: a lane
  that starts with `PhaseLane::cancelled()`, returns `Err`, or unwinds marks the
  control. hataori -> tenferro: a drop guard around the coordinator cancels the
  hataori control *and* calls `PhaseControl::cancel()`, so lanes stop when the
  coordinator returns, errors, or panics. tenferro keeps ownership of the join
  and of resuming a lane panic, so the adapter adds no `catch_unwind`.
- Liveness: tenferro can fail a phase before a lane body runs (a child-session
  entry or cleanup failure), which sets tenferro's stop signal but never reaches
  the adapter's lane wrapper. The coordinator therefore receives a
  `LaneControl::with_probe` view of the run's control whose probe is
  `|| phase_control.cancelled()`; the view shares the run's cancellation flag,
  so `cancel` from either side still reaches both, and the coordinator's waits
  also observe tenferro's own stop. The coordinator then closes its queue and the
  remaining lanes finish.
- Errors keep their sources: `PhaseRunError::Lane` ->
  `LanesError::Lane`, `Coordinator` -> `LanesError::Coordinator`, `Session(s)`
  -> `LanesError::Executor(Session(s))`, `SingleLane` -> `LanesError::SingleLane`
  (so hybrid `pmap` takes its explicit `Lanes::run` fallback), and a phase-open
  error such as `TargetPoolCaller` -> `LanesError::Executor(Phase(..))`.
- Deleted: `TenferroDomain::new`/`domain`, `with_backend`, the borrowed-pool
  backend, the per-domain `AtomicBool` entry gate, `MissingPool`,
  `ExternalDomain`, `OutsideDomainPool`, `ConcurrentEntry`, and the old
  integration test that depended on them. The held session already rejects
  reentry and contention with typed tenferro errors.

Alternatives rejected: a wrapper lane type (raw `PhaseLane` already exposes
`session()`/`cancelled()`, so the narrowest surface is the type itself); an
adapter-owned queue (hataori's `map_in_lanes`/`pmap_lanes` already own item
distribution and ordering); re-deriving the held session per run (the cache and
plan warmth is the point of holding it across a fan-out region).

## Evidence

- `cargo test -p hataori-tenferro` covers known-value tenferro operations through
  `map_in_lanes` (order, exactly once, two distinct lane indices, worker names
  all `tenferro-cpu-*`), the lowest failing index, a lane panic with a reusable
  session afterwards, a one-worker backend (`SingleLane` for a coordinator run,
  inline lane for `map_in_lanes`), and direct `Lanes` liveness: a normal
  coordinator return releases the lanes without an explicit `cancel`, and a
  coordinator error, coordinator panic, lane error, and lane unwind each release
  peers waiting on `LaneControl` and join every lane.
- `pmap_lanes(TenferroLanes)` runs at world size one through the adapter's
  `mpi`-gated example with an explicit `--workers` count: one worker asserts a
  single lane and the `SingleLane` shape the hybrid `pmap` falls back from, and
  two workers (with the launcher not pinning the rank) assert two lanes and the
  concurrent coordinator. The repository's `examples/mpi_*` harness belongs to
  the root `hataori` package and cannot host the adapter, because the dependency
  direction is adapter -> `hataori`.
- `scripts/check-tenferro.sh` now asserts the new pin from the adapter's own
  dependency tree, runs all-target tests, doctests, clippy and rustdoc with
  `-D warnings`, keeps the Rust 1.85 core checks, and runs the MPI smoke under a
  short process-group watchdog.
- Not claimed: multi-rank hybrid acceptance for the adapter, tensor4all
  contexts/reconstruction, three-repository integration. On this host only two
  CPUs are visible, so a three-worker doctest would be capped to two workers;
  the documentation asserts two.
