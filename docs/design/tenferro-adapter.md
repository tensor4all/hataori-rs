# tenferro adapter: lanes over a held session

**Status:** Implemented (hataori-rs#24). This document supersedes the Phase 20a
external-domain design, which tenferro-rs no longer supports.

**Upstream contract:** tenferro-rs
[#2004](https://github.com/tensor4all/tenferro-rs/pull/2004) decision D9 (tenferro
never borrows a caller pool) and
[#2202](https://github.com/tensor4all/tenferro-rs/pull/2202) (phase with a
caller-thread coordinator), merge commit
`74aa7fac31490f7773978e6fc2b1e80391018b73`.

**Scope:** the tenferro-only adapter. tensor4all explicit contexts, tensor
transport, and tensor reconstruction stay outside this change.

## 1. Crate and dependency boundary

`hataori-tenferro` is a non-publishable workspace member. It depends on:

- local `hataori` with `rayon` enabled (for the `Lanes` contract and
  `pmap_lanes`);
- `tenferro-cpu` (default features off, **no provider feature**) and
  `tenferro-tensor` at exactly `74aa7fac`. tenferro-cpu's CPU providers
  (`native`, `blas`, `provider-inject`, ...) are mutually exclusive, so the
  consumer selects one through its own tenferro-cpu dependency and Cargo feature
  unification; the adapter's tests, doctests and examples enable `native` through
  a dev-dependency only;
- an optional `mpi` feature (`hataori/mpi` plus upstream `mpi`) used only by the
  `pmap_lanes` world-size-1 smoke example.

The adapter declares `rust-version = "1.96"`, matching the pinned tenferro-rs
workspace. It adds no Rayon handle, pool handle, tensor4all crate, or MPI
backend to ordinary Hataori builds; the root package's opt-in `tenferro` example
feature is separate and outside the Rust 1.85 core matrix.

## 2. Ownership

| Layer | Owns | Does not own |
|---|---|---|
| tenferro | the CPU pool and its invariants (worker affinity, worker stack), root admission (held session), one owner-inheriting child session per worker lane | which item runs where |
| hataori | scheduling: work queues, item-to-lane assignment, ordering and error rules, MPI distribution | a pool, a domain, or a thread on the tensor path |
| application | the per-item function | parallelism machinery |

The adapter is the seam: it opens tenferro's held root session, lends
tenferro's phase lanes to Hataori's `Lanes` implementation, and maps the two
cancellation and error vocabularies. It creates no pool, no domain, and no
thread, installs no thread-local selector, and never gives tenferro a borrowed
pool. D9 stays intact: the only admission is tenferro's held session.

A lane is one worker of the backend's own pool plus one child session that stays
open for the whole run, so scratch and prepared plans are reused across the
items that lane pulls. Lane kernels are not forced to be serial: the child
session follows tenferro's own lane policy, so a kernel issued from a lane may
still use the pool. Hataori's contract covers only which item runs on which
lane; it says nothing about how a kernel uses the pool.

## 3. Public API

```rust
pub struct TenferroLanes<'b> { /* CpuHeldSession<'b> */ }

impl<'b> TenferroLanes<'b> {
    pub fn open(backend: &'b tenferro_cpu::CpuBackend) -> Result<Self, TenferroAdapterError>;
    pub fn close(self) -> Result<(), TenferroAdapterError>;
}

impl hataori::Lanes for TenferroLanes<'_> {
    type Lane<'l> = tenferro_cpu::PhaseLane<'l>;
    type Error = TenferroAdapterError;
    // run(lane)                -> held.phase(|p| p.run(..))
    // run_with_coordinator(..) -> held.phase(|p| p.run_with_coordinator(..))
}
```

`open` admits the backend's root session for the executor's lifetime and records
`backend.num_threads()` as the lane count, which is the count tenferro's phase
will drive. `close` releases the session and reports an affinity-restoration
failure instead of logging it; dropping the executor performs the same release
best-effort.

The executor is `!Send` exactly like the held session, because admission, the
execution-owner marker, and the caller's narrowed CPU mask belong to the opening
thread. A caller that needs to run on another thread opens a new executor there.

`TenferroAdapterError` has four typed variants that all keep their upstream
source:

| Variant | Meaning |
|---|---|
| `Open(SessionEntryError)` | the held root session could not be opened |
| `Phase(CpuPhaseError)` | tenferro refused to open a phase (for example `TargetPoolCaller`) |
| `Session(SessionEntryError)` | a lane could not enter or clean up its child session |
| `Close(CpuHeldSessionError)` | the caller's CPU affinity mask could not be restored |

## 4. One run

`map_in_lanes` and `pmap_lanes` drive the same Hataori `Lanes` methods, so the
adapter needs no queue of its own. (`map_in` takes a Hataori `Domain`, not a
`Lanes` executor, so it is not a tenferro entry.)

```text
hataori::map_in_lanes(map) / pmap_lanes
  -> Lanes::run                 -> held.phase(|p| p.run(lane))                 (no coordinator)
  -> Lanes::run_with_coordinator-> held.phase(|p| p.run_with_coordinator(c, lane))
       lane k = tenferro worker k + child session k
         loop { item = hataori's next index; f(item, &mut lane); write slot }
```

Every run creates one `LaneControl` and lends it to the lane bodies and to the
coordinator, and it bridges in both directions:

- **tenferro to Hataori:** if a lane observes `PhaseLane::cancelled()` when it
  starts, it marks the Hataori control; a lane body that returns `Err` or
  unwinds marks it before tenferro joins (an unwind guard also marks it).
- **Hataori to tenferro:** when the coordinator returns, errors, or unwinds, the
  Hataori control is cancelled *and* `PhaseControl::cancel()` is called, so
  lanes that poll either signal stop feeding. This is the `Lanes` contract:
  "when the coordinator returns, errors, or panics, the control is cancelled and
  every lane is joined".
- **tenferro's own stop, without a lane callback:** tenferro can fail a phase
  before a lane body runs (a child-session entry or cleanup failure), so the
  executor's stop signal may never reach `run_lane`. The coordinator therefore
  receives a `LaneControl::with_probe` view of the run's control whose probe is
  `|| phase_control.cancelled()`. The view shares the run's cancellation flag, so
  `cancel` from either side still reaches both, and it adds tenferro's own stop
  to `cancelled()`. A coordinator waiting on `LaneControl` is thereby released
  instead of waiting forever, and it closes its work queue so the remaining lanes
  finish.

tenferro owns the join: `CpuPhase::run` broadcasts to every worker and
`run_with_coordinator` runs the coordinator in an `in_place_scope`, so both
return only after every lane finished, and a lane panic is resumed on the
calling thread after the join. The adapter does not duplicate that join and
does not catch panics.

Error mapping keeps the `source()` chain intact:

| tenferro | Hataori |
|---|---|
| `PhaseRunError::Lane(e)` | `LanesError::Lane(e)` |
| `PhaseRunError::Coordinator(e)` | `LanesError::Coordinator(e)` |
| `PhaseRunError::Session(s)` | `LanesError::Executor(TenferroAdapterError::Session(s))` |
| `PhaseRunError::SingleLane` | `LanesError::SingleLane` |
| `CpuPhaseError::TargetPoolCaller` (or any phase-open error) | `LanesError::Executor(TenferroAdapterError::Phase(..))` |

`SingleLane` is forwarded rather than hidden so the hybrid `pmap` takes its
documented explicit fallback: a one-worker backend drives its single lane
inline on the caller, and `pmap_lanes` executes each batch through
`Lanes::run` between MPI progress steps. `map_in_lanes` needs no fallback,
because `Lanes::run` supports the inline lane directly.

## 5. Removed Phase 20a chain

tenferro-rs #2004 (D9) removed borrowing an external pool, so the adapter no
longer has:

- `TenferroDomain::new(Domain)`, `domain()`, and the borrowed-pool backend built
  from `ExternalCpuDomain::new_caller_managed` /
  `CpuBackend::from_external_managed_domains`;
- `with_backend` and its per-domain `AtomicBool` `ConcurrentEntry` gate, plus
  `OutsideDomainPool` pool-membership entry;
- `MissingPool` and `ExternalDomain` errors, the `LocalMode::Inner`-only entry
  contract, and the `rayon::ThreadPool` handle the old binding retained.

There is one engine and one admission now, and no adapter-owned entry gate is
needed: the held session already rejects reentry and contention with typed
tenferro errors, and lanes cannot open a root session at all.

## 6. Acceptance evidence

`scripts/check-tenferro.sh` is the single entry point. It proves:

- core default and `rayon` trees still contain no tenferro or tensor4all crate;
- the adapter tree carries exactly the pinned tenferro revision (no older rev,
  no tensor4all package);
- `cargo test -p hataori-tenferro` proves, on a two-worker backend:
  known-value tenferro operations through `map_in_lanes(TenferroLanes)` (every
  item once, input order preserved, two distinct lane indices, and a worker-name
  census where every observed worker name starts with `tenferro-cpu-`), the
  lowest callback-error index, a lane panic that is resumed only after the join
  with the same session reusable afterwards, and a one-worker backend whose
  `run_with_coordinator` returns `LanesError::SingleLane` without running either
  callback while `map_in_lanes` still evaluates every item on the inline lane;
- direct `Lanes` liveness cases with peers waiting on `LaneControl`: a normal
  coordinator return releases the lanes without an explicit `cancel` (the
  cancellation-on-exit guard is what releases them), and a coordinator error, a
  coordinator panic, a lane error, and a lane unwind each release the waiting
  peers and join every lane; the core `LaneControl::with_probe` unit test covers
  the shared flag and the external stop view;
- the adapter builds no pool and no thread: no worker name outside
  `tenferro-cpu-*` is observed, and the inline lane runs on the calling thread;
- `pmap_lanes(TenferroLanes)` succeeds at world size one with an explicit worker
  count: `--workers 1` asserts one lane and the `LanesError::SingleLane` shape
  the hybrid `pmap` falls back from, and `--workers 2` (launcher set not to pin
  the rank) asserts two lanes and the concurrent coordinator;
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
