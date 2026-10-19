# Hataori P0 core acceptance ledger

This ledger maps the core-only acceptance contract in `docs/design.md` §16 and
`docs/implementation-readiness.md` §5 to checked evidence. Adapter claims remain
outside this ledger: Phase 20a tenferro evidence is recorded in
`docs/tenferro-acceptance-ledger.md`, while the tensor4all layer remains blocked
on tensor4all-rs#663.

## Reproducible entry point

```bash
scripts/check-core.sh /absolute/path/to/libmpiwrapper.so \
  966f4231c96153a08295fc7d0bcbd65e916a73fd
```

`.github/workflows/ci.yml` builds that exact MPIwrapper source revision against
Open MPI and runs the same entry point. No prebuilt fixture or secret is used.

## Feature, dependency, lint, and documentation matrix

| Contract | Evidence |
|---|---|
| Dependency-free default | `check-core.sh`: default tests and `cargo tree` exclusion |
| Rayon without MPI/serde/bincode | `check-core.sh`: Rayon tests/tree, local borrowed and global-pool tests |
| Upstream MPI and hybrid | six-feature loop plus `check-hybrid.sh`, `check-placement.sh`, `check-faults.sh` |
| Runtime MPI and hybrid | same scripts plus `check-rsmpi-rt.sh` poisoned build/tree boundary |
| Backends mutually exclusive | intended compile failure checked by `check-core.sh` and downstream fixture |
| Rust 1.85 | `cargo +1.85.0 check` for every valid feature set |
| Formatting, lint, docs | fmt plus all-target clippy `-D warnings` and rustdoc for every valid set |
| Non-Linux Rayon build | `x86_64-apple-darwin` check when the installed CI/local target is present |

## Scheduling and protocol

| Acceptance criterion | Evidence |
|---|---|
| Empty, world size one, extra ranks, batch 1 and larger fixed batches | MPI-only smoke in both backends (upstream via `check-hybrid.sh`, runtime via `check-rsmpi-rt.sh`); explicit one-item n=4 and delayed batches in `mpi_pmap_smoke.rs` / `rsmpi_rt_pmap_smoke.rs` |
| Execute once and preserve order under delayed/reverse completions | delayed callback smoke with global call-count reduction; scheduler `reverse_completion_restores_order_and_metadata_is_exact` |
| Dynamic skew improves over static contiguous assignment | scheduler `dynamic_skew_beats_static_contiguous_assignment` |
| Capacity one, no prefetch, exact completion, one STOP/DRAIN | scheduler running-slot, ready/stop/drain, error/drain, and transactional-completion tests |
| Root-running dispatch is state preserving | scheduler `running_slots_are_capacity_one_without_prefetch` |
| Root computes, including n=1 | MPI-only smoke and hybrid root-participation assertion |
| FUNNELED root identity and fair local/remote progress | hybrid chooser/guard tests and `check-hybrid.sh` |
| Rendezvous progress while root computes | 2 MiB worker result; worker-side post-send Drop trace must appear before root callback completes |
| Callback panic aborts promptly | both hybrid backends' subprocess panic watchdog |
| Signed deterministic simultaneous user/decode winner | `check-faults.sh`: rank 1 malformed result plus rank 2 user error, both root counters asserted, rank 2 key winner asserted on all ranks |
| Drain, caller communicator reuse, no private unmatched frame | same fault test: post-failure barrier/raw caller ping and successful pmap reuse |
| Private communicator does not consume caller tags | placement smoke preserves caller tag 2 traffic on the original communicator |
| Wire version/kind/status/length/count/overflow/trailing errors typed | `wire.rs` unit suite plus placement announced-length test |
| Malformed frame payload is drained before decode | placement receive ordering comment/test; pmap fault decode counter increments only after payload receipt |
| Error-key overflow and signed ordering | `wire.rs` error-key validation/order tests and multi-rank signed-min fault winner |
| Same-rank codec count zero | filtered n=1 `size_one_root_paths_bypass_codec` for pmap and all placement helpers, using Serialize-always-error values and zero codec counters |
| Broadcast/scatter/gather ownership and order | `check-placement.sh`: arbitrary roots, rank shards/order, UTF-8, empty, large values, reuse, encode/decode faults, size-one no-codec and `Rc` non-`Send` values |

## Local lanes executor

| Contract | Evidence |
|---|---|
| `map_in_lanes` order, exactly-once, lowest error, panic join | `src/lanes/tests.rs` at fake-lane and Rayon worker counts 1/2/4; `map_in` `Outer` compatibility tests in `src/local.rs` |
| Lane resource reached per lane, no `Send`/`Sync` executor bound | fake `Lanes` impl with `Rc`-backed lane state, non-`Send` executor, and a coordinator capture/result that stay on one thread |
| Lane entry, cancellation, and join rules | `RayonLanes::run`/`run_with_coordinator` tests: overlap at one worker, coordinator error and panic, lane error, lane panic, immediate cancel on lane failure, panic-over-error precedence for both entries, same-pool-worker `run` and typed `Reentrant` for `run_with_coordinator`, and admission reuse after each failure |
| External stop probe for backend-owned cancellation | `LaneControl::with_probe` unit test: a probe view follows the external signal, shares the run's cancellation flag (`cancel` on either view reaches the other), and does not report an external stop as an own-flag cancellation |
| Idle lanes keep nested pool work moving | hybrid smoke `LocalMode::Inner` matrix at one and two workers asserts a lane-side `pool.broadcast` reaches every worker and `rayon::join` completes; `lanes::tests::batch` covers the queue-level case |
| Hybrid batches run on lanes for every mode and prefetch setting | `check-hybrid.sh` n=1/2/4 hybrid matrix, `LocalMode` x prefetch at one and two workers |
| One-lane typed rejection falls back explicitly | `HATAORI_SINGLE_LANE_FALLBACK` stage in `check-hybrid.sh` runs `pmap_lanes` with a `SingleLane` executor in both backends at n=1/2, and the smoke runs it at n=1/2/4 |
| Single-lane prefetch abort ordering | `HATAORI_SINGLE_LANE_PREFETCH_PANIC` stage: a panicking current batch must reach `MPI_Abort(71)` under a 15 s watchdog without executing the prefetched batch |
| Panic, prefetch-transfer, and user-error abort semantics unchanged | existing panic `MPI_Abort(70)` and live-prefetch `MPI_Abort(75)` watchdog stages; fault-convergence suite |

## P1 bounded prefetch

| Acceptance criterion | Evidence |
|---|---|
| Opt-in/default-off and P0 preservation | `PmapOptions::default`; unchanged no-prefetch worker loop; existing hybrid matrix still runs every local mode with `prefetch = false` |
| MPI-only rejection before execution | both MPI-only smoke examples request prefetch, observe collective `Preflight`, and reduce callback count to zero |
| Capacity and promotion | scheduler `bounded_prefetch_promotes_in_order_and_never_exceeds_one` and STOP/error/protocol companion tests |
| Task and result overlap | n=2 hybrid serialization handshake proves task N+1 is decoded during callback N and result N is decoded during callback N+1 |
| Ordered exactly-once execution | prefetched 16-item call reduces callback count and checks root order under both backends at n=1/2/4 |
| Failure, drain, and reuse | n=2 hybrid fixtures cover current+prefetched user errors, current/prefetched input decode failures, deterministic first error, assigned callback count, and successful reuse |
| Live-job transfer failure abort | subprocess fixture fails result-N serialization after callback N+1 starts and requires prompt init-thread `MPI_Abort(75)` before scoped join |
| FUNNELED and scoped bounds | all new MPI sites use `mpi_call!`; source scanner and test builds enforce `MPI_Is_thread_main`; both hybrid roles run their protocol loop as the `Lanes::run_with_coordinator` coordinator, and `RayonLanes` joins its lanes inside `in_place_scope` with borrowed, non-`'static` callbacks |
| Bounded memory | scheduler state structurally stores one current plus at most one prefetched `RunningMeta`; the local batch queue keeps one resident batch per domain and the coordinator holds at most one prefetched batch plus one outcome |

## MPI thread boundary

`mpi_call!(...)` surrounds every Hataori-owned MPI operation in `pmap.rs` and
`placement.rs`. Test builds assert `MPI_Is_thread_main` immediately before the
operation; release builds expand to the operation alone. The balanced scanner
`scripts/check-mpi-call-boundaries.py` fails on future unwrapped call tokens and
runs in `check-core.sh`/CI. Calls made internally by the selected rsmpi backend,
and MPI initialization in test/example harnesses, are intentionally outside
this Hataori-owned boundary.

## Domain, Rayon, and affinity

| Acceptance criterion | Evidence |
|---|---|
| Linux managed workers pinned to distinct allowed CPUs | managed affinity tests in `domain.rs` |
| Bounded start latch and typed pin/start failures | managed construction and injected pinner tests |
| Non-Linux mode reports `CallerDeclared` without claiming pinning | non-Linux cfg test plus cross-target check |
| External pool not repinned/shut down | external ownership test |
| Admission released after success/error/unwind | domain and local tests |
| Missing/foreign/busy precedence before callbacks | `local.rs` precondition tests |
| Sequential/Inner stop first; Outer executes all and reports lowest | shared local-kernel tests, exercised again by hybrid smoke |
| Global pool remains outside selected execution | explicit-pool callback assertions and global-pool sentinels |
| Rayon-worker hybrid entry guard | MPI-free context matrix for plain threads, install/scope/in-place-scope/global worker behavior |

## Compile-time boundaries

`scripts/check-compile-boundaries.sh` builds downstream crates proving:

- serial `map` accepts borrowed `Rc` state without serde/Send/Sync/`'static`;
- Rayon accepts borrowed scoped state;
- MPI-only `pmap` accepts custom-serde `Rc` values and `Rc` callback state;
- hybrid callbacks borrow non-`'static` synchronized state;
- a selected communicator cannot move into `std::thread::spawn`;
- enabling both MPI backends fails with the intended diagnostic.

No unsafe `Send`/`Sync` wrapper is present in Hataori.

## Delivery record

Core tasks were delivered as separate commits:

- #9 `b1da840` — crate/feature matrix
- #10 `dc6c81d` — serial map
- #11 `00099a2` — domain/admission
- #12 `3cf81b2` — Rayon/affinity
- #13 `87c0552` — wire/error keys
- #14 `d919460` — scheduler
- #15 `a8cb009` — upstream MPI pmap
- #16 `163f3d6` — rsmpi-rt parity (upstream fix `6db6a2d6`)
- #17 `09bde95` — hybrid pmap/root compute
- #18 `7ad7517` — placement helpers
- #19 — complete core acceptance matrix (the commit containing this ledger)
