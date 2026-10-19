# tenferro held-session lane adapter acceptance ledger

Run the complete adapter gate from the repository root:

```bash
scripts/check-tenferro.sh
```

| Contract | Evidence |
|---|---|
| Core default and Rayon builds contain no tenferro/tensor4all | root `cargo tree` assertions |
| Adapter pins tenferro-rs merge `74aa7fac` (phase with a caller-thread coordinator) and contains no tensor4all | adapter tree and `cargo metadata` pin assertions |
| Core retains Rust 1.85; adapter declares `rust-version = "1.96"` matching the pinned tenferro-rs workspace | `check-tenferro.sh` runs `cargo +1.85.0 check` for core only; the adapter is checked on current stable, and no separate 1.96 toolchain check exists |
| tenferro owns the pool, admission, and per-worker child sessions | adapter opens one held session and creates no pool, domain, thread, or TLS selector |
| Known-value tenferro work runs through `map_in_lanes(TenferroLanes)` | every item once, input order preserved, two distinct lane indices, `add_read` doubling per lane |
| Work runs on tenferro's own workers | worker-name census: every observed name starts with `tenferro-cpu-` |
| Lowest failing input index survives the lane fan-out | deterministic two-error fixture reporting index 1 |
| A lane panic joins before it is resumed and the session stays reusable | panic fixture followed by a successful two-item fan-out |
| Peers waiting on `LaneControl` are released and joined | direct-`Lanes` fixtures: normal coordinator return (no explicit `cancel`), coordinator error, coordinator panic, lane error, lane unwind |
| tenferro's own phase stop reaches a waiting coordinator | `LaneControl::with_probe` shared-flag/probe unit test; the adapter lends the coordinator a probe view of `PhaseControl::cancelled()` |
| A one-worker backend drives one inline lane | `run_with_coordinator` returns `LanesError::SingleLane` without running either callback; `map_in_lanes` evaluates every item on the calling thread |
| Coordinator exit stops the lanes | two-lane fixture where the coordinator returns normally after both lanes start; both lanes observe the Hataori control and tenferro's own stop signal |
| `pmap_lanes(TenferroLanes)` works at world size one | `mpi`-gated example under `mpiexec -n 1 --workers 1` (asserts one lane and `SingleLane`) and `--workers 2` with an unpinned launcher (asserts two lanes) |
| Public docs remain runnable | adapter and core doctests |
| Adapter lints clean with and without the `mpi` example feature | `cargo clippy -p hataori-tenferro [--features mpi] --all-targets -- -D warnings` |

The exact upstream revision separately verifies its own pool, admission, child
session, coordinator, and cancellation internals; Hataori does not duplicate
those tests. The adapter does not claim multi-rank hybrid acceptance, tensor4all
contexts, tensor wire formats, or three-repository integration; those belong to
Phases 20b and 20c.
