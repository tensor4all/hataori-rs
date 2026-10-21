# tenferro engine lane adapter acceptance ledger

Run the complete adapter gate from the repository root:

```bash
scripts/check-tenferro.sh
```

| Contract | Evidence |
|---|---|
| Core default and Rayon builds contain no tenferro/tensor4all | root `cargo tree` assertions |
| Adapter pins tenferro-rs `5f0ae0a3` (caller-held engines, no sessions) and contains no tensor4all | adapter tree and `cargo metadata` pin assertions |
| Core retains Rust 1.85; adapter declares `rust-version = "1.96"` matching the pinned tenferro-rs workspace | `check-tenferro.sh` runs `cargo +1.85.0 check` for core only; the adapter is checked on current stable, and no separate 1.96 toolchain check exists |
| The caller owns the engines and their pools; the adapter builds neither | adapter has no pool constructor and no Rayon dependency of its own; `TenferroLanes::new` takes the caller's `Vec<CpuEngine>` |
| One lane per engine, in the caller's order | `each_lane_is_lent_its_own_engine_on_its_own_thread` census over two engines of different worker counts |
| A lane is never a worker of the engine pool it uses | inside every lane, the engine pool reports `current_thread_index() == None`, and a real `add_read` issued there succeeds rather than returning tenferro's own-pool refusal (the census runs the doubling on every lane, so no lane is skipped) |
| Known-value tenferro work runs through `map_in_lanes(TenferroLanes)` | every item once, input order preserved, `add_read` doubling per lane |
| Lowest failing input index survives the lane fan-out | deterministic two-error fixture reporting index 1 |
| A callback panic joins before it is resumed and the engines stay reusable | panic fixture raised outside any engine operation, followed by a successful two-item fan-out |
| Peers waiting on `LaneControl` are released and joined | direct-`Lanes` fixtures: normal coordinator return (no explicit `cancel`), coordinator error, coordinator panic, lane error, lane unwind |
| A single lane still overlaps a coordinator | one-engine fixture where the lane and the coordinator meet at a barrier; the held-session design refused this with `SingleLane` |
| A lane executor needs an engine | `TenferroLanes::new(Vec::new())` returns `NoEngine` |
| `pmap_lanes(TenferroLanes)` works at world size one | `mpi`-gated example under `mpiexec -n 1 --lanes 1` (asserts one lane with an overlapping coordinator) and `--lanes 2` with an unpinned launcher (asserts two lanes) |
| Public docs remain runnable | adapter and core doctests |
| Adapter lints clean with and without the `mpi` example feature | `cargo clippy -p hataori-tenferro [--features mpi] --all-targets -- -D warnings` |

The exact upstream revision separately verifies its own pool, engine, plan-cache
and cancellation internals; Hataori does not duplicate those tests. The adapter
does not claim multi-rank hybrid acceptance, tensor4all contexts, tensor wire
formats, or three-repository integration; those belong to Phases 20b and 20c.
