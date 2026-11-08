# Parallel Execution Models

Hataori has two orthogonal axes of parallelism:

1. **Execution backend** — which processes and threads exist. This is chosen
   by Cargo features and by the function you call.
2. **Local mode** — how the callbacks of one operation are run inside a
   rank's thread pool. This is the [`LocalMode`](#local-modes) enum.

Every model shares one contract:

- input is a `Vec<T>`, output is a `Vec<U>` in the **same order**;
- each successful input is evaluated **exactly once**;
- the callback returns `Result<U, E>` with `E: Display`, and the first error
  is reported with a message truncated to at most 4096 bytes on a UTF-8
  boundary; `map` and `map_in` also expose the zero-based failing input index
  (`MapError::index()`), while `pmap` reports a `PmapErrorKind` plus the
  message and an opaque converged `key()`;
- calls are **synchronous** — the function returns when the whole operation
  has finished on the calling thread.

## Model 1: serial `map`

| | |
| --- | --- |
| Feature | none |
| Function | `hataori::map(items, f)` |
| Bounds | none beyond `E: Display`; no `Send`, `Sync`, `'static`, or serde |
| Error rule | stop at the first error; later items are never evaluated |

`map` runs on the calling thread. It is the reference semantics that the
parallel models preserve, and it is the right choice for tests, tiny inputs,
and callbacks that borrow non-thread-safe state.
Tutorial: [Serial map](../tutorials/serial-map.md).

## Model 2: Rayon `map_in` on a `Domain`

| | |
| --- | --- |
| Feature | `rayon` |
| Function | `hataori::map_in(&domain, local_mode, items, f)` |
| Bounds | the Rayon `Send`/`Sync` bounds; no `'static`, no serde |
| Error rule | depends on `LocalMode` (see below) |

A `Domain` is an explicit Rayon pool plus a declared CPU set:

- `allowed_cpus()` returns the CPU ids this process may run on (the Linux
  affinity mask, otherwise `0..available_parallelism`). It is the supported
  source for a `cpu_set`: `Domain::managed` rejects an id the process may not
  run on (`DomainBuildError::CpuNotAllowed`), and `0..workers` is wrong
  wherever the process does not start on cpu 0.
- `Domain::managed(cpu_set, workers)` creates and owns the pool. On Linux
  every worker is pinned to a CPU from `cpu_set` and the placement is verified
  before the constructor returns (`PlacementStatus::Verified`). On other
  platforms the set is only a declaration.
- `Domain::external(pool, cpu_set, workers)` wraps a pool the application
  already owns. Hataori never re-pins or shuts it down
  (`PlacementStatus::CallerDeclared`).

A domain admits **one coarse operation at a time**, and `map_in` must not be
called from inside another Rayon pool (`MapInError::ForeignPool`). The global
Rayon pool is never used implicitly.
Tutorial: [Rayon map_in](../tutorials/rayon-map-in.md).

## Model 3: MPI `pmap` (one sequential worker per rank)

| | |
| --- | --- |
| Feature | `mpi` or `rsmpi-rt`, **without** `rayon` |
| Function | `hataori::pmap(&world, &Domain::sequential(), options, root_items, f)` |
| Bounds | `T`/`U: Serialize + DeserializeOwned` (they cross ranks); the callback may borrow non-`'static` state |
| Error rule | sequential; the first error converges to the same `PmapError` on every rank |

`pmap` is a **collective**: every rank calls it in the same order with the
same communicator and `PmapOptions`. Only `options.root` passes
`Some(items)` and gets `Ok(Some(results))`; every other rank passes `None`
and gets `Ok(None)`. The root dynamically hands out batches of
`options.batch_size` items to whichever rank is idle (itself included), so
uneven work balances automatically, as in Distributed.jl's `pmap`.

This entry point requires `LocalMode::Sequential` and `prefetch: false`;
anything else fails collective preflight before any callback runs.
Tutorial: [MPI pmap](../tutorials/mpi-pmap.md).

The same features also provide the placement collectives `broadcast`,
`scatter`, and `gather` for owned serde values.
Tutorial: [MPI placement collectives](../tutorials/mpi-placement.md).

## Model 4: hybrid `pmap` (a Rayon pool on every rank)

| | |
| --- | --- |
| Feature | (`mpi` or `rsmpi-rt`) **and** `rayon` |
| Function | `hataori::pmap(&world, &domain, options, root_items, f)` with a pooled `Domain` |
| Bounds | serde on `T`/`U`, plus `Send`/`Sync` on the callback and values; no `'static` |
| Error rule | per batch, follows `LocalMode`; converges across ranks like model 3 |

When both feature groups are on, `hataori::pmap` **is** the hybrid entry
point. Each rank owns a `Domain` with a pool; the root distributes batches
across ranks and each rank evaluates its batch inside its pool using
`options.local_mode` (`Outer` spreads a batch over the workers). All MPI calls
stay on the calling thread, which must be the thread that initialized MPI with
`MPI_THREAD_FUNNELED` or stronger, and must not be a Rayon worker.

`PmapOptions::prefetch = true` lets each remote domain hold one extra batch so
transfer of the next batch overlaps computation of the current one (see the
[P1 bounded-prefetch design](../design/bounded-prefetch.md)). The root domain
always stays at capacity one.
Tutorial: [Hybrid MPI + Rayon pmap](../tutorials/hybrid-pmap.md).

::: {.callout-important}
## `rayon` changes what `pmap` means

With `rayon` enabled, `pmap` rejects `Domain::sequential()` during preflight
because the hybrid entry requires a pool. To run the pure-MPI model in a
binary that also links Rayon, build a one-worker managed domain and pass
`LocalMode::Sequential`; to get the MPI-only entry point, compile without the
`rayon` feature.
:::

## Local modes

`LocalMode` selects how the callbacks of one operation (a `map_in` call, or
one batch of a hybrid `pmap`) are executed inside the selected pool.

| Mode | Concurrency inside the pool | Nested Rayon in the callback | Error rule |
| --- | --- | --- | --- |
| `Sequential` | one callback at a time | not expected | stop at the first error |
| `Outer` | every callback in parallel across the workers | no (the pool is busy with the outer loop) | evaluate everything, then report the **lowest** failing index |
| `Inner` | one callback at a time | yes — `par_iter` etc. run in the same pool | stop at the first error |

Use `Outer` when items are independent and roughly uniform in cost. Use
`Inner` when each item is itself a parallel kernel (for example a tensor
contraction that parallelizes internally); this is also the mode integration
adapters such as `hataori-tenferro` enter through. `Sequential` is the
MPI-only default and a useful baseline.

## Choosing a model

| Situation | Model |
| --- | --- |
| Debugging, unit tests, non-thread-safe callbacks | serial `map` |
| One machine, CPU-bound independent items | Rayon `map_in` with `Outer` |
| One machine, each item is itself a parallel kernel | Rayon `map_in` with `Inner` |
| Several nodes, single-threaded kernels or a process per core | MPI `pmap` |
| Several nodes, one rank per node with all its cores | hybrid `pmap` with `Outer` (or `Inner` for parallel kernels) |
| Slow interconnect relative to callback cost in hybrid runs | hybrid `pmap` with `prefetch: true` |

## Current limits (P0/P1)

- Exactly one execution domain per rank: `DomainId::ZERO`. Other identifiers
  are rejected by `DomainId::try_from`.
- `pmap` is synchronous and the root participates as a worker.
- Batch size is fixed per call; there is no adaptive batching yet.
