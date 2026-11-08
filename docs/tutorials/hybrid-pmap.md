# 5. Hybrid MPI + Rayon pmap

**Model:** MPI ranks, each with its own Rayon pool. **Features:**
`mpi,rayon,tenferro` (or `rsmpi-rt,rayon,tenferro`). **Example:**
`mpi_mandelbrot_hybrid`.

With both feature groups enabled, `hataori::pmap` is the hybrid entry point.
The collective protocol is the same as in [tutorial 3](mpi-pmap.md): the root
owns the items and dynamically hands out batches. What changes is that each
rank evaluates its batch inside a pooled `Domain` with the requested
`LocalMode`, so in a one-rank-per-node deployment all cores of every node
participate.

## Initialize MPI for funneled threads

All MPI calls stay on the thread that calls `pmap`, while Rayon workers
evaluate callbacks concurrently. That is exactly `MPI_THREAD_FUNNELED`, so
ask for it explicitly and check what the implementation provided.

<!-- snippet-source: examples/mpi_mandelbrot_hybrid.rs#hybrid-init -->
```rust
/// Hybrid `pmap` keeps every MPI call on the thread that initialized MPI
/// while Rayon workers evaluate callbacks, so MPI must be initialized with
/// at least `MPI_THREAD_FUNNELED`.
fn initialize_funneled() -> Universe {
    let (universe, provided) =
        mandelbrot_common::mpi_api::initialize_with_threading(Threading::Funneled)
            .expect("MPI must not already be initialized or finalized");
    assert!(
        provided >= Threading::Funneled,
        "hybrid pmap requires MPI_THREAD >= Funneled"
    );
    universe
}
```
<!-- end-snippet-source -->

## The collective call with a pooled domain

<!-- snippet-source: examples/mpi_mandelbrot_hybrid.rs#hybrid-call -->
```rust
/// Every rank builds a Rayon domain and passes it to the collective. The
/// root hands out batches to itself and to remote ranks; each rank runs its
/// batch through its own pool with the requested `LocalMode`.
fn render<C: Communicator>(
    world: &C,
    param: &Param,
    workers: usize,
    batch_size: NonZeroUsize,
    prefetch: bool,
) -> Option<Tensor> {
    let rank = world.rank();
    let (x, y) = param.make_axes();

    // The worker count may differ per rank; the `PmapOptions` must not.
    // The CPU ids come from this rank's own affinity mask: `mpiexec
    // --bind-to core` (or a runner that reserves cpu 0) narrows it, so
    // `0..workers` is not guaranteed to be admissible.
    let domain = match managed_domain(workers) {
        Ok(domain) => domain,
        Err(message) => {
            eprintln!("rank {rank}: {message}");
            // Rank 0 alone cannot report this: every other rank stays in the
            // collective, so tear the whole job down instead of hanging.
            world.abort(1)
        }
    };

    let input = (rank == 0).then(|| (0..param.width).collect::<Vec<usize>>());
    let columns = pmap(
        world,
        &domain,
        PmapOptions {
            batch_size,
            // `Outer` spreads the columns of one batch over the pool.
            local_mode: LocalMode::Outer,
            // When true, each remote rank may hold one extra batch so that
            // the next transfer overlaps the current computation.
            prefetch,
            ..PmapOptions::default()
        },
        input,
        |col_idx| Ok::<_, String>(mandelbrot_common::compute_column(param, x[col_idx], &y)),
    )
    .expect("pmap must succeed");

    columns.map(|columns| {
        mandelbrot_common::tensor_from_ordered_columns(param.width, param.height, columns)
    })
}
```
<!-- end-snippet-source -->

- Every rank builds a `Domain::managed` (or `Domain::external`) pool; the
  worker count can differ per rank, but the `PmapOptions` must be identical.
- `LocalMode::Outer` spreads the columns of one batch over the pool, so
  `batch_size` should be at least the worker count. `Inner` is the choice
  when each callback is itself a parallel kernel, as in
  [tutorial 2](rayon-map-in.md).
- The callback and the values need the Rayon `Send`/`Sync` bounds in
  addition to serde, but still not `'static`: the closure borrows `param`,
  `x`, and `y`.
- The call must originate outside any Rayon worker; calling `pmap` from
  inside a pool is rejected in preflight.

## The pool's CPU set

The worker count can differ per rank, but the CPU ids cannot be chosen by
arithmetic. A launcher (`mpiexec --bind-to core`) or a runner that reserves a
CPU narrows the affinity mask, so `0..workers` can name an id the process may
not use and `Domain::managed` rejects it. Build the set from
`hataori::allowed_cpus()` instead, and abort the whole job when the mask is
smaller than `--workers`: rank 0 alone cannot report it, because the other
ranks stay in the collective.

<!-- snippet-source: examples/mpi_mandelbrot_hybrid.rs#hybrid-domain -->
```rust
/// Rejects a worker request this process cannot honour.
///
/// Asking for more workers than the affinity mask allows is a readable
/// error, not a panic: a panicking rank leaves the others in a collective
/// until the launcher is killed.
fn check_worker_request(workers: usize) -> Result<(), String> {
    let allowed = hataori::allowed_cpus().map_err(|error| format!("affinity query: {error}"))?;
    check_against(workers, &allowed)
}

fn check_against(workers: usize, allowed: &[usize]) -> Result<(), String> {
    if workers == 0 {
        return Err("--workers must be greater than zero".to_owned());
    }
    if allowed.len() < workers {
        return Err(format!(
            "{workers} workers requested, but this process may only use {} CPU(s) {allowed:?}; lower --workers or launch with --bind-to none",
            allowed.len()
        ));
    }
    Ok(())
}

/// Builds this rank's managed pool from the CPUs the process may use.
fn managed_domain(workers: usize) -> Result<Domain, String> {
    // One snapshot: validating and slicing different queries would leave a
    // window where the mask shrinks between them and the slice panics.
    let allowed = hataori::allowed_cpus().map_err(|error| format!("affinity query: {error}"))?;
    check_against(workers, &allowed)?;
    Domain::managed(allowed[..workers].to_vec(), workers)
        .map_err(|error| format!("rayon domain: {error}"))
}
```
<!-- end-snippet-source -->

## Batch size with a pool on every rank

<!-- snippet-source: examples/mpi_mandelbrot_hybrid.rs#hybrid-batch-size -->
```rust
/// Columns per `pmap` batch.
///
/// The batch size is `width / (size * workers * factor)`, clamped to at
/// least [`MIN_BATCH_COLUMNS`] columns and to at least `workers`, so that
/// `LocalMode::Outer` can spread one batch over the whole pool. The factor is
/// relative to the total number of compute threads (`size * workers`)
/// because in hybrid mode every batch is internally subdivided by Rayon.
fn batch_size(param: &Param, size: i32, workers: usize, factor: usize) -> NonZeroUsize {
    // Saturating: `--workers` and `--batch-factor` are user input; a wrapped
    // product would divide by zero before the worker request is validated.
    let threads = (size as usize)
        .max(1)
        .saturating_mul(workers.max(1))
        .saturating_mul(factor.max(1));
    NonZeroUsize::new(
        (param.width / threads)
            .max(MIN_BATCH_COLUMNS)
            .max(workers)
            .min(param.width),
    )
    .expect("width is positive")
}
```
<!-- end-snippet-source -->

The `--batch-factor` is now relative to the total number of compute threads
(`size × workers`), because each batch is subdivided again by Rayon inside
the rank.

## Prefetch

The example renders the image twice, with `prefetch: false` and
`prefetch: true`, and asserts that both give the identical image. With
prefetch on, each remote rank may hold one additional batch, so receiving the
next batch and sending the previous results overlap the current
computation. The root's own domain always stays at capacity one. The design
and its bounds are described in
[P1 bounded prefetch](../design/bounded-prefetch.md).

Source: [`examples/mpi_mandelbrot_hybrid.rs`](https://github.com/tensor4all/hataori-rs/blob/main/examples/mpi_mandelbrot_hybrid.rs)

## Build and run

```bash
cargo build --release --no-default-features --features mpi,rayon,tenferro \
  --example mpi_mandelbrot_hybrid
mpiexec -n 2 target/release/examples/mpi_mandelbrot_hybrid --workers 2 --width 1024 --height 1024
```

On a laptop keep `ranks × workers` at or below the core count; `--workers`
defaults to every logical core, which is the right value for one rank per
node.

::: {.callout-warning}
## Launcher core binding vs. managed domains

On Linux, `Domain::managed` pins each worker to a CPU from `cpu_set` and
rejects CPUs outside the process's affinity mask
(`DomainBuildError::CpuNotAllowed`). Open MPI binds each rank to a single
core by default for small jobs, which makes a two-worker managed domain
fail. Either launch with `mpiexec --bind-to none` (or `--map-by
node:PE=<workers>` to give each rank a core set), or use `Domain::external`
with a pool the application built itself. `scripts/check-tutorial-examples.sh`
passes `--oversubscribe --bind-to none` to Open MPI for this reason.
:::

Next: [6. Runtime-loaded MPI](rsmpi-rt.md).
