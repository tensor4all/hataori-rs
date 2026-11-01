# 3. MPI pmap

**Model:** dynamic scheduling across MPI ranks, one sequential worker per
rank. **Features:** `mpi,tenferro` (or `rsmpi-rt,tenferro`), without
`rayon`. **Example:** `mpi_mandelbrot_pmap`.

`hataori::pmap` is a collective. Every rank calls it in the same order with
the same communicator and `PmapOptions`; the root rank supplies the items
and receives the ordered results, and every other rank supplies `None` and
receives `None`. Internally the root hands out batches to idle ranks —
including itself — until the input is exhausted, so the uneven column costs
balance automatically.

## The collective call

<!-- snippet-source: examples/mpi_mandelbrot_pmap.rs#pmap-call -->
```rust
/// One collective `pmap` call. Every rank enters this function; only the
/// root supplies the column indices and only the root receives the columns.
fn render<C: Communicator>(world: &C, param: &Param, batch_size: NonZeroUsize) -> Option<Tensor> {
    let rank = world.rank();
    let (x, y) = param.make_axes();

    // Column indices and columns cross rank boundaries, so they must be
    // serde serializable; `usize` and `Vec<i64>` already are. The callback
    // itself only borrows `param`, `x`, and `y` on whichever rank runs it.
    let input = (rank == 0).then(|| (0..param.width).collect::<Vec<usize>>());
    let columns = pmap(
        world,
        // The MPI-only entry point always evaluates callbacks sequentially
        // in a one-worker domain.
        &Domain::sequential(),
        PmapOptions {
            batch_size,
            ..PmapOptions::default()
        },
        input,
        |col_idx| Ok::<_, String>(mandelbrot_common::compute_column(param, x[col_idx], &y)),
    )
    .expect("pmap must succeed");

    // The root receives `Some(columns)` in input order; other ranks `None`.
    columns.map(|columns| {
        mandelbrot_common::tensor_from_ordered_columns(param.width, param.height, columns)
    })
}
```
<!-- end-snippet-source -->

- The items (`usize` column indices) and results (`Vec<i64>` columns)
  travel between processes, so they need serde; the callback does not. It
  runs on whichever rank received the batch and borrows that rank's `param`,
  `x`, and `y`.
- `Domain::sequential()` is the only domain the MPI-only entry accepts: one
  worker, no pool.
- `local_mode` must be `Sequential` and `prefetch` must be `false` here
  (both are the defaults); other values fail collective preflight before
  any callback runs.
- `root` defaults to rank 0 and may be any rank in `[0, world.size())`.

## Choosing the batch size

<!-- snippet-source: examples/mpi_mandelbrot_pmap.rs#batch-size -->
```rust
/// Columns per `pmap` batch.
///
/// The batch size is `width / (world_size * factor)`, clamped to at least
/// [`MIN_BATCH_COLUMNS`] columns. A larger `--batch-factor` produces smaller
/// batches and therefore finer load balancing, but more per-batch protocol
/// overhead. The default of 32 measured fastest on a 4096-column image
/// across 2-8 ranks.
fn batch_size(param: &Param, size: i32, factor: usize) -> NonZeroUsize {
    let per_rank = (size as usize).max(1) * factor.max(1);
    NonZeroUsize::new(
        (param.width / per_rank)
            .max(MIN_BATCH_COLUMNS)
            .min(param.width),
    )
    .expect("width is positive")
}
```
<!-- end-snippet-source -->

`batch_size` is the number of items per message. Larger batches amortize
the per-batch handshake; smaller batches balance better. The `--batch-factor`
flag lets you measure the trade-off: `mpiexec -n 4 … --batch-factor 4`
sends 16 batches of 256 columns, `--batch-factor 64` sends 256 batches of 16.

## Errors converge on every rank

<!-- snippet-source: examples/mpi_mandelbrot_pmap.rs#pmap-errors -->
```rust
/// A callback error on any rank converges to one `PmapError` on every rank,
/// so all ranks can agree on the failure and keep using MPI afterwards.
fn reject_columns_outside<C: Communicator>(world: &C, param: &Param, limit: usize) {
    let rank = world.rank();
    let (x, y) = param.make_axes();
    let input = (rank == 0).then(|| (0..param.width).collect::<Vec<usize>>());
    let error = pmap(
        world,
        &Domain::sequential(),
        PmapOptions::default(),
        input,
        |col_idx| {
            if col_idx >= limit {
                return Err(format!("column {col_idx} is outside the requested range"));
            }
            Ok(mandelbrot_common::compute_column(param, x[col_idx], &y))
        },
    )
    .expect_err("column `limit` fails on whichever rank evaluates it");
    assert_eq!(error.kind(), PmapErrorKind::User);
    assert!(error.message().contains("outside the requested range"));
}
```
<!-- end-snippet-source -->

A user error on one rank becomes the same `PmapError` on all ranks —
`kind()` is `PmapErrorKind::User`, `message()` carries the callback's
`Display` output — so the program can agree on what happened and keep using
MPI afterwards (the example renders the image first and runs this failing
call second on the same communicator). Panics inside a callback abort the
whole MPI job instead.

Source: [`examples/mpi_mandelbrot_pmap.rs`](https://github.com/tensor4all/hataori-rs/blob/main/examples/mpi_mandelbrot_pmap.rs)

## Build and run

```bash
cargo build --release --no-default-features --features mpi,tenferro \
  --example mpi_mandelbrot_pmap
mpiexec -n 4 target/release/examples/mpi_mandelbrot_pmap --width 1024 --height 1024
```

::: {.callout-note}
Build this tutorial **without** the `rayon` feature. When `rayon` is enabled
`pmap` becomes the [hybrid entry point](hybrid-pmap.md) and requires a
domain with a pool.
:::

## Compare with hand-written MPI

[`examples/mpi_mandelbrot_raw.rs`](https://github.com/tensor4all/hataori-rs/blob/main/examples/mpi_mandelbrot_raw.rs)
renders the same image with a static round-robin split and explicit
tagged sends and receives. Run both at the default 4096 × 4096 size:

```bash
cargo build --release --no-default-features --features mpi,tenferro \
  --example mpi_mandelbrot_raw
mpiexec -n 4 target/release/examples/mpi_mandelbrot_raw
```

The raw version reports its computation and transfer time per rank; the
`pmap` version reports one total. With the columns' uneven cost, dynamic
scheduling finishes earlier than the static split as soon as the batches
are small enough to rebalance.

Next: [4. MPI placement collectives](mpi-placement.md).
