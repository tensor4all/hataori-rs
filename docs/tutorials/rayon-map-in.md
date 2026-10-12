# 2. Rayon map_in

**Model:** rank-local thread parallelism inside one process.
**Features:** `rayon,tenferro`. **Example:** `rayon_mandelbrot`.

`hataori::map_in` maps over items inside an explicit Rayon `Domain`. Compared
with calling `par_iter` directly you get: an explicit, verifiable CPU
placement; a single admission slot per domain; and the three
[`LocalMode`](../guides/parallel-models.md#local-modes)s with their
well-defined error rules.

## Build a domain

A managed domain creates and owns its pool. The CPU set must be non-empty
and at least as large as the worker count; on Linux the workers are pinned
and verified at startup.

<!-- snippet-source: examples/rayon_mandelbrot.rs#managed-domain -->
```rust
/// Build a Hataori-owned Rayon pool with `threads` workers.
///
/// The CPU set must be non-empty and at least as large as the worker count.
/// On Linux every worker is pinned to its CPU and the placement is verified
/// at startup; macOS does not enforce affinity, so there the set is only a
/// declaration.
fn managed_domain(threads: usize) -> Domain {
    let cpu_set: Vec<usize> = (0..threads).collect();
    let domain = Domain::managed(cpu_set, threads).expect("rayon domain");
    assert_eq!(domain.worker_count(), threads);
    assert_eq!(domain.pool_ownership(), Some(PoolOwnership::Managed));
    domain
}
```
<!-- end-snippet-source -->

If the application already has a `rayon::ThreadPool`, wrap it with
`Domain::external(pool, cpu_set, workers)` instead. Hataori then records the
CPU set as `PlacementStatus::CallerDeclared` and never re-pins or shuts down
the pool.

## Render with `Outer`

<!-- snippet-source: examples/rayon_mandelbrot.rs#outer -->
```rust
/// Render every column across the pool with `LocalMode::Outer`.
///
/// Each column is one Rayon task; work stealing balances the uneven column
/// costs, and the result vector still follows the input order.
fn render_outer(domain: &Domain, param: &Param) -> Result<Vec<Vec<i64>>, MapInError> {
    let (x, y) = param.make_axes();
    let columns: Vec<usize> = (0..param.width).collect();
    map_in(domain, LocalMode::Outer, columns, |col_idx| {
        Ok::<_, String>(mandelbrot_common::compute_column(param, x[col_idx], &y))
    })
}
```
<!-- end-snippet-source -->

This is the same call as in [tutorial 1](serial-map.md) with a domain and a
mode in front. Rayon's work stealing balances the uneven column costs
across the pool, and the result vector still follows the input order.

## The other two local modes

<!-- snippet-source: examples/rayon_mandelbrot.rs#local-modes -->
```rust
/// The same image through the other two local modes.
///
/// `Sequential` evaluates one column at a time on a pool thread. `Inner`
/// also evaluates one column at a time, but leaves the pool free for nested
/// Rayon work inside the callback — here the rows of one column.
fn render_sequential_and_inner(domain: &Domain, param: &Param) -> (Vec<Vec<i64>>, Vec<Vec<i64>>) {
    let (x, y) = param.make_axes();
    let columns: Vec<usize> = (0..param.width).collect();

    let sequential = map_in(domain, LocalMode::Sequential, columns.clone(), |col_idx| {
        Ok::<_, String>(mandelbrot_common::compute_column(param, x[col_idx], &y))
    })
    .expect("sequential map_in must succeed");

    let inner = map_in(domain, LocalMode::Inner, columns, |col_idx| {
        let column: Vec<i64> = y
            .par_iter()
            .map(|&y_val| mandelbrot_common::mandelbrot_kernel(param, x[col_idx], y_val))
            .collect();
        Ok::<_, String>(column)
    })
    .expect("inner map_in must succeed");

    (sequential, inner)
}
```
<!-- end-snippet-source -->

- `Sequential` runs one callback at a time on a pool thread.
- `Outer` runs every callback in parallel and still returns results in input
  order.
- `Inner` runs one callback at a time but leaves the pool free for nested
  Rayon work inside the callback — here a `par_iter` over the rows of one
  column. It is the right mode when each item is itself a parallel kernel.

The example asserts that all three modes produce the identical image.

## Error rules differ by mode

<!-- snippet-source: examples/rayon_mandelbrot.rs#outer-errors -->
```rust
/// Error rules differ by mode.
///
/// `Sequential` and `Inner` stop at the first failing input. `Outer`
/// evaluates every input exactly once and then reports the **lowest**
/// failing index, so the error is deterministic no matter which column
/// finished first.
fn demonstrate_error_rules(domain: &Domain, param: &Param, limit: usize) {
    let (x, y) = param.make_axes();
    let columns: Vec<usize> = (0..param.width).collect();
    let render_up_to_limit = |col_idx: usize| {
        if col_idx >= limit {
            Err(format!("column {col_idx} is outside the requested range"))
        } else {
            Ok(mandelbrot_common::compute_column(param, x[col_idx], &y))
        }
    };

    for mode in [LocalMode::Sequential, LocalMode::Outer] {
        let error = map_in(domain, mode, columns.clone(), render_up_to_limit)
            .expect_err("column `limit` must fail");
        match error {
            MapInError::Callback(inner) => {
                assert_eq!(inner.index(), limit);
                assert!(inner.message().contains("outside the requested range"));
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }
}
```
<!-- end-snippet-source -->

`Sequential` and `Inner` stop at the first failing input. `Outer` evaluates
every input exactly once and then reports the **lowest** failing index, so
the result is deterministic regardless of which column finished first.

## Preconditions

`map_in` checks, before calling the callback, that the domain has a pool
(`MapInError::MissingPool`), that the call does not originate on another
Rayon pool (`MapInError::ForeignPool`), and that no other operation is
admitted on the domain (`MapInError::DomainBusy`).

Source: [`examples/rayon_mandelbrot.rs`](https://github.com/tensor4all/hataori-rs/blob/main/examples/rayon_mandelbrot.rs)

## Run

```bash
cargo run --release --no-default-features --features rayon,tenferro \
  --example rayon_mandelbrot -- --threads 4 --width 1024 --height 1024
```

`--threads` defaults to every logical core. Compare the total time with
tutorial 1 at the same size.

Next: [3. MPI pmap](mpi-pmap.md) distributes the same loop across
processes.
