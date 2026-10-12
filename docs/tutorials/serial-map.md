# 1. Serial map

**Model:** one thread, one process. **Features:** `tenferro` (for the
image only). **Example:** `serial_mandelbrot`.

`hataori::map` is the semantic baseline for every other model: it applies a
fallible callback to a `Vec<T>` in order and returns a `Vec<U>` in the same
order. There are no `Send`, `Sync`, `'static`, or serde requirements, so the
callback can borrow anything, including `Rc` or `RefCell` state.

## Render the image

<!-- snippet-source: examples/serial_mandelbrot.rs#serial-map -->
```rust
/// Render every column in input order on the calling thread.
///
/// `map` takes a `Vec<T>` and a fallible callback and returns a `Vec<U>` in
/// the same order, or the first error together with the failing index. The
/// callback borrows `param`, `x`, and `y` from the caller's stack.
fn render(param: &Param) -> Result<Vec<Vec<i64>>, MapError> {
    let (x, y) = param.make_axes();
    let columns: Vec<usize> = (0..param.width).collect();
    map(columns, |col_idx| {
        Ok::<_, String>(mandelbrot_common::compute_column(param, x[col_idx], &y))
    })
}
```
<!-- end-snippet-source -->

The items are the column indices; the callback borrows the parameters and
the axes and returns one column. `map` evaluates the columns one at a time
on the calling thread.

## Errors stop at the first failure

<!-- snippet-source: examples/serial_mandelbrot.rs#serial-errors -->
```rust
/// `map` stops at the first error and reports its zero-based input index.
///
/// The callback error type only needs `Display`; `MapError` keeps the
/// index and the `Display` output (truncated to 4096 bytes).
fn reject_columns_outside(param: &Param, limit: usize) -> MapError {
    let (x, y) = param.make_axes();
    let mut evaluated = 0_usize;
    let error = map((0..param.width).collect::<Vec<usize>>(), |col_idx| {
        evaluated += 1;
        if col_idx >= limit {
            return Err(format!("column {col_idx} is outside the requested range"));
        }
        Ok(mandelbrot_common::compute_column(param, x[col_idx], &y))
    })
    .expect_err("column `limit` must fail");
    assert_eq!(error.index(), limit);
    // Columns after the failing one were never evaluated.
    assert_eq!(evaluated, limit + 1);
    error
}
```
<!-- end-snippet-source -->

- The error type only needs `Display`. `MapError` stores the failing input's
  index and the `Display` output, truncated to 4096 bytes at a UTF-8
  character boundary.
- `map` **stops at the first error**. The counter shows that the column
  after `limit` was never evaluated, and inputs after the failure are simply
  dropped.
- The callback is `FnMut`, so it can mutate captured state such as
  `evaluated`.

Source: [`examples/serial_mandelbrot.rs`](https://github.com/tensor4all/hataori-rs/blob/main/examples/serial_mandelbrot.rs)

## Run

```bash
cargo run --release --no-default-features --features tenferro \
  --example serial_mandelbrot -- --width 1024 --height 1024
```

The example prints the elapsed time, the first error from the failing run,
and writes `mandelbrot_serial.png`.

Next: [2. Rayon map_in](rayon-map-in.md) keeps this contract and adds a
thread pool.
