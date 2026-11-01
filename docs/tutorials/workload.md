# 0. The workload

All tutorials share one computation so that only the execution model
changes from page to page. It lives in
[`examples/support/mandelbrot_common.rs`](https://github.com/tensor4all/hataori-rs/blob/main/examples/support/mandelbrot_common.rs)
and every example includes it with `#[path]`.

## Parameters

<!-- snippet-source: examples/support/mandelbrot_common.rs#param -->
```rust
/// Parameters describing the Mandelbrot image region and resolution.
#[derive(Debug, Clone, Copy)]
pub struct Param {
    pub xmin: f64,
    pub xmax: f64,
    pub width: usize,
    pub ymin: f64,
    pub ymax: f64,
    pub height: usize,
    pub max_iter: i64,
}

impl Default for Param {
    fn default() -> Self {
        Self {
            xmin: -1.75,
            xmax: 0.75,
            width: 4096,
            ymin: -1.25,
            ymax: 1.25,
            height: 4096,
            max_iter: 500,
        }
    }
}
```
<!-- end-snippet-source -->

`Param::from_args()` applies `--width`, `--height`, and `--max-iter` from
the command line; `make_axes()` turns the region into one `f64` per column
and per row.

## The unit of work: one column

<!-- snippet-source: examples/support/mandelbrot_common.rs#kernel -->
```rust
/// Number of iterations until escape for a single complex coordinate.
pub fn mandelbrot_kernel(param: &Param, c_re: f64, c_im: f64) -> i64 {
    let mut z_re = c_re;
    let mut z_im = c_im;
    for i in 0..param.max_iter {
        let z_re_sq = z_re * z_re;
        let z_im_sq = z_im * z_im;
        if z_re_sq + z_im_sq > 4.0 {
            return i;
        }
        z_im = 2.0 * z_re * z_im + c_im;
        z_re = z_re_sq - z_im_sq + c_re;
    }
    param.max_iter
}

/// Compute one full column of the Mandelbrot set.
///
/// This is the unit of work every example distributes: columns near the
/// set's boundary need far more iterations than columns outside it, so the
/// cost per column is very uneven and dynamic scheduling pays off.
pub fn compute_column(param: &Param, x: f64, y: &[f64]) -> Vec<i64> {
    let mut col = Vec::with_capacity(y.len());
    for &y_val in y {
        col.push(mandelbrot_kernel(param, x, y_val));
    }
    col
}
```
<!-- end-snippet-source -->

Every tutorial maps `compute_column` over the column indices
`0..param.width`. Three properties of this workload matter for what
follows:

- **The cost per column is very uneven.** Columns that cross the set run
  every pixel to `max_iter`; columns outside it escape after a handful of
  iterations. A static split therefore leaves some workers idle, which is
  why the MPI tutorials contrast dynamic `pmap` scheduling with the static
  `scatter`/`gather` and raw-MPI versions.
- **The callback borrows.** Each closure captures `&param`, `&x`, and `&y`
  from the caller's stack. No Hataori entry point requires `'static`, and
  only the MPI ones require serde — for the *items and results*, not the
  callback.
- **The result is ordered.** Every entry point returns the columns in input
  order, so assembling the image is a `concat`
  (`tensor_from_ordered_columns`), and the serial, Rayon, MPI, and hybrid
  images are bit-for-bit identical.

## Output

`save_png` transposes the column-major `[width, height]` tensor to a
grayscale image; `max_iteration` asserts that at least one pixel reached
`max_iter`, which every example uses as its correctness check.

Next: [1. Serial map](serial-map.md).
