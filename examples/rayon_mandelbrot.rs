//! Mandelbrot set, Rayon-only dynamic scheduling (no MPI).
//!
//! This is the Rayon-only counterpart of `mpi_mandelbrot_pmap.rs`: the same
//! Mandelbrot workload, dynamically scheduled by Rayon's work-stealing
//! `par_iter` inside a single process via [`hataori::map_in`] with
//! [`LocalMode::Outer`]. There is no MPI and no rank coordination here; the
//! pool owns `--threads` worker threads and the call preserves input order.
//!
//! Run with:
//!
//! ```sh
//! cargo run --release --no-default-features --features rayon,tenferro \
//!     --example rayon_mandelbrot -- --threads 10
//! ```
use hataori::{map_in, Domain, LocalMode, MapInError, PoolOwnership};
use mandelbrot_common::{print_info, Param};
use rayon::prelude::*;
use std::time::Instant;

#[path = "support/mandelbrot_common.rs"]
mod mandelbrot_common;

// snippet-start:managed-domain
/// Build a Hataori-owned Rayon pool with `threads` workers.
///
/// The CPU set must be non-empty and at least as large as the worker count.
/// On Linux every worker is pinned to its CPU and the placement is verified
/// at startup; macOS does not enforce affinity, so there the set is only a
/// declaration.
fn managed_domain(threads: usize) -> Domain {
    // The mask decides which ids are admissible: a launcher or the runner may
    // reserve cpu 0, so `0..threads` is not guaranteed to be in it.
    let allowed = hataori::allowed_cpus().expect("affinity query");
    assert!(
        threads > 0 && allowed.len() >= threads,
        "{threads} threads requested, but this process may only use {} CPU(s) {allowed:?}",
        allowed.len()
    );
    let cpu_set: Vec<usize> = allowed[..threads].to_vec();
    let domain = Domain::managed(cpu_set, threads).expect("rayon domain");
    assert_eq!(domain.worker_count(), threads);
    assert_eq!(domain.pool_ownership(), Some(PoolOwnership::Managed));
    domain
}
// snippet-end:managed-domain

// snippet-start:outer
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
// snippet-end:outer

// snippet-start:local-modes
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
// snippet-end:local-modes

// snippet-start:outer-errors
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
// snippet-end:outer-errors

fn main() {
    let threads = mandelbrot_common::arg_thread_count("--threads");
    let param = Param::from_args();
    let domain = managed_domain(threads);

    print_info("Start processing Mandelbrot set...", "");
    print_info("Backend", "Rayon only (no MPI)");
    print_info("Threads", threads);
    print_info("Parameters", format!("{:?}", param));
    print_info("Computing...", "");

    let total_start = Instant::now();
    let columns = render_outer(&domain, &param).expect("map_in must succeed");
    let elapsed = total_start.elapsed();
    print_info("Total time", format!("{:.3} s", elapsed.as_secs_f64()));

    let (sequential, inner) = render_sequential_and_inner(&domain, &param);
    assert_eq!(
        sequential, columns,
        "every local mode renders the same image"
    );
    assert_eq!(inner, columns, "every local mode renders the same image");
    demonstrate_error_rules(&domain, &param, param.width / 2);

    let tensor = mandelbrot_common::tensor_from_ordered_columns(param.width, param.height, columns);
    print_info(
        "Max iteration count",
        mandelbrot_common::max_iteration(&tensor),
    );
    let png_path = mandelbrot_common::output_path("mandelbrot_rayon.png");
    mandelbrot_common::save_png(&tensor, &png_path);
    print_info("Saved", png_path);
}
