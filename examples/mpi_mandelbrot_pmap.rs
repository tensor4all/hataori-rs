//! Mandelbrot set, dynamically scheduled across MPI ranks with
//! [`hataori::pmap`], one sequential worker per rank.
//!
//! Rank 0 owns the list of column indices and hands out batches to idle
//! ranks — including itself — until the input is exhausted, so the uneven
//! column costs balance automatically. Compare with `mpi_mandelbrot_raw.rs`,
//! which distributes the same columns round-robin with hand-written sends
//! and receives.
//!
//! Build and run (link-time MPI):
//!
//! ```sh
//! cargo build --release --no-default-features --features mpi,tenferro \
//!     --example mpi_mandelbrot_pmap
//! mpiexec -n 4 target/release/examples/mpi_mandelbrot_pmap
//! ```
//!
//! The same source builds against the runtime-loaded backend as
//! `rsmpi_rt_mandelbrot_pmap` with `--features rsmpi-rt,tenferro`.
use hataori::{pmap, Domain, PmapErrorKind, PmapOptions};
use mandelbrot_common::mpi_api::traits::Communicator;
use mandelbrot_common::{print_info, Param};
use std::num::NonZeroUsize;
use std::time::Instant;
use tenferro_tensor::Tensor;

#[path = "support/mandelbrot_common.rs"]
mod mandelbrot_common;

/// Smallest allowed batch, in columns.
///
/// A column is `height` i64 values (32 KiB at 4096 rows), so a 16-column
/// batch is a 512 KiB message: large enough to amortize the per-batch
/// handshake and serialization. Without this floor the formula collapses to a
/// single column per batch once `world_size > width / factor`, turning every
/// column into a full request/response round trip and breaking scaling with
/// `-n`.
const MIN_BATCH_COLUMNS: usize = 16;

// snippet-start:batch-size
/// Columns per `pmap` batch.
///
/// The batch size is `width / (world_size * factor)`, clamped to at least
/// [`MIN_BATCH_COLUMNS`] columns. A larger `--batch-factor` produces smaller
/// batches and therefore finer load balancing, but more per-batch protocol
/// overhead. The default of 32 measured fastest on a 4096-column image
/// across 2-8 ranks.
fn batch_size(param: &Param, size: i32, factor: usize) -> NonZeroUsize {
    // Saturating: `--batch-factor` is user input, and a wrapped product would
    // turn `width / per_rank` into a division by zero.
    let per_rank = (size as usize).max(1).saturating_mul(factor.max(1));
    NonZeroUsize::new(
        (param.width / per_rank)
            .max(MIN_BATCH_COLUMNS)
            .min(param.width),
    )
    .expect("width is positive")
}
// snippet-end:batch-size

// snippet-start:pmap-call
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
// snippet-end:pmap-call

// snippet-start:pmap-errors
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
// snippet-end:pmap-errors

/// `pub` so that the `rsmpi_rt_*` wrapper example can reuse this file.
pub fn main() {
    let universe = mandelbrot_common::mpi_api::initialize()
        .expect("MPI must not already be initialized or finalized");
    let world = universe.world();
    let rank = world.rank();
    let size = world.size();

    let param = Param::from_args();
    let factor = mandelbrot_common::arg_usize("--batch-factor", 32);
    let batch_size = batch_size(&param, size, factor);
    if rank == 0 {
        print_info("Start processing Mandelbrot set...", "");
        print_info("Backend", mandelbrot_common::describe_backend());
        print_info("MPI size", size);
        print_info("Parameters", format!("{:?}", param));
        print_info("Batch factor", factor);
        print_info("Batch size", batch_size.get());
        print_info("Computing...", "");
    }

    let total_start = Instant::now();
    let result = render(&world, &param, batch_size);
    let total_time = total_start.elapsed().as_secs_f64();

    reject_columns_outside(&world, &param, param.width / 2);

    if rank == 0 {
        let result = result.expect("rank 0 must receive the full result");
        print_info("Performance metrics", "");
        print_info("  Total time", format!("{total_time} seconds"));
        print_info("  Transfer ratio", "N/A (included in total)");
        print_info(
            "  Max iteration count",
            mandelbrot_common::max_iteration(&result),
        );

        let png_path = mandelbrot_common::output_path("mandelbrot_pmap.png");
        print_info("Saving PNG", &png_path);
        mandelbrot_common::save_png(&result, &png_path);
        print_info("Done!", "");
    } else {
        assert!(result.is_none(), "only the root receives the result");
    }
}
