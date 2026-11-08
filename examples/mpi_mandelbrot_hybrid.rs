//! Mandelbrot set, symmetric hybrid execution: one MPI rank per node, with a
//! rank-local Rayon pool for the compute.
//!
//! This is the hybrid counterpart of `mpi_mandelbrot_pmap.rs` (sequential
//! pmap) and `rayon_mandelbrot.rs` (single-process Rayon). Each MPI rank owns
//! a Rayon pool of `--workers` threads; pmap hands batches to the root and to
//! every remote rank, and each rank computes its assigned batches inside its
//! pool while the rank's main thread keeps serving MPI messages. In a
//! homogeneous cluster this runs `--workers` compute threads per node.
//!
//! Requires `MPI_THREAD >= Funneled` (Open MPI provides it by default).
//!
//! Build and run:
//!
//! ```sh
//! cargo build --release --no-default-features --features mpi,rayon,tenferro \
//!     --example mpi_mandelbrot_hybrid
//! mpiexec -n 4 target/release/examples/mpi_mandelbrot_hybrid --workers 2
//! ```
//!
//! The same source builds against the runtime-loaded backend as
//! `rsmpi_rt_mandelbrot_hybrid` with `--features rsmpi-rt,rayon,tenferro`.
use hataori::{pmap, Domain, LocalMode, PmapOptions};
use mandelbrot_common::mpi_api::environment::{Threading, Universe};
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
/// batch is a 512 KiB message. The same floor as `mpi_mandelbrot_pmap.rs`.
const MIN_BATCH_COLUMNS: usize = 16;

// snippet-start:hybrid-init
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
// snippet-end:hybrid-init

// snippet-start:hybrid-batch-size
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
// snippet-end:hybrid-batch-size

// snippet-start:hybrid-call
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
// snippet-end:hybrid-call

// snippet-start:hybrid-domain
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
// snippet-end:hybrid-domain

/// `pub` so that the `rsmpi_rt_*` wrapper example can reuse this file.
pub fn main() {
    let universe = initialize_funneled();
    let world = universe.world();
    let rank = world.rank();
    let size = world.size();
    let workers = mandelbrot_common::arg_thread_count("--workers");
    // Validate the request before any batch-size arithmetic, which multiplies
    // the rank count, the worker count and the batch factor.
    if let Err(message) = check_worker_request(workers) {
        eprintln!("rank {rank}: {message}");
        world.abort(1);
    }

    let param = Param::from_args();
    let factor = mandelbrot_common::arg_usize("--batch-factor", 32);
    let batch_size = batch_size(&param, size, workers, factor);
    if rank == 0 {
        print_info("Start processing Mandelbrot set...", "");
        print_info("Backend", mandelbrot_common::describe_backend());
        print_info("MPI size", size);
        print_info("Workers per rank", workers);
        print_info("Parameters", format!("{:?}", param));
        print_info("Batch factor", factor);
        print_info("Batch size", batch_size.get());
        print_info("Computing...", "");
    }

    let total_start = Instant::now();
    let result = render(&world, &param, workers, batch_size, false);
    let elapsed = total_start.elapsed();
    if rank == 0 {
        print_info("Total time", format!("{:.3} s", elapsed.as_secs_f64()));
    }

    // The same call with bounded prefetch must produce the identical image.
    let prefetch_start = Instant::now();
    let with_prefetch = render(&world, &param, workers, batch_size, true);
    let prefetch_elapsed = prefetch_start.elapsed();
    if rank == 0 {
        print_info(
            "Total time (prefetch)",
            format!("{:.3} s", prefetch_elapsed.as_secs_f64()),
        );
    }

    match (result, with_prefetch) {
        (Some(tensor), Some(prefetched)) => {
            assert_eq!(
                tensor.as_slice::<i64>().expect("i64 image"),
                prefetched.as_slice::<i64>().expect("i64 image"),
                "prefetch must not change the result"
            );
            print_info(
                "Max iteration count",
                mandelbrot_common::max_iteration(&tensor),
            );
            let png_path = mandelbrot_common::output_path("mandelbrot_hybrid.png");
            mandelbrot_common::save_png(&tensor, &png_path);
            print_info("Saved", png_path);
        }
        (None, None) => {}
        _ => {
            eprintln!("rank {rank}: only the root receives the result");
            world.abort(1)
        }
    }
}
