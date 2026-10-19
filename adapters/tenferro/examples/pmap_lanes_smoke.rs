//! `pmap_lanes` over one tenferro held session, at world size one.
//!
//! Every rank opens its own tenferro backend and held session, then enters the
//! same hybrid collective. The root owns the input and the ordered output; the
//! MPI progress loop is the coordinator on the initialization thread while
//! tenferro phase lanes execute each local batch, so the held session never
//! leaves the thread that opened it.
//!
//! `--workers N` selects the backend's worker count. With one worker the
//! executor has one inline lane, so the coordinator run is rejected with
//! `SingleLane` and `pmap_lanes` takes its documented fallback through
//! `Lanes::run`. With two or more workers the coordinator overlaps the lanes.
//!
//! Build and run at world size one:
//!
//! ```sh
//! cargo build -p hataori-tenferro --features mpi --example pmap_lanes_smoke
//! mpiexec --oversubscribe --bind-to none -n 1 target/debug/examples/pmap_lanes_smoke --workers 2
//! mpiexec --oversubscribe -n 1 target/debug/examples/pmap_lanes_smoke --workers 1
//! ```
//!
//! `scripts/check-tenferro.sh` runs both shapes and asserts the lane count in
//! each one instead of relying on the launcher's core binding.
use hataori::{pmap_lanes, Lanes, LanesError, LocalMode, PmapOptions};
use hataori_tenferro::TenferroLanes;
use mpi_upstream::environment::Threading;
use mpi_upstream::traits::Communicator;
use std::num::NonZeroUsize;
use tenferro_cpu::CpuBackend;
use tenferro_tensor::{Tensor, TensorRead};

fn workers() -> usize {
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--workers" {
            return args
                .next()
                .expect("--workers needs a value")
                .parse()
                .expect("--workers takes a positive integer");
        }
    }
    2
}

fn main() {
    let (universe, provided) = mpi_upstream::initialize_with_threading(Threading::Funneled)
        .expect("MPI must not already be initialized or finalized");
    assert!(
        provided >= Threading::Funneled,
        "pmap_lanes keeps MPI progress on the initialization thread"
    );
    let world = universe.world();
    let rank = world.rank();

    let workers = workers();
    let backend = CpuBackend::with_threads(workers).expect("tenferro workers");
    let mut lanes = TenferroLanes::open(&backend).expect("held tenferro session");
    assert_eq!(
        lanes.lane_count(),
        workers,
        "the adapter reports the backend's worker count as its lane count"
    );
    if rank == 0 {
        eprintln!(
            "world size {} lane count {}",
            world.size(),
            lanes.lane_count()
        );
    }

    if workers == 1 {
        let rejected = lanes
            .run_with_coordinator(
                |_control| Ok::<(), String>(()),
                |_index, _lane, _control| Ok::<(), String>(()),
            )
            .expect_err("a one-lane phase cannot overlap a coordinator");
        assert!(
            matches!(rejected, LanesError::SingleLane),
            "one worker must report the fallback path: {rejected}"
        );
    }

    let items = (rank == 0).then(|| vec![1.0_f64, 2.0, 3.0, 4.0, 5.0]);
    let doubled = pmap_lanes(
        &world,
        &mut lanes,
        PmapOptions {
            root: 0,
            batch_size: NonZeroUsize::new(2).expect("positive batch size"),
            local_mode: LocalMode::Outer,
            prefetch: false,
        },
        items,
        |item, lane| -> Result<f64, String> {
            let x = Tensor::from_vec_col_major(vec![1], vec![item]).map_err(|e| e.to_string())?;
            let sum = lane
                .session()
                .add_read(TensorRead::from_tensor(&x), TensorRead::from_tensor(&x))
                .map_err(|e| e.to_string())?;
            Ok(sum.as_slice::<f64>().map_err(|e| e.to_string())?[0])
        },
    )
    .expect("pmap_lanes must succeed");

    if rank == 0 {
        assert_eq!(doubled, Some(vec![2.0, 4.0, 6.0, 8.0, 10.0]));
    } else {
        assert_eq!(doubled, None);
    }
    lanes.close().expect("close the held tenferro session");
    drop(universe);
}
