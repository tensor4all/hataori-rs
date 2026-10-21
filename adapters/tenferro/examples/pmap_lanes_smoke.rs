//! `pmap_lanes` over caller-held tenferro engines, at world size one.
//!
//! Every rank builds its own engines and lane executor, then enters the same
//! hybrid collective. The root owns the input and the ordered output; the MPI
//! progress loop is the coordinator on the initialization thread while the
//! lanes execute each local batch. A lane runs on a thread the adapter owns, so
//! the coordinator overlaps it whatever the lane count is.
//!
//! `--lanes N` selects the number of lanes and engines (default 2) and
//! `--threads M` the workers of each engine's own pool (default 1).
//!
//! Build and run at world size one:
//!
//! ```sh
//! cargo build -p hataori-tenferro --features mpi --example pmap_lanes_smoke
//! mpiexec --oversubscribe --bind-to none -n 1 target/debug/examples/pmap_lanes_smoke --lanes 2
//! mpiexec --oversubscribe -n 1 target/debug/examples/pmap_lanes_smoke --lanes 1
//! ```
//!
//! `scripts/check-tenferro.sh` runs both shapes and asserts the lane count in
//! each one instead of relying on the launcher's core binding.
use hataori::{pmap_lanes, LocalMode, PmapOptions};
use hataori_tenferro::TenferroLanes;
use mpi_upstream::environment::Threading;
use mpi_upstream::traits::Communicator;
use std::num::NonZeroUsize;
use tenferro_cpu::{thread_pool_builder, CpuEngine};
use tenferro_tensor::{Tensor, TensorElementwise, TensorRead};

fn positive(name: &str, default: usize) -> usize {
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == name {
            return args
                .next()
                .unwrap_or_else(|| panic!("{name} needs a value"))
                .parse()
                .expect("a positive integer");
        }
    }
    default
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

    let lane_count = positive("--lanes", 2);
    let threads = positive("--threads", 1);
    let engines = (0..lane_count)
        .map(|_| {
            thread_pool_builder()
                .num_threads(threads)
                .build()
                .map(CpuEngine::new)
                .expect("a Rayon pool for one engine")
        })
        .collect();
    let mut lanes = TenferroLanes::new(engines).expect("at least one lane");
    assert_eq!(
        lanes.lane_count(),
        lane_count,
        "the adapter reports one lane per engine"
    );
    if rank == 0 {
        eprintln!(
            "world size {} lane count {} engine threads {}",
            world.size(),
            lanes.lane_count(),
            threads
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
        |item, engine| -> Result<f64, String> {
            let x = Tensor::from_vec_col_major(vec![1], vec![item]).map_err(|e| e.to_string())?;
            let sum = engine
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
    drop(universe);
}
