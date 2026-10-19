//! Hataori: local data parallelism and an optional long-lived distributed runtime.

mod domain;
mod lanes;
#[cfg(feature = "rayon")]
mod local;
mod map;
#[cfg(any(feature = "mpi", feature = "rsmpi-rt"))]
mod mpi_check;
#[cfg(any(feature = "mpi", feature = "rsmpi-rt"))]
mod placement;
#[cfg(any(feature = "mpi", feature = "rsmpi-rt"))]
mod pmap;
#[cfg(any(feature = "mpi", feature = "rsmpi-rt"))]
mod scheduler;
#[cfg(any(feature = "mpi", feature = "rsmpi-rt"))]
mod wire;

pub use domain::{
    Domain, DomainAdmission, DomainBusy, DomainId, LocalMode, NegativeRank, Place,
    UnsupportedDomainId,
};
#[cfg(feature = "rayon")]
pub use domain::{DomainBuildError, PlacementStatus, PoolOwnership};
#[cfg(feature = "runtime")]
pub use hataori_algorithms as algorithms;
#[cfg(feature = "runtime")]
pub use hataori_algorithms::{
    pmap, pmap_at, pmap_colocated, pmap_preferred_colocated, AlgorithmError, AlgorithmRegistryExt,
    BatchActionToken, CollectiveRegistryExt, CollectiveToken, ControllerStats, PmapFuture,
    PmapOptions,
};
#[cfg(feature = "runtime")]
pub use hataori_runtime as runtime;

#[cfg(feature = "runtime")]
pub mod blocking {
    use super::{algorithms, runtime};

    pub fn pmap<A: runtime::Action>(
        runtime: &mut runtime::Runtime,
        options: algorithms::PmapOptions,
        items: Vec<A>,
        token: algorithms::BatchActionToken<A>,
    ) -> Result<(Vec<A::Output>, algorithms::ControllerStats), algorithms::AlgorithmError> {
        let future = algorithms::pmap(runtime, options, items, token)?;
        runtime.block_on(future)
    }
}
pub use lanes::{map_in_lanes, LaneControl, LaneExecutorError, Lanes, LanesError, MapInError};
#[cfg(feature = "rayon")]
pub use local::{map_in, RayonLanes, RayonLanesError};
pub use map::{map, MapError};
#[cfg(any(feature = "mpi", feature = "rsmpi-rt"))]
pub use placement::{broadcast, gather, scatter, PlacementError, PlacementErrorKind};
#[cfg(all(
    feature = "rayon",
    any(feature = "mpi", feature = "rsmpi-rt"),
    not(feature = "runtime")
))]
pub use pmap::pmap_lanes;
#[cfg(all(any(feature = "mpi", feature = "rsmpi-rt"), not(feature = "runtime")))]
pub use pmap::{pmap, PmapOptions};
#[cfg(any(feature = "mpi", feature = "rsmpi-rt"))]
pub use pmap::{PmapError, PmapErrorKind};

#[cfg(all(feature = "mpi", feature = "rsmpi-rt"))]
compile_error!("hataori: features `mpi` and `rsmpi-rt` are mutually exclusive");

#[cfg(all(feature = "mpi", not(feature = "rsmpi-rt")))]
#[allow(unused_imports)]
use mpi_upstream as mpi_backend;

#[cfg(all(feature = "rsmpi-rt", not(feature = "mpi")))]
#[allow(unused_imports)]
use mpi_runtime as mpi_backend;
