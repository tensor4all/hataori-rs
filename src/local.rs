use crate::domain::{Domain, DomainAdmission, DomainBusy, LocalMode};
use crate::lanes::{map_in_lanes, LaneControl, Lanes, LanesError, MapInError};
use crate::map::{map, MapError};
use std::any::Any;
use std::convert::Infallible;
use std::fmt;
use std::panic::AssertUnwindSafe;
use std::sync::{Mutex, PoisonError};

pub(crate) fn in_rayon_worker_context() -> bool {
    rayon::current_thread_index().is_some()
}

/// Runs `items` one at a time inside `pool` without stopping early.
///
/// [`LocalMode::Sequential`] and [`LocalMode::Inner`] share this path: both run
/// one callback at a time on a pool worker, and `Inner` keeps nested same-pool
/// work available because the callback already runs there.
fn run_sequential_in_pool<T, U, E, F>(
    pool: &rayon::ThreadPool,
    items: Vec<T>,
    f: F,
) -> Result<Vec<U>, MapError>
where
    T: Send,
    U: Send,
    E: fmt::Display + Send,
    F: Fn(T) -> Result<U, E> + Send + Sync,
{
    pool.install(|| map(items, f))
}

/// Why a [`RayonLanes`] executor could not be created for a domain.
#[derive(Debug)]
pub enum RayonLanesError {
    /// The domain has no explicit Rayon pool.
    MissingPool,
    /// The call originated on a different Rayon pool.
    ForeignPool,
    /// Another operation already owns the domain admission slot.
    DomainBusy(DomainBusy),
}

impl fmt::Display for RayonLanesError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingPool => formatter.write_str("domain has no Rayon pool"),
            Self::ForeignPool => formatter.write_str("lanes entered from a foreign Rayon pool"),
            Self::DomainBusy(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for RayonLanesError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::DomainBusy(error) => Some(error),
            Self::MissingPool | Self::ForeignPool => None,
        }
    }
}

impl From<RayonLanesError> for MapInError {
    fn from(error: RayonLanesError) -> Self {
        match error {
            RayonLanesError::MissingPool => Self::MissingPool,
            RayonLanesError::ForeignPool => Self::ForeignPool,
            RayonLanesError::DomainBusy(error) => Self::DomainBusy(error),
        }
    }
}

/// The [`Lanes`] implementation over a domain's Rayon pool.
///
/// Creating one admits the domain, so the executor holds the domain's one coarse
/// operation slot for its whole lifetime. Each lane is one Rayon worker; the
/// lane resource is `()` because the callback already runs on that worker.
///
/// Entry from a Rayon worker that does not belong to the domain pool is rejected,
/// and so is entry while another operation owns the domain.
///
/// # Example
///
/// ```
/// # #[cfg(feature = "rayon")]
/// # {
/// use hataori::{Domain, Lanes, RayonLanes};
/// use std::sync::Arc;
///
/// let pool = Arc::new(rayon::ThreadPoolBuilder::new().num_threads(2).build().unwrap());
/// let domain = Domain::external(Arc::clone(&pool), vec![0, 1], 2).unwrap();
/// let mut lanes = RayonLanes::new(&domain).unwrap();
/// assert_eq!(lanes.lane_count(), 2);
/// let visited = std::sync::atomic::AtomicUsize::new(0);
/// lanes
///     .run(|_index, (), _control| {
///         visited.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
///         Ok::<_, std::convert::Infallible>(())
///     })
///     .unwrap();
/// assert_eq!(visited.into_inner(), 2);
/// # }
/// ```
pub struct RayonLanes<'d> {
    pool: &'d rayon::ThreadPool,
    worker_count: usize,
    // Held for the executor's whole lifetime; dropping it releases admission.
    _admission: DomainAdmission<'d>,
}

impl<'d> RayonLanes<'d> {
    /// Admits `domain` and builds a lane executor over its pool.
    ///
    /// # Errors
    ///
    /// Returns [`RayonLanesError::MissingPool`] for a domain without a pool,
    /// [`RayonLanesError::ForeignPool`] when called from a different Rayon pool,
    /// and [`RayonLanesError::DomainBusy`] when the domain is already admitted.
    pub fn new(domain: &'d Domain) -> Result<Self, RayonLanesError> {
        let pool = domain.rayon_pool().ok_or(RayonLanesError::MissingPool)?;
        if in_rayon_worker_context() && pool.current_thread_index().is_none() {
            return Err(RayonLanesError::ForeignPool);
        }
        let admission = domain.try_admit().map_err(RayonLanesError::DomainBusy)?;
        Ok(Self {
            pool,
            worker_count: domain.worker_count(),
            _admission: admission,
        })
    }
}

/// What one lane reported back to the executor.
enum LaneReport<E> {
    Error(E),
    Panic(Box<dyn Any + Send + 'static>),
}

fn record_lane_report<E>(report: &Mutex<Option<LaneReport<E>>>, value: LaneReport<E>) {
    let mut report = report.lock().unwrap_or_else(PoisonError::into_inner);
    // A panic payload always wins over an earlier ordinary lane error.
    let replace = matches!(
        (&*report, &value),
        (None, _) | (Some(LaneReport::Error(_)), LaneReport::Panic(_))
    );
    if replace {
        *report = Some(value);
    }
}

impl Lanes for RayonLanes<'_> {
    type Lane<'l> = ();
    type Error = Infallible;

    fn lane_count(&self) -> usize {
        self.worker_count
    }

    fn run<E: Send>(
        &mut self,
        lane: impl Fn(usize, &mut (), &LaneControl<'_>) -> Result<(), E> + Sync,
    ) -> Result<(), LanesError<E, Self::Error>> {
        let control = LaneControl::new();
        let report: Mutex<Option<LaneReport<E>>> = Mutex::new(None);
        // `broadcast` returns only after every worker finished its callback, and
        // a failing lane cancels the shared control so peers waiting on it stop.
        self.pool.broadcast(|context| {
            let outcome = std::panic::catch_unwind(AssertUnwindSafe(|| {
                lane(context.index(), &mut (), &control)
            }));
            match outcome {
                Ok(Ok(())) => {}
                Ok(Err(error)) => {
                    control.cancel();
                    record_lane_report(&report, LaneReport::Error(error));
                }
                Err(payload) => {
                    control.cancel();
                    record_lane_report(&report, LaneReport::Panic(payload));
                }
            }
        });
        match report.into_inner().unwrap_or_else(PoisonError::into_inner) {
            Some(LaneReport::Panic(payload)) => std::panic::resume_unwind(payload),
            Some(LaneReport::Error(error)) => Err(LanesError::Lane(error)),
            None => Ok(()),
        }
    }

    fn run_with_coordinator<C, E: Send>(
        &mut self,
        coordinator: impl FnOnce(&LaneControl<'_>) -> Result<C, E>,
        lane: impl Fn(usize, &mut (), &LaneControl<'_>) -> Result<(), E> + Sync,
    ) -> Result<C, LanesError<E, Self::Error>> {
        if self.pool.current_thread_index().is_some() {
            // A coordinator on one of this pool's own workers cannot make
            // progress while its lanes are joined.
            return Err(LanesError::Reentrant);
        }
        let control = LaneControl::new();
        let report: Mutex<Option<LaneReport<E>>> = Mutex::new(None);
        // The calling thread stays outside the pool, so even a one-worker pool
        // runs its lane concurrently with the coordinator.
        let coordinator_outcome = self.pool.in_place_scope(|scope| {
            let lane = &lane;
            let control = &control;
            let report = &report;
            scope.spawn_broadcast(move |_scope, context| {
                let outcome = std::panic::catch_unwind(AssertUnwindSafe(|| {
                    lane(context.index(), &mut (), control)
                }));
                match outcome {
                    Ok(Ok(())) => {}
                    Ok(Err(error)) => {
                        control.cancel();
                        record_lane_report(report, LaneReport::Error(error));
                    }
                    Err(payload) => {
                        control.cancel();
                        record_lane_report(report, LaneReport::Panic(payload));
                    }
                }
            });
            let outcome = std::panic::catch_unwind(AssertUnwindSafe(|| coordinator(control)));
            // Lanes stop when the coordinator stops feeding them.
            control.cancel();
            outcome
        });

        // `in_place_scope` joined every lane before returning.
        let report = report.into_inner().unwrap_or_else(PoisonError::into_inner);
        match coordinator_outcome {
            Err(payload) => std::panic::resume_unwind(payload),
            Ok(result) => match report {
                Some(LaneReport::Panic(payload)) => std::panic::resume_unwind(payload),
                Some(LaneReport::Error(error)) => Err(LanesError::Lane(error)),
                None => result.map_err(LanesError::Coordinator),
            },
        }
    }
}

/// Maps `items` through an explicit Rayon domain.
///
/// The target domain must have a pool, the call must not originate in another
/// Rayon pool, and the domain must not already be admitted. These preconditions
/// are checked in that order, before `f` is called. The callback and input,
/// output, and error types must satisfy `Send`/`Sync` as required by Rayon, but
/// none needs to be `'static`.
///
/// [`LocalMode::Sequential`] and [`LocalMode::Inner`] run callbacks one at a
/// time in the target pool and stop at the first callback error. `Inner` leaves
/// nested same-pool Rayon work available. [`LocalMode::Outer`] fans out through
/// [`RayonLanes`] and [`map_in_lanes`]: every input is evaluated exactly once,
/// result order is preserved, and the lowest callback-error index is reported
/// only after all callbacks have finished.
///
/// # Errors
///
/// Returns [`MapInError::MissingPool`], [`MapInError::ForeignPool`], or
/// [`MapInError::DomainBusy`] for the preconditions above,
/// [`MapInError::Callback`] with the lowest failing input index when a callback
/// fails, and [`MapInError::Executor`] when the lane executor cannot run.
///
/// # Example
///
/// ```
/// # #[cfg(feature = "rayon")]
/// # {
/// use hataori::{map_in, Domain, LocalMode};
/// use std::sync::Arc;
/// use rayon::ThreadPoolBuilder;
///
/// let pool = Arc::new(ThreadPoolBuilder::new().num_threads(2).build().unwrap());
/// let domain = Domain::external(Arc::clone(&pool), vec![0, 1], 2).unwrap();
/// let result = map_in(&domain, LocalMode::Outer, vec![1, 2, 3], |item| {
///     Ok::<_, std::convert::Infallible>(item * 2)
/// }).unwrap();
/// assert_eq!(result, vec![2, 4, 6]);
/// # }
/// ```
pub fn map_in<T, U, E, F>(
    domain: &Domain,
    mode: LocalMode,
    items: Vec<T>,
    f: F,
) -> Result<Vec<U>, MapInError>
where
    T: Send,
    U: Send,
    E: fmt::Display + Send,
    F: Fn(T) -> Result<U, E> + Send + Sync,
{
    let mut lanes = RayonLanes::new(domain)?;
    match mode {
        LocalMode::Outer => map_in_lanes(&mut lanes, items, |item, ()| f(item)),
        LocalMode::Sequential | LocalMode::Inner => {
            let pool = domain.rayon_pool().ok_or(MapInError::MissingPool)?;
            run_sequential_in_pool(pool, items, f).map_err(MapInError::Callback)
        }
    }
}

#[cfg(test)]
mod tests;
