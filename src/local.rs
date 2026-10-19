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
mod tests {
    use super::{in_rayon_worker_context, map_in};
    use crate::{Domain, DomainBusy, LocalMode, MapInError};
    use rayon::ThreadPoolBuilder;
    use std::error::Error;
    use std::panic::{catch_unwind, AssertUnwindSafe};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{mpsc, Arc, Once};

    fn domain(worker_count: usize) -> (Domain, Arc<rayon::ThreadPool>) {
        let pool = Arc::new(
            ThreadPoolBuilder::new()
                .num_threads(worker_count)
                .build()
                .unwrap(),
        );
        let cpus = (0..worker_count).collect();
        let domain = Domain::external(Arc::clone(&pool), cpus, worker_count).unwrap();
        (domain, pool)
    }

    #[test]
    fn rayon_worker_guard_matches_entry_contexts() {
        assert!(!in_rayon_worker_context());
        assert!(!std::thread::spawn(in_rayon_worker_context).join().unwrap());

        let pool = ThreadPoolBuilder::new().num_threads(1).build().unwrap();
        assert!(pool.install(in_rayon_worker_context));
        assert!(pool.scope(|_| in_rayon_worker_context()));
        assert!(!pool.in_place_scope(|_| in_rayon_worker_context()));

        let (sender, receiver) = mpsc::channel();
        rayon::scope(|scope| {
            assert!(in_rayon_worker_context());
            scope.spawn(|_| sender.send(in_rayon_worker_context()).unwrap());
        });
        assert!(receiver.recv().unwrap());
    }

    #[test]
    fn all_modes_succeed_in_order() {
        let (domain, _) = domain(2);
        for mode in [LocalMode::Sequential, LocalMode::Inner, LocalMode::Outer] {
            let result = map_in(&domain, mode, vec![1, 2, 3], |item| {
                Ok::<_, &'static str>(item * 2)
            })
            .unwrap();
            assert_eq!(result, vec![2, 4, 6]);
        }
    }

    #[test]
    fn sequential_and_inner_stop_at_first_error() {
        for mode in [LocalMode::Sequential, LocalMode::Inner] {
            let (domain, _) = domain(2);
            let calls = Arc::new(AtomicUsize::new(0));
            let seen = Arc::clone(&calls);
            let error = map_in(&domain, mode, vec![0, 1, 2, 3], move |item| {
                seen.fetch_add(1, Ordering::Relaxed);
                if item == 2 {
                    Err::<usize, _>("callback failed")
                } else {
                    Ok(item)
                }
            })
            .unwrap_err();
            assert!(matches!(error, MapInError::Callback(ref error) if error.index() == 2));
            assert_eq!(calls.load(Ordering::Relaxed), 3);
        }
    }

    #[test]
    fn outer_evaluates_every_input_and_reports_lowest_error() {
        let (domain, _) = domain(2);
        let calls: Arc<Vec<AtomicUsize>> = Arc::new((0..6).map(|_| AtomicUsize::new(0)).collect());
        let seen = Arc::clone(&calls);
        let error = map_in(&domain, LocalMode::Outer, (0..6).collect(), move |item| {
            seen[item].fetch_add(1, Ordering::Relaxed);
            match item {
                1 => Err::<usize, _>("first"),
                4 => Err::<usize, _>("later"),
                _ => Ok(item),
            }
        })
        .unwrap_err();
        assert!(matches!(error, MapInError::Callback(ref error) if error.index() == 1));
        assert_eq!(error.to_string(), "map failed at index 1: first");
        assert!(calls.iter().all(|count| count.load(Ordering::Relaxed) == 1));
    }

    #[test]
    fn preconditions_have_fixed_order_and_never_call_back() {
        let calls = Arc::new(AtomicUsize::new(0));
        let sequential = Domain::sequential();
        let admission = sequential.try_admit().unwrap();
        let seen = Arc::clone(&calls);
        assert!(matches!(
            map_in(&sequential, LocalMode::Sequential, vec![()], move |_| {
                seen.fetch_add(1, Ordering::Relaxed);
                Ok::<_, &'static str>(())
            }),
            Err(MapInError::MissingPool)
        ));
        drop(admission);

        let (domain, pool) = domain(1);
        let foreign_pool = ThreadPoolBuilder::new().num_threads(1).build().unwrap();
        let admission = domain.try_admit().unwrap();
        let seen = Arc::clone(&calls);
        let foreign_and_busy = foreign_pool.install(|| {
            map_in(&domain, LocalMode::Sequential, vec![()], move |_| {
                seen.fetch_add(1, Ordering::Relaxed);
                Ok::<_, &'static str>(())
            })
        });
        assert!(matches!(foreign_and_busy, Err(MapInError::ForeignPool)));
        drop(admission);

        let admission = domain.try_admit().unwrap();
        let seen = Arc::clone(&calls);
        let busy = pool.install(|| {
            map_in(&domain, LocalMode::Sequential, vec![()], move |_| {
                seen.fetch_add(1, Ordering::Relaxed);
                Ok::<_, &'static str>(())
            })
        });
        assert!(matches!(busy, Err(MapInError::DomainBusy(_))));
        drop(admission);
        assert_eq!(calls.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn foreign_custom_and_global_pools_are_rejected_but_same_pool_is_accepted() {
        let (domain, pool) = domain(1);
        let foreign_pool = ThreadPoolBuilder::new().num_threads(1).build().unwrap();
        let custom_result = foreign_pool.install(|| {
            map_in(&domain, LocalMode::Sequential, vec![1], |_| {
                Ok::<_, &'static str>(1)
            })
        });
        assert!(matches!(custom_result, Err(MapInError::ForeignPool)));

        let same_pool_result = pool.install(|| {
            map_in(&domain, LocalMode::Sequential, vec![1], |_| {
                Ok::<_, &'static str>(1)
            })
        });
        assert_eq!(same_pool_result.unwrap(), vec![1]);

        static GLOBAL: Once = Once::new();
        GLOBAL.call_once(|| {
            let _ = ThreadPoolBuilder::new().num_threads(2).build_global();
        });
        let (sender, receiver) = mpsc::channel();
        rayon::scope(|scope| {
            scope.spawn(|_| {
                sender
                    .send(map_in(&domain, LocalMode::Sequential, vec![1], |_| {
                        Ok::<_, &'static str>(1)
                    }))
                    .unwrap();
            });
        });
        assert!(matches!(
            receiver.recv().unwrap(),
            Err(MapInError::ForeignPool)
        ));
    }

    #[test]
    fn inner_callback_can_use_the_target_pool() {
        let (domain, _) = domain(2);
        let result = map_in(&domain, LocalMode::Inner, vec![()], |_| {
            let (thread_count, in_pool) = rayon::join(rayon::current_num_threads, || {
                rayon::current_thread_index().is_some()
            });
            Ok::<_, &'static str>((thread_count, in_pool))
        })
        .unwrap();
        assert_eq!(result, vec![(2, true)]);
    }

    #[test]
    fn admission_is_released_after_success_error_and_unwind() {
        let (domain, _) = domain(1);
        map_in(&domain, LocalMode::Sequential, vec![()], |_| {
            Ok::<_, &'static str>(())
        })
        .unwrap();
        assert!(domain.try_admit().is_ok());

        let error = map_in(&domain, LocalMode::Sequential, vec![()], |_| {
            Err::<(), _>("callback failed")
        });
        assert!(error.is_err());
        assert!(domain.try_admit().is_ok());

        let result = catch_unwind(AssertUnwindSafe(|| {
            map_in(
                &domain,
                LocalMode::Sequential,
                vec![()],
                |_| -> Result<(), &'static str> {
                    panic!("callback panic");
                },
            )
            .unwrap();
        }));
        assert!(result.is_err());
        assert!(domain.try_admit().is_ok());
    }

    #[test]
    fn borrowed_inputs_and_sync_stack_capture_need_no_static_bound() {
        let (domain, _) = domain(2);
        let values = [1, 2, 3];
        let bias = 10;
        let captured = &bias;
        let items = values.iter().collect::<Vec<_>>();
        let result = map_in(&domain, LocalMode::Outer, items, |value| {
            Ok::<_, &'static str>(*value + *captured)
        })
        .unwrap();
        assert_eq!(result, vec![11, 12, 13]);
    }

    #[test]
    fn error_sources_and_conversions_are_typed() {
        let busy: MapInError = DomainBusy.into();
        assert!(busy.source().is_some());
        let map_error = crate::map::map(vec![()], |_| Err::<(), _>("bad")).unwrap_err();
        let callback: MapInError = map_error.into();
        assert!(callback.source().is_some());
    }
}
