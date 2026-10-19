//! Execution lanes and the coordinator/lane fan-out contract.
//!
//! A [`Lanes`] executor owns one execution resource per worker and lends it to a
//! local fan-out. [`map_in_lanes`] spreads independent items over those lanes;
//! the hybrid [`crate::pmap`] protocol runs its MPI progress loop on the calling
//! thread as a coordinator while local batches execute on the lanes.
//!
//! [`crate::RayonLanes`] is the Rayon implementation over an admitted
//! [`crate::Domain`] pool. A backend adapter supplies its own implementation so
//! that Hataori stays the scheduler while the adapter keeps ownership of its
//! pool and lane resources.

#[cfg(all(feature = "rayon", any(feature = "mpi", feature = "rsmpi-rt")))]
pub(crate) mod batch;

use crate::domain::DomainBusy;
use crate::map::{map, truncate_message, MapError};
use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Mutex, MutexGuard, PoisonError};

/// A cancellation capability shared by a coordinator and its lanes.
///
/// The executor creates one control per run and lends it to the coordinator and
/// to every lane, so cancellation is visible in both directions: a coordinator
/// that stops feeding work cancels the lanes, and an executor that observes a
/// lane failure cancels the coordinator. Cancellation is a request: already
/// running callback code is not stopped.
///
/// An adapter bridges its own cancellation state into the control it observes
/// and must treat [`LaneControl::cancelled`] as a request to stop feeding work.
/// A backend whose own executor can stop a run without reaching a lane callback
/// (a failed child-session entry, for example) lends the side that waits on
/// cancellation a [`LaneControl::with_probe`] view of the same control, so that
/// wait also observes the backend's stop signal.
///
/// # Example
///
/// ```
/// use hataori::LaneControl;
///
/// let control = LaneControl::new();
/// assert!(!control.cancelled());
/// control.cancel();
/// assert!(control.cancelled());
/// ```
///
/// A probe view shares the same cancellation flag and adds an external signal:
///
/// ```
/// use hataori::LaneControl;
/// use std::sync::atomic::{AtomicBool, Ordering};
///
/// let control = LaneControl::new();
/// let backend_stopped = AtomicBool::new(false);
/// let probe = || backend_stopped.load(Ordering::Acquire);
/// let probed = control.with_probe(&probe);
/// assert!(!probed.cancelled());
/// backend_stopped.store(true, Ordering::Release);
/// assert!(probed.cancelled());
/// assert!(!control.cancelled(), "the probe is not an own-flag cancellation");
/// probed.cancel();
/// assert!(control.cancelled(), "views share one cancellation flag");
/// ```
pub struct LaneControl<'a> {
    flag: LaneFlag<'a>,
    probe: Option<&'a (dyn Fn() -> bool + Sync)>,
}

/// Where a control's cancellation flag lives.
///
/// A control created by an executor owns its flag; a probe view borrows the flag
/// of the control it was created from, so `cancel` on either view is visible on
/// both.
#[derive(Debug)]
enum LaneFlag<'a> {
    Owned(AtomicBool),
    Borrowed(&'a AtomicBool),
}

impl LaneFlag<'_> {
    fn flag(&self) -> &AtomicBool {
        match self {
            Self::Owned(flag) => flag,
            Self::Borrowed(flag) => flag,
        }
    }
}

impl fmt::Debug for LaneControl<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("LaneControl")
            .field("cancelled", &self.flag.flag().load(Ordering::Acquire))
            .field("probed", &self.probe.is_some())
            .finish()
    }
}

impl LaneControl<'static> {
    /// Creates an un-cancelled control that owns its cancellation flag.
    pub const fn new() -> Self {
        Self {
            flag: LaneFlag::Owned(AtomicBool::new(false)),
            probe: None,
        }
    }
}

impl Default for LaneControl<'static> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a> LaneControl<'a> {
    /// Returns a view of this control that also reports `probe` as cancelled.
    ///
    /// The view shares this control's cancellation flag, so a lane and a
    /// coordinator cannot diverge on an ordinary `cancel`; the probe only adds a
    /// stop signal the executor owns. [`LaneControl::cancelled`] is evaluated as
    /// own flag first and then the probe, so a probe must be cheap and
    /// non-blocking.
    pub fn with_probe(&'a self, probe: &'a (dyn Fn() -> bool + Sync)) -> Self {
        Self {
            flag: LaneFlag::Borrowed(self.flag.flag()),
            probe: Some(probe),
        }
    }

    /// Returns whether cancellation has been requested.
    pub fn cancelled(&self) -> bool {
        self.flag.flag().load(Ordering::Acquire) || self.probe.is_some_and(|probe| probe())
    }

    /// Requests cancellation. This never blocks and never stops running code.
    pub fn cancel(&self) {
        self.flag.flag().store(true, Ordering::Release);
    }
}

/// A per-worker execution lanes executor plus a caller-thread coordinator.
///
/// The trait is implemented by execution backends; it has no `Send`, `Sync`, or
/// `'static` supertrait, so an executor, its lane resources, a coordinator
/// capture, and a coordinator result may all stay on one thread. Lane closures
/// are shared across the lanes and therefore require `Sync`.
///
/// Both methods join every lane before returning, and resume a lane panic only
/// after that join. Lanes observe cancellation through [`LaneControl`] and must
/// stop when it is cancelled; the executor cancels the control after the
/// coordinator returns. An executor whose backend can stop a run without
/// reaching a lane callback lends the side that waits on cancellation a
/// [`LaneControl::with_probe`] view, so that wait also observes the backend's
/// own stop signal.
///
/// # Example
///
/// ```
/// # #[cfg(feature = "rayon")]
/// # {
/// use hataori::{map_in_lanes, Domain, LocalMode, RayonLanes};
/// use std::sync::Arc;
///
/// let pool = Arc::new(rayon::ThreadPoolBuilder::new().num_threads(2).build().unwrap());
/// let domain = Domain::external(Arc::clone(&pool), vec![0, 1], 2).unwrap();
/// let mut lanes = RayonLanes::new(&domain).unwrap();
/// let doubled = map_in_lanes(&mut lanes, vec![1, 2, 3], |item, ()| {
///     Ok::<_, std::convert::Infallible>(item * 2)
/// })
/// .unwrap();
/// assert_eq!(doubled, vec![2, 4, 6]);
/// # }
/// ```
pub trait Lanes {
    /// The execution resource lent to one lane callback.
    type Lane<'l>;

    /// A lane entry, join, or cleanup failure.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Returns the number of lanes the executor runs.
    fn lane_count(&self) -> usize;

    /// Runs `lane` on every lane, joins all of them, then resumes a lane panic.
    ///
    /// # Errors
    ///
    /// A lane error or panic cancels the shared control immediately, so
    /// cooperatively waiting lanes are released before the join. Returns
    /// [`LanesError::Lane`] when a lane callback returns an error,
    /// [`LanesError::Executor`] when the executor itself cannot run or join a
    /// lane, or [`LanesError::SingleLane`] when the executor has no lane to run.
    /// A lane panic is resumed after the join and takes precedence over a lane
    /// error.
    fn run<E: Send>(
        &mut self,
        lane: impl Fn(usize, &mut Self::Lane<'_>, &LaneControl<'_>) -> Result<(), E> + Sync,
    ) -> Result<(), LanesError<E, Self::Error>>;

    /// Runs `coordinator` on the calling thread concurrently with every lane.
    ///
    /// The coordinator is not a lane: it gets no lane resource. Lanes and the
    /// coordinator communicate only through caller-owned synchronization. When
    /// the coordinator returns, errors, or panics, the control is cancelled and
    /// every lane is joined before this method returns.
    ///
    /// An executor that cannot overlap a coordinator with its lanes returns
    /// [`LanesError::SingleLane`] without running either callback, so the caller
    /// can fall back to a non-concurrent path explicitly. An executor that
    /// rejects being driven from one of its own workers returns
    /// [`LanesError::Reentrant`] before running either callback.
    ///
    /// # Errors
    ///
    /// Returns [`LanesError::SingleLane`] for a non-concurrent executor,
    /// [`LanesError::Reentrant`] for entry from one of the executor's own
    /// workers, [`LanesError::Coordinator`] when the coordinator returns an
    /// error, [`LanesError::Lane`] when a lane callback returns an error, or
    /// [`LanesError::Executor`] when the executor cannot run or join a lane. A
    /// lane panic is resumed after the join and takes precedence over every
    /// ordinary error.
    fn run_with_coordinator<C, E: Send>(
        &mut self,
        coordinator: impl FnOnce(&LaneControl<'_>) -> Result<C, E>,
        lane: impl Fn(usize, &mut Self::Lane<'_>, &LaneControl<'_>) -> Result<(), E> + Sync,
    ) -> Result<C, LanesError<E, Self::Error>>;
}

/// Failure from a [`Lanes`] run.
#[derive(Debug)]
pub enum LanesError<E, LE> {
    /// The executor cannot overlap a coordinator with its lanes.
    SingleLane,
    /// A coordinator run was requested from one of the executor's own workers.
    Reentrant,
    /// The coordinator callback returned an error.
    Coordinator(E),
    /// A lane callback returned an error.
    Lane(E),
    /// The executor could not run, join, or clean up a lane.
    Executor(LE),
}

impl<E: fmt::Display, LE: fmt::Display> fmt::Display for LanesError<E, LE> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SingleLane => formatter.write_str("executor has no lane for a concurrent run"),
            Self::Reentrant => {
                formatter.write_str("executor cannot run a coordinator from one of its workers")
            }
            Self::Coordinator(error) => write!(formatter, "coordinator failed: {error}"),
            Self::Lane(error) => write!(formatter, "lane failed: {error}"),
            Self::Executor(error) => write!(formatter, "lane executor failed: {error}"),
        }
    }
}

impl<E: std::error::Error + 'static, LE: std::error::Error + 'static> std::error::Error
    for LanesError<E, LE>
{
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::SingleLane | Self::Reentrant => None,
            Self::Coordinator(error) | Self::Lane(error) => Some(error),
            Self::Executor(error) => Some(error),
        }
    }
}

/// A lane executor failure during a local fan-out.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct LaneExecutorError {
    message: String,
}

impl LaneExecutorError {
    pub(crate) fn new(error: &(impl fmt::Display + ?Sized)) -> Self {
        Self {
            message: truncate_message(error.to_string()),
        }
    }

    pub(crate) fn from_message(message: impl fmt::Display) -> Self {
        Self::new(&message)
    }

    /// Returns the bounded executor failure message.
    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for LaneExecutorError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for LaneExecutorError {}

/// Failure from a local fan-out over [`Lanes`].
#[derive(Debug)]
pub enum MapInError {
    /// The domain has no explicit Rayon pool.
    MissingPool,
    /// The call originated on a different Rayon pool.
    ForeignPool,
    /// Another operation already owns the domain admission slot.
    DomainBusy(DomainBusy),
    /// A callback returned an error.
    Callback(MapError),
    /// The lane executor failed to enter or clean up a lane.
    Executor(LaneExecutorError),
}

impl fmt::Display for MapInError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingPool => formatter.write_str("domain has no Rayon pool"),
            Self::ForeignPool => formatter.write_str("map_in called from a foreign Rayon pool"),
            Self::DomainBusy(error) => error.fmt(formatter),
            Self::Callback(error) => error.fmt(formatter),
            Self::Executor(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for MapInError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::DomainBusy(error) => Some(error),
            Self::Callback(error) => Some(error),
            Self::Executor(error) => Some(error),
            Self::MissingPool | Self::ForeignPool => None,
        }
    }
}

impl From<DomainBusy> for MapInError {
    fn from(error: DomainBusy) -> Self {
        Self::DomainBusy(error)
    }
}

impl From<MapError> for MapInError {
    fn from(error: MapError) -> Self {
        Self::Callback(error)
    }
}

/// State shared by the lanes of one [`map_in_lanes`] fan-out.
struct FanOut<T, U, E> {
    items: Vec<Option<T>>,
    results: Vec<Option<Result<U, E>>>,
}

impl<T, U, E> FanOut<T, U, E> {
    fn new(items: Vec<T>) -> Self {
        let items: Vec<Option<T>> = items.into_iter().map(Some).collect();
        let results = (0..items.len()).map(|_| None).collect();
        Self { items, results }
    }
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Maps `items` through a [`Lanes`] executor with `LocalMode::Outer` semantics.
///
/// Every input is evaluated exactly once, results keep input order, and the
/// lowest failing input index is reported after every item has finished.
/// Lane callbacks pull the next input index from one shared counter and write
/// their result into the per-index slot, so no Rayon job and no extra heap
/// allocation is created per item. A callback panic resumes only after every
/// lane has been joined.
///
/// # Errors
///
/// Returns [`MapInError::Callback`] with the lowest failing input index, or
/// [`MapInError::Executor`] when the lane executor itself cannot run or join a
/// lane.
///
/// # Example
///
/// ```
/// # #[cfg(feature = "rayon")]
/// # {
/// use hataori::{map_in_lanes, Domain, RayonLanes};
/// use std::sync::Arc;
///
/// let pool = Arc::new(rayon::ThreadPoolBuilder::new().num_threads(2).build().unwrap());
/// let domain = Domain::external(Arc::clone(&pool), vec![0, 1], 2).unwrap();
/// let mut lanes = RayonLanes::new(&domain).unwrap();
/// let tripled = map_in_lanes(&mut lanes, vec![2, 4], |item, ()| {
///     Ok::<_, std::convert::Infallible>(item * 3)
/// })
/// .unwrap();
/// assert_eq!(tripled, vec![6, 12]);
/// # }
/// ```
pub fn map_in_lanes<L, T, U, E, F>(lanes: &mut L, items: Vec<T>, f: F) -> Result<Vec<U>, MapInError>
where
    L: Lanes,
    T: Send,
    U: Send,
    E: fmt::Display + Send,
    F: Fn(T, &mut L::Lane<'_>) -> Result<U, E> + Sync,
{
    let fan_out = Mutex::new(FanOut::new(items));
    let length = lock(&fan_out).items.len();
    let next = AtomicUsize::new(0);
    let outcome = lanes.run(|_lane_index, lane, control| {
        loop {
            if control.cancelled() {
                break;
            }
            let index = next.fetch_add(1, Ordering::Relaxed);
            if index >= length {
                break;
            }
            let item = {
                let mut state = lock(&fan_out);
                state.items[index]
                    .take()
                    .expect("one lane claims each input index once")
            };
            // The slot guard is dropped before the callback runs, so a callback
            // panic cannot poison the fan-out mutex.
            let result = f(item, lane);
            lock(&fan_out).results[index] = Some(result);
        }
        Ok::<(), std::convert::Infallible>(())
    });
    match outcome {
        Ok(()) => {}
        Err(LanesError::Executor(error)) => {
            return Err(MapInError::Executor(LaneExecutorError::new(&error)))
        }
        Err(LanesError::SingleLane) | Err(LanesError::Reentrant) => {
            return Err(MapInError::Executor(LaneExecutorError::from_message(
                "executor has no lane for a lane-only run",
            )))
        }
        Err(LanesError::Lane(never)) => match never {},
        Err(LanesError::Coordinator(never)) => match never {},
    }

    let mut results = Vec::with_capacity(length);
    for result in std::mem::take(&mut lock(&fan_out).results) {
        match result {
            Some(result) => results.push(result),
            None => {
                return Err(MapInError::Executor(LaneExecutorError::from_message(
                    "lanes stopped before every input was evaluated",
                )))
            }
        }
    }
    map(results, |result| result).map_err(MapInError::Callback)
}

#[cfg(test)]
mod tests;
