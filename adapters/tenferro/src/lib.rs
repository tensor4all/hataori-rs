//! Hataori [`Lanes`] over one tenferro held CPU session.
//!
//! tenferro owns the CPU pool, admission, and one owner-inheriting child session
//! per worker of that pool; Hataori owns which item runs where. Opening a
//! [`TenferroLanes`] takes tenferro's held root session, and each Hataori lane is
//! one worker of the backend's own pool plus that worker's child session.
//!
//! The adapter creates no pool, no domain, and no thread, installs no
//! thread-local selector, and never hands tenferro a borrowed pool. The held
//! session stays on the thread that opened it, so [`TenferroLanes`] is `!Send`
//! exactly like [`tenferro_cpu::CpuHeldSession`].
//!
//! Cancellation is bridged in both directions. A lane that fails, unwinds, or
//! starts after tenferro stopped the phase cancels the Hataori
//! [`LaneControl`]; a coordinator that returns, fails, or panics cancels that
//! control and tenferro's phase control. The coordinator also receives a
//! probe view of the control (`LaneControl::with_probe`) that follows
//! tenferro's own stop signal, so a phase that fails before any lane body runs
//! (a child-session entry or cleanup failure) still releases a coordinator
//! waiting on cancellation.
//!
//! # Examples
//!
//! ```
//! use hataori::map_in_lanes;
//! use hataori_tenferro::TenferroLanes;
//! use tenferro_cpu::CpuBackend;
//! use tenferro_tensor::{Tensor, TensorRead};
//!
//! let backend = CpuBackend::with_threads(2)?;
//! let mut lanes = TenferroLanes::open(&backend)?;
//! let doubled = map_in_lanes(&mut lanes, vec![1.0_f64, 2.0, 3.0], |item, lane| {
//!     let x = Tensor::from_vec_col_major(vec![1], vec![item]).map_err(|e| e.to_string())?;
//!     let sum = lane
//!         .session()
//!         .add_read(TensorRead::from_tensor(&x), TensorRead::from_tensor(&x))
//!         .map_err(|e| e.to_string())?;
//!     Ok::<f64, String>(sum.as_slice::<f64>().map_err(|e| e.to_string())?[0])
//! })?;
//! assert_eq!(doubled, vec![2.0, 4.0, 6.0]);
//! lanes.close()?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use hataori::{LaneControl, Lanes, LanesError};
use std::fmt;
use tenferro_cpu::{
    CpuBackend, CpuHeldSession, CpuHeldSessionError, CpuPhaseError, PhaseControl, PhaseLane,
    PhaseRunError,
};
use tenferro_tensor::SessionEntryError;

/// Failure to open, drive, or close a [`TenferroLanes`].
///
/// # Examples
///
/// ```
/// use hataori_tenferro::TenferroLanes;
/// use tenferro_cpu::CpuBackend;
///
/// let backend = CpuBackend::with_threads(1)?;
/// let lanes = TenferroLanes::open(&backend)?;
/// assert!(lanes.close().is_ok());
/// // The reservation is released, so a second session is admitted.
/// let reopened = TenferroLanes::open(&backend)?;
/// assert!(reopened.close().is_ok());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug)]
pub enum TenferroAdapterError {
    /// tenferro could not open the held root session.
    Open(SessionEntryError),
    /// A phase lane could not enter or clean up its child session.
    Session(SessionEntryError),
    /// tenferro refused to open a phase over the held session.
    Phase(CpuPhaseError),
    /// tenferro could not restore the caller's CPU affinity on close.
    Close(CpuHeldSessionError),
}

impl fmt::Display for TenferroAdapterError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Open(error) => {
                write!(formatter, "could not open a tenferro held session: {error}")
            }
            Self::Session(error) => {
                write!(formatter, "tenferro phase lane session failed: {error}")
            }
            Self::Phase(error) => write!(formatter, "could not open a tenferro phase: {error}"),
            Self::Close(error) => {
                write!(
                    formatter,
                    "could not close the tenferro held session: {error}"
                )
            }
        }
    }
}

impl std::error::Error for TenferroAdapterError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Open(error) | Self::Session(error) => Some(error),
            Self::Phase(error) => Some(error),
            Self::Close(error) => Some(error),
        }
    }
}

/// Hataori lanes over one tenferro held session.
///
/// Opening the executor admits the backend's root session for its whole
/// lifetime. Each [`Lanes::run`] borrows a phase from that session, so lanes are
/// workers of the backend's own pool and every lane keeps one child session warm
/// across the items it pulls. The executor holds no queue of its own: Hataori
/// pulls items from its own scheduler and hands them to the lane callback.
///
/// The held session keeps tenferro's execution admission, its execution-owner
/// marker, and the caller's narrowed CPU affinity on the opening thread, so the
/// executor is `!Send` and never migrates. Closing it reports an affinity
/// restoration failure; dropping it performs the same release best-effort.
///
/// # Examples
///
/// ```
/// use hataori::{Lanes, LaneControl};
/// use hataori_tenferro::TenferroLanes;
/// use tenferro_cpu::CpuBackend;
/// use tenferro_tensor::{Tensor, TensorRead};
///
/// let backend = CpuBackend::with_threads(2)?;
/// let mut lanes = TenferroLanes::open(&backend)?;
/// assert_eq!(lanes.lane_count(), 2);
/// let x = Tensor::from_vec_col_major(vec![1], vec![5.0_f64])?;
/// let total = std::sync::atomic::AtomicUsize::new(0);
/// lanes.run(|_index, lane, control| {
///     let _ = (control, lane.cancelled());
///     let value = lane
///         .session()
///         .add_read(TensorRead::from_tensor(&x), TensorRead::from_tensor(&x))
///         .map_err(|error| error.to_string())?;
///     assert_eq!(value.as_slice::<f64>().map_err(|error| error.to_string())?, &[10.0]);
///     total.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
///     Ok::<(), String>(())
/// })
/// .expect("lane fan-out");
/// assert_eq!(total.into_inner(), 2);
/// lanes.close()?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct TenferroLanes<'b> {
    session: CpuHeldSession<'b>,
    lane_count: usize,
}

impl fmt::Debug for TenferroLanes<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("TenferroLanes")
            .field("lane_count", &self.lane_count)
            .finish_non_exhaustive()
    }
}

impl<'b> TenferroLanes<'b> {
    /// Opens tenferro's held root session on `backend`.
    ///
    /// The caller builds and owns the backend. This creates no pool, domain, or
    /// thread, and it takes no Hataori domain: Hataori schedules items, tenferro
    /// owns admission and the workers that run them.
    ///
    /// # Errors
    ///
    /// Returns [`TenferroAdapterError::Open`] with tenferro's typed session
    /// entry error when a CPU execution is already active on this thread,
    /// another held session owns the backend's resources, or the caller's CPU
    /// mask cannot be narrowed.
    ///
    /// # Examples
    ///
    /// ```
    /// use hataori::Lanes;
    /// use hataori_tenferro::TenferroLanes;
    /// use tenferro_cpu::CpuBackend;
    ///
    /// let backend = CpuBackend::with_threads(2)?;
    /// let lanes = TenferroLanes::open(&backend)?;
    /// assert_eq!(lanes.lane_count(), 2);
    /// lanes.close()?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn open(backend: &'b CpuBackend) -> Result<Self, TenferroAdapterError> {
        let session = backend.open_session().map_err(TenferroAdapterError::Open)?;
        Ok(Self {
            session,
            // A phase drives one lane per worker of the backend's own pool, and
            // a one-worker backend drives its single lane inline.
            lane_count: backend.num_threads(),
        })
    }

    /// Closes the held session and reports a failed affinity restoration.
    ///
    /// Releasing resources, the execution-owner marker, and the admission
    /// reservation cannot fail; only restoring the caller's CPU mask can, and
    /// that failure is reported here instead of being logged.
    ///
    /// # Errors
    ///
    /// Returns [`TenferroAdapterError::Close`] with
    /// [`tenferro_cpu::CpuHeldSessionError::Restore`] when the caller's CPU
    /// affinity mask cannot be restored. Everything else is released either way.
    ///
    /// # Examples
    ///
    /// ```
    /// use hataori_tenferro::TenferroLanes;
    /// use tenferro_cpu::CpuBackend;
    ///
    /// let backend = CpuBackend::with_threads(1)?;
    /// TenferroLanes::open(&backend)?.close()?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn close(self) -> Result<(), TenferroAdapterError> {
        self.session.close().map_err(TenferroAdapterError::Close)
    }
}

/// Cancels the Hataori control if the lane unwinds.
struct CancelOnUnwind<'a, 'b>(&'a LaneControl<'b>);

impl Drop for CancelOnUnwind<'_, '_> {
    fn drop(&mut self) {
        if std::thread::panicking() {
            self.0.cancel();
        }
    }
}

/// Stops a phase's lanes when the coordinator returns, errors, or unwinds.
///
/// The coordinator's own exit is the end of the work it can feed, and the
/// Hataori contract also cancels the shared control at that point. tenferro's
/// `PhaseControl::cancel` is the signal its lanes observe, so both sides are set
/// on every exit path including unwinding.
struct CancelPhaseOnExit<'a, 'b, 'c, 'd> {
    control: &'a LaneControl<'b>,
    phase: &'c PhaseControl<'d>,
}

impl Drop for CancelPhaseOnExit<'_, '_, '_, '_> {
    fn drop(&mut self) {
        self.control.cancel();
        self.phase.cancel();
    }
}

/// Runs one Hataori lane body on one tenferro lane.
///
/// tenferro's lane stop signal and Hataori's control are bridged in both
/// directions: a lane that tenferro already cancelled marks the Hataori control,
/// and a lane body that fails or unwinds marks it before the join.
fn run_lane<E>(
    control: &LaneControl<'_>,
    lane_resource: &mut PhaseLane<'_>,
    index: usize,
    lane_body: &(impl Fn(usize, &mut PhaseLane<'_>, &LaneControl<'_>) -> Result<(), E> + Sync),
) -> Result<(), E> {
    if lane_resource.cancelled() {
        control.cancel();
    }
    let _unwind = CancelOnUnwind(control);
    match lane_body(index, lane_resource, control) {
        Ok(()) => Ok(()),
        Err(error) => {
            control.cancel();
            Err(error)
        }
    }
}

/// Maps tenferro's phase outcome onto the Hataori lanes outcome, keeping sources.
fn phase_error<E>(error: PhaseRunError<E>) -> LanesError<E, TenferroAdapterError> {
    match error {
        PhaseRunError::Lane(error) => LanesError::Lane(error),
        PhaseRunError::Coordinator(error) => LanesError::Coordinator(error),
        PhaseRunError::Session(error) => LanesError::Executor(TenferroAdapterError::Session(error)),
        PhaseRunError::SingleLane { .. } => LanesError::SingleLane,
    }
}

/// A phase that could not be opened is an executor failure, not a lane one.
fn executor_error<E>(error: CpuPhaseError) -> LanesError<E, TenferroAdapterError> {
    LanesError::Executor(TenferroAdapterError::Phase(error))
}

impl Lanes for TenferroLanes<'_> {
    type Lane<'l> = PhaseLane<'l>;
    type Error = TenferroAdapterError;

    fn lane_count(&self) -> usize {
        self.lane_count
    }

    fn run<E: Send>(
        &mut self,
        lane: impl Fn(usize, &mut PhaseLane<'_>, &LaneControl<'_>) -> Result<(), E> + Sync,
    ) -> Result<(), LanesError<E, Self::Error>> {
        let control = LaneControl::new();
        let outcome = self.session.phase(|phase| {
            phase.run(|index, lane_resource| run_lane(&control, lane_resource, index, &lane))
        });
        match outcome {
            Ok(result) => result.map_err(phase_error),
            Err(error) => Err(executor_error(error)),
        }
    }

    fn run_with_coordinator<C, E: Send>(
        &mut self,
        coordinator: impl FnOnce(&LaneControl<'_>) -> Result<C, E>,
        lane: impl Fn(usize, &mut PhaseLane<'_>, &LaneControl<'_>) -> Result<(), E> + Sync,
    ) -> Result<C, LanesError<E, Self::Error>> {
        let control = LaneControl::new();
        // The coordinator closure must own `coordinator`, while both closures
        // share the one control's cancellation flag by reference.
        let control_ref = &control;
        let outcome = self.session.phase(|phase| {
            phase.run_with_coordinator(
                move |phase_control| {
                    // tenferro can stop a phase without reaching a lane body (a
                    // child-session entry or cleanup failure). The coordinator
                    // observes that stop through a probe view of the same
                    // control, so its waits on `LaneControl` cannot outlive the
                    // phase.
                    let probe = || phase_control.cancelled();
                    let probed = control_ref.with_probe(&probe);
                    let _stop = CancelPhaseOnExit {
                        control: &probed,
                        phase: phase_control,
                    };
                    coordinator(&probed)
                },
                |index, lane_resource| run_lane(control_ref, lane_resource, index, &lane),
            )
        });
        match outcome {
            Ok(result) => result.map_err(phase_error),
            Err(error) => Err(executor_error(error)),
        }
    }
}
