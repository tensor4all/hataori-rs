//! Hataori [`Lanes`] over the caller's tenferro CPU [`CpuEngine`]s.
//!
//! Hataori owns which item runs where and tenferro owns nothing but the engine
//! a lane runs on. The caller builds one engine per lane — each with its own
//! Rayon pool, sized by the caller — and this adapter runs the lanes on threads
//! of its own. A lane thread is never a worker of any lane's engine pool, so
//! tenferro's "never called from one of its engine's own workers" rule holds by
//! construction rather than by a runtime check.
//!
//! The adapter creates no pool, no session, and no thread-local state. An
//! engine is an ordinary value: it is lent to one lane for the length of one
//! run and returned to the caller's vector afterwards, so a run costs one
//! thread per lane and nothing is held between runs.
//!
//! # Examples
//!
//! Two lanes, each with its own two-thread engine, doubling three items:
//!
//! ```
//! use hataori::{map_in_lanes, Lanes};
//! use hataori_tenferro::TenferroLanes;
//! use tenferro_cpu::{CpuEngine, Pool};
//! use tenferro_tensor::{Tensor, TensorElementwise, TensorRead};
//!
//! let engines = (0..2)
//!     .map(|_| Pool::builder().threads(2).build().map(CpuEngine::new))
//!     .collect::<Result<Vec<_>, _>>()?;
//! let mut lanes = TenferroLanes::new(engines)?;
//! assert_eq!(lanes.lane_count(), 2);
//! let doubled = map_in_lanes(&mut lanes, vec![1.0_f64, 2.0, 3.0], |item, engine| {
//!     let x = Tensor::from_vec_col_major(vec![1], vec![item]).map_err(|e| e.to_string())?;
//!     let sum = engine
//!         .add_read(TensorRead::from_tensor(&x), TensorRead::from_tensor(&x))
//!         .map_err(|e| e.to_string())?;
//!     Ok::<f64, String>(sum.as_slice::<f64>().map_err(|e| e.to_string())?[0])
//! })?;
//! assert_eq!(doubled, vec![2.0, 4.0, 6.0]);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use hataori::{LaneControl, Lanes, LanesError};
use std::fmt;
use std::thread::{self, ScopedJoinHandle};
use tenferro_cpu::CpuEngine;

/// A lane body runs provider code on its own thread, so it gets the same 16 MiB
/// stack `strided-exec`'s pool default gives its workers.
const LANE_STACK_BYTES: usize = 16 << 20;

/// The lane executor was built without an engine.
///
/// # Examples
///
/// ```
/// use hataori_tenferro::{NoEngine, TenferroLanes};
///
/// assert_eq!(TenferroLanes::new(Vec::new()).unwrap_err(), NoEngine);
/// ```
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct NoEngine;

impl fmt::Display for NoEngine {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a lane needs an engine and none was given")
    }
}

impl std::error::Error for NoEngine {}

/// A lane thread could not be started.
///
/// # Examples
///
/// ```
/// use hataori_tenferro::LaneThreadError;
///
/// let error = LaneThreadError::from(std::io::Error::other("no thread"));
/// assert!(error.to_string().contains("lane thread"));
/// ```
#[derive(Debug)]
pub struct LaneThreadError {
    source: std::io::Error,
}

impl From<std::io::Error> for LaneThreadError {
    fn from(source: std::io::Error) -> Self {
        Self { source }
    }
}

impl fmt::Display for LaneThreadError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "could not start a lane thread: {}", self.source)
    }
}

impl std::error::Error for LaneThreadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.source)
    }
}

/// Hataori lanes over the caller's tenferro CPU engines.
///
/// One lane per engine, in the order the engines were given. The executor holds
/// no lane resource of its own: [`Lanes::run`] lends engine `i` to lane `i` and
/// takes it back when every lane has been joined.
///
/// # Examples
///
/// ```
/// use hataori::{Lanes, LaneControl};
/// use hataori_tenferro::TenferroLanes;
/// use std::sync::atomic::{AtomicUsize, Ordering};
/// use tenferro_cpu::{CpuEngine, Pool};
///
/// let engine = CpuEngine::new(Pool::builder().threads(1).build()?);
/// let mut lanes = TenferroLanes::new(vec![engine])?;
/// let entered = AtomicUsize::new(0);
/// lanes.run(|index, engine, control| {
///     let _ = (index, engine.num_threads(), control.cancelled());
///     entered.fetch_add(1, Ordering::AcqRel);
///     Ok::<(), std::convert::Infallible>(())
/// })?;
/// assert_eq!(entered.into_inner(), 1);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct TenferroLanes {
    engines: Vec<CpuEngine>,
}

impl fmt::Debug for TenferroLanes {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("TenferroLanes")
            .field("lane_count", &self.engines.len())
            .field("engines", &self.engines)
            .finish_non_exhaustive()
    }
}

impl TenferroLanes {
    /// Makes one lane per engine, keeping the engines' order.
    ///
    /// The caller owns the pools: build each engine with
    /// [`tenferro_cpu::Pool`] and the worker count it wants. The
    /// adapter never resizes, replaces, or drops an engine's pool, and it adds
    /// no compute threads of its own beyond the lane threads a run needs.
    ///
    /// # Errors
    ///
    /// Returns [`NoEngine`] when `engines` is empty: a lane executor without an
    /// engine could only report that no item was ever evaluated.
    ///
    /// # Examples
    ///
    /// ```
    /// use hataori_tenferro::TenferroLanes;
    /// use tenferro_cpu::CpuEngine;
    ///
    /// let lanes = TenferroLanes::new(vec![CpuEngine::serial(), CpuEngine::serial()])?;
    /// assert_eq!(lanes.lane_count(), 2);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn new(engines: Vec<CpuEngine>) -> Result<Self, NoEngine> {
        if engines.is_empty() {
            return Err(NoEngine);
        }
        Ok(Self { engines })
    }

    /// Returns the number of lanes, one per engine.
    ///
    /// # Examples
    ///
    /// ```
    /// use hataori_tenferro::TenferroLanes;
    /// use tenferro_cpu::CpuEngine;
    ///
    /// assert_eq!(TenferroLanes::new(vec![CpuEngine::serial()])?.lane_count(), 1);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn lane_count(&self) -> usize {
        self.engines.len()
    }

    /// Starts one lane thread per engine, in engine order.
    fn start_lanes<'scope, 'env, E: Send + 'env>(
        &'env mut self,
        scope: &'scope thread::Scope<'scope, 'env>,
        control: &'env LaneControl<'env>,
        body: &'env LaneBody<'env, E>,
    ) -> Result<Vec<ScopedJoinHandle<'scope, Result<(), E>>>, LaneThreadError> {
        let mut handles = Vec::with_capacity(self.engines.len());
        for (index, engine) in self.engines.iter_mut().enumerate() {
            handles.push(
                thread::Builder::new()
                    .stack_size(LANE_STACK_BYTES)
                    .spawn_scoped(scope, move || run_lane(control, engine, index, body))?,
            );
        }
        Ok(handles)
    }
}

/// Cancels the shared control if the lane body unwinds.
struct CancelOnUnwind<'a, 'b>(&'a LaneControl<'b>);

impl Drop for CancelOnUnwind<'_, '_> {
    fn drop(&mut self) {
        if std::thread::panicking() {
            self.0.cancel();
        }
    }
}

/// Cancels the shared control when the coordinator leaves, either way.
///
/// The coordinator's exit is the end of the work it can feed, and it may leave
/// by panicking, in which case the lanes must be stopped before the join.
struct CancelOnExit<'a, 'b>(&'a LaneControl<'b>);

impl Drop for CancelOnExit<'_, '_> {
    fn drop(&mut self) {
        self.0.cancel();
    }
}

type LaneBody<'a, E> = dyn Fn(usize, &mut CpuEngine, &LaneControl<'_>) -> Result<(), E> + Sync + 'a;

/// Runs one lane body on one engine, stopping the other lanes if it fails.
fn run_lane<E>(
    control: &LaneControl<'_>,
    engine: &mut CpuEngine,
    index: usize,
    body: &LaneBody<'_, E>,
) -> Result<(), E> {
    let _unwind = CancelOnUnwind(control);
    match body(index, engine, control) {
        Ok(()) => Ok(()),
        Err(error) => {
            control.cancel();
            Err(error)
        }
    }
}

/// Joins every lane, reporting the first lane error and resuming the first panic.
fn join_lanes<E: Send>(
    handles: Vec<ScopedJoinHandle<'_, Result<(), E>>>,
    control: &LaneControl<'_>,
) -> Result<(), LanesError<E, LaneThreadError>> {
    let mut outcome = Ok(());
    let mut panicked = None;
    for handle in handles {
        match handle.join() {
            Ok(Ok(())) => {}
            Ok(Err(error)) => {
                control.cancel();
                if outcome.is_ok() {
                    outcome = Err(LanesError::Lane(error));
                }
            }
            Err(payload) => {
                control.cancel();
                if panicked.is_none() {
                    panicked = Some(payload);
                }
            }
        }
    }
    match panicked {
        Some(payload) => std::panic::resume_unwind(payload),
        None => outcome,
    }
}

impl Lanes for TenferroLanes {
    type Lane<'l> = CpuEngine;
    type Error = LaneThreadError;

    fn lane_count(&self) -> usize {
        self.engines.len()
    }

    fn run<E: Send>(
        &mut self,
        lane: impl Fn(usize, &mut CpuEngine, &LaneControl<'_>) -> Result<(), E> + Sync,
    ) -> Result<(), LanesError<E, Self::Error>> {
        let control = LaneControl::new();
        let control_ref = &control;
        let body: &LaneBody<'_, E> = &lane;
        thread::scope(|scope| {
            let handles = match self.start_lanes(scope, control_ref, body) {
                Ok(handles) => handles,
                Err(error) => {
                    // The lanes already running have nothing left to feed them,
                    // so they are cancelled and the scope's own join returns.
                    control_ref.cancel();
                    return Err(LanesError::Executor(error));
                }
            };
            join_lanes(handles, control_ref)
        })
    }

    fn run_with_coordinator<C, E: Send>(
        &mut self,
        coordinator: impl FnOnce(&LaneControl<'_>) -> Result<C, E>,
        lane: impl Fn(usize, &mut CpuEngine, &LaneControl<'_>) -> Result<(), E> + Sync,
    ) -> Result<C, LanesError<E, Self::Error>> {
        let control = LaneControl::new();
        let control_ref = &control;
        let body: &LaneBody<'_, E> = &lane;
        thread::scope(|scope| {
            let handles = match self.start_lanes(scope, control_ref, body) {
                Ok(handles) => handles,
                Err(error) => {
                    // The lanes already running have nothing left to feed them,
                    // so they are cancelled and the scope's own join returns.
                    control_ref.cancel();
                    return Err(LanesError::Executor(error));
                }
            };
            // The coordinator runs on the calling thread while the lanes run.
            // Its exit cancels them, so a waiting lane cannot outlive it.
            let coordinated = {
                let _stop = CancelOnExit(control_ref);
                coordinator(control_ref)
            };
            join_lanes(handles, control_ref)?;
            coordinated.map_err(LanesError::Coordinator)
        })
    }
}
