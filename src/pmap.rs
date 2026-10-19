use crate::domain::{Domain, LocalMode};
#[cfg(feature = "rayon")]
use crate::lanes::batch::{
    idle_backoff, run_lane, LocalBatch, LocalBatchOutcome, LocalBatchResult, LocalQueue,
};
#[cfg(feature = "rayon")]
use crate::lanes::{LaneControl, Lanes, LanesError};
#[cfg(feature = "rayon")]
use crate::local::{in_rayon_worker_context, RayonLanes};
use crate::map::truncate_message;
#[cfg(feature = "rayon")]
use crate::mpi_backend;
use crate::mpi_backend::collective::SystemOperation;
use crate::mpi_backend::traits::*;
use crate::mpi_check::mpi_call;
use crate::scheduler::{BatchId, Coordinator, Dispatch, ItemResult, SchedulerError};
use crate::wire::{self, ErrorClass, ErrorKey, Header, MessageKind, MessageStatus};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::cell::Cell;
#[cfg(feature = "rayon")]
use std::convert::Infallible;
use std::fmt::{self, Display, Formatter};
use std::num::NonZeroUsize;
#[cfg(all(test, not(feature = "rayon")))]
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

const HEADER_TAG: i32 = 0;
const PAYLOAD_TAG: i32 = 1;

#[cfg(all(test, not(feature = "rayon")))]
static CORRUPT_NEXT_RESULT: AtomicBool = AtomicBool::new(false);
#[cfg(all(test, not(feature = "rayon")))]
static FAULT_PHASE: AtomicBool = AtomicBool::new(false);
#[cfg(all(test, not(feature = "rayon")))]
static TEST_ASSIGNED_REMOTES: AtomicU64 = AtomicU64::new(0);
#[cfg(all(test, not(feature = "rayon")))]
static TEST_USER_RESULTS: AtomicUsize = AtomicUsize::new(0);
#[cfg(all(test, not(feature = "rayon")))]
static TEST_DECODE_FAILURES: AtomicUsize = AtomicUsize::new(0);

#[cfg(all(test, not(feature = "rayon")))]
fn reset_fault_trace() {
    CORRUPT_NEXT_RESULT.store(false, Ordering::SeqCst);
    FAULT_PHASE.store(false, Ordering::SeqCst);
    TEST_ASSIGNED_REMOTES.store(0, Ordering::SeqCst);
    TEST_USER_RESULTS.store(0, Ordering::SeqCst);
    TEST_DECODE_FAILURES.store(0, Ordering::SeqCst);
}

#[cfg(all(test, not(feature = "rayon")))]
fn wait_for_marker(path: &std::path::Path) {
    let observed = (0..500).any(|_| {
        if path.exists() {
            true
        } else {
            std::thread::sleep(std::time::Duration::from_millis(10));
            false
        }
    });
    assert!(observed, "timed out waiting for fault-test marker");
}

type LocalError = (PmapErrorKind, String);
type RootOutcome<U> = (Vec<U>, i64, Option<LocalError>);

#[cfg(feature = "rayon")]
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
enum RootEvent {
    Local,
    Remote,
    Neither,
}

#[cfg(feature = "rayon")]
fn choose_root_event(
    local_ready: bool,
    remote_ready: bool,
    prefer_local: bool,
) -> (RootEvent, bool) {
    match (local_ready, remote_ready) {
        (true, true) if prefer_local => (RootEvent::Local, false),
        (true, true) => (RootEvent::Remote, true),
        (true, false) => (RootEvent::Local, prefer_local),
        (false, true) => (RootEvent::Remote, prefer_local),
        (false, false) => (RootEvent::Neither, prefer_local),
    }
}

/// Options shared by every rank in one collective [`pmap`] call.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct PmapOptions {
    /// Rank that owns the input and receives the ordered output.
    pub root: i32,
    /// Fixed positive number of items assigned in one batch.
    pub batch_size: NonZeroUsize,
    /// Rank-local execution mode; MPI-only calls require [`LocalMode::Sequential`].
    pub local_mode: LocalMode,
    /// Allows one prefetched batch per remote hybrid domain.
    ///
    /// The default is `false`. MPI-only calls reject `true` during collective
    /// preflight; hybrid calls keep the root domain at capacity one.
    pub prefetch: bool,
}

impl Default for PmapOptions {
    fn default() -> Self {
        Self {
            root: 0,
            batch_size: NonZeroUsize::new(1).unwrap(),
            local_mode: LocalMode::Sequential,
            prefetch: false,
        }
    }
}

/// Stable category of a collective [`pmap`] failure.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum PmapErrorKind {
    Reentrant,
    Preflight,
    User,
    Wire,
    Protocol,
}

/// Error converged across every rank in a collective [`pmap`] call.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct PmapError {
    kind: PmapErrorKind,
    key: Option<i64>,
    message: String,
}

impl PmapError {
    fn new(kind: PmapErrorKind, message: impl Display) -> Self {
        Self {
            kind,
            key: None,
            message: truncate_message(message.to_string()),
        }
    }

    fn keyed(kind: PmapErrorKind, key: i64, message: impl Display) -> Self {
        Self {
            kind,
            key: Some(key),
            message: truncate_message(message.to_string()),
        }
    }

    pub fn kind(&self) -> PmapErrorKind {
        self.kind
    }

    pub fn key(&self) -> Option<i64> {
        self.key
    }

    pub fn message(&self) -> &str {
        &self.message
    }
}

impl Display for PmapError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.message())
    }
}

impl std::error::Error for PmapError {}

thread_local! {
    static ACTIVE: Cell<bool> = const { Cell::new(false) };
}

pub struct Active;

impl Active {
    fn acquire() -> Result<Self, PmapError> {
        ACTIVE.with(|active| {
            if active.replace(true) {
                Err(PmapError::new(
                    PmapErrorKind::Reentrant,
                    "pmap is already active on this thread",
                ))
            } else {
                Ok(Self)
            }
        })
    }
}

impl Drop for Active {
    fn drop(&mut self) {
        ACTIVE.with(|active| active.set(false));
    }
}

fn wire_error(error: impl Display) -> PmapError {
    PmapError::new(PmapErrorKind::Wire, error)
}

fn protocol_error(error: impl Display) -> PmapError {
    PmapError::new(PmapErrorKind::Protocol, error)
}

fn scheduler_error(error: SchedulerError) -> PmapError {
    protocol_error(error)
}

fn header(
    kind: MessageKind,
    status: MessageStatus,
    batch_id: u64,
    item_count: u64,
    payload_len: usize,
) -> Result<Header, PmapError> {
    Header::new(kind, status, batch_id, item_count, payload_len as u64).map_err(wire_error)
}

fn send_header<C: Communicator>(comm: &C, rank: i32, value: Header) {
    mpi_call!(comm
        .process_at_rank(rank)
        .send_with_tag(&value.encode(), HEADER_TAG));
}

fn receive_header<C: Communicator>(comm: &C, rank: i32) -> Result<Header, PmapError> {
    let (bytes, _) = mpi_call!(comm
        .process_at_rank(rank)
        .receive_vec_with_tag::<u8>(HEADER_TAG));
    Header::decode(&bytes).map_err(wire_error)
}

#[cfg(not(feature = "rayon"))]
fn receive_any_header<C: Communicator>(comm: &C) -> Result<(i32, Header), PmapError> {
    let (bytes, status) = mpi_call!(comm.any_process().receive_vec_with_tag::<u8>(HEADER_TAG));
    Header::decode(&bytes)
        .map(|value| (status.source_rank(), value))
        .map_err(wire_error)
}

fn receive_payload<C: Communicator, T: DeserializeOwned>(
    comm: &C,
    rank: i32,
    value: Header,
) -> Result<T, PmapError> {
    let count = wire::checked_mpi_count(value.payload_len()).map_err(wire_error)?;
    let _length = wire::checked_usize_length(value.payload_len()).map_err(wire_error)?;
    let (bytes, _) = mpi_call!(comm
        .process_at_rank(rank)
        .receive_vec_with_tag::<u8>(PAYLOAD_TAG));
    if bytes.len() != count as usize {
        return Err(protocol_error("payload length does not match its header"));
    }
    wire::decode_payload(&bytes).map_err(wire_error)
}

fn send_frame<C: Communicator, T: Serialize>(
    comm: &C,
    rank: i32,
    kind: MessageKind,
    status: MessageStatus,
    batch_id: u64,
    item_count: usize,
    payload: &T,
) -> Result<(), PmapError> {
    let bytes = wire::encode_payload(payload).map_err(wire_error)?;
    #[cfg(all(test, not(feature = "rayon")))]
    let mut bytes = bytes;
    #[cfg(all(test, not(feature = "rayon")))]
    if kind == MessageKind::Result
        && mpi_call!(comm.rank()) == 1
        && CORRUPT_NEXT_RESULT.swap(false, Ordering::SeqCst)
    {
        bytes = vec![0xff];
    }
    let value = header(kind, status, batch_id, item_count as u64, bytes.len())?;
    send_header(comm, rank, value);
    mpi_call!(comm
        .process_at_rank(rank)
        .send_with_tag(bytes.as_slice(), PAYLOAD_TAG));
    Ok(())
}

fn send_empty<C: Communicator>(
    comm: &C,
    rank: i32,
    kind: MessageKind,
    batch_id: u64,
) -> Result<(), PmapError> {
    let value = header(kind, MessageStatus::None, batch_id, 0, 0)?;
    send_header(comm, rank, value);
    Ok(())
}

#[derive(Serialize, Deserialize)]
struct CallbackFailure {
    index: usize,
    kind: i32,
    message: String,
}

#[derive(Serialize, Deserialize)]
enum ResultPayload<U> {
    Values(Vec<(usize, U)>),
    Error(CallbackFailure),
}

fn error_key(
    task_key: usize,
    class: ErrorClass,
    rank: i32,
    world_size: i32,
) -> Result<i64, PmapError> {
    ErrorKey::new(task_key as u64, class, rank, world_size)
        .map(|key| key.get())
        .map_err(wire_error)
}

fn convergence<C: Communicator>(comm: &C, local_key: i64) -> i64 {
    let mut selected = local_key;
    mpi_call!(comm.all_reduce_into(&local_key, &mut selected, SystemOperation::min()));
    selected
}

fn broadcast_error<C: Communicator>(
    comm: &C,
    selected: i64,
    local_key: i64,
    local: Option<(PmapErrorKind, String)>,
) -> Result<Option<(PmapErrorKind, String)>, PmapError> {
    if selected == ErrorKey::NO_ERROR_KEY {
        return Ok(None);
    }
    if selected < 0 || mpi_call!(comm.size()) <= 0 {
        return Err(protocol_error("invalid selected error key"));
    }
    let winner = (selected % i64::from(mpi_call!(comm.size()))) as i32;
    let is_winner = mpi_call!(comm.rank()) == winner;
    let winner_local = if is_winner && local_key == selected {
        local
    } else if is_winner {
        return Err(protocol_error("winner does not own selected error"));
    } else {
        None
    };

    let mut kind = winner_local
        .as_ref()
        .map_or(PmapErrorKind::Protocol.code(), |(kind, _)| kind.code());
    let mut key = selected;
    mpi_call!(comm.process_at_rank(winner).broadcast_into(&mut kind));
    mpi_call!(comm.process_at_rank(winner).broadcast_into(&mut key));
    if key != selected {
        return Err(protocol_error("broadcast error key mismatch"));
    }

    let mut message = winner_local
        .map(|(_, message)| message.into_bytes())
        .unwrap_or_default();
    let mut length = i32::try_from(message.len())
        .map_err(|_| protocol_error("error message length overflow"))?;
    mpi_call!(comm.process_at_rank(winner).broadcast_into(&mut length));
    if length < 0 || length as usize > crate::map::MAX_ERROR_MESSAGE_BYTES {
        return Err(protocol_error("invalid broadcast error message length"));
    }
    if !is_winner {
        message.resize(length as usize, 0);
    }
    mpi_call!(comm
        .process_at_rank(winner)
        .broadcast_into(message.as_mut_slice()));

    let kind =
        PmapErrorKind::from_code(kind).ok_or_else(|| protocol_error("invalid error kind"))?;
    let message = String::from_utf8(message).map_err(protocol_error)?;
    Ok(Some((kind, message)))
}

fn preflight<C, T>(
    comm: &C,
    options: &PmapOptions,
    root_items: Option<&Vec<T>>,
    execution_valid: bool,
) -> Result<(), PmapError>
where
    C: Communicator,
{
    let rank = mpi_call!(comm.rank());
    let size = mpi_call!(comm.size());
    let local_batch = options.batch_size.get() as u64;
    let local_mode = match options.local_mode {
        LocalMode::Sequential => 0_i32,
        LocalMode::Outer => 1,
        LocalMode::Inner => 2,
    };
    let local_prefetch = i32::from(options.prefetch);

    let mut min_root = options.root;
    let mut max_root = options.root;
    let mut min_batch = local_batch;
    let mut max_batch = local_batch;
    let mut min_mode = local_mode;
    let mut max_mode = local_mode;
    let mut min_prefetch = local_prefetch;
    let mut max_prefetch = local_prefetch;
    mpi_call!(comm.all_reduce_into(&options.root, &mut min_root, SystemOperation::min()));
    mpi_call!(comm.all_reduce_into(&options.root, &mut max_root, SystemOperation::max()));
    mpi_call!(comm.all_reduce_into(&local_batch, &mut min_batch, SystemOperation::min()));
    mpi_call!(comm.all_reduce_into(&local_batch, &mut max_batch, SystemOperation::max()));
    mpi_call!(comm.all_reduce_into(&local_mode, &mut min_mode, SystemOperation::min()));
    mpi_call!(comm.all_reduce_into(&local_mode, &mut max_mode, SystemOperation::max()));
    mpi_call!(comm.all_reduce_into(&local_prefetch, &mut min_prefetch, SystemOperation::min()));
    mpi_call!(comm.all_reduce_into(&local_prefetch, &mut max_prefetch, SystemOperation::max()));

    let local_valid = mpi_call!(crate::mpi_check::is_thread_main())
        && options.root >= 0
        && options.root < size
        && ((rank == options.root) == root_items.is_some())
        && execution_valid
        && min_root == max_root
        && min_batch == max_batch
        && min_mode == max_mode
        && min_prefetch == max_prefetch;
    let candidate = wire::preflight_candidate(!local_valid, rank, size).map_err(wire_error)?;
    let mut failing_rank = candidate;
    mpi_call!(comm.all_reduce_into(&candidate, &mut failing_rank, SystemOperation::min()));
    if failing_rank != size {
        return Err(PmapError::new(
            PmapErrorKind::Preflight,
            format!("pmap preflight failed on rank {failing_rank}"),
        ));
    }
    Ok(())
}

#[cfg(not(feature = "rayon"))]
fn worker_loop<C, T, U, E, F>(
    comm: &C,
    root: i32,
    mut f: F,
) -> Result<(i64, Option<LocalError>), PmapError>
where
    C: Communicator,
    T: Serialize + DeserializeOwned,
    U: Serialize + DeserializeOwned,
    E: Display,
    F: FnMut(T) -> Result<U, E>,
{
    let rank = mpi_call!(comm.rank());
    let size = mpi_call!(comm.size());
    let mut local_key = ErrorKey::NO_ERROR_KEY;
    let mut local_error = None;

    loop {
        #[cfg(all(test, not(feature = "rayon")))]
        if FAULT_PHASE.load(Ordering::SeqCst) && rank >= 2 {
            if let Some(base) = std::env::var_os("HATAORI_FAULT_MARKER") {
                let assigned = std::path::PathBuf::from(base).with_extension("assigned");
                wait_for_marker(&assigned);
            }
        }
        send_empty(comm, root, MessageKind::Ready, 0)?;
        let value = receive_header(comm, root)?;
        match value.kind() {
            MessageKind::Task => {
                let items: Vec<(usize, T)> = match receive_payload(comm, root, value) {
                    Ok(items) => items,
                    Err(error) => {
                        let index = usize::try_from(value.item_count()).unwrap_or(usize::MAX);
                        let key = error_key(index, ErrorClass::WireProtocol, rank, size)?;
                        local_key = local_key.min(key);
                        local_error = Some((PmapErrorKind::Wire, error.message().to_owned()));
                        let failure = CallbackFailure {
                            index,
                            kind: PmapErrorKind::Wire.code(),
                            message: error.message().to_owned(),
                        };
                        send_frame(
                            comm,
                            root,
                            MessageKind::Result,
                            MessageStatus::Error,
                            value.batch_id(),
                            1,
                            &ResultPayload::<U>::Error(failure),
                        )?;
                        continue;
                    }
                };
                let mut results = Vec::with_capacity(items.len());
                let mut failure = None;
                for (index, item) in items {
                    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| f(item))) {
                        Ok(Ok(result)) => results.push((index, result)),
                        Ok(Err(error)) => {
                            failure = Some(CallbackFailure {
                                index,
                                kind: PmapErrorKind::User.code(),
                                message: truncate_message(error.to_string()),
                            });
                            break;
                        }
                        Err(_) => mpi_call!(comm.abort(71)),
                    }
                }
                match failure {
                    Some(failure) => {
                        let key = error_key(failure.index, ErrorClass::Callback, rank, size)?;
                        local_key = local_key.min(key);
                        local_error = Some((PmapErrorKind::User, failure.message.clone()));
                        send_frame(
                            comm,
                            root,
                            MessageKind::Result,
                            MessageStatus::Error,
                            value.batch_id(),
                            value.item_count() as usize,
                            &ResultPayload::<U>::Error(failure),
                        )?;
                    }
                    None => send_frame(
                        comm,
                        root,
                        MessageKind::Result,
                        MessageStatus::Ok,
                        value.batch_id(),
                        results.len(),
                        &ResultPayload::Values(results),
                    )?,
                }
            }
            MessageKind::Stop => {
                send_empty(comm, root, MessageKind::Drain, 0)?;
                return Ok((local_key, local_error));
            }
            _ => {
                return Err(protocol_error("worker received an unexpected message"));
            }
        }
    }
}

#[cfg(feature = "rayon")]
enum PrefetchedBatch<T, U> {
    Work {
        batch_id: u64,
        items: Vec<(usize, T)>,
    },
    Ready(LocalBatchOutcome<U>),
    Stop,
}

#[cfg(feature = "rayon")]
fn receive_prefetched_batch<C, T, U>(
    comm: &C,
    root: i32,
) -> Result<PrefetchedBatch<T, U>, PmapError>
where
    C: Communicator,
    T: DeserializeOwned,
{
    let value = receive_header(comm, root)?;
    match value.kind() {
        MessageKind::Task => {
            let batch_id = value.batch_id();
            match receive_payload::<_, Vec<(usize, T)>>(comm, root, value) {
                Ok(items) => Ok(PrefetchedBatch::Work { batch_id, items }),
                // The task transfer failed after its header: report one wire
                // failure for this batch instead of handing it to a lane.
                Err(error) => Ok(PrefetchedBatch::Ready(LocalBatchOutcome::wire_error(
                    batch_id,
                    1,
                    usize::try_from(value.item_count()).unwrap_or(usize::MAX),
                    error.message().to_owned(),
                ))),
            }
        }
        MessageKind::Stop => Ok(PrefetchedBatch::Stop),
        _ => Err(protocol_error("worker received an unexpected message")),
    }
}

#[cfg(feature = "rayon")]
fn send_worker_outcome<C, U>(
    comm: &C,
    root: i32,
    rank: i32,
    size: i32,
    outcome: LocalBatchOutcome<U>,
    local_key: &mut i64,
    local_error: &mut Option<LocalError>,
) -> Result<(), PmapError>
where
    C: Communicator,
    U: Serialize,
{
    let failure = match outcome.result {
        LocalBatchResult::Values(values) => {
            return send_frame(
                comm,
                root,
                MessageKind::Result,
                MessageStatus::Ok,
                outcome.batch_id,
                values.len(),
                &ResultPayload::Values(values),
            )
        }
        LocalBatchResult::CallbackError { index, message } => BatchFailure {
            index,
            message,
            class: ErrorClass::Callback,
            kind: PmapErrorKind::User,
        },
        LocalBatchResult::WireError { index, message } => BatchFailure {
            index,
            message,
            class: ErrorClass::WireProtocol,
            kind: PmapErrorKind::Wire,
        },
        LocalBatchResult::Panic => mpi_call!(comm.abort(71)),
    };
    let key = error_key(failure.index, failure.class, rank, size)?;
    if key < *local_key {
        *local_key = key;
        *local_error = Some((failure.kind, failure.message.clone()));
    }
    let payload = CallbackFailure {
        index: failure.index,
        kind: failure.kind.code(),
        message: failure.message,
    };
    send_frame(
        comm,
        root,
        MessageKind::Result,
        MessageStatus::Error,
        outcome.batch_id,
        outcome.item_count,
        &ResultPayload::<U>::Error(payload),
    )
}

#[cfg(feature = "rayon")]
struct BatchFailure {
    index: usize,
    message: String,
    class: ErrorClass,
    kind: PmapErrorKind,
}

/// How the coordinator hands one local batch to the execution resource.
#[cfg(feature = "rayon")]
trait BatchSink<T, U> {
    /// Submits `batch`. A concurrent sink returns immediately; an inline sink
    /// has already executed the batch when this returns.
    fn submit(&mut self, batch: LocalBatch<T>);

    /// Returns the outcome of `batch_id` if the lanes already produced it.
    fn try_wait(&mut self, batch_id: u64) -> Option<LocalBatchOutcome<U>>;

    /// Waits for the outcome of `batch_id`, bounded by executor cancellation.
    fn wait(&mut self, batch_id: u64) -> Option<LocalBatchOutcome<U>> {
        loop {
            if let Some(outcome) = self.try_wait(batch_id) {
                return Some(outcome);
            }
            if self.cancelled() {
                return None;
            }
            idle_backoff();
        }
    }

    /// Whether the executor cancelled the run (lane infrastructure failure).
    fn cancelled(&self) -> bool;
}

#[cfg(feature = "rayon")]
struct ConcurrentSink<'a, T, U> {
    queue: &'a LocalQueue<T, U>,
    control: &'a LaneControl<'a>,
}

#[cfg(feature = "rayon")]
impl<T, U> BatchSink<T, U> for ConcurrentSink<'_, T, U> {
    fn submit(&mut self, batch: LocalBatch<T>) {
        self.queue.submit(batch);
    }

    fn try_wait(&mut self, batch_id: u64) -> Option<LocalBatchOutcome<U>> {
        self.queue.take_outcome(batch_id)
    }

    fn cancelled(&self) -> bool {
        self.control.cancelled()
    }
}

/// A sink that executes each batch on the coordinator thread through `lanes.run`.
#[cfg(feature = "rayon")]
struct InlineSink<S, U> {
    run_batch: S,
    outcome: Option<LocalBatchOutcome<U>>,
}

#[cfg(feature = "rayon")]
impl<S, T, U> BatchSink<T, U> for InlineSink<S, U>
where
    S: FnMut(LocalBatch<T>) -> LocalBatchOutcome<U>,
{
    fn submit(&mut self, batch: LocalBatch<T>) {
        let outcome = (self.run_batch)(batch);
        self.outcome = Some(outcome);
    }

    fn try_wait(&mut self, batch_id: u64) -> Option<LocalBatchOutcome<U>> {
        self.outcome
            .take()
            .filter(|outcome| outcome.batch_id == batch_id)
    }

    fn cancelled(&self) -> bool {
        false
    }
}

/// Runs one batch on the calling thread through the lanes executor.
#[cfg(feature = "rayon")]
fn run_batch_inline<L, T, U, E, F>(
    lanes: &mut L,
    f: &F,
    batch: LocalBatch<T>,
) -> LocalBatchOutcome<U>
where
    L: Lanes,
    T: Send,
    U: Send,
    E: Display + Send,
    F: Fn(T, &mut L::Lane<'_>) -> Result<U, E> + Sync,
{
    let batch_id = batch.batch_id;
    let item_count = batch.item_count;
    let queue = LocalQueue::<T, U>::new();
    queue.submit(batch);
    queue.close();
    let joined = lanes.run(|_index, lane, control| {
        run_lane(&queue, f, control, lane);
        Ok::<(), Infallible>(())
    });
    let joined = match joined {
        Ok(()) => true,
        Err(LanesError::Lane(never)) => match never {},
        Err(LanesError::Coordinator(never)) => match never {},
        // The coordinator turns an executor failure into the ordinary lane abort.
        Err(LanesError::SingleLane) | Err(LanesError::Reentrant) | Err(LanesError::Executor(_)) => {
            false
        }
    };
    if joined {
        if let Some(outcome) = queue.take_outcome(batch_id) {
            return outcome;
        }
    }
    LocalBatchOutcome {
        batch_id,
        item_count,
        result: LocalBatchResult::Panic,
    }
}

#[cfg(feature = "rayon")]
fn lane_executor_failure(error: impl Display) -> PmapError {
    PmapError::new(
        PmapErrorKind::Protocol,
        format!("lane executor failed: {error}"),
    )
}

/// The worker-side MPI protocol, generic over how local batches execute.
#[cfg(feature = "rayon")]
fn hybrid_worker_protocol<C, S, T, U>(
    comm: &C,
    root: i32,
    mode: LocalMode,
    prefetch: bool,
    sink: &mut S,
) -> Result<(i64, Option<LocalError>), PmapError>
where
    C: Communicator,
    S: BatchSink<T, U>,
    T: Serialize + DeserializeOwned + Send,
    U: Serialize + DeserializeOwned + Send,
{
    if prefetch {
        hybrid_worker_protocol_prefetch(comm, root, mode, sink)
    } else {
        hybrid_worker_protocol_plain(comm, root, mode, sink)
    }
}

#[cfg(feature = "rayon")]
fn hybrid_worker_protocol_plain<C, S, T, U>(
    comm: &C,
    root: i32,
    mode: LocalMode,
    sink: &mut S,
) -> Result<(i64, Option<LocalError>), PmapError>
where
    C: Communicator,
    S: BatchSink<T, U>,
    T: Serialize + DeserializeOwned + Send,
    U: Serialize + DeserializeOwned + Send,
{
    let rank = mpi_call!(comm.rank());
    let size = mpi_call!(comm.size());
    let mut local_key = ErrorKey::NO_ERROR_KEY;
    let mut local_error = None;

    loop {
        send_empty(comm, root, MessageKind::Ready, 0)?;
        let value = receive_header(comm, root)?;
        match value.kind() {
            MessageKind::Task => {
                let items: Vec<(usize, T)> = match receive_payload(comm, root, value) {
                    Ok(items) => items,
                    Err(error) => {
                        let index = usize::try_from(value.item_count()).unwrap_or(usize::MAX);
                        let key = error_key(index, ErrorClass::WireProtocol, rank, size)?;
                        local_key = local_key.min(key);
                        local_error = Some((PmapErrorKind::Wire, error.message().to_owned()));
                        let failure = CallbackFailure {
                            index,
                            kind: PmapErrorKind::Wire.code(),
                            message: error.message().to_owned(),
                        };
                        send_frame(
                            comm,
                            root,
                            MessageKind::Result,
                            MessageStatus::Error,
                            value.batch_id(),
                            1,
                            &ResultPayload::<U>::Error(failure),
                        )?;
                        continue;
                    }
                };
                let batch_id = value.batch_id();
                sink.submit(LocalBatch::new(batch_id, mode, items));
                let outcome = match sink.wait(batch_id) {
                    Some(outcome) => outcome,
                    None => mpi_call!(comm.abort(71)),
                };
                send_worker_outcome(
                    comm,
                    root,
                    rank,
                    size,
                    outcome,
                    &mut local_key,
                    &mut local_error,
                )?;
            }
            MessageKind::Stop => {
                send_empty(comm, root, MessageKind::Drain, 0)?;
                return Ok((local_key, local_error));
            }
            _ => return Err(protocol_error("worker received an unexpected message")),
        }
    }
}

#[cfg(feature = "rayon")]
fn hybrid_worker_protocol_prefetch<C, S, T, U>(
    comm: &C,
    root: i32,
    mode: LocalMode,
    sink: &mut S,
) -> Result<(i64, Option<LocalError>), PmapError>
where
    C: Communicator,
    S: BatchSink<T, U>,
    T: Serialize + DeserializeOwned + Send,
    U: Serialize + DeserializeOwned + Send,
{
    let rank = mpi_call!(comm.rank());
    let size = mpi_call!(comm.size());
    let mut local_key = ErrorKey::NO_ERROR_KEY;
    let mut local_error = None;

    'idle: loop {
        send_empty(comm, root, MessageKind::Ready, 0)?;
        let mut current = match receive_prefetched_batch::<_, T, U>(comm, root)? {
            PrefetchedBatch::Work { batch_id, items } => {
                sink.submit(LocalBatch::new(batch_id, mode, items));
                batch_id
            }
            PrefetchedBatch::Ready(outcome) => {
                send_worker_outcome(
                    comm,
                    root,
                    rank,
                    size,
                    outcome,
                    &mut local_key,
                    &mut local_error,
                )?;
                continue 'idle;
            }
            PrefetchedBatch::Stop => {
                send_empty(comm, root, MessageKind::Drain, 0)?;
                return Ok((local_key, local_error));
            }
        };

        loop {
            // Overlap the next blocking transfer with the resident batch.
            if send_empty(comm, root, MessageKind::Ready, 0).is_err() {
                mpi_call!(comm.abort(75));
            }
            let next = match receive_prefetched_batch::<_, T, U>(comm, root) {
                Ok(next) => next,
                Err(_) => mpi_call!(comm.abort(75)),
            };
            let outcome = match sink.wait(current) {
                Some(outcome) => outcome,
                None => mpi_call!(comm.abort(71)),
            };
            // A panicking callback aborts before the prefetched batch is handed
            // to the lanes or executed inline, so a hung peer callback can never
            // delay the abort.
            if matches!(outcome.result, LocalBatchResult::Panic) {
                mpi_call!(comm.abort(71));
            }

            match next {
                PrefetchedBatch::Work { batch_id, items } => {
                    // The resident batch has completed, so the lanes can start
                    // the prefetched batch while this result is transferred.
                    sink.submit(LocalBatch::new(batch_id, mode, items));
                    current = batch_id;
                    if send_worker_outcome(
                        comm,
                        root,
                        rank,
                        size,
                        outcome,
                        &mut local_key,
                        &mut local_error,
                    )
                    .is_err()
                    {
                        mpi_call!(comm.abort(75));
                    }
                }
                PrefetchedBatch::Ready(next_outcome) => {
                    if send_worker_outcome(
                        comm,
                        root,
                        rank,
                        size,
                        outcome,
                        &mut local_key,
                        &mut local_error,
                    )
                    .is_err()
                    {
                        mpi_call!(comm.abort(75));
                    }
                    send_worker_outcome(
                        comm,
                        root,
                        rank,
                        size,
                        next_outcome,
                        &mut local_key,
                        &mut local_error,
                    )?;
                    continue 'idle;
                }
                PrefetchedBatch::Stop => {
                    if send_worker_outcome(
                        comm,
                        root,
                        rank,
                        size,
                        outcome,
                        &mut local_key,
                        &mut local_error,
                    )
                    .is_err()
                    {
                        mpi_call!(comm.abort(75));
                    }
                    send_empty(comm, root, MessageKind::Drain, 0)?;
                    return Ok((local_key, local_error));
                }
            }
        }
    }
}

/// Runs the hybrid worker protocol with a caller-thread coordinator.
///
/// The lanes execute local batches while the calling thread keeps MPI progress.
/// An executor that rejects a concurrent coordinator run (a one-lane backend)
/// falls back explicitly to executing each batch on the coordinator thread
/// through [`Lanes::run`] between MPI progress steps.
#[cfg(feature = "rayon")]
fn hybrid_worker_loop<L, C, T, U, E, F>(
    comm: &C,
    root: i32,
    mode: LocalMode,
    prefetch: bool,
    lanes: &mut L,
    f: &F,
) -> Result<(i64, Option<LocalError>), PmapError>
where
    L: Lanes,
    C: Communicator,
    T: Serialize + DeserializeOwned + Send,
    U: Serialize + DeserializeOwned + Send,
    E: Display + Send,
    F: Fn(T, &mut L::Lane<'_>) -> Result<U, E> + Sync,
{
    let queue = LocalQueue::<T, U>::new();
    let concurrent = lanes.run_with_coordinator(
        |control| {
            let mut sink = ConcurrentSink {
                queue: &queue,
                control,
            };
            hybrid_worker_protocol(comm, root, mode, prefetch, &mut sink)
        },
        // A lane callback returns only infrastructure failures; item and batch
        // failures travel through the queue as data.
        |_index, lane, control| {
            run_lane(&queue, f, control, lane);
            Ok::<(), PmapError>(())
        },
    );
    match concurrent {
        Ok(outcome) => Ok(outcome),
        Err(LanesError::SingleLane) => {
            let mut sink = InlineSink {
                run_batch: |batch| run_batch_inline(lanes, f, batch),
                outcome: None,
            };
            hybrid_worker_protocol(comm, root, mode, prefetch, &mut sink)
        }
        Err(LanesError::Coordinator(error)) => Err(error),
        Err(LanesError::Executor(error)) => Err(lane_executor_failure(error)),
        Err(LanesError::Reentrant) => Err(lane_executor_failure(
            "executor cannot run a coordinator from one of its own workers",
        )),
        Err(LanesError::Lane(error)) => Err(error),
    }
}

#[cfg(not(feature = "rayon"))]
fn execute_root_batch<C, T, U, E, F>(
    comm: &C,
    scheduler: &mut Coordinator<T, U>,
    f: &mut F,
    rank: i32,
    size: i32,
    local_key: &mut i64,
    local_error: &mut Option<LocalError>,
) -> Result<bool, PmapError>
where
    C: Communicator,
    E: Display,
    F: FnMut(T) -> Result<U, E>,
{
    let Some(batch) = scheduler.next_root_batch().map_err(scheduler_error)? else {
        return Ok(false);
    };
    let batch_id = batch.id();
    let mut results = Vec::new();
    let mut failure = None;
    for item in batch.into_items() {
        let index = item.original_index();
        let callback =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| f(item.into_value())));
        match callback {
            Ok(Ok(result)) => results.push(ItemResult::new(index, result)),
            Ok(Err(error)) => {
                failure = Some((index, truncate_message(error.to_string())));
                break;
            }
            Err(_) => mpi_call!(comm.abort(70)),
        }
    }
    match failure {
        Some((index, message)) => {
            let key = error_key(index, ErrorClass::Callback, rank, size)?;
            *local_key = (*local_key).min(key);
            *local_error = Some((PmapErrorKind::User, message));
            scheduler.on_root_error(batch_id).map_err(scheduler_error)?;
        }
        None => {
            scheduler
                .on_root_success(batch_id, results)
                .map_err(scheduler_error)?;
        }
    }
    Ok(true)
}

struct RootErrorState<'a> {
    call_task_key: usize,
    rank: i32,
    size: i32,
    local_key: &'a mut i64,
    local_error: &'a mut Option<LocalError>,
}

fn process_remote_header<C, T, U>(
    comm: &C,
    scheduler: &mut Coordinator<T, U>,
    source: i32,
    value: Header,
    errors: &mut RootErrorState<'_>,
) -> Result<(), PmapError>
where
    C: Communicator,
    T: Serialize + DeserializeOwned,
    U: Serialize + DeserializeOwned,
{
    let call_task_key = errors.call_task_key;
    let rank = errors.rank;
    let size = errors.size;
    let local_key = &mut *errors.local_key;
    let local_error = &mut *errors.local_error;
    match value.kind() {
        MessageKind::Ready => match scheduler.on_remote_ready(source).map_err(scheduler_error)? {
            Dispatch::Task(batch) => {
                #[cfg(all(test, not(feature = "rayon")))]
                if (0..64).contains(&source) {
                    TEST_ASSIGNED_REMOTES.fetch_or(1_u64 << source, Ordering::SeqCst);
                }
                let batch_id = batch.id();
                let items: Vec<(usize, T)> = batch
                    .into_items()
                    .into_iter()
                    .map(|item| (item.original_index(), item.into_value()))
                    .collect();
                send_frame(
                    comm,
                    source,
                    MessageKind::Task,
                    MessageStatus::None,
                    batch_id.get(),
                    items.len(),
                    &items,
                )?;
            }
            Dispatch::Stop => send_empty(comm, source, MessageKind::Stop, 0)?,
        },
        MessageKind::Result => {
            let payload: ResultPayload<U> = match receive_payload(comm, source, value) {
                Ok(payload) => payload,
                Err(error) => {
                    // receive_payload consumed the actual payload before decode failed.
                    #[cfg(all(test, not(feature = "rayon")))]
                    TEST_DECODE_FAILURES.fetch_add(1, Ordering::SeqCst);
                    let key = error_key(call_task_key, ErrorClass::WireProtocol, rank, size)?;
                    *local_key = (*local_key).min(key);
                    *local_error = Some((error.kind(), error.message().to_owned()));
                    scheduler
                        .on_remote_protocol_error(source)
                        .map_err(scheduler_error)?;
                    return Ok(());
                }
            };
            match (value.status(), payload) {
                (MessageStatus::Ok, ResultPayload::Values(values)) => {
                    let results = values
                        .into_iter()
                        .map(|(index, result)| ItemResult::new(index, result))
                        .collect();
                    scheduler
                        .on_remote_success(source, BatchId::from_raw(value.batch_id()), results)
                        .map_err(scheduler_error)?;
                }
                (MessageStatus::Error, ResultPayload::Error(_failure)) => {
                    #[cfg(all(test, not(feature = "rayon")))]
                    TEST_USER_RESULTS.fetch_add(1, Ordering::SeqCst);
                    scheduler
                        .on_remote_error(source, BatchId::from_raw(value.batch_id()))
                        .map_err(scheduler_error)?;
                }
                _ => {
                    let key = error_key(call_task_key, ErrorClass::WireProtocol, rank, size)?;
                    *local_key = (*local_key).min(key);
                    *local_error =
                        Some((PmapErrorKind::Protocol, "invalid result status".to_owned()));
                    scheduler
                        .on_remote_protocol_error(source)
                        .map_err(scheduler_error)?;
                }
            }
        }
        MessageKind::Drain => scheduler.on_remote_drain(source).map_err(scheduler_error)?,
        _ => {
            let key = error_key(call_task_key, ErrorClass::WireProtocol, rank, size)?;
            *local_key = (*local_key).min(key);
            *local_error = Some((
                PmapErrorKind::Protocol,
                "unexpected message from worker".to_owned(),
            ));
            scheduler
                .on_remote_protocol_error(source)
                .map_err(scheduler_error)?;
        }
    }
    Ok(())
}

#[cfg(not(feature = "rayon"))]
fn root_loop<C, T, U, E, F>(
    comm: &C,
    options: &PmapOptions,
    root_items: Vec<T>,
    mut f: F,
) -> Result<RootOutcome<U>, PmapError>
where
    C: Communicator,
    T: Serialize + DeserializeOwned,
    U: Serialize + DeserializeOwned,
    E: Display,
    F: FnMut(T) -> Result<U, E>,
{
    let root = options.root;
    let rank = mpi_call!(comm.rank());
    let size = mpi_call!(comm.size());
    let remotes: Vec<i32> = (0..size).filter(|candidate| *candidate != root).collect();
    let call_task_key = root_items.len();
    let mut scheduler = Coordinator::<T, U>::new_with_prefetch(
        root,
        remotes,
        root_items,
        options.batch_size,
        options.prefetch,
    )
    .map_err(scheduler_error)?;
    let mut local_key = ErrorKey::NO_ERROR_KEY;
    let mut local_error = None;

    loop {
        if size == 1 {
            if !execute_root_batch(
                comm,
                &mut scheduler,
                &mut f,
                rank,
                size,
                &mut local_key,
                &mut local_error,
            )? {
                break;
            }
            continue;
        }

        if scheduler.is_finished() && scheduler.is_quiescent() {
            break;
        }

        // Serve every worker whose message is already pending. The root must
        // not race the workers for batches: taking one batch per received
        // message makes it execute the majority of the work itself, and rank 0
        // becomes the serialized bottleneck that prevents strong scaling with
        // `-n`. The root therefore serves first and loops back while messages
        // keep arriving; it computes a local batch only when the pending
        // queue is empty (every worker is busy computing), and then blocks
        // for the next message to pace itself instead of taking another batch
        // back-to-back.
        let mut served = 0usize;
        while mpi_call!(comm.any_process().immediate_probe_with_tag(HEADER_TAG)).is_some() {
            let (source, value) = receive_any_header(comm)?;
            process_remote_header(
                comm,
                &mut scheduler,
                source,
                value,
                &mut RootErrorState {
                    call_task_key,
                    rank,
                    size,
                    local_key: &mut local_key,
                    local_error: &mut local_error,
                },
            )?;
            served += 1;
        }
        // `is_finished` only turns true once every worker has drained; the
        // top-of-loop break then terminates the loop, so never block on a
        // message after completion.
        if scheduler.is_finished() {
            continue;
        }
        if served == 0 {
            let _ = execute_root_batch(
                comm,
                &mut scheduler,
                &mut f,
                rank,
                size,
                &mut local_key,
                &mut local_error,
            )?;
            // `execute_root_batch` may have just stopped the root lane (no
            // pending work left, or the run failed); re-check completion
            // before blocking, or the root waits forever for a Drain that
            // already arrived.
            if scheduler.is_finished() {
                continue;
            }
            let (source, value) = receive_any_header(comm)?;
            process_remote_header(
                comm,
                &mut scheduler,
                source,
                value,
                &mut RootErrorState {
                    call_task_key,
                    rank,
                    size,
                    local_key: &mut local_key,
                    local_error: &mut local_error,
                },
            )?;
            continue;
        }
    }

    let results = if scheduler.failed() {
        Vec::new()
    } else {
        scheduler.into_results().map_err(scheduler_error)?
    };
    Ok((results, local_key, local_error))
}

/// The root-side MPI protocol, generic over how local batches execute.
///
/// This is the shared root loop: the coordinator on the calling thread keeps
/// scheduler and MPI progress while root-local batches execute on the lanes.
#[cfg(feature = "rayon")]
fn hybrid_root_protocol<C, S, T, U>(
    comm: &C,
    options: &PmapOptions,
    root_items: Vec<T>,
    sink: &mut S,
) -> Result<RootOutcome<U>, PmapError>
where
    C: Communicator,
    S: BatchSink<T, U>,
    T: Serialize + DeserializeOwned + Send,
    U: Serialize + DeserializeOwned + Send,
{
    let root = options.root;
    let rank = mpi_call!(comm.rank());
    let size = mpi_call!(comm.size());
    let remotes: Vec<i32> = (0..size).filter(|candidate| *candidate != root).collect();
    let call_task_key = root_items.len();
    let mut scheduler = Coordinator::<T, U>::new_with_prefetch(
        root,
        remotes,
        root_items,
        options.batch_size,
        options.prefetch,
    )
    .map_err(scheduler_error)?;
    let mut local_key = ErrorKey::NO_ERROR_KEY;
    let mut local_error = None;
    let mut running: Option<u64> = None;
    let mut pending_local: Option<LocalBatchOutcome<U>> = None;
    let mut prefer_local = true;

    loop {
        if sink.cancelled() {
            // A lane infrastructure failure is unrecoverable on this rank.
            mpi_call!(comm.abort(70));
        }
        if running.is_none() {
            if let Some(batch) = scheduler.next_root_batch().map_err(scheduler_error)? {
                let batch_id = batch.id().get();
                let items = batch
                    .into_items()
                    .into_iter()
                    .map(|item| (item.original_index(), item.into_value()))
                    .collect();
                sink.submit(LocalBatch::new(batch_id, options.local_mode, items));
                running = Some(batch_id);
            }
        }
        if running.is_none() && scheduler.is_finished() && scheduler.is_quiescent() {
            break;
        }
        if pending_local.is_none() {
            if let Some(batch_id) = running {
                pending_local = sink.try_wait(batch_id);
            }
        }
        let remote_source = if size > 1 {
            mpi_call!(comm.any_process().immediate_probe_with_tag(HEADER_TAG))
                .map(|status| status.source_rank())
        } else {
            None
        };
        let (event, next_preference) = choose_root_event(
            pending_local.is_some(),
            remote_source.is_some(),
            prefer_local,
        );
        prefer_local = next_preference;

        match event {
            RootEvent::Local => {
                let outcome = pending_local
                    .take()
                    .ok_or_else(|| protocol_error("missing root-local outcome"))?;
                running = None;
                let batch_id = BatchId::from_raw(outcome.batch_id);
                match outcome.result {
                    LocalBatchResult::Values(values) => {
                        let results = values
                            .into_iter()
                            .map(|(index, value)| ItemResult::new(index, value))
                            .collect();
                        scheduler
                            .on_root_success(batch_id, results)
                            .map_err(scheduler_error)?;
                    }
                    LocalBatchResult::CallbackError { index, message } => {
                        let key = error_key(index, ErrorClass::Callback, rank, size)?;
                        local_key = local_key.min(key);
                        local_error = Some((PmapErrorKind::User, message));
                        scheduler.on_root_error(batch_id).map_err(scheduler_error)?;
                    }
                    LocalBatchResult::WireError { .. } => {
                        return Err(protocol_error("root lane produced a wire error"))
                    }
                    LocalBatchResult::Panic => mpi_call!(comm.abort(70)),
                }
            }
            RootEvent::Remote => {
                let source =
                    remote_source.ok_or_else(|| protocol_error("missing probed remote source"))?;
                let value = receive_header(comm, source)?;
                process_remote_header(
                    comm,
                    &mut scheduler,
                    source,
                    value,
                    &mut RootErrorState {
                        call_task_key,
                        rank,
                        size,
                        local_key: &mut local_key,
                        local_error: &mut local_error,
                    },
                )?;
            }
            RootEvent::Neither => std::thread::yield_now(),
        }
    }

    let results = if scheduler.failed() {
        Vec::new()
    } else {
        scheduler.into_results().map_err(scheduler_error)?
    };
    Ok((results, local_key, local_error))
}

/// Runs the hybrid root protocol with a caller-thread coordinator.
///
/// See [`hybrid_worker_loop`] for the one-lane fallback rule.
#[cfg(feature = "rayon")]
fn hybrid_root_loop<L, C, T, U, E, F>(
    comm: &C,
    options: &PmapOptions,
    root_items: Vec<T>,
    lanes: &mut L,
    f: &F,
) -> Result<RootOutcome<U>, PmapError>
where
    L: Lanes,
    C: Communicator,
    T: Serialize + DeserializeOwned + Send,
    U: Serialize + DeserializeOwned + Send,
    E: Display + Send,
    F: Fn(T, &mut L::Lane<'_>) -> Result<U, E> + Sync,
{
    let queue = LocalQueue::<T, U>::new();
    let mut root_input = Some(root_items);
    let concurrent = lanes.run_with_coordinator(
        |control| {
            let mut sink = ConcurrentSink {
                queue: &queue,
                control,
            };
            hybrid_root_protocol(
                comm,
                options,
                root_input.take().expect("root input is passed once"),
                &mut sink,
            )
        },
        // A lane callback returns only infrastructure failures; item and batch
        // failures travel through the queue as data.
        |_index, lane, control| {
            run_lane(&queue, f, control, lane);
            Ok::<(), PmapError>(())
        },
    );
    match concurrent {
        Ok(outcome) => Ok(outcome),
        Err(LanesError::SingleLane) => {
            let mut sink = InlineSink {
                run_batch: |batch| run_batch_inline(lanes, f, batch),
                outcome: None,
            };
            hybrid_root_protocol(
                comm,
                options,
                root_input.take().expect("root input is passed once"),
                &mut sink,
            )
        }
        Err(LanesError::Coordinator(error)) => Err(error),
        Err(LanesError::Executor(error)) => Err(lane_executor_failure(error)),
        Err(LanesError::Reentrant) => Err(lane_executor_failure(
            "executor cannot run a coordinator from one of its own workers",
        )),
        Err(LanesError::Lane(error)) => Err(error),
    }
}

/// Runs one hybrid `pmap` call on an already duplicated private communicator.
#[cfg(feature = "rayon")]
fn run_collective<L, C, T, U, E, F>(
    comm: &C,
    lanes: &mut L,
    options: &PmapOptions,
    root_items: Option<Vec<T>>,
    f: &F,
) -> Result<Option<Vec<U>>, PmapError>
where
    L: Lanes,
    C: Communicator,
    T: Serialize + DeserializeOwned + Send,
    U: Serialize + DeserializeOwned + Send,
    E: Display + Send,
    F: Fn(T, &mut L::Lane<'_>) -> Result<U, E> + Sync,
{
    let rank = mpi_call!(comm.rank());
    let outcome = if rank == options.root {
        hybrid_root_loop(
            comm,
            options,
            root_items.expect("successful preflight guarantees root input"),
            lanes,
            f,
        )
    } else {
        hybrid_worker_loop(
            comm,
            options.root,
            options.local_mode,
            options.prefetch,
            lanes,
            f,
        )
        .map(|(key, error)| (Vec::new(), key, error))
    };
    let (results, local_key, local_error) = match outcome {
        Ok(outcome) => outcome,
        Err(_) => mpi_call!(comm.abort(72)),
    };

    let selected = convergence(comm, local_key);
    let error = match broadcast_error(comm, selected, local_key, local_error) {
        Ok(error) => error,
        Err(_) => mpi_call!(comm.abort(73)),
    };
    match error {
        Some((kind, message)) => Err(PmapError::keyed(kind, selected, message)),
        None if rank == options.root => Ok(Some(results)),
        None => Ok(None),
    }
}

/// Dynamically maps root-owned items across MPI ranks and caller-supplied lanes
/// for each rank's local batches.
///
/// Every rank must call this collective in the same order with identical
/// options. Only `options.root` supplies items and receives ordered results.
/// The call must originate on the MPI initialization/main thread, and MPI must
/// provide `MPI_THREAD_FUNNELED` or stronger, because the coordinator keeps MPI
/// progress on that thread while the lanes compute. Do not enter it from a
/// worker of the executor's own pool.
///
/// The coordinator is the protocol loop on the calling thread: it dispatches
/// scheduler batches, keeps MPI progress, and applies the root's immediate abort
/// rules before anything is joined. Lanes execute the local batches with the
/// requested [`LocalMode`]: `Outer` spreads a batch over the lanes, while
/// `Sequential` and `Inner` hand the batch to one lane. When
/// [`PmapOptions::prefetch`] is true, each remote domain may hold one additional
/// batch so blocking task/result transfer overlaps adjacent batch execution; the
/// root domain remains capacity one.
///
/// The per-item callback receives the lane resource of the lane that runs the
/// item (`&mut L::Lane<'_>`), so a backend adapter can hand its own per-worker
/// context, such as a tenferro child session, to the callback. [`pmap`] adapts
/// the ordinary `Fn(T)` shape to `RayonLanes`' unit lane.
///
/// A lanes executor that reports [`LanesError::SingleLane`] falls back
/// explicitly to executing each batch on the coordinator thread through
/// [`Lanes::run`] between MPI progress steps. This is the documented one-lane
/// path; it never happens silently and never runs work twice.
///
/// # Errors
///
/// Returns [`PmapErrorKind::Preflight`] when the collective preconditions fail
/// (including entry from a Rayon worker), [`PmapErrorKind::User`] with a
/// converged error key when a callback fails, [`PmapErrorKind::Wire`] or
/// [`PmapErrorKind::Protocol`] for transport, protocol, or lane-executor
/// failures, and [`PmapErrorKind::Reentrant`] when `pmap` is already active on
/// this thread.
#[cfg(feature = "rayon")]
pub fn pmap_lanes<L, C, T, U, E, F>(
    world: &C,
    lanes: &mut L,
    options: PmapOptions,
    root_items: Option<Vec<T>>,
    f: F,
) -> Result<Option<Vec<U>>, PmapError>
where
    L: Lanes,
    C: Communicator,
    T: Serialize + DeserializeOwned + Send,
    U: Serialize + DeserializeOwned + Send,
    E: Display + Send,
    F: Fn(T, &mut L::Lane<'_>) -> Result<U, E> + Sync,
{
    if in_rayon_worker_context() {
        return Err(PmapError::new(
            PmapErrorKind::Preflight,
            "hybrid pmap cannot enter from a Rayon worker",
        ));
    }
    execute(world, Some(lanes), options, root_items, &f)
}

/// Runs one hybrid collective call on an already built lane executor.
///
/// A missing or lane-less executor is folded into collective preflight instead
/// of returning locally, so peers are never stranded.
#[cfg(feature = "rayon")]
fn execute<L, C, T, U, E, F>(
    world: &C,
    lanes: Option<&mut L>,
    options: PmapOptions,
    root_items: Option<Vec<T>>,
    f: &F,
) -> Result<Option<Vec<U>>, PmapError>
where
    L: Lanes,
    C: Communicator,
    T: Serialize + DeserializeOwned + Send,
    U: Serialize + DeserializeOwned + Send,
    E: Display + Send,
    F: Fn(T, &mut L::Lane<'_>) -> Result<U, E> + Sync,
{
    let _active = Active::acquire()?;
    let executor_valid = lanes.as_ref().is_some_and(|lanes| lanes.lane_count() > 0);
    let execution_valid = executor_valid
        && mpi_call!(mpi_backend::environment::threading_support())
            >= mpi_backend::environment::Threading::Funneled;
    let comm = mpi_call!(world.duplicate());
    preflight(&comm, &options, root_items.as_ref(), execution_valid)?;
    let lanes = lanes.expect("successful preflight guarantees a lane executor");
    run_collective(&comm, lanes, &options, root_items, f)
}
impl PmapErrorKind {
    fn code(self) -> i32 {
        match self {
            Self::Reentrant => 0,
            Self::Preflight => 1,
            Self::User => 2,
            Self::Wire => 3,
            Self::Protocol => 4,
        }
    }

    fn from_code(code: i32) -> Option<Self> {
        Some(match code {
            0 => Self::Reentrant,
            1 => Self::Preflight,
            2 => Self::User,
            3 => Self::Wire,
            4 => Self::Protocol,
            _ => return None,
        })
    }
}

/// Dynamically maps root-owned items across all ranks of `world`.
///
/// Every rank must call this function in the same collective order with the
/// same communicator and options. Only `options.root` supplies `Some(items)`
/// and receives `Ok(Some(results))`; every other rank supplies `None` and
/// receives `Ok(None)`. Results preserve input order. The root rank must lie
/// in `[0, world.size())`.
///
/// This MPI-only entry is synchronous and adds no `Send`, `Sync`, or `'static`
/// bounds to the callback or values. It rejects [`PmapOptions::prefetch`] during
/// collective preflight.
#[cfg(not(feature = "rayon"))]
pub fn pmap<C, T, U, E, F>(
    world: &C,
    domain: &Domain,
    options: PmapOptions,
    root_items: Option<Vec<T>>,
    f: F,
) -> Result<Option<Vec<U>>, PmapError>
where
    C: Communicator,
    T: Serialize + DeserializeOwned,
    U: Serialize + DeserializeOwned,
    E: Display,
    F: FnMut(T) -> Result<U, E>,
{
    let _active = Active::acquire()?;
    let admission = domain.try_admit();
    let comm = mpi_call!(world.duplicate());
    preflight(
        &comm,
        &options,
        root_items.as_ref(),
        admission.is_ok()
            && domain.id().get() == 0
            && domain.worker_count() == 1
            && options.local_mode == LocalMode::Sequential
            && !options.prefetch,
    )?;
    let _admission = admission.expect("successful preflight guarantees domain admission");

    let rank = mpi_call!(comm.rank());
    let outcome = if rank == options.root {
        root_loop(&comm, &options, root_items.unwrap(), f)
    } else {
        worker_loop(&comm, options.root, f).map(|(key, error)| (Vec::new(), key, error))
    };
    let (results, local_key, local_error) = match outcome {
        Ok(outcome) => outcome,
        Err(_) => mpi_call!(comm.abort(72)),
    };

    let selected = convergence(&comm, local_key);
    let error = match broadcast_error(&comm, selected, local_key, local_error) {
        Ok(error) => error,
        Err(_) => mpi_call!(comm.abort(73)),
    };
    match error {
        Some((kind, message)) => Err(PmapError::keyed(kind, selected, message)),
        None if rank == options.root => Ok(Some(results)),
        None => Ok(None),
    }
}

/// Dynamically maps root-owned items across MPI ranks and each rank's explicit
/// Rayon domain.
///
/// Every rank must call this collective in the same order with identical
/// options. Only `options.root` supplies items and receives ordered results.
/// The call must originate on the MPI initialization/main thread outside every
/// Rayon worker, and MPI must provide `MPI_THREAD_FUNNELED` or stronger.
/// Callback state may borrow non-`'static` data while satisfying the stated
/// Rayon `Send`/`Sync` bounds; values retain the serialization bounds required
/// by remote execution.
///
/// Local execution is [`pmap_lanes`] with [`RayonLanes`]: the coordinator keeps MPI progress on the calling thread
/// while lanes execute local batches, so when [`PmapOptions::prefetch`] is true
/// each remote domain may hold one additional batch and blocking task/result
/// transfer overlaps adjacent batch execution. The root domain remains capacity
/// one, and all MPI calls remain on the calling thread.
///
/// # Errors
///
/// Returns [`PmapErrorKind::Preflight`] when a rank lacks a pooled, admitted
/// domain, enters from a Rayon worker, or disagrees about options;
/// [`PmapErrorKind::User`] with a converged error key when a callback fails;
/// [`PmapErrorKind::Wire`] or [`PmapErrorKind::Protocol`] for transport or
/// protocol failures; and [`PmapErrorKind::Reentrant`] when `pmap` is already
/// active on this thread.
#[cfg(feature = "rayon")]
pub fn pmap<C, T, U, E, F>(
    world: &C,
    domain: &Domain,
    options: PmapOptions,
    root_items: Option<Vec<T>>,
    f: F,
) -> Result<Option<Vec<U>>, PmapError>
where
    C: Communicator,
    T: Serialize + DeserializeOwned + Send,
    U: Serialize + DeserializeOwned + Send,
    E: Display + Send,
    F: Fn(T) -> Result<U, E> + Send + Sync,
{
    if in_rayon_worker_context() {
        return Err(PmapError::new(
            PmapErrorKind::Preflight,
            "hybrid pmap cannot enter from a Rayon worker",
        ));
    }
    // Building the executor admits the domain. A construction failure is a
    // collective preflight failure, never a local early return that would
    // strand peers.
    let mut lanes = RayonLanes::new(domain).ok();
    execute(
        world,
        lanes.as_mut(),
        options,
        root_items,
        &|item: T, _lane: &mut ()| f(item),
    )
}

#[cfg(all(test, not(feature = "rayon")))]
mod mpi_fault_tests {
    use super::*;
    use serde::{Deserialize, Serialize, Serializer};
    use std::path::PathBuf;

    #[derive(Debug, Deserialize)]
    struct NoCodec(i32);

    impl Serialize for NoCodec {
        fn serialize<S: Serializer>(&self, _serializer: S) -> Result<S::Ok, S::Error> {
            Err(serde::ser::Error::custom(
                "codec must not run at world size one",
            ))
        }
    }

    #[test]
    fn size_one_root_paths_bypass_codec() {
        if std::env::var_os("HATAORI_CODEC_TEST").is_none() {
            return;
        }
        let universe =
            crate::mpi_backend::initialize().expect("codec test owns MPI initialization");
        let world = universe.world();
        assert_eq!(mpi_call!(world.size()), 1);
        crate::wire::reset_codec_counts();

        let domain = Domain::sequential();
        let mapped = pmap(
            &world,
            &domain,
            PmapOptions::default(),
            Some(vec![NoCodec(1)]),
            |value| Ok::<_, String>(NoCodec(value.0 + 1)),
        )
        .unwrap()
        .unwrap();
        assert_eq!(mapped[0].0, 2);
        assert_eq!(crate::broadcast(&world, 0, Some(NoCodec(3))).unwrap().0, 3);
        assert_eq!(
            crate::scatter(&world, 0, Some(vec![NoCodec(4)])).unwrap().0,
            4
        );
        assert_eq!(
            crate::gather(&world, 0, NoCodec(5)).unwrap().unwrap()[0].0,
            5
        );
        assert_eq!(crate::wire::codec_counts(), (0, 0));
    }

    #[test]
    fn simultaneous_user_and_decode_faults_drain_and_reuse() {
        let Some(base) = std::env::var_os("HATAORI_FAULT_MARKER") else {
            return;
        };
        let universe =
            crate::mpi_backend::initialize().expect("fault test owns MPI initialization");
        let world = universe.world();
        let rank = mpi_call!(world.rank());
        let size = mpi_call!(world.size());
        assert!(size >= 4, "fault test requires at least four ranks");
        let base = PathBuf::from(base);
        let assigned = base.with_extension("assigned");
        let errored = base.with_extension("error");
        if rank == 0 {
            let _ = std::fs::remove_file(&assigned);
            let _ = std::fs::remove_file(&errored);
        }
        mpi_call!(world.barrier());

        reset_fault_trace();
        let domain = Domain::sequential();
        let options = PmapOptions::default();
        let control = pmap(
            &world,
            &domain,
            options,
            (rank == 0).then(|| (0..12).collect::<Vec<i32>>()),
            |item| Ok::<_, String>(item * 2),
        )
        .unwrap();
        if rank == 0 {
            assert_eq!(
                control.unwrap(),
                (0..12).map(|item| item * 2).collect::<Vec<_>>()
            );
        } else {
            assert!(control.is_none());
        }

        reset_fault_trace();
        FAULT_PHASE.store(true, Ordering::SeqCst);
        if rank == 1 {
            CORRUPT_NEXT_RESULT.store(true, Ordering::SeqCst);
        }
        mpi_call!(world.barrier());
        let assigned_for_callback = assigned.clone();
        let error_for_callback = errored.clone();
        let failure = pmap(
            &world,
            &domain,
            options,
            (rank == 0).then(|| (0..32).collect::<Vec<i32>>()),
            move |item| {
                match rank {
                    0 => std::thread::sleep(std::time::Duration::from_millis(100)),
                    1 => {
                        std::fs::write(&assigned_for_callback, b"rank1-running").unwrap();
                        wait_for_marker(&error_for_callback);
                    }
                    2 => {
                        std::fs::write(&error_for_callback, b"rank2-error").unwrap();
                        return Err::<i32, _>("simultaneous user failure".to_owned());
                    }
                    _ => {}
                }
                Ok(item)
            },
        )
        .unwrap_err();
        assert_eq!(failure.kind(), PmapErrorKind::User);
        assert_eq!(failure.message(), "simultaneous user failure");
        assert_eq!(failure.key().unwrap() % i64::from(size), 2);
        if rank == 0 {
            let assigned_mask = TEST_ASSIGNED_REMOTES.load(Ordering::SeqCst);
            assert_ne!(assigned_mask & (1_u64 << 1), 0);
            assert_ne!(assigned_mask & (1_u64 << 2), 0);
            assert_eq!(TEST_USER_RESULTS.load(Ordering::SeqCst), 1);
            assert_eq!(TEST_DECODE_FAILURES.load(Ordering::SeqCst), 1);
        }

        FAULT_PHASE.store(false, Ordering::SeqCst);
        mpi_call!(world.barrier());
        if rank == 0 {
            mpi_call!(world.process_at_rank(1).send_with_tag(&[77_u8], 77));
        } else if rank == 1 {
            let (message, _) = mpi_call!(world.process_at_rank(0).receive_vec_with_tag::<u8>(77));
            assert_eq!(message, vec![77]);
        }
        mpi_call!(world.barrier());

        let reused = pmap(
            &world,
            &domain,
            options,
            (rank == 0).then(|| vec![3_i32, 1, 4]),
            |item| Ok::<_, String>(item + 1),
        )
        .unwrap();
        if rank == 0 {
            assert_eq!(reused.unwrap(), vec![4, 2, 5]);
            std::fs::remove_file(&assigned).unwrap();
            std::fs::remove_file(&errored).unwrap();
        }
        mpi_call!(world.barrier());
    }
}

#[cfg(all(test, feature = "rayon"))]
mod tests {
    #[test]
    fn prefetch_is_opt_in() {
        assert!(!super::PmapOptions::default().prefetch);
    }

    #[test]
    fn root_event_chooser_alternates_when_both_sources_are_ready() {
        let mut prefer_local = true;
        let mut events = Vec::new();
        for _ in 0..4 {
            let (event, next) = super::choose_root_event(true, true, prefer_local);
            events.push(event);
            prefer_local = next;
        }
        assert_eq!(
            events,
            [
                super::RootEvent::Local,
                super::RootEvent::Remote,
                super::RootEvent::Local,
                super::RootEvent::Remote
            ]
        );
    }

    #[test]
    fn root_event_chooser_keeps_one_sided_progress() {
        let (local, turn) = super::choose_root_event(true, false, true);
        assert_eq!(local, super::RootEvent::Local);
        let (remote, _) = super::choose_root_event(false, true, turn);
        assert_eq!(remote, super::RootEvent::Remote);
    }
}
