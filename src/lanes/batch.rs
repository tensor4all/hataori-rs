//! Bounded local batch queue shared by the hybrid `pmap` coordinator and lanes.
//!
//! Exactly one batch is resident for the lanes at a time, which is the same
//! capacity the P0/P1 protocol allows per domain (`running <= 1`, plus at most
//! one prefetched batch held by the coordinator rather than by a lane). Lanes
//! claim items from that batch, so `LocalMode::Outer` spreads a batch over the
//! lanes while `Sequential` and `Inner` hand the whole batch to one lane.
//!
//! Waiting never uses a condition variable. An idle lane yields to the executor
//! on every poll so that nested same-pool work still finds a worker, and the
//! coordinator rechecks cancellation instead of blocking unconditionally.

use crate::domain::LocalMode;
use crate::lanes::LaneControl;
use crate::map::truncate_message;
use std::fmt::Display;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{Mutex, MutexGuard, PoisonError};
use std::time::Duration;

/// How long an idle lane or a waiting coordinator parks before rechecking.
const IDLE_PARK: Duration = Duration::from_micros(200);

/// Backs off one idle poll while keeping the executor able to run nested work.
pub(crate) fn idle_backoff() {
    // Run one pending executor job even when the caller is not a pool worker.
    let _ = rayon::yield_now();
    std::thread::park_timeout(IDLE_PARK);
}

/// One local batch handed to the lanes.
pub(crate) struct LocalBatch<T> {
    pub(crate) batch_id: u64,
    pub(crate) item_count: usize,
    mode: LocalMode,
    items: Vec<Option<(usize, T)>>,
    next: usize,
    remaining: usize,
    claimed: bool,
}

impl<T> LocalBatch<T> {
    pub(crate) fn new(batch_id: u64, mode: LocalMode, items: Vec<(usize, T)>) -> Self {
        let remaining = items.len();
        Self {
            batch_id,
            item_count: remaining,
            mode,
            items: items.into_iter().map(Some).collect(),
            next: 0,
            remaining,
            claimed: false,
        }
    }
}

/// The work one lane claimed from the resident batch.
pub(crate) enum LaneWork<T> {
    /// One item of an `Outer` batch.
    Item {
        batch_id: u64,
        index: usize,
        item: T,
    },
    /// A whole `Sequential` or `Inner` batch.
    Batch {
        batch_id: u64,
        items: Vec<(usize, T)>,
    },
}

/// The result of one local batch.
pub(crate) enum LocalBatchResult<U> {
    /// Every item succeeded; entries are in original input order.
    Values(Vec<(usize, U)>),
    /// A callback failed; the lowest failing input index wins.
    CallbackError { index: usize, message: String },
    /// A transported batch could not be decoded before it reached a lane.
    WireError { index: usize, message: String },
    /// A callback panicked; the coordinator aborts the job.
    Panic,
}

/// The completed outcome of one local batch.
pub(crate) struct LocalBatchOutcome<U> {
    pub(crate) batch_id: u64,
    pub(crate) item_count: usize,
    pub(crate) result: LocalBatchResult<U>,
}

impl<U> LocalBatchOutcome<U> {
    pub(crate) fn wire_error(
        batch_id: u64,
        item_count: usize,
        index: usize,
        message: String,
    ) -> Self {
        Self {
            batch_id,
            item_count,
            result: LocalBatchResult::WireError { index, message },
        }
    }
}

struct BatchSlot<T, U> {
    batch: LocalBatch<T>,
    values: Vec<(usize, U)>,
    error: Option<(usize, String)>,
    panicked: bool,
}

struct QueueState<T, U> {
    slot: Option<BatchSlot<T, U>>,
    // One resident batch produces exactly one outcome, which the coordinator
    // takes before it submits the next batch.
    outcome: Option<LocalBatchOutcome<U>>,
    closed: bool,
}

/// Bounded hand-off between one coordinator and its lanes.
pub(crate) struct LocalQueue<T, U> {
    state: Mutex<QueueState<T, U>>,
}

impl<T, U> LocalQueue<T, U> {
    pub(crate) fn new() -> Self {
        Self {
            state: Mutex::new(QueueState {
                slot: None,
                outcome: None,
                closed: false,
            }),
        }
    }

    pub(crate) fn submit(&self, batch: LocalBatch<T>) {
        let mut state = lock(&self.state);
        assert!(
            state.slot.is_none(),
            // INVARIANT: the protocol keeps one local batch resident per domain.
            "a local batch is already resident for the lanes"
        );
        assert!(
            !state.closed,
            "local batch submitted after the queue closed"
        );
        assert!(
            state.outcome.is_none(),
            "a completed local outcome was not collected"
        );
        state.slot = Some(BatchSlot {
            batch,
            values: Vec::new(),
            error: None,
            panicked: false,
        });
    }

    /// Closes the queue for new work; lanes drain the resident batch and exit.
    pub(crate) fn close(&self) {
        lock(&self.state).closed = true;
    }

    pub(crate) fn is_closed(&self) -> bool {
        lock(&self.state).closed
    }

    /// Claims the next unit of work, if any.
    pub(crate) fn claim(&self) -> Option<LaneWork<T>> {
        let mut state = lock(&self.state);
        let slot = state.slot.as_mut()?;
        match slot.batch.mode {
            LocalMode::Outer => {
                let index = slot.batch.next;
                let entry = slot.batch.items.get_mut(index)?.take();
                let (original_index, item) = entry?;
                slot.batch.next += 1;
                Some(LaneWork::Item {
                    batch_id: slot.batch.batch_id,
                    index: original_index,
                    item,
                })
            }
            LocalMode::Sequential | LocalMode::Inner => {
                if slot.batch.claimed {
                    return None;
                }
                slot.batch.claimed = true;
                let items = slot
                    .batch
                    .items
                    .iter_mut()
                    .filter_map(|entry| entry.take())
                    .collect();
                Some(LaneWork::Batch {
                    batch_id: slot.batch.batch_id,
                    items,
                })
            }
        }
    }

    pub(crate) fn complete_value(&self, batch_id: u64, index: usize, value: U) {
        let mut state = lock(&self.state);
        {
            let Some(slot) = state.slot.as_mut() else {
                return;
            };
            if slot.batch.batch_id != batch_id {
                return;
            }
            slot.values.push((index, value));
            slot.batch.remaining -= 1;
            if slot.batch.remaining != 0 {
                return;
            }
        }
        state.finish();
    }

    pub(crate) fn complete_error(&self, batch_id: u64, index: usize, message: String) {
        let mut state = lock(&self.state);
        let stop = {
            let Some(slot) = state.slot.as_mut() else {
                return;
            };
            if slot.batch.batch_id != batch_id {
                return;
            }
            if slot.error.as_ref().is_none_or(|(first, _)| index < *first) {
                slot.error = Some((index, message));
            }
            slot.batch.remaining -= 1;
            slot.batch.remaining == 0 || slot.batch.mode != LocalMode::Outer
        };
        if stop {
            state.finish();
        }
    }

    pub(crate) fn complete_panic(&self, batch_id: u64) {
        let mut state = lock(&self.state);
        {
            let Some(slot) = state.slot.as_mut() else {
                return;
            };
            if slot.batch.batch_id != batch_id {
                return;
            }
            slot.panicked = true;
        }
        state.finish();
    }

    /// Takes the completed outcome of `batch_id`, if the lanes produced it.
    pub(crate) fn take_outcome(&self, batch_id: u64) -> Option<LocalBatchOutcome<U>> {
        let mut state = lock(&self.state);
        match state.outcome.as_ref() {
            Some(outcome) if outcome.batch_id == batch_id => state.outcome.take(),
            _ => None,
        }
    }
}

impl<T, U> QueueState<T, U> {
    fn finish(&mut self) {
        let Some(slot) = self.slot.take() else {
            return;
        };
        let result = if slot.panicked {
            LocalBatchResult::Panic
        } else if let Some((index, message)) = slot.error {
            LocalBatchResult::CallbackError { index, message }
        } else {
            let mut values = slot.values;
            values.sort_unstable_by_key(|(index, _)| *index);
            LocalBatchResult::Values(values)
        };
        self.outcome = Some(LocalBatchOutcome {
            batch_id: slot.batch.batch_id,
            item_count: slot.batch.item_count,
            result,
        });
    }
}

/// Runs one lane of a local batch queue until the queue closes or cancels.
///
/// A callback panic becomes a [`LocalBatchResult::Panic`] outcome for the batch
/// instead of unwinding the lane, so the coordinator can abort immediately as
/// the protocol requires.
pub(crate) fn run_lane<T, U, E, F, Lane>(
    queue: &LocalQueue<T, U>,
    f: &F,
    control: &LaneControl<'_>,
    lane: &mut Lane,
) where
    T: Send,
    U: Send,
    E: Display + Send,
    F: Fn(T, &mut Lane) -> Result<U, E>,
{
    loop {
        if control.cancelled() {
            return;
        }
        match queue.claim() {
            Some(LaneWork::Item {
                batch_id,
                index,
                item,
            }) => match catch_unwind(AssertUnwindSafe(|| f(item, lane))) {
                Ok(Ok(value)) => queue.complete_value(batch_id, index, value),
                Ok(Err(error)) => {
                    queue.complete_error(batch_id, index, truncate_message(error.to_string()))
                }
                Err(_) => queue.complete_panic(batch_id),
            },
            Some(LaneWork::Batch { batch_id, items }) => {
                for (index, item) in items {
                    if control.cancelled() {
                        return;
                    }
                    match catch_unwind(AssertUnwindSafe(|| f(item, lane))) {
                        Ok(Ok(value)) => queue.complete_value(batch_id, index, value),
                        Ok(Err(error)) => {
                            queue.complete_error(
                                batch_id,
                                index,
                                truncate_message(error.to_string()),
                            );
                            break;
                        }
                        Err(_) => {
                            queue.complete_panic(batch_id);
                            break;
                        }
                    }
                }
            }
            None => {
                if queue.is_closed() {
                    return;
                }
                idle_backoff();
            }
        }
    }
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}
