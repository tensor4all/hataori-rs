use std::collections::{BTreeMap, HashSet, VecDeque};
use std::fmt;
use std::num::NonZeroUsize;

#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub(crate) struct BatchId(u64);

impl BatchId {
    const ZERO: Self = Self(0);

    pub(crate) const fn get(self) -> u64 {
        self.0
    }

    pub(crate) const fn from_raw(value: u64) -> Self {
        Self(value)
    }

    fn checked_increment(self) -> Result<Self, SchedulerError> {
        self.0
            .checked_add(1)
            .map(Self)
            .ok_or(SchedulerError::BatchIdOverflow)
    }
}

pub(crate) struct ScheduledItem<T> {
    original_index: usize,
    value: T,
}

impl<T> ScheduledItem<T> {
    fn new(original_index: usize, value: T) -> Self {
        Self {
            original_index,
            value,
        }
    }

    pub(crate) fn original_index(&self) -> usize {
        self.original_index
    }

    pub(crate) fn into_value(self) -> T {
        self.value
    }
}

pub(crate) struct Batch<T> {
    id: BatchId,
    items: Vec<ScheduledItem<T>>,
}

impl<T> Batch<T> {
    pub(crate) fn id(&self) -> BatchId {
        self.id
    }

    pub(crate) fn into_items(self) -> Vec<ScheduledItem<T>> {
        self.items
    }
}

pub(crate) struct ItemResult<U> {
    original_index: usize,
    value: U,
}

impl<U> ItemResult<U> {
    pub(crate) fn new(original_index: usize, value: U) -> Self {
        Self {
            original_index,
            value,
        }
    }

    pub(crate) fn original_index(&self) -> usize {
        self.original_index
    }

    pub(crate) fn into_value(self) -> U {
        self.value
    }
}

pub(crate) enum Dispatch<T> {
    Task(Batch<T>),
    Stop,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub(crate) enum Lane {
    Root,
    Remote(i32),
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub(crate) struct CompletionMeta {
    lane: Lane,
    batch_id: BatchId,
}

impl CompletionMeta {
    #[cfg(test)]
    pub(crate) const fn lane(self) -> Lane {
        self.lane
    }

    #[cfg(test)]
    pub(crate) const fn batch_id(self) -> BatchId {
        self.batch_id
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
enum LaneStateKind {
    Idle,
    Running,
    Stopped,
    Drained,
}

struct RunningMeta {
    batch_id: BatchId,
    original_indices: Vec<usize>,
}

enum LaneState {
    Idle,
    Running {
        current: RunningMeta,
        prefetched: Option<RunningMeta>,
        stopping: bool,
    },
    Stopped,
    Drained,
}

impl LaneState {
    fn kind(&self) -> LaneStateKind {
        match self {
            Self::Idle => LaneStateKind::Idle,
            Self::Running { .. } => LaneStateKind::Running,
            Self::Stopped => LaneStateKind::Stopped,
            Self::Drained => LaneStateKind::Drained,
        }
    }
}

#[derive(Debug, Eq, PartialEq)]
pub(crate) enum SchedulerError {
    InvalidRoot,
    InvalidRemote,
    DuplicateRemote,
    RootInRemotes,
    UnknownRank,
    WrongLaneState,
    BatchIdOverflow,
    BatchMismatch,
    ResultCount,
    IndexSequence,
    IndexOutOfBounds,
    AlreadyFilled,
    DuplicateResultIndex,
    IndexOverflow,
    NotFinished,
    Failed,
    MissingResult,
}

impl fmt::Display for SchedulerError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::InvalidRoot => "invalid root rank",
            Self::InvalidRemote => "invalid remote rank",
            Self::DuplicateRemote => "duplicate remote rank",
            Self::RootInRemotes => "root rank is listed as remote",
            Self::UnknownRank => "unknown scheduler rank",
            Self::WrongLaneState => "scheduler lane is in the wrong state",
            Self::BatchIdOverflow => "scheduler batch ID overflow",
            Self::BatchMismatch => "scheduler batch ID mismatch",
            Self::ResultCount => "scheduler result count mismatch",
            Self::IndexSequence => "scheduler result index sequence mismatch",
            Self::IndexOutOfBounds => "scheduler result index is out of bounds",
            Self::AlreadyFilled => "scheduler result slot is already filled",
            Self::DuplicateResultIndex => "scheduler result index is duplicated",
            Self::IndexOverflow => "scheduler input index overflow",
            Self::NotFinished => "scheduler is not finished",
            Self::Failed => "scheduler failed",
            Self::MissingResult => "scheduler result is missing",
        };
        formatter.write_str(message)
    }
}

impl std::error::Error for SchedulerError {}

pub(crate) struct Coordinator<T, U> {
    pending: VecDeque<ScheduledItem<T>>,
    ordered: Vec<Option<U>>,
    batch_size: NonZeroUsize,
    next_batch_id: BatchId,
    root: LaneState,
    remotes: BTreeMap<i32, LaneState>,
    prefetch: bool,
    failed: bool,
}

impl<T, U> Coordinator<T, U> {
    #[cfg(test)]
    pub(crate) fn new(
        root_rank: i32,
        remote_ranks: Vec<i32>,
        items: Vec<T>,
        batch_size: NonZeroUsize,
    ) -> Result<Self, SchedulerError> {
        Self::new_with_prefetch(root_rank, remote_ranks, items, batch_size, false)
    }

    pub(crate) fn new_with_prefetch(
        root_rank: i32,
        remote_ranks: Vec<i32>,
        items: Vec<T>,
        batch_size: NonZeroUsize,
        prefetch: bool,
    ) -> Result<Self, SchedulerError> {
        if root_rank < 0 {
            return Err(SchedulerError::InvalidRoot);
        }

        let mut remotes = BTreeMap::new();
        for rank in remote_ranks {
            if rank < 0 {
                return Err(SchedulerError::InvalidRemote);
            }
            if rank == root_rank {
                return Err(SchedulerError::RootInRemotes);
            }
            if remotes.insert(rank, LaneState::Idle).is_some() {
                return Err(SchedulerError::DuplicateRemote);
            }
        }

        let mut pending = VecDeque::with_capacity(items.len());
        for (index, value) in items.into_iter().enumerate() {
            u64::try_from(index).map_err(|_| SchedulerError::IndexOverflow)?;
            pending.push_back(ScheduledItem::new(index, value));
        }

        let input_len = pending.len();
        Ok(Self {
            pending,
            ordered: (0..input_len).map(|_| None).collect(),
            batch_size,
            next_batch_id: BatchId::ZERO,
            root: LaneState::Idle,
            remotes,
            prefetch,
            failed: false,
        })
    }

    pub(crate) fn on_remote_ready(&mut self, rank: i32) -> Result<Dispatch<T>, SchedulerError> {
        let prefetch_request = match self.remotes.get(&rank) {
            Some(LaneState::Idle) => false,
            Some(LaneState::Running {
                prefetched: None,
                stopping: false,
                ..
            }) if self.prefetch => true,
            Some(_) => return Err(SchedulerError::WrongLaneState),
            None => return Err(SchedulerError::UnknownRank),
        };

        if self.failed || self.pending.is_empty() {
            if prefetch_request {
                let state = self
                    .remotes
                    .get_mut(&rank)
                    .ok_or(SchedulerError::UnknownRank)?;
                match state {
                    LaneState::Running { stopping, .. } => *stopping = true,
                    _ => unreachable!("prefetch request state was validated"),
                }
            } else {
                self.remotes.insert(rank, LaneState::Stopped);
            }
            return Ok(Dispatch::Stop);
        }

        let batch = self.make_batch()?.expect("pending work was checked above");
        let metadata = Self::metadata(&batch);
        if prefetch_request {
            let state = self
                .remotes
                .get_mut(&rank)
                .ok_or(SchedulerError::UnknownRank)?;
            match state {
                LaneState::Running { prefetched, .. } => *prefetched = Some(metadata),
                _ => unreachable!("prefetch request state was validated"),
            }
        } else {
            self.remotes.insert(
                rank,
                LaneState::Running {
                    current: metadata,
                    prefetched: None,
                    stopping: false,
                },
            );
        }
        Ok(Dispatch::Task(batch))
    }

    pub(crate) fn next_root_batch(&mut self) -> Result<Option<Batch<T>>, SchedulerError> {
        if self.root.kind() == LaneStateKind::Running {
            return Err(SchedulerError::WrongLaneState);
        }
        if self.root.kind() == LaneStateKind::Stopped {
            return Ok(None);
        }
        if self.failed || self.pending.is_empty() {
            self.root = LaneState::Stopped;
            return Ok(None);
        }

        let batch = self.make_batch()?.expect("pending work was checked above");
        let metadata = Self::metadata(&batch);
        self.root = LaneState::Running {
            current: metadata,
            prefetched: None,
            stopping: false,
        };
        Ok(Some(batch))
    }

    pub(crate) fn on_root_success(
        &mut self,
        batch_id: BatchId,
        results: Vec<ItemResult<U>>,
    ) -> Result<CompletionMeta, SchedulerError> {
        let metadata = match &self.root {
            LaneState::Running {
                current,
                prefetched: None,
                stopping: false,
            } => current,
            _ => return Err(SchedulerError::WrongLaneState),
        };
        self.validate_completion(metadata, batch_id, &results)?;
        let completion = CompletionMeta {
            lane: Lane::Root,
            batch_id,
        };
        self.root = LaneState::Idle;
        if !self.failed {
            for result in results {
                let original_index = result.original_index();
                self.ordered[original_index] = Some(result.into_value());
            }
        }
        Ok(completion)
    }

    pub(crate) fn on_remote_success(
        &mut self,
        rank: i32,
        batch_id: BatchId,
        results: Vec<ItemResult<U>>,
    ) -> Result<CompletionMeta, SchedulerError> {
        let metadata = self.remote_current(rank)?;
        self.validate_completion(metadata, batch_id, &results)?;
        let completion = CompletionMeta {
            lane: Lane::Remote(rank),
            batch_id,
        };
        self.advance_remote(rank)?;
        if !self.failed {
            for result in results {
                let original_index = result.original_index();
                self.ordered[original_index] = Some(result.into_value());
            }
        }
        Ok(completion)
    }

    pub(crate) fn on_root_error(
        &mut self,
        batch_id: BatchId,
    ) -> Result<CompletionMeta, SchedulerError> {
        let metadata = match &self.root {
            LaneState::Running {
                current,
                prefetched: None,
                stopping: false,
            } => current,
            _ => return Err(SchedulerError::WrongLaneState),
        };
        if metadata.batch_id != batch_id {
            return Err(SchedulerError::BatchMismatch);
        }
        self.root = LaneState::Idle;
        self.failed = true;
        self.pending.clear();
        Ok(CompletionMeta {
            lane: Lane::Root,
            batch_id,
        })
    }

    pub(crate) fn on_remote_error(
        &mut self,
        rank: i32,
        batch_id: BatchId,
    ) -> Result<CompletionMeta, SchedulerError> {
        if self.remote_current(rank)?.batch_id != batch_id {
            return Err(SchedulerError::BatchMismatch);
        }
        self.advance_remote(rank)?;
        self.failed = true;
        self.pending.clear();
        Ok(CompletionMeta {
            lane: Lane::Remote(rank),
            batch_id,
        })
    }

    pub(crate) fn on_remote_protocol_error(
        &mut self,
        rank: i32,
    ) -> Result<CompletionMeta, SchedulerError> {
        let batch_id = self.remote_current(rank)?.batch_id;
        self.advance_remote(rank)?;
        self.failed = true;
        self.pending.clear();
        Ok(CompletionMeta {
            lane: Lane::Remote(rank),
            batch_id,
        })
    }

    fn remote_current(&self, rank: i32) -> Result<&RunningMeta, SchedulerError> {
        match self.remotes.get(&rank) {
            Some(LaneState::Running { current, .. }) => Ok(current),
            Some(_) => Err(SchedulerError::WrongLaneState),
            None => Err(SchedulerError::UnknownRank),
        }
    }

    fn advance_remote(&mut self, rank: i32) -> Result<(), SchedulerError> {
        let state = self
            .remotes
            .get_mut(&rank)
            .ok_or(SchedulerError::UnknownRank)?;
        let valid = matches!(
            state,
            LaneState::Running {
                prefetched: Some(_),
                stopping: false,
                ..
            } | LaneState::Running {
                prefetched: None,
                ..
            }
        );
        if !valid {
            return Err(SchedulerError::WrongLaneState);
        }
        let prior = std::mem::replace(state, LaneState::Idle);
        *state = match prior {
            LaneState::Running {
                prefetched: Some(current),
                stopping: false,
                ..
            } => LaneState::Running {
                current,
                prefetched: None,
                stopping: false,
            },
            LaneState::Running {
                prefetched: None,
                stopping: true,
                ..
            } => LaneState::Stopped,
            LaneState::Running {
                prefetched: None,
                stopping: false,
                ..
            } => LaneState::Idle,
            _ => unreachable!("remote running state was validated"),
        };
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn fail(&mut self) {
        self.failed = true;
        self.pending.clear();
    }

    pub(crate) fn on_remote_drain(&mut self, rank: i32) -> Result<(), SchedulerError> {
        let state = self.remotes.get(&rank).ok_or(SchedulerError::UnknownRank)?;
        if state.kind() != LaneStateKind::Stopped {
            return Err(SchedulerError::WrongLaneState);
        }
        self.remotes.insert(rank, LaneState::Drained);
        Ok(())
    }

    pub(crate) fn is_quiescent(&self) -> bool {
        self.pending.is_empty() && self.running_count() == 0
    }

    pub(crate) fn is_finished(&self) -> bool {
        self.root.kind() == LaneStateKind::Stopped
            && self
                .remotes
                .values()
                .all(|state| state.kind() == LaneStateKind::Drained)
            && self.running_count() == 0
    }

    pub(crate) fn into_results(self) -> Result<Vec<U>, SchedulerError> {
        if self.failed {
            return Err(SchedulerError::Failed);
        }
        if !self.is_finished() {
            return Err(SchedulerError::NotFinished);
        }
        self.ordered
            .into_iter()
            .map(|result| result.ok_or(SchedulerError::MissingResult))
            .collect()
    }

    #[cfg(test)]
    fn pending_count(&self) -> usize {
        self.pending.len()
    }

    fn running_count(&self) -> usize {
        usize::from(self.root.kind() == LaneStateKind::Running)
            + self
                .remotes
                .values()
                .filter(|state| state.kind() == LaneStateKind::Running)
                .count()
    }

    #[cfg(test)]
    fn prefetched_count(&self) -> usize {
        self.remotes
            .values()
            .filter(|state| {
                matches!(
                    state,
                    LaneState::Running {
                        prefetched: Some(_),
                        ..
                    }
                )
            })
            .count()
    }

    #[cfg(test)]
    fn result_count(&self) -> usize {
        self.ordered
            .iter()
            .filter(|result| result.is_some())
            .count()
    }

    pub(crate) fn failed(&self) -> bool {
        self.failed
    }

    fn make_batch(&mut self) -> Result<Option<Batch<T>>, SchedulerError> {
        if self.failed || self.pending.is_empty() {
            return Ok(None);
        }
        let id = self.next_batch_id;
        let next_id = id.checked_increment()?;
        let count = self.batch_size.get().min(self.pending.len());
        let items = self.pending.drain(..count).collect();
        self.next_batch_id = next_id;
        Ok(Some(Batch { id, items }))
    }

    fn metadata(batch: &Batch<T>) -> RunningMeta {
        RunningMeta {
            batch_id: batch.id,
            original_indices: batch
                .items
                .iter()
                .map(ScheduledItem::original_index)
                .collect(),
        }
    }

    fn validate_completion(
        &self,
        metadata: &RunningMeta,
        batch_id: BatchId,
        results: &[ItemResult<U>],
    ) -> Result<(), SchedulerError> {
        if metadata.batch_id != batch_id {
            return Err(SchedulerError::BatchMismatch);
        }
        if metadata.original_indices.len() != results.len() {
            return Err(SchedulerError::ResultCount);
        }

        let mut seen = HashSet::with_capacity(results.len());
        for (position, result) in results.iter().enumerate() {
            let index = result.original_index;
            if index >= self.ordered.len() {
                return Err(SchedulerError::IndexOutOfBounds);
            }
            if !seen.insert(index) {
                return Err(SchedulerError::DuplicateResultIndex);
            }
            if metadata.original_indices[position] != index {
                return Err(SchedulerError::IndexSequence);
            }
            if self.ordered[index].is_some() {
                return Err(SchedulerError::AlreadyFilled);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
