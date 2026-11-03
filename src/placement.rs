use crate::map::{truncate_message, MAX_ERROR_MESSAGE_BYTES};
use crate::mpi_backend::collective::SystemOperation;
use crate::mpi_backend::traits::*;
use crate::mpi_check::{is_thread_main, mpi_call};
use crate::wire::{self, Header, MessageKind, MessageStatus};
use serde::{de::DeserializeOwned, Serialize};
use std::fmt::{self, Display, Formatter};

const HEADER_TAG: i32 = 2;
const PAYLOAD_TAG: i32 = 3;
const FRAME_ID: u64 = 0;
const FRAME_ITEMS: u64 = 1;

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
enum Operation {
    Broadcast,
    Scatter,
    Gather,
}

impl Operation {
    const fn code(self) -> i32 {
        match self {
            Self::Broadcast => 0,
            Self::Scatter => 1,
            Self::Gather => 2,
        }
    }

    const fn message_kind(self) -> MessageKind {
        match self {
            Self::Broadcast => MessageKind::Broadcast,
            Self::Scatter => MessageKind::Scatter,
            Self::Gather => MessageKind::Gather,
        }
    }
}

/// Stable category of a collective placement failure.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum PlacementErrorKind {
    /// Collective arguments or calling-thread requirements disagreed or were invalid.
    Preflight,
    /// Serialization, deserialization, or wire framing failed.
    Wire,
    /// A received placement frame violated its operation schema.
    Protocol,
}

/// Error converged across every rank in a placement collective.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct PlacementError {
    kind: PlacementErrorKind,
    rank: Option<i32>,
    message: String,
}

impl PlacementError {
    fn new(kind: PlacementErrorKind, rank: Option<i32>, message: impl Display) -> Self {
        Self {
            kind,
            rank,
            message: truncate_message(message.to_string()),
        }
    }

    /// Returns the stable failure category.
    pub const fn kind(&self) -> PlacementErrorKind {
        self.kind
    }

    /// Returns the lowest rank that reported the converged failure.
    pub const fn rank(&self) -> Option<i32> {
        self.rank
    }

    /// Returns the bounded diagnostic message.
    pub fn message(&self) -> &str {
        &self.message
    }
}

impl Display for PlacementError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.message())
    }
}

impl std::error::Error for PlacementError {}

struct Failure {
    kind: PlacementErrorKind,
    message: String,
}

impl Failure {
    fn new(kind: PlacementErrorKind, message: impl Display) -> Self {
        Self {
            kind,
            message: truncate_message(message.to_string()),
        }
    }
}

fn record_failure(slot: &mut Option<Failure>, failure: Failure) {
    if slot.is_none() {
        *slot = Some(failure);
    }
}

impl PlacementErrorKind {
    const fn code(self) -> i32 {
        match self {
            Self::Preflight => 0,
            Self::Wire => 1,
            Self::Protocol => 2,
        }
    }

    const fn from_code(code: i32) -> Option<Self> {
        match code {
            0 => Some(Self::Preflight),
            1 => Some(Self::Wire),
            2 => Some(Self::Protocol),
            _ => None,
        }
    }
}

fn converge<C: Communicator>(comm: &C, local: Option<Failure>) -> Option<PlacementError> {
    let rank = mpi_call!(comm.rank());
    let size = mpi_call!(comm.size());
    let candidate = if local.is_some() { rank } else { size };
    let mut winner = candidate;
    mpi_call!(comm.all_reduce_into(&candidate, &mut winner, SystemOperation::min()));
    if winner == size {
        return None;
    }
    if winner < 0 || winner >= size {
        mpi_call!(comm.abort(74));
    }

    let is_winner = rank == winner;
    if is_winner && local.is_none() {
        mpi_call!(comm.abort(74));
    }
    let mut kind = local
        .as_ref()
        .filter(|_| is_winner)
        .map_or(PlacementErrorKind::Protocol.code(), |failure| {
            failure.kind.code()
        });
    let mut bytes = local
        .filter(|_| is_winner)
        .map_or_else(Vec::new, |failure| failure.message.into_bytes());
    let mut length = match i32::try_from(bytes.len()) {
        Ok(length) => length,
        Err(_) => mpi_call!(comm.abort(74)),
    };

    mpi_call!(comm.process_at_rank(winner).broadcast_into(&mut kind));
    mpi_call!(comm.process_at_rank(winner).broadcast_into(&mut length));
    if length < 0 || length as usize > MAX_ERROR_MESSAGE_BYTES {
        mpi_call!(comm.abort(74));
    }
    if !is_winner {
        bytes.resize(length as usize, 0);
    }
    mpi_call!(comm
        .process_at_rank(winner)
        .broadcast_into(bytes.as_mut_slice()));

    let Some(kind) = PlacementErrorKind::from_code(kind) else {
        mpi_call!(comm.abort(74));
    };
    let message = match String::from_utf8(bytes) {
        Ok(message) => message,
        Err(_) => mpi_call!(comm.abort(74)),
    };
    Some(PlacementError::new(kind, Some(winner), message))
}

fn preflight<C: Communicator>(
    comm: &C,
    operation: Operation,
    root: i32,
    local_shape: bool,
) -> Result<(), PlacementError> {
    let local_operation = operation.code();
    let mut min_operation = local_operation;
    let mut max_operation = local_operation;
    mpi_call!(comm.all_reduce_into(&local_operation, &mut min_operation, SystemOperation::min()));
    mpi_call!(comm.all_reduce_into(&local_operation, &mut max_operation, SystemOperation::max()));

    let mut min_root = root;
    let mut max_root = root;
    mpi_call!(comm.all_reduce_into(&root, &mut min_root, SystemOperation::min()));
    mpi_call!(comm.all_reduce_into(&root, &mut max_root, SystemOperation::max()));

    let size = mpi_call!(comm.size());
    let local = if min_operation != max_operation {
        Some(Failure::new(
            PlacementErrorKind::Preflight,
            "placement operation mismatch",
        ))
    } else if min_root != max_root {
        Some(Failure::new(
            PlacementErrorKind::Preflight,
            "placement root mismatch",
        ))
    } else if root < 0 || root >= size {
        Some(Failure::new(
            PlacementErrorKind::Preflight,
            "placement root is out of range",
        ))
    } else if !local_shape {
        Some(Failure::new(
            PlacementErrorKind::Preflight,
            "placement input shape is invalid",
        ))
    } else if !is_thread_main() {
        Some(Failure::new(
            PlacementErrorKind::Preflight,
            "placement helper must run on the MPI main thread",
        ))
    } else {
        None
    };
    match converge(comm, local) {
        Some(error) => Err(error),
        None => Ok(()),
    }
}

fn encode<T: Serialize>(value: &T) -> Result<Vec<u8>, Failure> {
    wire::encode_payload(value).map_err(|error| Failure::new(PlacementErrorKind::Wire, error))
}

fn send_frame<C: Communicator>(comm: &C, destination: i32, operation: Operation, payload: &[u8]) {
    let header = match Header::new(
        operation.message_kind(),
        MessageStatus::None,
        FRAME_ID,
        FRAME_ITEMS,
        payload.len() as u64,
    ) {
        Ok(header) => header,
        Err(_) => mpi_call!(comm.abort(74)),
    };
    mpi_call!(comm
        .process_at_rank(destination)
        .send_with_tag(&header.encode(), HEADER_TAG));
    mpi_call!(comm
        .process_at_rank(destination)
        .send_with_tag(payload, PAYLOAD_TAG));
}

fn receive_frame<C: Communicator>(comm: &C, source: i32) -> (Vec<u8>, Vec<u8>) {
    // Drain both actual MPI messages before inspecting an untrusted header.
    let (header, _) = mpi_call!(comm
        .process_at_rank(source)
        .receive_vec_with_tag::<u8>(HEADER_TAG));
    let (payload, _) = mpi_call!(comm
        .process_at_rank(source)
        .receive_vec_with_tag::<u8>(PAYLOAD_TAG));
    (header, payload)
}

fn decode_received<T: DeserializeOwned>(
    header_bytes: &[u8],
    payload: &[u8],
    operation: Operation,
) -> Result<T, Failure> {
    let header = Header::decode(header_bytes)
        .map_err(|error| Failure::new(PlacementErrorKind::Wire, error))?;
    if header.kind() != operation.message_kind()
        || header.status() != MessageStatus::None
        || header.batch_id() != FRAME_ID
        || header.item_count() != FRAME_ITEMS
    {
        return Err(Failure::new(
            PlacementErrorKind::Protocol,
            "invalid placement frame schema",
        ));
    }
    wire::checked_mpi_count(header.payload_len())
        .map_err(|error| Failure::new(PlacementErrorKind::Wire, error))?;
    wire::checked_usize_length(header.payload_len())
        .map_err(|error| Failure::new(PlacementErrorKind::Wire, error))?;
    if payload.len() as u64 != header.payload_len() {
        return Err(Failure::new(
            PlacementErrorKind::Protocol,
            "placement payload length does not match its header",
        ));
    }
    wire::checked_mpi_count(payload.len() as u64)
        .map_err(|error| Failure::new(PlacementErrorKind::Wire, error))?;
    wire::decode_payload(payload).map_err(|error| Failure::new(PlacementErrorKind::Wire, error))
}

fn preencode_scatter<T: Serialize>(
    shards: &mut [Option<T>],
    root: i32,
    size: i32,
) -> (Option<T>, Vec<Option<Vec<u8>>>, Option<Failure>) {
    let local = shards[root as usize].take();
    let mut encoded = (0..size).map(|_| None).collect::<Vec<_>>();
    let mut failure = None;
    for destination in 0..size {
        if destination == root {
            continue;
        }
        match shards[destination as usize].as_ref() {
            Some(value) => match encode(value) {
                Ok(bytes) => encoded[destination as usize] = Some(bytes),
                Err(error) => record_failure(&mut failure, error),
            },
            None => record_failure(
                &mut failure,
                Failure::new(PlacementErrorKind::Protocol, "missing scatter shard"),
            ),
        }
    }
    (local, encoded, failure)
}

/// Copies one root-owned value to every rank.
///
/// Every rank calls this collective with the same communicator and root. Only
/// the root supplies `Some`; every rank receives an owned value.
pub fn broadcast<C, T>(world: &C, root: i32, root_value: Option<T>) -> Result<T, PlacementError>
where
    C: Communicator,
    T: Serialize + DeserializeOwned,
{
    let comm = mpi_call!(world.duplicate());
    let rank = mpi_call!(comm.rank());
    let size = mpi_call!(comm.size());
    preflight(
        &comm,
        Operation::Broadcast,
        root,
        (rank == root) == root_value.is_some(),
    )?;

    if size == 1 {
        return Ok(root_value.unwrap_or_else(|| mpi_call!(comm.abort(74))));
    }

    let (encoded, encode_failure) = if rank == root {
        match root_value.as_ref() {
            Some(value) => match encode(value) {
                Ok(bytes) => (Some(bytes), None),
                Err(error) => (None, Some(error)),
            },
            None => (
                None,
                Some(Failure::new(
                    PlacementErrorKind::Protocol,
                    "missing root broadcast value",
                )),
            ),
        }
    } else {
        (None, None)
    };
    if let Some(error) = converge(&comm, encode_failure) {
        return Err(error);
    }

    let mut received = None;
    let mut local_failure = None;
    if rank == root {
        let payload = encoded
            .as_deref()
            .unwrap_or_else(|| mpi_call!(comm.abort(74)));
        for destination in 0..size {
            if destination != root {
                send_frame(&comm, destination, Operation::Broadcast, payload);
            }
        }
    } else {
        let (header, payload) = receive_frame(&comm, root);
        match decode_received(&header, &payload, Operation::Broadcast) {
            Ok(value) => received = Some(value),
            Err(error) => local_failure = Some(error),
        }
    }
    if let Some(error) = converge(&comm, local_failure) {
        return Err(error);
    }

    if rank == root {
        Ok(root_value.unwrap_or_else(|| mpi_call!(comm.abort(74))))
    } else {
        Ok(received.unwrap_or_else(|| mpi_call!(comm.abort(74))))
    }
}

/// Distributes one owned shard from the root to each rank.
///
/// Only the root supplies `Some`, containing exactly one shard per rank. Every
/// rank receives its rank-indexed owned shard.
pub fn scatter<C, T>(world: &C, root: i32, root_shards: Option<Vec<T>>) -> Result<T, PlacementError>
where
    C: Communicator,
    T: Serialize + DeserializeOwned,
{
    let comm = mpi_call!(world.duplicate());
    let rank = mpi_call!(comm.rank());
    let size = mpi_call!(comm.size());
    let local_shape = (rank == root) == root_shards.is_some()
        && (rank != root
            || root_shards
                .as_ref()
                .is_some_and(|shards| shards.len() == size as usize));
    preflight(&comm, Operation::Scatter, root, local_shape)?;

    let mut shards = root_shards.map(|shards| shards.into_iter().map(Some).collect::<Vec<_>>());
    if size == 1 {
        return Ok(shards
            .as_mut()
            .and_then(|shards| shards[0].take())
            .unwrap_or_else(|| mpi_call!(comm.abort(74))));
    }

    let (local_value, encoded, encode_failure) = if rank == root {
        preencode_scatter(
            shards.as_mut().unwrap_or_else(|| mpi_call!(comm.abort(74))),
            root,
            size,
        )
    } else {
        (None, Vec::new(), None)
    };
    if let Some(error) = converge(&comm, encode_failure) {
        return Err(error);
    }

    let mut received = None;
    let mut local_failure = None;
    if rank == root {
        for destination in 0..size {
            if destination != root {
                let payload = encoded[destination as usize]
                    .as_deref()
                    .unwrap_or_else(|| mpi_call!(comm.abort(74)));
                send_frame(&comm, destination, Operation::Scatter, payload);
            }
        }
    } else {
        let (header, payload) = receive_frame(&comm, root);
        match decode_received(&header, &payload, Operation::Scatter) {
            Ok(value) => received = Some(value),
            Err(error) => local_failure = Some(error),
        }
    }
    if let Some(error) = converge(&comm, local_failure) {
        return Err(error);
    }

    if rank == root {
        Ok(local_value.unwrap_or_else(|| mpi_call!(comm.abort(74))))
    } else {
        Ok(received.unwrap_or_else(|| mpi_call!(comm.abort(74))))
    }
}

/// Collects one value from every rank in rank order on the root.
///
/// Every rank supplies one owned value. Only the root receives `Some(values)`;
/// all other ranks receive `None`.
pub fn gather<C, T>(world: &C, root: i32, value: T) -> Result<Option<Vec<T>>, PlacementError>
where
    C: Communicator,
    T: Serialize + DeserializeOwned,
{
    let comm = mpi_call!(world.duplicate());
    let rank = mpi_call!(comm.rank());
    let size = mpi_call!(comm.size());
    preflight(&comm, Operation::Gather, root, true)?;

    if size == 1 {
        return Ok(Some(vec![value]));
    }

    let (encoded, encode_failure) = if rank == root {
        (None, None)
    } else {
        match encode(&value) {
            Ok(bytes) => (Some(bytes), None),
            Err(error) => (None, Some(error)),
        }
    };
    if let Some(error) = converge(&comm, encode_failure) {
        return Err(error);
    }

    let mut gathered = if rank == root {
        let mut values = (0..size).map(|_| None).collect::<Vec<Option<T>>>();
        values[root as usize] = Some(value);
        Some(values)
    } else {
        None
    };
    let mut local_failure = None;
    if rank == root {
        for source in 0..size {
            if source == root {
                continue;
            }
            let (header, payload) = receive_frame(&comm, source);
            match decode_received(&header, &payload, Operation::Gather) {
                Ok(value) => {
                    gathered
                        .as_mut()
                        .unwrap_or_else(|| mpi_call!(comm.abort(74)))[source as usize] =
                        Some(value);
                }
                Err(error) => record_failure(&mut local_failure, error),
            }
        }
    } else {
        send_frame(
            &comm,
            root,
            Operation::Gather,
            encoded
                .as_deref()
                .unwrap_or_else(|| mpi_call!(comm.abort(74))),
        );
    }
    if let Some(error) = converge(&comm, local_failure) {
        return Err(error);
    }

    if rank == root {
        let values = gathered.unwrap_or_else(|| mpi_call!(comm.abort(74)));
        let mut output = Vec::with_capacity(size as usize);
        for value in values {
            output.push(value.unwrap_or_else(|| mpi_call!(comm.abort(74))));
        }
        Ok(Some(output))
    } else {
        Ok(None)
    }
}

#[cfg(test)]
mod tests;
