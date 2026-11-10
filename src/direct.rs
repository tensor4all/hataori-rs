//! Direct data-plane transport for contiguous primitive numeric buffers.
//!
//! Slice A of issue #16 item 2: one raw, typed MPI buffer instead of a serde
//! codec and an intermediate `Vec<u8>`. The frame kind, its descriptor, and the
//! control-message encoding live in [`crate::wire`]; this module owns only the
//! two MPI messages.
//!
//! The public facade does not use this path yet: the issue's item 3 adapters and
//! the `pmap`/placement callers land later, so the only caller today is the test
//! suite.

use crate::mpi_backend::datatype::Equivalence;
use crate::mpi_backend::traits::*;
use crate::mpi_check::mpi_call;
use crate::wire::{self, ByteOrder, ElemType, Header, MessageKind, WireError};

/// Control message tag, the pair `src/placement.rs` uses for its frames.
const CONTROL_TAG: i32 = 2;
/// Data message tag, the pair `src/placement.rs` uses for its frames.
const DATA_TAG: i32 = 3;

/// One primitive element type the direct data plane can carry.
pub(crate) trait DirectElem: Copy + Default + 'static + Equivalence {
    /// The element type tag written into the control message.
    const TYPE: ElemType;
}

impl DirectElem for f32 {
    const TYPE: ElemType = ElemType::F32;
}

impl DirectElem for f64 {
    const TYPE: ElemType = ElemType::F64;
}

/// Sends one contiguous buffer to `destination` as a raw typed MPI buffer.
///
/// An empty buffer sends only the control message: there is no zero-length data
/// message to match.
pub(crate) fn send<C: Communicator, T: DirectElem>(
    comm: &C,
    destination: i32,
    values: &[T],
) -> Result<(), WireError> {
    let header = Header::new_direct(T::TYPE, values.len() as u64)?;
    let control = wire::encode_direct_control(header, T::TYPE);
    mpi_call!(comm
        .process_at_rank(destination)
        .send_with_tag(&control[..], CONTROL_TAG));
    if !values.is_empty() {
        mpi_call!(comm
            .process_at_rank(destination)
            .send_with_tag(values, DATA_TAG));
    }
    Ok(())
}

/// Receives one contiguous buffer from `source`.
///
/// The control message is probed, so its size is checked before anything is
/// allocated; the element buffer is allocated only after the announced element
/// count and the probed data count agree.
pub(crate) fn receive<C: Communicator, T: DirectElem>(
    comm: &C,
    source: i32,
) -> Result<Vec<T>, WireError> {
    let (control, len) = receive_control(comm, source)?;
    match decode_control(&control[..len], T::TYPE) {
        // An empty buffer has no data message, so there is nothing to receive.
        Ok(0) => Ok(Vec::new()),
        Ok(elements) => receive_elements(comm, source, elements),
        Err(error) => {
            // The data message is already in flight. Receive and discard it so a
            // blocking sender completes and the next exchange starts clean. A
            // rejection that cannot be drained is reported as such, never as a
            // recoverable mismatch: the caller may reuse the communicator only
            // after the original error.
            drain(comm, source, &control[..len])?;
            Err(error)
        }
    }
}

/// Reads one control message into a stack buffer; the probed size is checked
/// before the buffer is filled, so a control message allocates nothing.
fn receive_control<C: Communicator>(
    comm: &C,
    source: i32,
) -> Result<([u8; wire::DIRECT_CONTROL_LEN], usize), WireError> {
    let status = mpi_call!(comm.process_at_rank(source).probe_with_tag(CONTROL_TAG));
    let count = mpi_call!(status.count(u8::equivalent_datatype()));
    if count != wire::HEADER_LEN as i32 && count != wire::DIRECT_CONTROL_LEN as i32 {
        return Err(WireError::InvalidHeaderSize {
            actual: usize::try_from(count).unwrap_or(usize::MAX),
        });
    }
    let len = count as usize;
    let mut control = [0; wire::DIRECT_CONTROL_LEN];
    mpi_call!(comm
        .process_at_rank(source)
        .receive_into_with_tag(&mut control[..len], CONTROL_TAG));
    Ok((control, len))
}

fn receive_elements<C: Communicator, T: DirectElem>(
    comm: &C,
    source: i32,
    elements: usize,
) -> Result<Vec<T>, WireError> {
    let received = probe_elements::<C, T>(comm, source);
    if received != elements as i32 {
        return Err(WireError::DirectCountMismatch {
            expected: elements as u64,
            actual: i64::from(received),
        });
    }
    let mut values = vec![T::default(); elements];
    mpi_call!(comm
        .process_at_rank(source)
        .receive_into_with_tag(values.as_mut_slice(), DATA_TAG));
    Ok(values)
}

/// Returns the element count of the pending data message, without receiving it.
fn probe_elements<C: Communicator, T: DirectElem>(comm: &C, source: i32) -> i32 {
    let status = mpi_call!(comm.process_at_rank(source).probe_with_tag(DATA_TAG));
    mpi_call!(status.count(T::equivalent_datatype()))
}

/// Consumes the data message of a direct frame whose control message failed
/// validation, so the sender and the next exchange are not affected.
///
/// Only a length-consistent `Direct` frame whose element-type byte is readable
/// is drained. Anything else means the two ends disagree about the wire format,
/// so the message boundary is unknowable: that is [`WireError::DirectDrainFailed`],
/// and the caller must not reuse the communicator.
fn drain<C: Communicator>(comm: &C, source: i32, control: &[u8]) -> Result<(), WireError> {
    let header = Header::decode(control.get(..wire::HEADER_LEN).unwrap_or(control))
        .map_err(|_| WireError::DirectDrainFailed)?;
    if header.kind() != MessageKind::Direct {
        return Err(WireError::DirectDrainFailed);
    }
    if header.item_count() == 0 {
        return Ok(());
    }
    let elem = control
        .get(wire::HEADER_LEN)
        .and_then(|byte| ElemType::try_from(*byte).ok())
        .ok_or(WireError::DirectDrainFailed)?;
    if header.item_count().checked_mul(elem.size() as u64) != Some(header.payload_len()) {
        return Err(WireError::DirectDrainFailed);
    }
    let elements = wire::checked_usize_length(header.item_count())
        .map_err(|_| WireError::DirectDrainFailed)?;
    if elements > i32::MAX as usize {
        return Err(WireError::DirectDrainFailed);
    }
    match elem {
        ElemType::F32 => receive_elements::<C, f32>(comm, source, elements).map(|_| ()),
        ElemType::F64 => receive_elements::<C, f64>(comm, source, elements).map(|_| ()),
    }
}

/// Validates a direct control message against the requested element type and
/// returns its element count.
fn decode_control(control: &[u8], expected: ElemType) -> Result<usize, WireError> {
    let (header, meta) = Header::decode_control(control)?;
    let Some(meta) = meta else {
        return Err(WireError::UnexpectedDirectKind {
            kind: header.kind(),
        });
    };
    if meta.elem() != expected {
        return Err(WireError::DirectTypeMismatch {
            expected,
            actual: meta.elem(),
        });
    }
    if meta.order() != ByteOrder::NATIVE {
        return Err(WireError::DirectByteOrderMismatch {
            expected: ByteOrder::NATIVE,
            actual: meta.order(),
        });
    }
    let elements = header.item_count();
    elements
        .checked_mul(expected.size() as u64)
        .filter(|bytes| *bytes == header.payload_len())
        .ok_or(WireError::DirectLengthMismatch {
            elements,
            bytes: header.payload_len(),
        })?;
    wire::checked_mpi_count(elements)?;
    wire::checked_usize_length(elements)
}

#[cfg(test)]
mod tests;
