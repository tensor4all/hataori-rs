use crate::codec;
use crate::map::truncate_message;
use serde::{de::DeserializeOwned, Serialize};
use std::fmt;
#[cfg(all(test, not(feature = "rayon")))]
use std::sync::atomic::{AtomicUsize, Ordering};

pub(crate) const PROTOCOL_VERSION: u16 = 1;
pub(crate) const MAX_WIRE_BYTES: u64 = i32::MAX as u64;
pub(crate) const HEADER_LEN: usize = 28;
/// Element-type tag plus byte-order marker of a direct data-plane frame.
pub(crate) const DIRECT_META_LEN: usize = 2;
pub(crate) const DIRECT_CONTROL_LEN: usize = HEADER_LEN + DIRECT_META_LEN;
pub(crate) const NO_ERROR_KEY: i64 = i64::MAX;

#[cfg(all(test, not(feature = "rayon")))]
static ENCODE_CALLS: AtomicUsize = AtomicUsize::new(0);
#[cfg(all(test, not(feature = "rayon")))]
static DECODE_CALLS: AtomicUsize = AtomicUsize::new(0);

#[cfg(all(test, not(feature = "rayon")))]
pub(crate) fn reset_codec_counts() {
    ENCODE_CALLS.store(0, Ordering::SeqCst);
    DECODE_CALLS.store(0, Ordering::SeqCst);
}

#[cfg(all(test, not(feature = "rayon")))]
pub(crate) fn codec_counts() -> (usize, usize) {
    (
        ENCODE_CALLS.load(Ordering::SeqCst),
        DECODE_CALLS.load(Ordering::SeqCst),
    )
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub(crate) enum MessageKind {
    Ready,
    Task,
    Result,
    Stop,
    Drain,
    Broadcast,
    Scatter,
    Gather,
    /// A raw data-plane payload: `item_count` elements of one primitive type.
    Direct,
}

impl MessageKind {
    pub(crate) const fn code(self) -> u8 {
        match self {
            Self::Ready => 0,
            Self::Task => 1,
            Self::Result => 2,
            Self::Stop => 3,
            Self::Drain => 4,
            Self::Broadcast => 5,
            Self::Scatter => 6,
            Self::Gather => 7,
            Self::Direct => 8,
        }
    }
}

impl TryFrom<u8> for MessageKind {
    type Error = WireError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Ready),
            1 => Ok(Self::Task),
            2 => Ok(Self::Result),
            3 => Ok(Self::Stop),
            4 => Ok(Self::Drain),
            5 => Ok(Self::Broadcast),
            6 => Ok(Self::Scatter),
            7 => Ok(Self::Gather),
            8 => Ok(Self::Direct),
            value => Err(WireError::InvalidKind(value)),
        }
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub(crate) enum MessageStatus {
    None,
    Ok,
    Error,
}

impl MessageStatus {
    pub(crate) const fn code(self) -> u8 {
        match self {
            Self::None => 0,
            Self::Ok => 1,
            Self::Error => 2,
        }
    }
}

impl TryFrom<u8> for MessageStatus {
    type Error = WireError;

    fn try_from(value: u8) -> Result<Self, WireError> {
        match value {
            0 => Ok(Self::None),
            1 => Ok(Self::Ok),
            2 => Ok(Self::Error),
            value => Err(WireError::InvalidStatus(value)),
        }
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub(crate) struct Header {
    version: u16,
    kind: MessageKind,
    status: MessageStatus,
    batch_id: u64,
    item_count: u64,
    payload_len: u64,
}

impl Header {
    pub(crate) fn new(
        kind: MessageKind,
        status: MessageStatus,
        batch_id: u64,
        item_count: u64,
        payload_len: u64,
    ) -> Result<Self, WireError> {
        validate_status(kind, status)?;
        validate_payload_len(payload_len)?;
        Ok(Self {
            version: PROTOCOL_VERSION,
            kind,
            status,
            batch_id,
            item_count,
            payload_len,
        })
    }

    /// Header of a direct data-plane frame carrying `elements` values of `elem`.
    pub(crate) fn new_direct(elem: ElemType, elements: u64) -> Result<Self, WireError> {
        let bytes = elements
            .checked_mul(elem.size() as u64)
            .filter(|bytes| *bytes <= MAX_WIRE_BYTES)
            .ok_or(WireError::PayloadTooLarge {
                bytes: elements.saturating_mul(elem.size() as u64),
            })?;
        Self::new(MessageKind::Direct, MessageStatus::None, 0, elements, bytes)
    }

    pub(crate) fn encode(self) -> [u8; HEADER_LEN] {
        let mut bytes = [0; HEADER_LEN];
        bytes[0..2].copy_from_slice(&self.version.to_le_bytes());
        bytes[2] = self.kind.code();
        bytes[3] = self.status.code();
        bytes[4..12].copy_from_slice(&self.batch_id.to_le_bytes());
        bytes[12..20].copy_from_slice(&self.item_count.to_le_bytes());
        bytes[20..28].copy_from_slice(&self.payload_len.to_le_bytes());
        bytes
    }

    pub(crate) fn decode(bytes: &[u8]) -> Result<Self, WireError> {
        if bytes.len() != HEADER_LEN {
            return Err(WireError::InvalidHeaderSize {
                actual: bytes.len(),
            });
        }
        let version = u16::from_le_bytes([bytes[0], bytes[1]]);
        if version != PROTOCOL_VERSION {
            return Err(WireError::InvalidVersion {
                expected: PROTOCOL_VERSION,
                actual: version,
            });
        }
        let kind = MessageKind::try_from(bytes[2])?;
        let status = MessageStatus::try_from(bytes[3])?;
        validate_status(kind, status)?;
        let batch_id = u64::from_le_bytes(bytes[4..12].try_into().unwrap());
        let item_count = u64::from_le_bytes(bytes[12..20].try_into().unwrap());
        let payload_len = u64::from_le_bytes(bytes[20..28].try_into().unwrap());
        validate_payload_len(payload_len)?;
        Ok(Self {
            version,
            kind,
            status,
            batch_id,
            item_count,
            payload_len,
        })
    }

    /// Decodes one control message.
    ///
    /// A control message is the 28-byte header, plus the two-byte direct
    /// descriptor only for [`MessageKind::Direct`]. The kind and the length must
    /// agree, so a framed kind can never carry a descriptor and a direct frame
    /// can never omit it.
    pub(crate) fn decode_control(bytes: &[u8]) -> Result<(Self, Option<DirectMeta>), WireError> {
        if !(HEADER_LEN..=DIRECT_CONTROL_LEN).contains(&bytes.len()) {
            return Err(WireError::InvalidHeaderSize {
                actual: bytes.len(),
            });
        }
        let header = Self::decode(&bytes[..HEADER_LEN])?;
        let expected = match header.kind {
            MessageKind::Direct => DIRECT_CONTROL_LEN,
            _ => HEADER_LEN,
        };
        if bytes.len() != expected {
            return Err(WireError::InvalidHeaderSize {
                actual: bytes.len(),
            });
        }
        let meta = (header.kind == MessageKind::Direct)
            .then(|| DirectMeta::decode(&bytes[HEADER_LEN..]))
            .transpose()?;
        Ok((header, meta))
    }

    #[cfg(test)]
    pub(crate) const fn version(self) -> u16 {
        self.version
    }
    pub(crate) const fn kind(self) -> MessageKind {
        self.kind
    }
    pub(crate) const fn status(self) -> MessageStatus {
        self.status
    }
    pub(crate) const fn batch_id(self) -> u64 {
        self.batch_id
    }
    pub(crate) const fn item_count(self) -> u64 {
        self.item_count
    }
    pub(crate) const fn payload_len(self) -> u64 {
        self.payload_len
    }
}

/// Element type of a direct data-plane payload.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub(crate) enum ElemType {
    F32,
    F64,
}

impl ElemType {
    pub(crate) const fn code(self) -> u8 {
        match self {
            Self::F32 => 1,
            Self::F64 => 2,
        }
    }

    pub(crate) const fn size(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F64 => 8,
        }
    }
}

impl TryFrom<u8> for ElemType {
    type Error = WireError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            1 => Ok(Self::F32),
            2 => Ok(Self::F64),
            value => Err(WireError::InvalidDirectDescriptor(value)),
        }
    }
}

/// Byte order of the sender's direct data-plane payload.
///
/// The data plane sends typed MPI elements and performs no Hataori-side byte
/// conversion, so a receiver must share the sender's byte order; the marker lets
/// it reject a mismatching peer instead of reading values wrongly. The framed
/// control plane stays explicitly little-endian.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub(crate) enum ByteOrder {
    Little,
    Big,
}

impl ByteOrder {
    pub(crate) const NATIVE: Self = if cfg!(target_endian = "little") {
        Self::Little
    } else {
        Self::Big
    };

    pub(crate) const fn code(self) -> u8 {
        match self {
            Self::Little => 1,
            Self::Big => 2,
        }
    }
}

impl TryFrom<u8> for ByteOrder {
    type Error = WireError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            1 => Ok(Self::Little),
            2 => Ok(Self::Big),
            value => Err(WireError::InvalidDirectDescriptor(value)),
        }
    }
}

/// The two control-plane bytes that describe a direct data-plane payload.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub(crate) struct DirectMeta {
    elem: ElemType,
    order: ByteOrder,
}

impl DirectMeta {
    pub(crate) const fn new(elem: ElemType) -> Self {
        Self {
            elem,
            order: ByteOrder::NATIVE,
        }
    }

    pub(crate) const fn elem(self) -> ElemType {
        self.elem
    }

    pub(crate) const fn order(self) -> ByteOrder {
        self.order
    }

    fn encode(self) -> [u8; DIRECT_META_LEN] {
        [self.elem.code(), self.order.code()]
    }

    fn decode(bytes: &[u8]) -> Result<Self, WireError> {
        match bytes {
            [elem, order] => Ok(Self {
                elem: ElemType::try_from(*elem)?,
                order: ByteOrder::try_from(*order)?,
            }),
            _ => Err(WireError::InvalidHeaderSize {
                actual: bytes.len(),
            }),
        }
    }
}

/// The control message of a direct frame: header plus the 2-byte descriptor.
///
/// Stack-allocated, so sending a direct frame allocates nothing on the Hataori
/// side.
pub(crate) fn encode_direct_control(header: Header, elem: ElemType) -> [u8; DIRECT_CONTROL_LEN] {
    debug_assert_eq!(
        header.kind(),
        MessageKind::Direct,
        "a framed kind has no descriptor"
    );
    let mut bytes = [0; DIRECT_CONTROL_LEN];
    bytes[..HEADER_LEN].copy_from_slice(&header.encode());
    bytes[HEADER_LEN..].copy_from_slice(&DirectMeta::new(elem).encode());
    bytes
}

fn validate_status(kind: MessageKind, status: MessageStatus) -> Result<(), WireError> {
    let valid = match kind {
        MessageKind::Ready
        | MessageKind::Task
        | MessageKind::Stop
        | MessageKind::Drain
        | MessageKind::Broadcast
        | MessageKind::Scatter
        | MessageKind::Gather
        | MessageKind::Direct => status == MessageStatus::None,
        MessageKind::Result => matches!(status, MessageStatus::Ok | MessageStatus::Error),
    };
    valid
        .then_some(())
        .ok_or(WireError::StatusKindMismatch { kind, status })
}

fn validate_payload_len(payload_len: u64) -> Result<(), WireError> {
    (payload_len <= MAX_WIRE_BYTES)
        .then_some(())
        .ok_or(WireError::PayloadTooLarge { bytes: payload_len })
}

/// Encodes a payload with `C`, then checks the protocol byte cap.
pub(crate) fn encode_payload_with<T: Serialize, C: codec::PayloadCodec>(
    value: &T,
) -> Result<Vec<u8>, WireError> {
    let bytes = C::encode(value)?;
    validate_payload_len(bytes.len() as u64)?;
    Ok(bytes)
}

/// Decodes a payload with `C` after checking the protocol byte cap.
pub(crate) fn decode_payload_with<T: DeserializeOwned, C: codec::PayloadCodec>(
    bytes: &[u8],
) -> Result<T, WireError> {
    validate_payload_len(bytes.len() as u64)?;
    C::decode(bytes)
}

/// Encodes a payload with the default codec.
pub(crate) fn encode_payload<T: Serialize>(value: &T) -> Result<Vec<u8>, WireError> {
    #[cfg(all(test, not(feature = "rayon")))]
    ENCODE_CALLS.fetch_add(1, Ordering::SeqCst);
    encode_payload_with::<T, codec::Bincode>(value)
}

/// Decodes a payload with the default codec.
pub(crate) fn decode_payload<T: DeserializeOwned>(bytes: &[u8]) -> Result<T, WireError> {
    #[cfg(all(test, not(feature = "rayon")))]
    DECODE_CALLS.fetch_add(1, Ordering::SeqCst);
    decode_payload_with::<T, codec::Bincode>(bytes)
}

pub(crate) fn checked_mpi_count(value: u64) -> Result<i32, WireError> {
    (value <= i32::MAX as u64)
        .then_some(value as i32)
        .ok_or(WireError::CountOverflow { count: value })
}

pub(crate) fn checked_usize_length(value: u64) -> Result<usize, WireError> {
    usize::try_from(value).map_err(|_| WireError::UsizeOverflow { length: value })
}

#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd)]
pub(crate) enum ErrorClass {
    WireProtocol,
    Callback,
    #[allow(dead_code, reason = "Task #19 drain-fault convergence uses this class")]
    Drain,
}

impl ErrorClass {
    pub(crate) const COUNT: u8 = 3;

    pub(crate) const fn code(self) -> u8 {
        match self {
            Self::WireProtocol => 0,
            Self::Callback => 1,
            Self::Drain => 2,
        }
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd)]
pub(crate) struct ErrorKey(i64);

impl ErrorKey {
    pub(crate) const NO_ERROR_KEY: i64 = NO_ERROR_KEY;

    pub(crate) fn new(
        task_key: u64,
        class: ErrorClass,
        reporting_rank: i32,
        world_size: i32,
    ) -> Result<Self, WireError> {
        validate_world_rank(reporting_rank, world_size)?;
        let value = ((task_key as u128 * ErrorClass::COUNT as u128 + class.code() as u128)
            * world_size as u128)
            + reporting_rank as u128;
        if value >= NO_ERROR_KEY as u128 {
            return Err(WireError::ErrorKeyOverflow);
        }
        Ok(Self(value as i64))
    }

    pub(crate) const fn get(self) -> i64 {
        self.0
    }

    #[cfg(test)]
    pub(crate) fn checked_reporting_rank(self, world_size: i32) -> Result<i32, WireError> {
        if world_size <= 0 {
            return Err(WireError::InvalidWorldSize { world_size });
        }
        if self.0 == NO_ERROR_KEY {
            return Err(WireError::NoErrorKey);
        }
        Ok((self.0 % world_size as i64) as i32)
    }
}

fn validate_world_rank(rank: i32, world_size: i32) -> Result<(), WireError> {
    if world_size <= 0 {
        return Err(WireError::InvalidWorldSize { world_size });
    }
    if rank < 0 || rank >= world_size {
        return Err(WireError::InvalidRank { rank, world_size });
    }
    Ok(())
}

pub(crate) fn preflight_candidate(
    failed: bool,
    reporting_rank: i32,
    world_size: i32,
) -> Result<i32, WireError> {
    validate_world_rank(reporting_rank, world_size)?;
    Ok(if failed { reporting_rank } else { world_size })
}

pub(crate) enum WireError {
    InvalidHeaderSize {
        actual: usize,
    },
    InvalidVersion {
        expected: u16,
        actual: u16,
    },
    InvalidKind(u8),
    InvalidStatus(u8),
    StatusKindMismatch {
        kind: MessageKind,
        status: MessageStatus,
    },
    PayloadTooLarge {
        bytes: u64,
    },
    CountOverflow {
        count: u64,
    },
    UsizeOverflow {
        length: u64,
    },
    Encode {
        detail: String,
    },
    Decode {
        detail: String,
    },
    TrailingBytes {
        used: usize,
        total: usize,
    },
    InvalidWorldSize {
        world_size: i32,
    },
    InvalidDirectDescriptor(u8),
    UnexpectedDirectKind {
        kind: MessageKind,
    },
    DirectTypeMismatch {
        expected: ElemType,
        actual: ElemType,
    },
    DirectByteOrderMismatch {
        expected: ByteOrder,
        actual: ByteOrder,
    },
    DirectLengthMismatch {
        elements: u64,
        bytes: u64,
    },
    DirectCountMismatch {
        expected: u64,
        actual: i64,
    },
    DirectDrainFailed,
    InvalidRank {
        rank: i32,
        world_size: i32,
    },
    ErrorKeyOverflow,
    #[cfg(test)]
    NoErrorKey,
}

impl fmt::Debug for WireError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.to_string())
    }
}

impl fmt::Display for WireError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidHeaderSize { actual } => {
                write!(formatter, "invalid wire header size: {actual}")
            }
            Self::InvalidVersion { expected, actual } => write!(
                formatter,
                "invalid wire version: expected {expected}, got {actual}"
            ),
            Self::InvalidKind(value) => write!(formatter, "invalid wire message kind: {value}"),
            Self::InvalidStatus(value) => write!(formatter, "invalid wire message status: {value}"),
            Self::StatusKindMismatch { kind, status } => write!(
                formatter,
                "wire status is incompatible with message kind: {kind:?}/{status:?}"
            ),
            Self::PayloadTooLarge { bytes } => {
                write!(formatter, "wire payload is too large: {bytes} bytes")
            }
            Self::CountOverflow { count } => write!(formatter, "MPI count overflows i32: {count}"),
            Self::UsizeOverflow { length } => {
                write!(formatter, "wire length overflows usize: {length}")
            }
            Self::Encode { detail } => write!(
                formatter,
                "wire payload encoding failed: {}",
                truncate_message(detail.clone())
            ),
            Self::Decode { detail } => write!(
                formatter,
                "wire payload decoding failed: {}",
                truncate_message(detail.clone())
            ),
            Self::TrailingBytes { used, total } => write!(
                formatter,
                "wire payload has trailing bytes: used {used} of {total}"
            ),
            Self::InvalidWorldSize { world_size } => {
                write!(formatter, "invalid MPI world size: {world_size}")
            }
            Self::InvalidDirectDescriptor(value) => {
                write!(formatter, "invalid direct-payload descriptor byte: {value}")
            }
            Self::UnexpectedDirectKind { kind } => write!(
                formatter,
                "direct receive got a {kind:?} frame instead of a Direct frame"
            ),
            Self::DirectTypeMismatch { expected, actual } => write!(
                formatter,
                "direct payload element type mismatch: expected {expected:?}, got {actual:?}"
            ),
            Self::DirectByteOrderMismatch { expected, actual } => write!(
                formatter,
                "direct payload byte order mismatch: receiver {expected:?}, sender {actual:?}"
            ),
            Self::DirectLengthMismatch { elements, bytes } => write!(
                formatter,
                "direct payload length does not match its element count: {elements} elements in {bytes} bytes"
            ),
            Self::DirectCountMismatch { expected, actual } => write!(
                formatter,
                "direct payload element count mismatch: expected {expected}, received {actual}"
            ),
            Self::DirectDrainFailed => formatter.write_str(
                "direct payload rejection could not be drained; the exchange is desynchronised \
                 and the communicator must not be reused",
            ),
            Self::InvalidRank { rank, world_size } => write!(
                formatter,
                "invalid MPI rank {rank} for world size {world_size}"
            ),
            Self::ErrorKeyOverflow => formatter.write_str("deterministic error key overflows i64"),
            #[cfg(test)]
            Self::NoErrorKey => formatter.write_str("no-error key has no reporting rank"),
        }
    }
}

impl std::error::Error for WireError {}

#[cfg(test)]
mod tests;
