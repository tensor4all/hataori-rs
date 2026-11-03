use super::*;
use serde::{Deserialize, Serialize};

#[test]
fn header_golden_and_roundtrip() {
    let header = Header::new(
        MessageKind::Result,
        MessageStatus::Error,
        0x0102_0304_0506_0708,
        0x1112_1314_1516_1718,
        123,
    )
    .unwrap();
    assert_eq!(
        header.encode(),
        [
            1, 0, 2, 2, 8, 7, 6, 5, 4, 3, 2, 1, 0x18, 0x17, 0x16, 0x15, 0x14, 0x13, 0x12, 0x11,
            123, 0, 0, 0, 0, 0, 0, 0,
        ]
    );
    assert_eq!(Header::decode(&header.encode()).unwrap(), header);
    assert_eq!(header.version(), PROTOCOL_VERSION);
    assert_eq!(header.kind(), MessageKind::Result);
    assert_eq!(header.status(), MessageStatus::Error);
    assert_eq!(header.batch_id(), 0x0102_0304_0506_0708);
    assert_eq!(header.item_count(), 0x1112_1314_1516_1718);
    assert_eq!(header.payload_len(), 123);
}

#[test]
fn header_rejects_bad_size_fields_pairs_and_payload_cap() {
    assert!(matches!(
        Header::decode(&[0; HEADER_LEN - 1]),
        Err(WireError::InvalidHeaderSize { .. })
    ));
    assert!(matches!(
        Header::decode(&[0; HEADER_LEN + 1]),
        Err(WireError::InvalidHeaderSize { .. })
    ));

    let mut bytes = [0; HEADER_LEN];
    bytes[0] = 2;
    assert!(matches!(
        Header::decode(&bytes),
        Err(WireError::InvalidVersion { .. })
    ));
    bytes[0] = 1;
    bytes[2] = 99;
    assert!(matches!(
        Header::decode(&bytes),
        Err(WireError::InvalidKind(99))
    ));
    bytes[2] = 0;
    bytes[3] = 99;
    assert!(matches!(
        Header::decode(&bytes),
        Err(WireError::InvalidStatus(99))
    ));

    assert!(matches!(
        Header::new(MessageKind::Ready, MessageStatus::Ok, 0, 0, 0),
        Err(WireError::StatusKindMismatch { .. })
    ));
    assert!(matches!(
        Header::new(MessageKind::Result, MessageStatus::None, 0, 0, 0),
        Err(WireError::StatusKindMismatch { .. })
    ));
    bytes[2] = MessageKind::Ready.code();
    bytes[3] = MessageStatus::Ok.code();
    assert!(matches!(
        Header::decode(&bytes),
        Err(WireError::StatusKindMismatch { .. })
    ));
    assert!(Header::new(MessageKind::Stop, MessageStatus::None, 0, 0, MAX_WIRE_BYTES).is_ok());
    assert!(matches!(
        Header::new(
            MessageKind::Stop,
            MessageStatus::None,
            0,
            0,
            MAX_WIRE_BYTES + 1
        ),
        Err(WireError::PayloadTooLarge { .. })
    ));
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
struct Pair {
    left: u16,
    right: String,
}

#[test]
fn codec_uses_fixed_little_endian_and_roundtrips() {
    assert_eq!(encode_payload(&0x1234_u16).unwrap(), vec![0x34, 0x12]);
    let value = Pair {
        left: 7,
        right: "wire".into(),
    };
    let bytes = encode_payload(&value).unwrap();
    assert_eq!(decode_payload::<Pair>(&bytes).unwrap(), value);
}

#[test]
fn codec_rejects_malformed_trailing_and_oversized_payloads() {
    assert!(matches!(
        decode_payload::<u16>(&[1]),
        Err(WireError::Decode { .. })
    ));
    let mut bytes = encode_payload(&1_u16).unwrap();
    bytes.push(0);
    assert!(matches!(
        decode_payload::<u16>(&bytes),
        Err(WireError::TrailingBytes { .. })
    ));
}

#[test]
fn conversion_boundaries_are_checked_without_large_allocations() {
    assert_eq!(checked_mpi_count(i32::MAX as u64).unwrap(), i32::MAX);
    assert!(matches!(
        checked_mpi_count(i32::MAX as u64 + 1),
        Err(WireError::CountOverflow { .. })
    ));
    assert_eq!(checked_usize_length(usize::MAX as u64).unwrap(), usize::MAX);
    #[cfg(target_pointer_width = "32")]
    assert!(matches!(
        checked_usize_length(u64::from(u32::MAX) + 1),
        Err(WireError::UsizeOverflow { .. })
    ));
}

#[test]
fn error_keys_order_and_rank_decode_are_signed() {
    let world = 4;
    let a = ErrorKey::new(2, ErrorClass::WireProtocol, 3, world).unwrap();
    let b = ErrorKey::new(2, ErrorClass::Callback, 0, world).unwrap();
    let c = ErrorKey::new(2, ErrorClass::Callback, 1, world).unwrap();
    assert!(a < b && b < c);
    assert_eq!(c.checked_reporting_rank(world).unwrap(), 1);
    assert_eq!(ErrorKey::NO_ERROR_KEY, i64::MAX);
    assert_eq!(std::cmp::min(a, ErrorKey(ErrorKey::NO_ERROR_KEY)), a);
    assert!(
        ErrorKey::new(0, ErrorClass::WireProtocol, 0, world)
            .unwrap()
            .get()
            >= 0
    );
}

#[test]
fn error_keys_validate_world_rank_and_overflow() {
    assert!(matches!(
        ErrorKey::new(0, ErrorClass::WireProtocol, 0, 0),
        Err(WireError::InvalidWorldSize { .. })
    ));
    assert!(matches!(
        ErrorKey::new(0, ErrorClass::WireProtocol, -1, 2),
        Err(WireError::InvalidRank { .. })
    ));
    assert!(matches!(
        ErrorKey::new(0, ErrorClass::WireProtocol, 2, 2),
        Err(WireError::InvalidRank { .. })
    ));
    assert!(matches!(
        ErrorKey::new(i64::MAX as u64, ErrorClass::WireProtocol, 0, i32::MAX),
        Err(WireError::ErrorKeyOverflow)
    ));
    let world = 2;
    let max_task = ((i64::MAX as u128 - 1 - (ErrorClass::Drain.code() as u128 * world as u128 + 1))
        / (ErrorClass::COUNT as u128 * world as u128)) as u64;
    let largest = ErrorKey::new(max_task, ErrorClass::Drain, 1, world).unwrap();
    assert!(largest.get() < i64::MAX);
    assert!(ErrorKey::new(max_task + 1, ErrorClass::Drain, 1, world).is_err());
    assert!(matches!(
        ErrorKey(i64::MAX).checked_reporting_rank(2),
        Err(WireError::NoErrorKey)
    ));
    assert!(matches!(
        ErrorKey::new(u64::MAX, ErrorClass::Drain, 1, 2),
        Err(WireError::ErrorKeyOverflow)
    ));
}

#[test]
fn preflight_candidate_uses_signed_rank_or_world_sentinel() {
    assert_eq!(preflight_candidate(false, 1, 4).unwrap(), 4);
    assert_eq!(preflight_candidate(true, 1, 4).unwrap(), 1);
    assert!(matches!(
        preflight_candidate(true, -1, 4),
        Err(WireError::InvalidRank { .. })
    ));
    assert!(matches!(
        preflight_candidate(false, 0, 0),
        Err(WireError::InvalidWorldSize { .. })
    ));
}

#[test]
fn error_display_is_bounded_and_accessors_are_available() {
    let detail = "é".repeat(5000);
    let error = WireError::Decode {
        detail: truncate_message(detail),
    };
    let WireError::Decode { detail } = &error else {
        unreachable!()
    };
    assert!(detail.len() <= 4096);
    assert!(detail.is_char_boundary(detail.len()));
    assert!(std::error::Error::source(&error).is_none());
    assert_eq!(MessageKind::Task.code(), 1);
    assert_eq!(MessageStatus::Ok.code(), 1);
    assert_eq!(ErrorClass::Drain.code(), 2);
}
