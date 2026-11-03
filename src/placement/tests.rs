use super::*;

#[test]
fn error_accessors_and_bound_are_stable() {
    let error = PlacementError::new(
        PlacementErrorKind::Wire,
        Some(3),
        "é".repeat(MAX_ERROR_MESSAGE_BYTES),
    );
    assert_eq!(error.kind(), PlacementErrorKind::Wire);
    assert_eq!(error.rank(), Some(3));
    assert!(error.message().len() <= MAX_ERROR_MESSAGE_BYTES);
    assert!(error.message().is_char_boundary(error.message().len()));
    assert_eq!(error.to_string(), error.message());
}

#[test]
fn operation_kinds_have_checked_none_status_headers() {
    for operation in [Operation::Broadcast, Operation::Scatter, Operation::Gather] {
        let header = Header::new(
            operation.message_kind(),
            MessageStatus::None,
            FRAME_ID,
            FRAME_ITEMS,
            0,
        )
        .unwrap();
        assert_eq!(header.kind(), operation.message_kind());
        assert_eq!(Header::decode(&header.encode()).unwrap(), header);
    }
}

#[test]
fn malformed_announced_length_is_rejected_after_payload_is_available() {
    let header = Header::new(
        MessageKind::Broadcast,
        MessageStatus::None,
        FRAME_ID,
        FRAME_ITEMS,
        8,
    )
    .unwrap();
    let error =
        decode_received::<i32>(&header.encode(), &[0_u8; 4], Operation::Broadcast).unwrap_err();
    assert_eq!(error.kind, PlacementErrorKind::Protocol);
    assert_eq!(
        error.message,
        "placement payload length does not match its header"
    );
}

#[test]
fn scatter_preencode_takes_root_shard_without_cloning() {
    let mut shards = vec![Some(String::from("root")), Some(String::from("remote"))];
    let (local, encoded, failure) = preencode_scatter(&mut shards, 0, 2);
    assert_eq!(local.as_deref(), Some("root"));
    assert!(shards[0].is_none());
    assert!(shards[1].is_some());
    assert!(encoded[1].is_some());
    assert!(failure.is_none());
}

#[test]
fn operation_discriminators_are_distinct() {
    assert_ne!(Operation::Broadcast.code(), Operation::Scatter.code());
    assert_ne!(Operation::Scatter.code(), Operation::Gather.code());
    assert_eq!(Operation::Gather.message_kind().code(), 7);
}
