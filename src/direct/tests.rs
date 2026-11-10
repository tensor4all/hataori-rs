use super::*;
use crate::wire::{
    ElemType, Header, MessageKind, MessageStatus, WireError, DIRECT_CONTROL_LEN, HEADER_LEN,
};

fn framed(kind: MessageKind, batch_id: u64, item_count: u64, payload_len: u64) -> Header {
    let status = if kind == MessageKind::Result {
        MessageStatus::Ok
    } else {
        MessageStatus::None
    };
    Header::new(kind, status, batch_id, item_count, payload_len).unwrap()
}

fn direct_control(item_count: u64, payload_len: u64) -> [u8; DIRECT_CONTROL_LEN] {
    let header = framed(MessageKind::Direct, 0, item_count, payload_len);
    crate::wire::encode_direct_control(header, ElemType::F64)
}

/// Literal wire bytes of every framed kind at the base commit, copied from the
/// old header layout. They pin the existing control plane independently of the
/// current encoder, so changing an encoding and its decoder together still
/// fails here.
const LEGACY_FRAMES: [(MessageKind, [u8; HEADER_LEN]); 8] = [
    (
        MessageKind::Ready,
        [
            1, 0, 0, 0, 8, 7, 6, 5, 4, 3, 2, 1, 17, 0, 0, 0, 0, 0, 0, 0, 123, 0, 0, 0, 0, 0, 0, 0,
        ],
    ),
    (
        MessageKind::Task,
        [
            1, 0, 1, 0, 8, 7, 6, 5, 4, 3, 2, 1, 17, 0, 0, 0, 0, 0, 0, 0, 123, 0, 0, 0, 0, 0, 0, 0,
        ],
    ),
    (
        MessageKind::Result,
        [
            1, 0, 2, 1, 8, 7, 6, 5, 4, 3, 2, 1, 17, 0, 0, 0, 0, 0, 0, 0, 123, 0, 0, 0, 0, 0, 0, 0,
        ],
    ),
    (
        MessageKind::Stop,
        [
            1, 0, 3, 0, 8, 7, 6, 5, 4, 3, 2, 1, 17, 0, 0, 0, 0, 0, 0, 0, 123, 0, 0, 0, 0, 0, 0, 0,
        ],
    ),
    (
        MessageKind::Drain,
        [
            1, 0, 4, 0, 8, 7, 6, 5, 4, 3, 2, 1, 17, 0, 0, 0, 0, 0, 0, 0, 123, 0, 0, 0, 0, 0, 0, 0,
        ],
    ),
    (
        MessageKind::Broadcast,
        [
            1, 0, 5, 0, 8, 7, 6, 5, 4, 3, 2, 1, 17, 0, 0, 0, 0, 0, 0, 0, 123, 0, 0, 0, 0, 0, 0, 0,
        ],
    ),
    (
        MessageKind::Scatter,
        [
            1, 0, 6, 0, 8, 7, 6, 5, 4, 3, 2, 1, 17, 0, 0, 0, 0, 0, 0, 0, 123, 0, 0, 0, 0, 0, 0, 0,
        ],
    ),
    (
        MessageKind::Gather,
        [
            1, 0, 7, 0, 8, 7, 6, 5, 4, 3, 2, 1, 17, 0, 0, 0, 0, 0, 0, 0, 123, 0, 0, 0, 0, 0, 0, 0,
        ],
    ),
];

/// A control message of an existing framed kind is exactly the unchanged
/// 28-byte header, and adding the direct descriptor to one is rejected.
#[test]
fn non_direct_control_messages_are_the_unchanged_header() {
    assert_legacy_frames();
    for (kind, legacy) in LEGACY_FRAMES {
        let encoded = framed(kind, 0x0102_0304_0506_0708, 0x11, 123).encode();
        let (decoded, meta) = Header::decode_control(&legacy).unwrap();
        assert_eq!(decoded.kind(), kind);
        assert_eq!(meta, None);

        let mut padded = encoded.to_vec();
        padded.extend_from_slice(&[ElemType::F64.code(), ByteOrder::NATIVE.code()]);
        assert!(matches!(
            Header::decode_control(&padded),
            Err(WireError::InvalidHeaderSize {
                actual: DIRECT_CONTROL_LEN
            })
        ));
    }
}

fn assert_legacy_frames() {
    for (kind, legacy) in LEGACY_FRAMES {
        assert_eq!(
            framed(kind, 0x0102_0304_0506_0708, 0x11, 123).encode(),
            legacy,
            "{kind:?} control bytes changed"
        );
    }
}

#[test]
fn direct_control_carries_the_descriptor_and_rejects_bad_lengths() {
    let header = framed(MessageKind::Direct, 0, 3, 24);
    let control = crate::wire::encode_direct_control(header, ElemType::F64);
    assert_eq!(control.len(), DIRECT_CONTROL_LEN);
    assert_eq!(
        &control[HEADER_LEN..],
        &[ElemType::F64.code(), ByteOrder::NATIVE.code()]
    );
    assert_eq!(Header::decode_control(&control).unwrap().0, header);

    // A Direct frame without its descriptor and a framed kind with one are both
    // invalid: the kind and the control length must agree.
    assert!(matches!(
        Header::decode_control(&header.encode()),
        Err(WireError::InvalidHeaderSize { .. })
    ));
    let mut padded = framed(MessageKind::Task, 0, 0, 0).encode().to_vec();
    padded.extend_from_slice(&[1, 1]);
    assert!(matches!(
        Header::decode_control(&padded),
        Err(WireError::InvalidHeaderSize { .. })
    ));

    let mut bad_type = direct_control(3, 24);
    bad_type[HEADER_LEN] = 9;
    assert!(matches!(
        Header::decode_control(&bad_type),
        Err(WireError::InvalidDirectDescriptor(9))
    ));
    let mut bad_order = direct_control(3, 24);
    bad_order[HEADER_LEN + 1] = 9;
    assert!(matches!(
        Header::decode_control(&bad_order),
        Err(WireError::InvalidDirectDescriptor(9))
    ));
}

/// Every rejection happens before the receiver allocates the element buffer.
#[test]
fn direct_receive_validates_the_control_message_before_allocating() {
    let values = direct_control(3, 24);
    assert_eq!(decode_control(&values, ElemType::F64).unwrap(), 3);

    assert!(matches!(
        decode_control(&values, ElemType::F32),
        Err(WireError::DirectTypeMismatch {
            expected: ElemType::F32,
            actual: ElemType::F64,
        })
    ));

    let mut other_order = direct_control(3, 24);
    other_order[HEADER_LEN + 1] = match ByteOrder::NATIVE {
        ByteOrder::Little => ByteOrder::Big,
        ByteOrder::Big => ByteOrder::Little,
    }
    .code();
    assert!(matches!(
        decode_control(&other_order, ElemType::F64),
        Err(WireError::DirectByteOrderMismatch { .. })
    ));

    // Three f64 elements are 24 bytes; anything else is a length mismatch.
    assert!(matches!(
        decode_control(&direct_control(3, 25), ElemType::F64),
        Err(WireError::DirectLengthMismatch {
            elements: 3,
            bytes: 25,
        })
    ));

    let task = framed(MessageKind::Task, 0, 0, 0).encode();
    assert!(matches!(
        decode_control(&task, ElemType::F64),
        Err(WireError::UnexpectedDirectKind {
            kind: MessageKind::Task,
        })
    ));

    // The empty buffer is a valid frame: zero elements, zero bytes.
    assert_eq!(
        decode_control(&direct_control(0, 0), ElemType::F64).unwrap(),
        0
    );

    // `i32::MAX` elements cannot fit an MPI count, and the byte estimate would
    // overflow `MAX_WIRE_BYTES` anyway; both must fail instead of allocating.
    assert!(decode_control(&direct_control(i64::MAX as u64, 0), ElemType::F64).is_err());
}

/// Real MPI, both directions, with the three buffers and the empty case in the
/// same exchange. Launched by `scripts/check-faults.sh`.
#[test]
fn direct_payload_roundtrip() {
    if std::env::var_os("HATAORI_DIRECT_PAYLOAD").is_none() {
        return;
    }
    let universe =
        crate::mpi_backend::initialize().expect("direct payload test owns MPI initialization");
    let world = universe.world();
    let rank = mpi_call!(world.rank());
    let size = mpi_call!(world.size());
    assert!(size >= 2, "direct payload test needs at least two ranks");

    // The control plane must be byte-identical on every participating rank.
    assert_legacy_frames();

    let values: Vec<f64> = (0..1000).map(|index| index as f64 * 0.5 - 7.0).collect();
    let narrow: Vec<f32> = vec![1.5, -2.25, 0.0];
    let empty: Vec<f64> = Vec::new();
    let after_empty: Vec<f64> = (0..17).map(|index| index as f64).collect();

    // Root to every other rank: nonempty, f32, empty, nonempty again.
    if rank == 0 {
        for destination in 1..size {
            send(&world, destination, &values).unwrap();
            send(&world, destination, &narrow).unwrap();
            send(&world, destination, &empty).unwrap();
            send(&world, destination, &after_empty).unwrap();
        }
        println!(
            "rank 0/{size} sent f64/1000, f32/3, f64/0, f64/17 to {} ranks",
            size - 1
        );
    } else {
        let received: Vec<f64> = receive(&world, 0).unwrap();
        assert_eq!(received, values, "rank {rank} f64/1000 mismatch");
        let received: Vec<f32> = receive(&world, 0).unwrap();
        assert_eq!(received, narrow, "rank {rank} f32/3 mismatch");
        let received: Vec<f64> = receive(&world, 0).unwrap();
        assert!(received.is_empty(), "rank {rank} f64/0 was not empty");
        let received: Vec<f64> = receive(&world, 0).unwrap();
        assert_eq!(received, after_empty, "rank {rank} f64/17 mismatch");
        println!("rank {rank}/{size} received f64/1000, f32/3, f64/0, f64/17");
    }

    // Every other rank back to the root, with a rank-dependent element count.
    let reply: Vec<f64> = (0..(100 * rank as usize))
        .map(|index| index as f64 + rank as f64)
        .collect();
    if rank == 0 {
        for source in 1..size {
            let received: Vec<f64> = receive(&world, source).unwrap();
            let expected: Vec<f64> = (0..(100 * source as usize))
                .map(|index| index as f64 + source as f64)
                .collect();
            assert_eq!(received, expected, "rank {source} reply mismatch");
        }
        println!(
            "rank 0/{size} received one f64 reply from each of {} ranks",
            size - 1
        );
    } else {
        send(&world, 0, &reply).unwrap();
    }

    // A rejected frame's data message is drained, so the next frame arrives.
    if rank <= 1 {
        if rank == 0 {
            send(&world, 1, &[1.0, 2.0]).unwrap();
            send(&world, 1, &[3.0, 4.0]).unwrap();
        } else {
            assert!(
                matches!(
                    receive::<_, f32>(&world, 0),
                    Err(WireError::DirectTypeMismatch { .. })
                ),
                "a mismatched element type was accepted"
            );
            let received: Vec<f64> = receive(&world, 0).unwrap();
            assert_eq!(received, vec![3.0, 4.0], "the stream was not drained");
            println!("rank 1/2 rejected f32/2 as f64, drained it, then received f64/2");
        }
    }

    // A frame that announces two elements but carries one cannot be drained, and
    // the error must say so instead of looking like a recoverable mismatch.
    if rank <= 1 {
        if rank == 0 {
            let header = Header::new_direct(ElemType::F64, 2).unwrap();
            let control = crate::wire::encode_direct_control(header, ElemType::F64);
            mpi_call!(world
                .process_at_rank(1)
                .send_with_tag(&control[..], CONTROL_TAG));
            mpi_call!(world
                .process_at_rank(1)
                .send_with_tag(&[9.0_f64][..], DATA_TAG));
        } else {
            assert!(
                matches!(
                    receive::<_, f32>(&world, 0),
                    Err(WireError::DirectCountMismatch {
                        expected: 2,
                        actual: 1
                    })
                ),
                "an undrainable frame was not reported as a count mismatch"
            );
            println!(
                "rank 1/2 reported the undrainable f64/2-as-f64/1 frame as DirectCountMismatch"
            );
        }
    }

    mpi_call!(world.barrier());
}

/// Counts every allocation the encoding side makes, so the table below is the
/// Hataori-side allocation cost of one send; MPI's own internals are not
/// instrumented. The counted window excludes building the input `Vec<f64>`.
mod allocation_counter {
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::sync::atomic::{AtomicUsize, Ordering};

    static ALLOCS: AtomicUsize = AtomicUsize::new(0);
    static BYTES: AtomicUsize = AtomicUsize::new(0);

    pub(super) struct CountingAllocator;

    // SAFETY: every method forwards to `System` unchanged and only adds
    // counters, so the allocator contract is the system allocator's.
    unsafe impl GlobalAlloc for CountingAllocator {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            ALLOCS.fetch_add(1, Ordering::Relaxed);
            BYTES.fetch_add(layout.size(), Ordering::Relaxed);
            unsafe { System.alloc(layout) }
        }

        unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
            ALLOCS.fetch_add(1, Ordering::Relaxed);
            BYTES.fetch_add(layout.size(), Ordering::Relaxed);
            unsafe { System.alloc_zeroed(layout) }
        }

        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            unsafe { System.dealloc(ptr, layout) }
        }

        unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
            ALLOCS.fetch_add(1, Ordering::Relaxed);
            BYTES.fetch_add(new_size.saturating_sub(layout.size()), Ordering::Relaxed);
            unsafe { System.realloc(ptr, layout, new_size) }
        }
    }

    #[global_allocator]
    static ALLOCATOR: CountingAllocator = CountingAllocator;

    pub(super) fn reset() {
        ALLOCS.store(0, Ordering::Relaxed);
        BYTES.store(0, Ordering::Relaxed);
    }

    pub(super) fn counts() -> (usize, usize) {
        (
            ALLOCS.load(Ordering::Relaxed),
            BYTES.load(Ordering::Relaxed),
        )
    }
}

/// Measurement harness, not a regression test. Run one exact test on one test
/// thread so libtest activity stays outside the counted window:
///
/// ```sh
/// cargo test --release --no-default-features --features mpi --lib \
///     direct::tests::direct_payload_measurement -- --exact --ignored --nocapture --test-threads=1
/// ```
///
/// Counts are Hataori/Rust-side only: the encoded copy in the codec path, and
/// nothing in the direct path. The MPI send adds no Hataori-side allocation in
/// either path, so this window is where the two differ.
#[test]
#[ignore = "measurement: run explicitly with --release --ignored --nocapture"]
fn direct_payload_measurement() {
    use std::hint::black_box;

    println!("elements\tpath\tcontrol_bytes\tdata_bytes\tencode_allocs\tencode_retained_bytes");
    for elements in [1_000_usize, 100_000, 10_000_000] {
        let values: Vec<f64> = (0..elements).map(|index| index as f64 * 1.5).collect();

        allocation_counter::reset();
        let encoded = crate::wire::encode_payload(&values).unwrap();
        let (allocs, allocated) = allocation_counter::counts();
        println!(
            "{elements}\tbincode\t{HEADER_LEN}\t{}\t{allocs}\t{allocated}",
            encoded.len()
        );

        allocation_counter::reset();
        let header = Header::new_direct(ElemType::F64, elements as u64).unwrap();
        let control = crate::wire::encode_direct_control(header, ElemType::F64);
        let (allocs, allocated) = allocation_counter::counts();
        println!(
            "{elements}\tdirect\t{}\t{}\t{allocs}\t{allocated}",
            control.len(),
            elements * std::mem::size_of::<f64>()
        );

        black_box(&encoded);
        black_box(&control);
        black_box(&values);
    }
}
