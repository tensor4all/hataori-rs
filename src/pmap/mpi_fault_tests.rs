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
    let universe = crate::mpi_backend::initialize().expect("codec test owns MPI initialization");
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
    let universe = crate::mpi_backend::initialize().expect("fault test owns MPI initialization");
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
