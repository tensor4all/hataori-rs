use super::mpi_api::collective::SystemOperation;
use super::mpi_api::traits::*;
use hataori::{
    pmap, pmap_lanes, Domain, LaneControl, Lanes, LanesError, LocalMode, PmapErrorKind, PmapOptions,
};
use rayon::ThreadPoolBuilder;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

#[derive(Serialize, Deserialize)]
struct RendezvousValue {
    bytes: Vec<u8>,
    mark_on_drop: bool,
    trace_path: String,
}

impl Drop for RendezvousValue {
    fn drop(&mut self) {
        if self.mark_on_drop {
            let _ = std::fs::write(&self.trace_path, b"remote-result-send-completed");
        }
    }
}

#[derive(Serialize)]
struct PrefetchTraceValue {
    value: i32,
    mark_on_deserialize: Option<String>,
    fail_on_deserialize: bool,
}

#[derive(Deserialize)]
struct PrefetchTraceWire {
    value: i32,
    mark_on_deserialize: Option<String>,
    fail_on_deserialize: bool,
}

impl<'de> Deserialize<'de> for PrefetchTraceValue {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let wire = PrefetchTraceWire::deserialize(deserializer)?;
        if let Some(path) = &wire.mark_on_deserialize {
            std::fs::write(path, b"deserialized").map_err(serde::de::Error::custom)?;
        }
        if wire.fail_on_deserialize {
            return Err(serde::de::Error::custom("expected prefetch decode failure"));
        }
        Ok(Self {
            value: wire.value,
            mark_on_deserialize: wire.mark_on_deserialize,
            fail_on_deserialize: false,
        })
    }
}

fn wait_for_trace(path: &str, message: &str) {
    let observed = (0..500).any(|_| {
        if std::path::Path::new(path).exists() {
            true
        } else {
            std::thread::sleep(Duration::from_millis(10));
            false
        }
    });
    assert!(observed, "{message}");
}

#[derive(Deserialize)]
struct AbortSerializeValue {
    value: i32,
    wait_before_failure: Option<String>,
}

impl Serialize for AbortSerializeValue {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        if let Some(path) = &self.wait_before_failure {
            wait_for_trace(
                path,
                "prefetched callback did not start before result failure",
            );
            return Err(serde::ser::Error::custom(
                "expected live-prefetch result serialization failure",
            ));
        }
        (&self.value, &self.wait_before_failure).serialize(serializer)
    }
}

/// The lane resource a fallback callback receives: created by the executor on
/// the lane's own thread, so the callback proves the lane reached it.
struct FallbackLane {
    index: usize,
    calls: usize,
}

/// A one-lane executor that cannot overlap a coordinator with its lane.
///
/// It is the tenferro-style inline case: `run_with_coordinator` reports the
/// typed single-lane rejection and `run` executes the lane on the calling thread.
struct CoordinatorThreadLanes;

impl Lanes for CoordinatorThreadLanes {
    type Lane<'l> = FallbackLane;
    type Error = std::convert::Infallible;

    fn lane_count(&self) -> usize {
        1
    }

    fn run<E: Send>(
        &mut self,
        lane: impl Fn(usize, &mut FallbackLane, &LaneControl<'_>) -> Result<(), E> + Sync,
    ) -> Result<(), LanesError<E, Self::Error>> {
        let control = LaneControl::new();
        let mut state = FallbackLane { index: 0, calls: 0 };
        lane(0, &mut state, &control).map_err(LanesError::Lane)
    }

    fn run_with_coordinator<C, E: Send>(
        &mut self,
        _coordinator: impl FnOnce(&LaneControl<'_>) -> Result<C, E>,
        _lane: impl Fn(usize, &mut FallbackLane, &LaneControl<'_>) -> Result<(), E> + Sync,
    ) -> Result<C, LanesError<E, Self::Error>> {
        Err(LanesError::SingleLane)
    }
}

pub fn run<C: Communicator>(world: &C) {
    let rank = world.rank();
    let main_thread = std::thread::current().id();

    static GLOBAL_WORKERS: AtomicUsize = AtomicUsize::new(0);
    ThreadPoolBuilder::new()
        .num_threads(1)
        .start_handler(|_| {
            GLOBAL_WORKERS.fetch_add(1, Ordering::SeqCst);
        })
        .build_global()
        .expect("global Rayon pool must be unused before the hybrid smoke");
    let pool = Arc::new(ThreadPoolBuilder::new().num_threads(1).build().unwrap());
    let domain = Domain::external(Arc::clone(&pool), vec![0], 1).unwrap();
    let bias = 7_i32;

    if std::env::var_os("HATAORI_HYBRID_PANIC").is_some() {
        let _ = pmap(
            world,
            &domain,
            PmapOptions::default(),
            (rank == 0).then(|| vec![0_i32]),
            |_| -> Result<i32, String> { panic!("expected hybrid callback panic") },
        );
        panic!("hybrid callback panic did not abort the MPI job");
    }

    if let Some(base) = std::env::var_os("HATAORI_PREFETCH_TRANSFER_ABORT") {
        assert_eq!(world.size(), 2);
        let base = base.to_string_lossy();
        let task_trace = format!("{base}.task");
        let callback_trace = format!("{base}.callback");
        if rank == 0 {
            let _ = std::fs::remove_file(&task_trace);
            let _ = std::fs::remove_file(&callback_trace);
        }
        world.barrier();
        let callback_task_trace = task_trace.clone();
        let callback_started_trace = callback_trace.clone();
        let _ = pmap(
            world,
            &domain,
            PmapOptions {
                prefetch: true,
                ..PmapOptions::default()
            },
            (rank == 0).then(|| {
                vec![
                    PrefetchTraceValue {
                        value: 0,
                        mark_on_deserialize: None,
                        fail_on_deserialize: false,
                    },
                    PrefetchTraceValue {
                        value: 1,
                        mark_on_deserialize: None,
                        fail_on_deserialize: false,
                    },
                    PrefetchTraceValue {
                        value: 2,
                        mark_on_deserialize: Some(task_trace),
                        fail_on_deserialize: false,
                    },
                ]
            }),
            |item| {
                if item.value <= 1 {
                    wait_for_trace(
                        &callback_task_trace,
                        "prefetched task did not arrive before current callback",
                    );
                }
                if item.value == 2 {
                    std::fs::write(&callback_started_trace, b"started").unwrap();
                    std::thread::sleep(Duration::from_secs(30));
                }
                Ok::<_, String>(AbortSerializeValue {
                    value: item.value,
                    wait_before_failure: (item.value == 1).then(|| callback_started_trace.clone()),
                })
            },
        );
        panic!("prefetched result serialization failure did not abort the MPI job");
    }

    // Worker-count x LocalMode x prefetch matrix: the lane executor must behave
    // identically for one and two workers, in every mode, with and without
    // prefetch.
    for worker_count in [1_usize, 2] {
        let matrix_pool = Arc::new(
            ThreadPoolBuilder::new()
                .num_threads(worker_count)
                .build()
                .unwrap(),
        );
        let cpus = (0..worker_count).collect::<Vec<usize>>();
        let matrix_domain = Domain::external(Arc::clone(&matrix_pool), cpus, worker_count).unwrap();
        for mode in [LocalMode::Sequential, LocalMode::Outer, LocalMode::Inner] {
            for prefetch in [false, true] {
                let calls = AtomicUsize::new(0);
                let root_participated = AtomicBool::new(false);
                let callback_pool = Arc::clone(&matrix_pool);
                let input = (rank == 0).then(|| (0..16).collect::<Vec<i32>>());
                let result = pmap(
                    world,
                    &matrix_domain,
                    PmapOptions {
                        root: 0,
                        batch_size: NonZeroUsize::new(3).unwrap(),
                        local_mode: mode,
                        prefetch,
                    },
                    input,
                    |item| {
                        assert!(callback_pool.current_thread_index().is_some());
                        calls.fetch_add(1, Ordering::Relaxed);
                        if rank == 0 && item == 0 {
                            root_participated.store(true, Ordering::Relaxed);
                        }
                        if mode == LocalMode::Inner {
                            // Nested same-pool work, including a broadcast that
                            // needs every worker, must complete while the idle
                            // lanes wait for their next batch.
                            let visited = AtomicUsize::new(0);
                            callback_pool.broadcast(|_| {
                                visited.fetch_add(1, Ordering::SeqCst);
                            });
                            assert_eq!(visited.load(Ordering::SeqCst), worker_count);
                            let (left, right) = rayon::join(|| item, || item + 1);
                            assert_eq!(left + 1, right);
                        }
                        Ok::<_, String>(item + bias)
                    },
                )
                .unwrap();
                let local_calls = calls.load(Ordering::Relaxed) as i64;
                let mut total_calls = 0_i64;
                world.all_reduce_into(&local_calls, &mut total_calls, SystemOperation::sum());
                assert_eq!(total_calls, 16);
                if rank == 0 {
                    assert_eq!(
                        result.unwrap(),
                        (0..16).map(|item| item + bias).collect::<Vec<_>>()
                    );
                    assert!(root_participated.load(Ordering::Relaxed));
                } else {
                    assert!(result.is_none());
                }
            }
        }
    }

    if std::env::var_os("HATAORI_SINGLE_LANE_PREFETCH_PANIC").is_some() {
        assert_eq!(world.size(), 2);
        // The single-lane fallback executes batches on the coordinator thread, so
        // a panicking current batch must abort before the prefetched batch is
        // executed. Every remote callback after the first one sleeps, so a wrong
        // ordering is caught by this stage's watchdog instead of finishing.
        let mut lanes = CoordinatorThreadLanes;
        let calls = AtomicUsize::new(0);
        let _ = pmap_lanes(
            world,
            &mut lanes,
            PmapOptions {
                root: 0,
                batch_size: NonZeroUsize::new(1).unwrap(),
                local_mode: LocalMode::Sequential,
                prefetch: true,
            },
            (rank == 0).then(|| (0..16).collect::<Vec<i32>>()),
            |item, lane: &mut FallbackLane| {
                assert_eq!(lane.index, 0);
                lane.calls += 1;
                let previous = calls.fetch_add(1, Ordering::Relaxed);
                if rank != 0 {
                    if previous == 0 {
                        panic!("expected single-lane prefetched-batch callback panic");
                    }
                    std::thread::sleep(Duration::from_secs(30));
                }
                // A slow root keeps the input queue non-empty while the remote
                // rank asks for its prefetched batch.
                std::thread::sleep(Duration::from_millis(20));
                Ok::<_, String>(item)
            },
        );
        panic!("single-lane prefetch callback panic did not abort the MPI job");
    }

    if std::env::var_os("HATAORI_SINGLE_LANE_FALLBACK").is_some() {
        // An executor that reports `SingleLane` must fall back explicitly to
        // running batches on the coordinator thread, for every mode and both
        // prefetch settings, without duplicating any work.
        for mode in [LocalMode::Sequential, LocalMode::Outer, LocalMode::Inner] {
            for prefetch in [false, true] {
                let mut lanes = CoordinatorThreadLanes;
                let calls = AtomicUsize::new(0);
                let input = (rank == 0).then(|| (0..9).collect::<Vec<i32>>());
                let result = pmap_lanes(
                    world,
                    &mut lanes,
                    PmapOptions {
                        root: 0,
                        batch_size: NonZeroUsize::new(2).unwrap(),
                        local_mode: mode,
                        prefetch,
                    },
                    input,
                    |item, lane| {
                        // The executor's lane resource reaches a pmap callback.
                        assert_eq!(lane.index, 0);
                        lane.calls += 1;
                        calls.fetch_add(1, Ordering::Relaxed);
                        Ok::<_, String>(item * 3)
                    },
                )
                .unwrap();
                let local_calls = calls.load(Ordering::Relaxed) as i64;
                let mut total_calls = 0_i64;
                world.all_reduce_into(&local_calls, &mut total_calls, SystemOperation::sum());
                assert_eq!(total_calls, 9);
                if rank == 0 {
                    assert_eq!(
                        result.unwrap(),
                        (0..9).map(|item| item * 3).collect::<Vec<_>>()
                    );
                }
            }
        }
    }

    let prefetched_calls = AtomicUsize::new(0);
    let prefetched_input = (rank == 0).then(|| (0..16).collect::<Vec<i32>>());
    let prefetched = pmap(
        world,
        &domain,
        PmapOptions {
            root: 0,
            batch_size: NonZeroUsize::new(3).unwrap(),
            local_mode: LocalMode::Sequential,
            prefetch: true,
        },
        prefetched_input,
        |item| {
            prefetched_calls.fetch_add(1, Ordering::Relaxed);
            Ok::<_, String>(item * 3)
        },
    )
    .unwrap();
    let local_prefetched_calls = prefetched_calls.load(Ordering::Relaxed) as i64;
    let mut total_prefetched_calls = 0_i64;
    world.all_reduce_into(
        &local_prefetched_calls,
        &mut total_prefetched_calls,
        SystemOperation::sum(),
    );
    assert_eq!(total_prefetched_calls, 16);
    if rank == 0 {
        assert_eq!(
            prefetched.unwrap(),
            (0..16).map(|item| item * 3).collect::<Vec<_>>()
        );
    }

    let empty = pmap(
        world,
        &domain,
        PmapOptions {
            prefetch: true,
            ..PmapOptions::default()
        },
        (rank == 0).then(Vec::<i32>::new),
        Ok::<_, String>,
    )
    .unwrap();
    assert_eq!(empty.is_some(), rank == 0);
    let one = pmap(
        world,
        &domain,
        PmapOptions {
            prefetch: true,
            ..PmapOptions::default()
        },
        (rank == 0).then(|| vec![41_i32]),
        |item| Ok::<_, String>(item + 1),
    )
    .unwrap();
    if rank == 0 {
        assert_eq!(one.unwrap(), vec![42]);
    }

    let error_calls = AtomicUsize::new(0);
    let error = pmap(
        world,
        &domain,
        PmapOptions {
            root: 0,
            batch_size: NonZeroUsize::new(8).unwrap(),
            local_mode: LocalMode::Outer,
            prefetch: false,
        },
        (rank == 0).then(|| (0..8).collect::<Vec<i32>>()),
        |item| {
            error_calls.fetch_add(1, Ordering::Relaxed);
            match item {
                1 => Err::<i32, _>("first hybrid failure".to_owned()),
                4 => Err("later hybrid failure".to_owned()),
                _ => Ok(item),
            }
        },
    )
    .unwrap_err();
    let local_error_calls = error_calls.load(Ordering::Relaxed) as i64;
    let mut total_error_calls = 0_i64;
    world.all_reduce_into(
        &local_error_calls,
        &mut total_error_calls,
        SystemOperation::sum(),
    );
    assert_eq!(total_error_calls, 8);
    assert_eq!(error.kind(), PmapErrorKind::User);
    assert_eq!(error.message(), "first hybrid failure");

    if world.size() > 1 {
        let mismatch_calls = AtomicUsize::new(0);
        let mismatch = pmap(
            world,
            &domain,
            PmapOptions {
                prefetch: rank == 0,
                ..PmapOptions::default()
            },
            (rank == 0).then(|| vec![1_i32]),
            |item| {
                mismatch_calls.fetch_add(1, Ordering::Relaxed);
                Ok::<_, String>(item)
            },
        )
        .unwrap_err();
        assert_eq!(mismatch.kind(), PmapErrorKind::Preflight);
        let local_mismatch_calls = mismatch_calls.load(Ordering::Relaxed) as i64;
        let mut total_mismatch_calls = 0_i64;
        world.all_reduce_into(
            &local_mismatch_calls,
            &mut total_mismatch_calls,
            SystemOperation::sum(),
        );
        assert_eq!(total_mismatch_calls, 0);
    }

    let missing = Domain::sequential();
    let missing_error = pmap(
        world,
        &missing,
        PmapOptions::default(),
        (rank == 0).then(|| vec![1_i32]),
        Ok::<_, String>,
    )
    .unwrap_err();
    assert_eq!(missing_error.kind(), PmapErrorKind::Preflight);

    let busy = domain.try_admit().unwrap();
    let busy_error = pmap(
        world,
        &domain,
        PmapOptions::default(),
        (rank == 0).then(|| vec![2_i32]),
        Ok::<_, String>,
    )
    .unwrap_err();
    assert_eq!(busy_error.kind(), PmapErrorKind::Preflight);
    drop(busy);

    let reused = pmap(
        world,
        &domain,
        PmapOptions::default(),
        (rank == 0).then(|| vec![3_i32, 1, 4]),
        |item| Ok::<_, String>(item * 2),
    )
    .unwrap();
    if rank == 0 {
        assert_eq!(reused.unwrap(), vec![6, 2, 8]);
    }

    if world.size() == 2 {
        if let Ok(trace_path) = std::env::var("HATAORI_HYBRID_TRACE") {
            let task_trace = format!("{trace_path}.prefetched-task");
            let result_trace = format!("{trace_path}.overlapped-result");
            if rank == 0 {
                let _ = std::fs::remove_file(&task_trace);
                let _ = std::fs::remove_file(&result_trace);
            }
            world.barrier();

            let callback_task_trace = task_trace.clone();
            let callback_result_trace = result_trace.clone();
            let overlapped = pmap(
                world,
                &domain,
                PmapOptions {
                    root: 0,
                    batch_size: NonZeroUsize::new(1).unwrap(),
                    local_mode: LocalMode::Sequential,
                    prefetch: true,
                },
                (rank == 0).then(|| {
                    vec![
                        PrefetchTraceValue {
                            value: 0,
                            mark_on_deserialize: None,
                            fail_on_deserialize: false,
                        },
                        PrefetchTraceValue {
                            value: 1,
                            mark_on_deserialize: None,
                            fail_on_deserialize: false,
                        },
                        PrefetchTraceValue {
                            value: 2,
                            mark_on_deserialize: Some(task_trace.clone()),
                            fail_on_deserialize: false,
                        },
                    ]
                }),
                |item| {
                    if item.value <= 1 {
                        wait_for_trace(
                            &callback_task_trace,
                            "task N+1 was not transferred while callback N ran",
                        );
                    }
                    if item.value == 2 {
                        wait_for_trace(
                            &callback_result_trace,
                            "result N was not transferred while callback N+1 ran",
                        );
                    }
                    Ok::<_, String>(PrefetchTraceValue {
                        value: item.value + 10,
                        mark_on_deserialize: (item.value == 1)
                            .then(|| callback_result_trace.clone()),
                        fail_on_deserialize: false,
                    })
                },
            )
            .unwrap();
            if rank == 0 {
                assert_eq!(
                    overlapped
                        .unwrap()
                        .into_iter()
                        .map(|item| item.value)
                        .collect::<Vec<_>>(),
                    vec![10, 11, 12]
                );
                assert!(std::path::Path::new(&task_trace).exists());
                assert!(std::path::Path::new(&result_trace).exists());
                std::fs::remove_file(&task_trace).unwrap();
                std::fs::remove_file(&result_trace).unwrap();
            }
            world.barrier();

            let user_error_calls = AtomicUsize::new(0);
            let user_error_task_trace = format!("{trace_path}.user-prefetched-task");
            let user_error_callback_trace = format!("{trace_path}.user-prefetched-callback");
            if rank == 0 {
                let _ = std::fs::remove_file(&user_error_task_trace);
                let _ = std::fs::remove_file(&user_error_callback_trace);
            }
            world.barrier();
            let callback_task_trace = user_error_task_trace.clone();
            let callback_ran_trace = user_error_callback_trace.clone();
            let user_error = pmap(
                world,
                &domain,
                PmapOptions {
                    root: 0,
                    batch_size: NonZeroUsize::new(1).unwrap(),
                    local_mode: LocalMode::Sequential,
                    prefetch: true,
                },
                (rank == 0).then(|| {
                    vec![
                        PrefetchTraceValue {
                            value: 0,
                            mark_on_deserialize: None,
                            fail_on_deserialize: false,
                        },
                        PrefetchTraceValue {
                            value: 1,
                            mark_on_deserialize: None,
                            fail_on_deserialize: false,
                        },
                        PrefetchTraceValue {
                            value: 2,
                            mark_on_deserialize: Some(user_error_task_trace.clone()),
                            fail_on_deserialize: false,
                        },
                    ]
                }),
                |item| {
                    if item.value <= 1 {
                        wait_for_trace(
                            &callback_task_trace,
                            "prefetched task was not resident before current user error",
                        );
                    }
                    user_error_calls.fetch_add(1, Ordering::Relaxed);
                    match item.value {
                        1 => Err("expected current-batch user error".to_owned()),
                        2 => {
                            std::fs::write(&callback_ran_trace, b"ran").unwrap();
                            Err("later prefetched user error".to_owned())
                        }
                        _ => Ok(item.value),
                    }
                },
            )
            .unwrap_err();
            assert_eq!(user_error.kind(), PmapErrorKind::User);
            assert_eq!(user_error.message(), "expected current-batch user error");
            let local_user_error_calls = user_error_calls.load(Ordering::Relaxed) as i64;
            let mut total_user_error_calls = 0_i64;
            world.all_reduce_into(
                &local_user_error_calls,
                &mut total_user_error_calls,
                SystemOperation::sum(),
            );
            assert_eq!(total_user_error_calls, 3);
            if rank == 0 {
                assert!(std::path::Path::new(&user_error_callback_trace).exists());
                std::fs::remove_file(&user_error_task_trace).unwrap();
                std::fs::remove_file(&user_error_callback_trace).unwrap();
            }
            world.barrier();
            let after_user_error = pmap(
                world,
                &domain,
                PmapOptions {
                    prefetch: true,
                    ..PmapOptions::default()
                },
                (rank == 0).then(|| vec![7_i32]),
                |item| Ok::<_, String>(item + 1),
            )
            .unwrap();
            if rank == 0 {
                assert_eq!(after_user_error.unwrap(), vec![8]);
            }

            for failed_value in [1_i32, 2] {
                let decode_trace = format!("{trace_path}.decode-{failed_value}");
                if rank == 0 {
                    let _ = std::fs::remove_file(&decode_trace);
                }
                world.barrier();
                let callback_decode_trace = decode_trace.clone();
                let decode_calls = AtomicUsize::new(0);
                let decode_error = pmap(
                    world,
                    &domain,
                    PmapOptions {
                        root: 0,
                        batch_size: NonZeroUsize::new(1).unwrap(),
                        local_mode: LocalMode::Sequential,
                        prefetch: true,
                    },
                    (rank == 0).then(|| {
                        (0..3)
                            .map(|value| PrefetchTraceValue {
                                value,
                                mark_on_deserialize: (value == failed_value)
                                    .then(|| decode_trace.clone()),
                                fail_on_deserialize: value == failed_value,
                            })
                            .collect::<Vec<_>>()
                    }),
                    |item| {
                        if item.value == 0 || (failed_value == 2 && item.value == 1) {
                            wait_for_trace(
                                &callback_decode_trace,
                                "decode-failure batch was not transferred while work ran",
                            );
                        }
                        decode_calls.fetch_add(1, Ordering::Relaxed);
                        Ok::<_, String>(item.value)
                    },
                )
                .unwrap_err();
                assert_eq!(decode_error.kind(), PmapErrorKind::Wire);
                let local_decode_calls = decode_calls.load(Ordering::Relaxed) as i64;
                let mut total_decode_calls = 0_i64;
                world.all_reduce_into(
                    &local_decode_calls,
                    &mut total_decode_calls,
                    SystemOperation::sum(),
                );
                assert_eq!(total_decode_calls, i64::from(failed_value));
                if rank == 0 {
                    assert!(std::path::Path::new(&decode_trace).exists());
                    std::fs::remove_file(&decode_trace).unwrap();
                }
                world.barrier();

                let after_decode = pmap(
                    world,
                    &domain,
                    PmapOptions {
                        prefetch: true,
                        ..PmapOptions::default()
                    },
                    (rank == 0).then(|| vec![failed_value]),
                    |item| Ok::<_, String>(item + 10),
                )
                .unwrap();
                if rank == 0 {
                    assert_eq!(after_decode.unwrap(), vec![failed_value + 10]);
                }
            }
        }
    }

    if world.size() > 1 {
        if let Ok(trace_path) = std::env::var("HATAORI_HYBRID_TRACE") {
            if rank == 0 {
                let _ = std::fs::remove_file(&trace_path);
            }
            world.barrier();
            let callback_trace = trace_path.clone();
            let result = pmap(
                world,
                &domain,
                PmapOptions::default(),
                (rank == 0).then(|| vec![false, true]),
                |remote_item| {
                    if remote_item {
                        Ok::<_, String>(RendezvousValue {
                            bytes: vec![0x5a; 2 * 1024 * 1024],
                            mark_on_drop: true,
                            trace_path: callback_trace.clone(),
                        })
                    } else {
                        let observed = (0..500).any(|_| {
                            if std::path::Path::new(&callback_trace).exists() {
                                true
                            } else {
                                std::thread::sleep(Duration::from_millis(10));
                                false
                            }
                        });
                        assert!(
                            observed,
                            "remote rendezvous result was not serviced before root completion"
                        );
                        Ok(RendezvousValue {
                            bytes: vec![0x11; 16],
                            mark_on_drop: false,
                            trace_path: callback_trace.clone(),
                        })
                    }
                },
            )
            .unwrap();
            if rank == 0 {
                let mut values = result.unwrap();
                assert_eq!(values[0].bytes, vec![0x11; 16]);
                assert_eq!(values[1].bytes.len(), 2 * 1024 * 1024);
                assert!(values[1].bytes.iter().all(|byte| *byte == 0x5a));
                values[1].mark_on_drop = false;
                drop(values);
            }
            world.barrier();
            if rank == 0 {
                std::fs::remove_file(&trace_path).unwrap();
            }
        }
    }

    assert_eq!(std::thread::current().id(), main_thread);
    assert_eq!(GLOBAL_WORKERS.load(Ordering::SeqCst), 1);
}
