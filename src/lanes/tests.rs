use super::{map_in_lanes, LaneControl, Lanes, LanesError, MapInError};
use std::cell::Cell;
use std::fmt;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

/// A lane entry or join failure used by the fake executor.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
struct TestLanesError;

impl fmt::Display for TestLanesError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("test lanes failed")
    }
}

impl std::error::Error for TestLanesError {}

/// Lane-local state created on the worker thread and deliberately not `Send`.
struct LaneState {
    index: usize,
    calls: Cell<usize>,
}

impl LaneState {
    fn new(index: usize) -> Self {
        Self {
            index,
            calls: Cell::new(0),
        }
    }

    fn count(&self) -> usize {
        self.calls.get()
    }
}

/// A fake executor: `Rc`-backed lane state created on the worker, a non-`Send`
/// executor value, and a typed rejection for a concurrent coordinator run.
struct TestLanes {
    worker_count: usize,
    reject_coordinator: bool,
    // Proves that neither the executor nor its lane resource needs `Send`.
    _not_send: Rc<Cell<usize>>,
    entries: Arc<AtomicUsize>,
}

impl TestLanes {
    fn new(worker_count: usize) -> Self {
        Self {
            worker_count,
            reject_coordinator: false,
            _not_send: Rc::new(Cell::new(0)),
            entries: Arc::new(AtomicUsize::new(0)),
        }
    }

    fn rejecting(worker_count: usize) -> Self {
        Self {
            reject_coordinator: true,
            ..Self::new(worker_count)
        }
    }
}

/// What one fake lane reported back to the executor.
enum LaneFailure<E> {
    Error(E),
    Panic(Box<dyn std::any::Any + Send + 'static>),
}

/// A panic payload always wins over an earlier ordinary lane error.
fn prefer_lane_failure<E>(
    current: Option<LaneFailure<E>>,
    next: LaneFailure<E>,
) -> Option<LaneFailure<E>> {
    match (&current, &next) {
        (None, _) | (Some(LaneFailure::Error(_)), LaneFailure::Panic(_)) => Some(next),
        _ => current,
    }
}

fn join<X>(handle: std::thread::ScopedJoinHandle<'_, X>) -> X {
    handle
        .join()
        .unwrap_or_else(|payload| std::panic::resume_unwind(payload))
}

impl Lanes for TestLanes {
    type Lane<'l> = Rc<LaneState>;
    type Error = TestLanesError;

    fn lane_count(&self) -> usize {
        self.worker_count
    }

    fn run<E: Send>(
        &mut self,
        lane: impl Fn(usize, &mut Rc<LaneState>, &LaneControl<'_>) -> Result<(), E> + Sync,
    ) -> Result<(), LanesError<E, Self::Error>> {
        let control = LaneControl::new();
        let lane = &lane;
        let entries = &self.entries;
        let outcomes = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..self.worker_count)
                .map(|index| {
                    let control = &control;
                    scope.spawn(move || {
                        let mut state = Rc::new(LaneState::new(index));
                        entries.fetch_add(1, Ordering::SeqCst);
                        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            lane(index, &mut state, control)
                        })) {
                            Ok(Ok(())) => None,
                            Ok(Err(error)) => {
                                control.cancel();
                                Some(Err(error))
                            }
                            Err(payload) => {
                                control.cancel();
                                Some(Ok(payload))
                            }
                        }
                    })
                })
                .collect();
            handles.into_iter().map(join).collect::<Vec<_>>()
        });
        let mut failure: Option<LaneFailure<E>> = None;
        for outcome in outcomes {
            match outcome {
                None => {}
                Some(Err(error)) => {
                    failure = prefer_lane_failure(failure, LaneFailure::Error(error))
                }
                Some(Ok(payload)) => {
                    failure = prefer_lane_failure(failure, LaneFailure::Panic(payload))
                }
            }
        }
        match failure {
            None => Ok(()),
            Some(LaneFailure::Error(error)) => Err(LanesError::Lane(error)),
            Some(LaneFailure::Panic(payload)) => std::panic::resume_unwind(payload),
        }
    }

    fn run_with_coordinator<C, E: Send>(
        &mut self,
        coordinator: impl FnOnce(&LaneControl<'_>) -> Result<C, E>,
        lane: impl Fn(usize, &mut Rc<LaneState>, &LaneControl<'_>) -> Result<(), E> + Sync,
    ) -> Result<C, LanesError<E, Self::Error>> {
        if self.reject_coordinator {
            return Err(LanesError::SingleLane);
        }
        let control = LaneControl::new();
        let lane = &lane;
        let entries = &self.entries;
        std::thread::scope(|scope| {
            let handles: Vec<_> = (0..self.worker_count)
                .map(|index| {
                    let control = &control;
                    scope.spawn(move || {
                        let mut state = Rc::new(LaneState::new(index));
                        entries.fetch_add(1, Ordering::SeqCst);
                        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            lane(index, &mut state, control)
                        })) {
                            Ok(Ok(())) => None,
                            Ok(Err(error)) => {
                                control.cancel();
                                Some(Err(error))
                            }
                            Err(payload) => {
                                control.cancel();
                                Some(Ok(payload))
                            }
                        }
                    })
                })
                .collect();
            let coordinator_outcome = coordinator(&control);
            control.cancel();
            let mut failure: Option<LaneFailure<E>> = None;
            for handle in handles {
                match join(handle) {
                    None => {}
                    Some(Err(error)) => {
                        failure = prefer_lane_failure(failure, LaneFailure::Error(error))
                    }
                    Some(Ok(payload)) => {
                        failure = prefer_lane_failure(failure, LaneFailure::Panic(payload))
                    }
                }
            }
            match (coordinator_outcome, failure) {
                (Ok(value), None) => Ok(value),
                (Ok(_), Some(LaneFailure::Error(error))) => Err(LanesError::Lane(error)),
                (Ok(_), Some(LaneFailure::Panic(payload))) => std::panic::resume_unwind(payload),
                (Err(error), _) => Err(LanesError::Coordinator(error)),
            }
        })
    }
}

#[test]
fn map_in_lanes_preserves_order_and_evaluates_every_item_once() {
    let mut lanes = TestLanes::new(3);
    let calls: Arc<Mutex<Vec<usize>>> = Arc::new(Mutex::new(Vec::new()));
    let seen = Arc::clone(&calls);
    let items = (0..64).collect::<Vec<usize>>();
    let result = map_in_lanes(&mut lanes, items, move |item, lane| {
        lane.calls.set(lane.count() + 1);
        seen.lock().unwrap().push(item);
        let mut value = item;
        value += 100;
        Ok::<_, &'static str>(value)
    })
    .unwrap();
    assert_eq!(result, (0..64).map(|item| item + 100).collect::<Vec<_>>());

    let mut calls = calls.lock().unwrap().clone();
    calls.sort_unstable();
    assert_eq!(calls, (0..64).collect::<Vec<_>>());
    assert!(lanes.entries.load(Ordering::SeqCst) >= 1);
    assert!(lanes.entries.load(Ordering::SeqCst) <= 3);
}

#[test]
fn map_in_lanes_lane_state_is_per_lane_and_reaches_the_callback() {
    let mut lanes = TestLanes::new(3);
    // The fake executor creates `Rc<LaneState>` on the worker, tagged with the
    // lane index. Recording `(lane, count)` proves the worker-made resource
    // reaches the callback and that each lane keeps its own counter: per lane the
    // counts are consecutive from one, and no lane sees another lane's state.
    let observed: Arc<Mutex<Vec<(usize, usize)>>> = Arc::new(Mutex::new(Vec::new()));
    let seen = Arc::clone(&observed);
    let items = (0..48).collect::<Vec<usize>>();
    map_in_lanes(&mut lanes, items, move |_item, lane| {
        let index = lane.index;
        lane.calls.set(lane.count() + 1);
        seen.lock().unwrap().push((index, lane.count()));
        Ok::<_, &'static str>(())
    })
    .unwrap();

    let observed = observed.lock().unwrap();
    assert_eq!(observed.len(), 48);
    let mut per_lane: std::collections::BTreeMap<usize, usize> = std::collections::BTreeMap::new();
    for (index, count) in observed.iter() {
        assert!(*index < 3, "lane state carries an unexpected lane identity");
        let observed_count = per_lane.entry(*index).or_insert(0);
        assert_eq!(
            *count,
            *observed_count + 1,
            "lane {index} observed another lane's counter"
        );
        *observed_count = *count;
    }
    assert!(!per_lane.is_empty());
}

#[test]
fn map_in_lanes_reports_the_lowest_error_index_after_every_item() {
    let mut lanes = TestLanes::new(3);
    let calls: Arc<Mutex<Vec<usize>>> = Arc::new(Mutex::new(Vec::new()));
    let seen = Arc::clone(&calls);
    let error = map_in_lanes(
        &mut lanes,
        (0..24).collect::<Vec<usize>>(),
        move |item, _lane| {
            seen.lock().unwrap().push(item);
            match item {
                7 => Err("early failure"),
                19 => Err("late failure"),
                _ => Ok(item),
            }
        },
    )
    .unwrap_err();
    assert!(matches!(error, MapInError::Callback(ref error) if error.index() == 7));
    assert_eq!(error.to_string(), "map failed at index 7: early failure");
    let mut calls = calls.lock().unwrap().clone();
    calls.sort_unstable();
    assert_eq!(calls, (0..24).collect::<Vec<_>>());
}

#[test]
fn map_in_lanes_resumes_a_callback_panic_after_joining_every_lane() {
    let mut lanes = TestLanes::new(2);
    let result = catch_unwind(AssertUnwindSafe(|| {
        map_in_lanes(&mut lanes, vec![0_usize, 1, 2, 3], |item, _lane| {
            if item == 1 {
                panic!("lane callback panic");
            }
            Ok::<_, &'static str>(item)
        })
        .unwrap()
    }));
    assert!(result.is_err());
    // The executor joined every lane and remains usable.
    let mut lanes = TestLanes::new(2);
    assert_eq!(
        map_in_lanes(&mut lanes, vec![4_usize], |item, _lane| {
            Ok::<_, &'static str>(item)
        })
        .unwrap(),
        vec![4]
    );
}

#[test]
fn run_with_coordinator_reports_single_lane_without_running_either_callback() {
    let mut lanes = TestLanes::rejecting(2);
    let coordinator_calls = AtomicUsize::new(0);
    let lane_calls = AtomicUsize::new(0);
    let outcome: Result<(), LanesError<&'static str, TestLanesError>> = lanes.run_with_coordinator(
        |_control| {
            coordinator_calls.fetch_add(1, Ordering::SeqCst);
            Ok(())
        },
        |_index, _lane, _control| {
            lane_calls.fetch_add(1, Ordering::SeqCst);
            Ok(())
        },
    );
    assert!(matches!(outcome, Err(LanesError::SingleLane)));
    assert_eq!(coordinator_calls.load(Ordering::SeqCst), 0);
    assert_eq!(lane_calls.load(Ordering::SeqCst), 0);
}

#[test]
fn run_with_coordinator_overlaps_lanes_and_keeps_non_send_state_on_one_thread() {
    let mut lanes = TestLanes::new(2);
    // Non-`Send` coordinator capture and non-`Send` coordinator result.
    let capture = Rc::new(Cell::new(0));
    let coordinator_capture = Rc::clone(&capture);
    let observed = Arc::new(AtomicUsize::new(0));
    let coordinator_seen = Arc::clone(&observed);
    let lane_seen = Arc::clone(&observed);
    let result = lanes
        .run_with_coordinator(
            move |control| {
                // Wait until a lane has run while the coordinator is active.
                while coordinator_seen.load(Ordering::SeqCst) == 0 {
                    if control.cancelled() {
                        break;
                    }
                    std::thread::yield_now();
                }
                coordinator_capture.set(coordinator_capture.get() + 1);
                while coordinator_seen.load(Ordering::SeqCst) < 2 {
                    std::thread::yield_now();
                }
                let message = Rc::new(format!("lanes={}", coordinator_seen.load(Ordering::SeqCst)));
                Ok::<_, &'static str>(message)
            },
            move |_index, _lane, _control| {
                lane_seen.fetch_add(1, Ordering::SeqCst);
                Ok(())
            },
        )
        .unwrap();
    assert_eq!(capture.get(), 1);
    assert_eq!(result.as_str(), "lanes=2");
}

#[test]
fn run_reports_a_lane_error_and_still_joins_every_lane() {
    let mut lanes = TestLanes::new(2);
    let entries = Arc::clone(&lanes.entries);
    let outcome: Result<(), LanesError<&'static str, TestLanesError>> =
        lanes.run(|index, _lane, _control| {
            if index == 1 {
                return Err("lane one failed");
            }
            Ok(())
        });
    assert!(matches!(outcome, Err(LanesError::Lane("lane one failed"))));
    assert_eq!(entries.load(Ordering::SeqCst), 2);
}

#[test]
fn lane_control_is_shared_between_coordinator_and_lanes() {
    let mut lanes = TestLanes::new(2);
    let observed = Arc::new(AtomicUsize::new(0));
    let seen = Arc::clone(&observed);
    lanes
        .run_with_coordinator(
            |control| {
                control.cancel();
                Ok::<_, &'static str>(())
            },
            move |_index, _lane, control| {
                while !control.cancelled() {
                    std::thread::yield_now();
                }
                seen.fetch_add(1, Ordering::SeqCst);
                Ok(())
            },
        )
        .unwrap();
    assert_eq!(observed.load(Ordering::SeqCst), 2);
}

#[test]
fn a_probe_view_shares_the_flag_and_follows_an_external_stop() {
    let control = LaneControl::new();
    let backend_stopped = AtomicBool::new(false);
    let probe = || backend_stopped.load(Ordering::SeqCst);
    let probed = control.with_probe(&probe);
    assert!(!probed.cancelled());
    backend_stopped.store(true, Ordering::SeqCst);
    assert!(probed.cancelled(), "the probe is observed");
    assert!(
        !control.cancelled(),
        "an external stop is not an own-flag cancellation"
    );

    probed.cancel();
    assert!(
        control.cancelled(),
        "views share one cancellation flag, so a coordinator cancels its lanes"
    );

    let other = LaneControl::new();
    let view = other.with_probe(&|| false);
    other.cancel();
    assert!(view.cancelled(), "the shared flag is visible on the view");
}

#[cfg(feature = "rayon")]
mod rayon_lanes {
    use super::super::{map_in_lanes, LaneControl, Lanes, LanesError};
    use crate::{Domain, RayonLanes};
    use std::panic::{catch_unwind, AssertUnwindSafe};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    fn domain(worker_count: usize) -> (Domain, Arc<rayon::ThreadPool>) {
        let pool = Arc::new(
            rayon::ThreadPoolBuilder::new()
                .num_threads(worker_count)
                .build()
                .unwrap(),
        );
        let cpus = (0..worker_count).collect();
        let domain = Domain::external(Arc::clone(&pool), cpus, worker_count).unwrap();
        (domain, pool)
    }

    #[test]
    fn map_in_lanes_matches_outer_semantics_at_every_worker_count() {
        for worker_count in [1, 2, 4] {
            let (domain, _) = domain(worker_count);
            let mut lanes = RayonLanes::new(&domain).unwrap();
            let calls = Arc::new(AtomicUsize::new(0));
            let seen = Arc::clone(&calls);
            let result = map_in_lanes(
                &mut lanes,
                (0..32).collect::<Vec<usize>>(),
                move |item, ()| {
                    seen.fetch_add(1, Ordering::SeqCst);
                    Ok::<_, &'static str>(item * 2)
                },
            )
            .unwrap();
            assert_eq!(result, (0..32).map(|item| item * 2).collect::<Vec<_>>());
            assert_eq!(calls.load(Ordering::SeqCst), 32);
        }
    }

    #[test]
    fn map_in_lanes_reports_the_lowest_error_index_at_every_worker_count() {
        for worker_count in [1, 2, 4] {
            let (domain, _) = domain(worker_count);
            let mut lanes = RayonLanes::new(&domain).unwrap();
            let calls = Arc::new(AtomicUsize::new(0));
            let seen = Arc::clone(&calls);
            let error = map_in_lanes(
                &mut lanes,
                (0..16).collect::<Vec<usize>>(),
                move |item, ()| {
                    seen.fetch_add(1, Ordering::SeqCst);
                    if item == 5 {
                        Err("first")
                    } else if item == 13 {
                        Err("later")
                    } else {
                        Ok(item)
                    }
                },
            )
            .unwrap_err();
            assert!(matches!(
                error,
                super::super::MapInError::Callback(ref error) if error.index() == 5
            ));
            assert_eq!(calls.load(Ordering::SeqCst), 16);
        }
    }

    #[test]
    fn map_in_lanes_resumes_a_callback_panic_after_the_join() {
        for worker_count in [1, 2, 4] {
            let (domain, _) = domain(worker_count);
            let mut lanes = RayonLanes::new(&domain).unwrap();
            let result = catch_unwind(AssertUnwindSafe(|| {
                map_in_lanes(&mut lanes, (0..8).collect::<Vec<usize>>(), |item, ()| {
                    if item == 3 {
                        panic!("rayon lane panic");
                    }
                    Ok::<_, &'static str>(item)
                })
                .unwrap()
            }));
            assert!(result.is_err());
            drop(lanes);
            // Admission is released after the unwind.
            assert!(domain.try_admit().is_ok());
        }
    }

    #[test]
    fn map_in_outer_accepts_a_same_pool_worker_entry() {
        let (domain, pool) = domain(2);
        let result = pool
            .install(|| {
                crate::map_in(
                    &domain,
                    crate::LocalMode::Outer,
                    (0..8).collect::<Vec<usize>>(),
                    |item| Ok::<_, &'static str>(item * 2),
                )
            })
            .unwrap();
        assert_eq!(result, (0..8).map(|item| item * 2).collect::<Vec<_>>());
    }

    #[test]
    fn one_worker_still_runs_a_coordinator_concurrently_with_its_lane() {
        let (domain, _) = domain(1);
        let mut lanes = RayonLanes::new(&domain).unwrap();
        let observed = AtomicUsize::new(0);
        let started = Instant::now();
        let value = lanes
            .run_with_coordinator(
                |control| {
                    while observed.load(Ordering::SeqCst) == 0 {
                        assert!(!control.cancelled());
                        assert!(
                            started.elapsed() < Duration::from_secs(10),
                            "the lane never ran while the coordinator was active"
                        );
                        std::thread::yield_now();
                    }
                    Ok::<_, &'static str>(7)
                },
                |_index, (), _control| {
                    observed.fetch_add(1, Ordering::SeqCst);
                    Ok(())
                },
            )
            .unwrap();
        assert_eq!(value, 7);
        assert_eq!(observed.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn coordinator_error_and_panic_cancel_and_join_the_lanes() {
        let (domain, _) = domain(2);
        let mut lanes = RayonLanes::new(&domain).unwrap();
        let stopped = Arc::new(AtomicUsize::new(0));
        let seen = Arc::clone(&stopped);
        let outcome: Result<(), LanesError<&'static str, std::convert::Infallible>> = lanes
            .run_with_coordinator(
                |_control| Err("coordinator failed"),
                move |_index, (), control| {
                    let deadline = Instant::now() + Duration::from_secs(10);
                    while !control.cancelled() {
                        assert!(Instant::now() < deadline, "lane was never cancelled");
                        std::thread::yield_now();
                    }
                    seen.fetch_add(1, Ordering::SeqCst);
                    Ok(())
                },
            );
        assert!(matches!(
            outcome,
            Err(LanesError::Coordinator("coordinator failed"))
        ));
        assert_eq!(stopped.load(Ordering::SeqCst), 2);

        let result = catch_unwind(AssertUnwindSafe(|| {
            let _ = lanes.run_with_coordinator(
                |_control| -> Result<(), &'static str> { panic!("coordinator panic") },
                |_index, (), control: &LaneControl<'_>| {
                    while !control.cancelled() {
                        std::thread::yield_now();
                    }
                    Ok(())
                },
            );
        }));
        assert!(result.is_err());
        drop(lanes);
        assert!(domain.try_admit().is_ok());
    }

    #[test]
    fn lane_panic_is_resumed_after_the_join() {
        let (domain, _) = domain(2);
        let mut lanes = RayonLanes::new(&domain).unwrap();
        let result = catch_unwind(AssertUnwindSafe(|| {
            let _ = lanes.run(|index, (), _control| {
                if index == 0 {
                    panic!("lane panic");
                }
                Ok::<_, std::convert::Infallible>(())
            });
        }));
        assert!(result.is_err());
        drop(lanes);
        assert!(domain.try_admit().is_ok());
    }

    fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
        payload
            .downcast_ref::<&'static str>()
            .map(|message| message.to_string())
            .or_else(|| payload.downcast_ref::<String>().cloned())
            .unwrap_or_else(|| String::from("<non-string panic>"))
    }

    #[test]
    fn a_lane_error_cancels_the_control_so_waiting_lanes_are_released() {
        let (domain, _) = domain(2);
        let mut lanes = RayonLanes::new(&domain).unwrap();
        // Lane 1 cooperatively waits for cancellation; only an immediate cancel
        // on lane 0's error releases it and lets the run finish.
        let deadline = Instant::now() + Duration::from_secs(10);
        let outcome: Result<(), LanesError<&'static str, std::convert::Infallible>> =
            lanes.run(|index, (), control| {
                if index == 0 {
                    return Err("lane 0 failed");
                }
                while !control.cancelled() {
                    assert!(
                        Instant::now() < deadline,
                        "lane error never cancelled the run"
                    );
                    std::thread::yield_now();
                }
                Ok(())
            });
        assert!(matches!(outcome, Err(LanesError::Lane("lane 0 failed"))));
        drop(lanes);
        assert!(domain.try_admit().is_ok());
    }

    #[test]
    fn run_prefers_a_lane_panic_over_an_earlier_lane_error() {
        let (domain, _) = domain(2);
        let mut lanes = RayonLanes::new(&domain).unwrap();
        let errored = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let seen = Arc::clone(&errored);
        let result = catch_unwind(AssertUnwindSafe(|| {
            let outcome: Result<(), LanesError<&'static str, std::convert::Infallible>> = lanes
                .run(|index, (), _control| {
                    if index == 0 {
                        seen.store(true, Ordering::SeqCst);
                        return Err("lane 0 failed");
                    }
                    while !seen.load(Ordering::SeqCst) {
                        std::thread::yield_now();
                    }
                    panic!("later lane panic");
                });
            assert!(outcome.is_ok());
        }));
        let payload = result.expect_err("the lane panic must be resumed");
        assert_eq!(panic_message(payload), "later lane panic");
        drop(lanes);
        assert!(domain.try_admit().is_ok());
    }

    #[test]
    fn run_with_coordinator_prefers_a_lane_panic_over_an_earlier_lane_error() {
        let (domain, _) = domain(2);
        let mut lanes = RayonLanes::new(&domain).unwrap();
        let errored = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let seen = Arc::clone(&errored);
        let observed = Arc::clone(&errored);
        let result = catch_unwind(AssertUnwindSafe(|| {
            let outcome: Result<(), LanesError<&'static str, std::convert::Infallible>> = lanes
                .run_with_coordinator(
                    |_control| {
                        while !observed.load(Ordering::SeqCst) {
                            std::thread::yield_now();
                        }
                        Ok(())
                    },
                    |index, (), _control| {
                        if index == 0 {
                            seen.store(true, Ordering::SeqCst);
                            return Err("lane 0 failed");
                        }
                        while !seen.load(Ordering::SeqCst) {
                            std::thread::yield_now();
                        }
                        panic!("later lane panic");
                    },
                );
            assert!(outcome.is_ok());
        }));
        let payload = result.expect_err("the lane panic must be resumed");
        assert_eq!(panic_message(payload), "later lane panic");
        drop(lanes);
        assert!(domain.try_admit().is_ok());
    }

    #[test]
    fn run_with_coordinator_rejects_a_same_pool_worker_before_spawning() {
        let (domain, pool) = domain(2);
        let mut lanes = RayonLanes::new(&domain).unwrap();
        let coordinator_calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let lane_calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let coordinator_seen = Arc::clone(&coordinator_calls);
        let lane_seen = Arc::clone(&lane_calls);
        let outcome = pool.install(|| {
            lanes.run_with_coordinator(
                move |_control| {
                    coordinator_seen.fetch_add(1, Ordering::SeqCst);
                    Ok::<_, &'static str>(())
                },
                move |_index, (), _control| {
                    lane_seen.fetch_add(1, Ordering::SeqCst);
                    Ok(())
                },
            )
        });
        assert!(matches!(outcome, Err(LanesError::Reentrant)));
        assert_eq!(coordinator_calls.load(Ordering::SeqCst), 0);
        assert_eq!(lane_calls.load(Ordering::SeqCst), 0);
        // The lane-only entry still supports a same-pool worker caller.
        let visited = AtomicUsize::new(0);
        let value = pool.install(|| {
            lanes.run(|_index, (), _control| {
                visited.fetch_add(1, Ordering::SeqCst);
                Ok::<_, std::convert::Infallible>(())
            })
        });
        assert!(value.is_ok());
        assert_eq!(visited.load(Ordering::SeqCst), 2);
    }
}

#[cfg(all(feature = "rayon", any(feature = "mpi", feature = "rsmpi-rt")))]
mod batch {
    use super::super::batch::{
        idle_backoff, run_lane, LocalBatch, LocalBatchOutcome, LocalBatchResult, LocalQueue,
    };
    use crate::LocalMode;
    use crate::{Domain, Lanes, RayonLanes};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    fn domain(worker_count: usize) -> (Domain, Arc<rayon::ThreadPool>) {
        let pool = Arc::new(
            rayon::ThreadPoolBuilder::new()
                .num_threads(worker_count)
                .build()
                .unwrap(),
        );
        let cpus = (0..worker_count).collect();
        let domain = Domain::external(Arc::clone(&pool), cpus, worker_count).unwrap();
        (domain, pool)
    }

    /// Runs one batch through the lanes and returns its outcome.
    fn run_batch<T, U, E, F>(
        lanes: &mut RayonLanes<'_>,
        mode: LocalMode,
        items: Vec<(usize, T)>,
        f: F,
    ) -> LocalBatchOutcome<U>
    where
        T: Send,
        U: Send,
        E: std::fmt::Display + Send,
        F: Fn(T, &mut ()) -> Result<U, E> + Sync,
    {
        let queue = LocalQueue::<T, U>::new();
        queue.submit(LocalBatch::new(0, mode, items));
        lanes
            .run_with_coordinator(
                |control| loop {
                    if let Some(outcome) = queue.take_outcome(0) {
                        queue.close();
                        return Ok::<_, std::convert::Infallible>(outcome);
                    }
                    assert!(!control.cancelled());
                    idle_backoff();
                },
                |_index, (), control| {
                    run_lane(&queue, &f, control, &mut ());
                    Ok(())
                },
            )
            .unwrap()
    }

    #[test]
    fn outer_batch_spreads_over_lanes_and_orders_values() {
        let (domain, _) = domain(2);
        let mut lanes = RayonLanes::new(&domain).unwrap();
        let outcome = run_batch(
            &mut lanes,
            LocalMode::Outer,
            (0..8).map(|index| (index, index as i32)).collect(),
            |item, ()| Ok::<_, &'static str>(item * 10),
        );
        let LocalBatchResult::Values(values) = outcome.result else {
            panic!("expected values");
        };
        assert_eq!(
            values,
            (0..8)
                .map(|index| (index, index as i32 * 10))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn outer_batch_reports_the_lowest_error_after_every_item() {
        let (domain, _) = domain(2);
        let mut lanes = RayonLanes::new(&domain).unwrap();
        let calls = Arc::new(AtomicUsize::new(0));
        let seen = Arc::clone(&calls);
        let outcome = run_batch(
            &mut lanes,
            LocalMode::Outer,
            (0..6).map(|index| (index, index)).collect(),
            move |item, ()| {
                seen.fetch_add(1, Ordering::SeqCst);
                match item {
                    2 => Err("first"),
                    5 => Err("later"),
                    _ => Ok(item),
                }
            },
        );
        assert_eq!(calls.load(Ordering::SeqCst), 6);
        assert!(matches!(
            outcome.result,
            LocalBatchResult::CallbackError { index: 2, .. }
        ));
    }

    #[test]
    fn sequential_batch_runs_on_one_lane_and_stops_at_the_first_error() {
        let (domain, _) = domain(2);
        let mut lanes = RayonLanes::new(&domain).unwrap();
        let calls = Arc::new(AtomicUsize::new(0));
        let seen = Arc::clone(&calls);
        let outcome = run_batch(
            &mut lanes,
            LocalMode::Sequential,
            (0..5).map(|index| (index, index)).collect(),
            move |item, ()| {
                seen.fetch_add(1, Ordering::SeqCst);
                if item == 1 {
                    Err("stop")
                } else {
                    Ok(item)
                }
            },
        );
        assert_eq!(calls.load(Ordering::SeqCst), 2);
        assert!(matches!(
            outcome.result,
            LocalBatchResult::CallbackError { index: 1, .. }
        ));
    }

    #[test]
    fn a_callback_panic_becomes_a_panic_outcome() {
        let (domain, _) = domain(2);
        let mut lanes = RayonLanes::new(&domain).unwrap();
        let outcome = run_batch(
            &mut lanes,
            LocalMode::Outer,
            (0..4).map(|index| (index, index)).collect(),
            |item, ()| {
                if item == 1 {
                    panic!("batch callback panic");
                }
                Ok::<_, &'static str>(item)
            },
        );
        assert!(matches!(outcome.result, LocalBatchResult::Panic));
    }

    #[test]
    fn an_idle_lane_still_runs_nested_pool_work_from_another_lane() {
        let (domain, pool) = domain(2);
        let mut lanes = RayonLanes::new(&domain).unwrap();
        let nested = Arc::clone(&pool);
        let items = vec![(0_usize, 1_i32)];
        let outcome = run_batch(&mut lanes, LocalMode::Sequential, items, move |item, ()| {
            // A `broadcast` from a lane needs every worker of this pool,
            // including the lane that is waiting for work.
            let visited = AtomicUsize::new(0);
            nested.broadcast(|_| {
                visited.fetch_add(1, Ordering::SeqCst);
            });
            let (left, right) = rayon::join(|| item, || item + 1);
            Ok::<_, &'static str>((visited.load(Ordering::SeqCst), left + right))
        });
        let LocalBatchResult::Values(values) = outcome.result else {
            panic!("expected values");
        };
        assert_eq!(values, vec![(0, (2, 3))]);
    }

    #[test]
    fn queue_drains_after_close_and_rejects_a_second_resident_batch() {
        let queue = LocalQueue::<usize, usize>::new();
        queue.submit(LocalBatch::new(0, LocalMode::Outer, vec![(0, 1), (1, 2)]));
        assert!(matches!(
            queue.claim(),
            Some(super::super::batch::LaneWork::Item { index: 0, .. })
        ));
        queue.close();
        assert!(queue.is_closed());
        let remaining = queue.claim();
        assert!(matches!(
            remaining,
            Some(super::super::batch::LaneWork::Item { index: 1, .. })
        ));
        assert!(queue.claim().is_none());
    }
}
