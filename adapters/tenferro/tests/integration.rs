use hataori::{map_in_lanes, Lanes, LanesError, MapInError};
use hataori_tenferro::{NoEngine, TenferroLanes};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Barrier, Mutex};
use std::thread::ThreadId;
use tenferro_cpu::{CpuEngine, Pool};
use tenferro_tensor::{Tensor, TensorElementwise, TensorRead};

/// Doubles one value through the lane's own engine.
///
/// This is a real tenferro operation, not a placeholder: it runs on the engine
/// lent to the lane, on a thread that is not one of that pool's workers, so the
/// providers fan out instead of running serially (threadteam's scheduler does
/// not fan out from inside a pool).
fn double_in_lane(item: f64, engine: &mut CpuEngine) -> Result<f64, String> {
    let x = Tensor::from_vec_col_major(vec![1], vec![item]).map_err(|error| error.to_string())?;
    let sum = engine
        .add_read(TensorRead::from_tensor(&x), TensorRead::from_tensor(&x))
        .map_err(|error| error.to_string())?;
    let values = sum.as_slice::<f64>().map_err(|error| error.to_string())?;
    values
        .first()
        .copied()
        .ok_or_else(|| "a doubled scalar has one element".to_owned())
}

/// Builds one engine with `threads` workers of its own, recording the thread id
/// of every worker it starts.
fn engine_recording(threads: usize) -> (CpuEngine, Arc<Mutex<Vec<ThreadId>>>) {
    let workers = Arc::new(Mutex::new(Vec::new()));
    let sink = Arc::clone(&workers);
    let pool = Pool::builder()
        .threads(threads)
        .on_start(move |_| sink.lock().unwrap().push(std::thread::current().id()))
        .build()
        .expect("a pool for one engine");
    (CpuEngine::new(pool), workers)
}

/// Builds one engine with `threads` workers of its own.
fn engine(threads: usize) -> CpuEngine {
    engine_recording(threads).0
}

/// Two engines whose worker counts differ, so a lane can be told apart by the
/// engine it was lent.
fn two_engines() -> Vec<CpuEngine> {
    vec![engine(2), engine(1)]
}

#[test]
fn map_in_lanes_runs_known_values_on_the_lanes_engines() {
    let mut lanes = TenferroLanes::new(two_engines()).expect("two engines");
    assert_eq!(lanes.lane_count(), 2);

    let visits = Mutex::new(Vec::new());
    let items: Vec<usize> = (1..=6).collect();
    let doubled = map_in_lanes(&mut lanes, items.clone(), |item, engine| {
        let value = double_in_lane(item as f64, engine)?;
        visits.lock().unwrap().push((item, value));
        Ok::<f64, String>(value)
    })
    .expect("map_in_lanes over the lanes' engines");

    assert_eq!(doubled, vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0]);
    let visits = visits.into_inner().unwrap();
    assert_eq!(visits.len(), items.len(), "every item runs exactly once");
    for visit in &visits {
        assert_eq!(visit.1, visit.0 as f64 * 2.0);
    }
}

#[test]
fn each_lane_is_lent_its_own_engine_on_its_own_thread() {
    let (lane0, workers0) = engine_recording(2);
    let (lane1, workers1) = engine_recording(1);
    let workers = [workers0, workers1];
    let mut lanes = TenferroLanes::new(vec![lane0, lane1]).expect("two engines");
    let seen = Mutex::new(Vec::new());
    lanes
        .run(|index, engine, control| {
            assert!(!control.cancelled());
            // A lane is never a worker of the pool it is about to use, so the
            // providers on that engine fan out rather than run serially.
            let thread = std::thread::current().id();
            assert!(
                !workers[index].lock().unwrap().contains(&thread),
                "lane {index} runs inside its engine's pool"
            );
            assert_eq!(
                engine.has_team(),
                engine.num_threads() > 1,
                "a multi-thread lane engine leads a threadteam team"
            );
            seen.lock()
                .unwrap()
                .push((index, engine.num_threads(), thread));
            Ok::<(), String>(())
        })
        .expect("lane census");
    let mut seen = seen.into_inner().unwrap();
    seen.sort_by_key(|row| row.0);
    assert_eq!(seen.len(), 2);
    assert_eq!((seen[0].0, seen[0].1), (0, 2), "lane 0 holds engine 0");
    assert_eq!((seen[1].0, seen[1].1), (1, 1), "lane 1 holds engine 1");
    assert_ne!(seen[0].2, seen[1].2, "lanes run on distinct threads");
}

#[test]
fn map_in_lanes_reports_the_lowest_failing_index() {
    let mut lanes = TenferroLanes::new(two_engines()).expect("two engines");
    let error = map_in_lanes(&mut lanes, vec![0_usize, 1, 2, 3, 4], |item, engine| {
        double_in_lane(item as f64, engine)?;
        if item == 1 || item == 3 {
            return Err(format!("item {item} failed"));
        }
        Ok(item as f64)
    })
    .expect_err("two items fail");
    match error {
        MapInError::Callback(error) => assert_eq!(error.index(), 1, "{error}"),
        other => panic!("expected a callback error, got {other}"),
    }
}

#[test]
fn lane_panic_joins_and_the_executor_is_reusable() {
    let mut lanes = TenferroLanes::new(two_engines()).expect("two engines");
    let panic = catch_unwind(AssertUnwindSafe(|| {
        let _ = map_in_lanes(&mut lanes, vec![0_usize, 1, 2, 3], |item, engine| {
            double_in_lane(item as f64, engine)?;
            assert!(item < 2, "fixture panic on item {item}");
            Ok::<f64, String>(item as f64)
        });
    }));
    assert!(panic.is_err(), "the lane panic is resumed on the caller");

    // Every lane was joined before the panic was resumed, and the caller's
    // engines are still usable for the next fan-out.
    let runs = AtomicUsize::new(0);
    let doubled = map_in_lanes(&mut lanes, vec![2_usize, 3], |item, engine| {
        runs.fetch_add(1, Ordering::AcqRel);
        double_in_lane(item as f64, engine)
    })
    .expect("the engines are reusable after a lane panic");
    assert_eq!(doubled, vec![4.0, 6.0]);
    assert_eq!(runs.load(Ordering::Acquire), 2);
}

#[test]
fn the_lane_count_is_the_number_of_engines() {
    assert_eq!(
        TenferroLanes::new(Vec::new()).unwrap_err(),
        NoEngine,
        "a lane executor needs an engine"
    );
    let one = TenferroLanes::new(vec![engine(1)]).expect("one engine");
    assert_eq!(one.lane_count(), 1);
    let three = TenferroLanes::new(vec![engine(1), engine(1), engine(1)]).expect("three engines");
    assert_eq!(three.lane_count(), 3);
}

#[test]
fn one_lane_still_overlaps_a_coordinator() {
    let mut lanes = TenferroLanes::new(vec![engine(1)]).expect("one engine");
    assert_eq!(lanes.lane_count(), 1);

    // A lane runs on a thread the adapter owns, so the coordinator can always
    // overlap it, whatever the width.
    // Neither party can pass the barrier alone, so reaching it proves the
    // coordinator overlapped the lane.
    let started = Barrier::new(2);
    let coordinator_saw = lanes
        .run_with_coordinator(
            |_control| {
                started.wait();
                Ok::<usize, String>(7)
            },
            |_index, engine, control| {
                assert_eq!(engine.num_threads(), 1);
                started.wait();
                while !control.cancelled() {
                    std::thread::yield_now();
                }
                Ok::<(), String>(())
            },
        )
        .expect("a single lane overlaps the coordinator");
    assert_eq!(coordinator_saw, 7);

    let doubled = map_in_lanes(&mut lanes, vec![1_usize, 2, 3], |item, engine| {
        double_in_lane(item as f64, engine)
    })
    .expect("the single lane still evaluates every item");
    assert_eq!(doubled, vec![2.0, 4.0, 6.0]);
}

#[test]
fn coordinator_return_releases_the_lanes_without_an_explicit_cancel() {
    let mut lanes = TenferroLanes::new(two_engines()).expect("two engines");
    // The coordinator returns normally only after every lane is running, and no
    // callback cancels the control explicitly: the executor's
    // cancellation-on-exit must release the waiting lanes.
    let started = Barrier::new(3);
    let exited = AtomicUsize::new(0);
    let result = lanes
        .run_with_coordinator(
            |_control| {
                started.wait();
                Ok::<usize, String>(17)
            },
            |_index, _engine, control| {
                started.wait();
                while !control.cancelled() {
                    std::thread::yield_now();
                }
                exited.fetch_add(1, Ordering::AcqRel);
                Ok::<(), String>(())
            },
        )
        .expect("the coordinator result survives the lane release");
    assert_eq!(result, 17);
    assert_eq!(exited.load(Ordering::Acquire), 2);
}

#[test]
fn coordinator_error_releases_waiting_lanes() {
    let mut lanes = TenferroLanes::new(two_engines()).expect("two engines");
    let started = Barrier::new(3);
    let exited = AtomicUsize::new(0);
    let error = lanes
        .run_with_coordinator(
            |_control| {
                started.wait();
                Err::<(), String>("coordinator failed".to_owned())
            },
            |_index, _engine, control| {
                started.wait();
                while !control.cancelled() {
                    std::thread::yield_now();
                }
                exited.fetch_add(1, Ordering::AcqRel);
                Ok::<(), String>(())
            },
        )
        .expect_err("the coordinator error is reported");
    assert!(
        matches!(error, LanesError::Coordinator(ref message) if message == "coordinator failed"),
        "{error}"
    );
    assert_eq!(exited.load(Ordering::Acquire), 2);
}

#[test]
fn coordinator_panic_releases_waiting_lanes_and_is_resumed() {
    let mut lanes = TenferroLanes::new(two_engines()).expect("two engines");
    let started = Barrier::new(3);
    let exited = AtomicUsize::new(0);
    let panic = catch_unwind(AssertUnwindSafe(|| {
        let _ = lanes.run_with_coordinator(
            |_control| -> Result<(), String> {
                started.wait();
                panic!("fixture coordinator panic")
            },
            |_index, _engine, control| {
                started.wait();
                while !control.cancelled() {
                    std::thread::yield_now();
                }
                exited.fetch_add(1, Ordering::AcqRel);
                Ok::<(), String>(())
            },
        );
    }));
    assert!(panic.is_err(), "the coordinator panic is resumed");
    assert_eq!(exited.load(Ordering::Acquire), 2);
}

#[test]
fn lane_error_releases_waiting_peers_and_still_joins() {
    let mut lanes = TenferroLanes::new(two_engines()).expect("two engines");
    let started = Barrier::new(2);
    let exited = AtomicUsize::new(0);
    let error = lanes
        .run(|index, _engine, control| {
            started.wait();
            if index == 0 {
                return Err("lane zero failed".to_owned());
            }
            while !control.cancelled() {
                std::thread::yield_now();
            }
            exited.fetch_add(1, Ordering::AcqRel);
            Ok(())
        })
        .expect_err("the lane error is reported");
    assert!(
        matches!(error, LanesError::Lane(ref message) if message == "lane zero failed"),
        "{error}"
    );
    assert_eq!(exited.load(Ordering::Acquire), 1);
}

#[test]
fn lane_unwind_releases_waiting_peers_and_is_resumed() {
    let mut lanes = TenferroLanes::new(two_engines()).expect("two engines");
    let started = Barrier::new(2);
    let exited = AtomicUsize::new(0);
    let panic = catch_unwind(AssertUnwindSafe(|| {
        let _ = lanes.run(|index, _engine, control| {
            started.wait();
            assert!(index != 0, "fixture lane panic");
            while !control.cancelled() {
                std::thread::yield_now();
            }
            exited.fetch_add(1, Ordering::AcqRel);
            Ok::<(), String>(())
        });
    }));
    assert!(panic.is_err(), "the lane panic is resumed after the join");
    assert_eq!(exited.load(Ordering::Acquire), 1);
}
