use hataori::{map_in_lanes, Lanes, LanesError, MapInError};
use hataori_tenferro::TenferroLanes;
use std::collections::{BTreeMap, BTreeSet};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Barrier, Mutex};
use tenferro_cpu::{CpuBackend, PhaseLane};
use tenferro_tensor::{Tensor, TensorRead};

/// Doubles one value through the lane's own child session.
///
/// This is a real tenferro operation, not a placeholder: the tensor is built on
/// the lane's worker and the add runs on that lane's session, so a successful
/// result also proves the item really executed there.
fn double_in_lane(item: f64, lane: &mut PhaseLane<'_>) -> Result<f64, String> {
    let x = Tensor::from_vec_col_major(vec![1], vec![item]).map_err(|error| error.to_string())?;
    let sum = lane
        .session()
        .add_read(TensorRead::from_tensor(&x), TensorRead::from_tensor(&x))
        .map_err(|error| error.to_string())?;
    let values = sum.as_slice::<f64>().map_err(|error| error.to_string())?;
    values
        .first()
        .copied()
        .ok_or_else(|| "a doubled scalar has one element".to_owned())
}

fn thread_name() -> String {
    std::thread::current().name().unwrap_or("").to_owned()
}

/// Builds a backend whose lane count the adapter must report and needs two
/// placeable workers for the multi-lane fixtures.
fn two_worker_backend() -> CpuBackend {
    let backend = CpuBackend::with_threads(2).expect("two tenferro workers");
    assert_eq!(
        backend.num_threads(),
        2,
        "the fixture needs two CPUs the backend can place on"
    );
    backend
}

#[test]
fn map_in_lanes_runs_known_values_on_tenferro_workers() {
    let backend = two_worker_backend();
    let mut lanes = TenferroLanes::open(&backend).expect("held tenferro session");
    assert_eq!(lanes.lane_count(), backend.num_threads());

    let visits = Mutex::new(Vec::new());
    let items: Vec<usize> = (1..=6).collect();
    let doubled = map_in_lanes(&mut lanes, items.clone(), |item, lane| {
        let value = double_in_lane(item as f64, lane)?;
        visits.lock().unwrap().push((item, value, thread_name()));
        Ok::<f64, String>(value)
    })
    .expect("map_in_lanes over tenferro lanes");

    assert_eq!(doubled, vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0]);
    let visits = visits.into_inner().unwrap();
    assert_eq!(visits.len(), items.len(), "every item runs exactly once");
    let seen: BTreeMap<usize, usize> = visits.iter().fold(BTreeMap::new(), |mut seen, visit| {
        *seen.entry(visit.0).or_default() += 1;
        seen
    });
    for item in &items {
        assert_eq!(seen.get(item), Some(&1), "item {item} ran exactly once");
    }
    for visit in &visits {
        assert_eq!(visit.1, visit.0 as f64 * 2.0);
        assert!(
            visit.2.starts_with("tenferro-cpu-"),
            "work runs on a tenferro worker, not on {0:?}",
            visit.2
        );
    }

    // The lane index and worker binding are visible through `Lanes::run`; the
    // adapter creates no worker of its own, so every index names a tenferro
    // worker and the two indices are distinct workers.
    let by_index = Mutex::new(BTreeMap::new());
    lanes
        .run(|index, lane, control| {
            assert!(!control.cancelled());
            assert!(!lane.cancelled());
            let worker = thread_name();
            assert!(
                worker.starts_with("tenferro-cpu-"),
                "lane {index} ran on {worker:?}"
            );
            by_index.lock().unwrap().insert(index, worker);
            Ok::<(), String>(())
        })
        .expect("lane census");
    let by_index = by_index.into_inner().unwrap();
    assert_eq!(
        by_index.keys().copied().collect::<BTreeSet<_>>(),
        BTreeSet::from([0, 1])
    );
    assert_eq!(
        by_index.values().collect::<BTreeSet<_>>().len(),
        2,
        "lane indices name distinct workers"
    );

    lanes.close().expect("close the held session");
}

#[test]
fn map_in_lanes_reports_the_lowest_failing_index() {
    let backend = two_worker_backend();
    let mut lanes = TenferroLanes::open(&backend).expect("held tenferro session");
    let error = map_in_lanes(&mut lanes, vec![0_usize, 1, 2, 3, 4], |item, lane| {
        double_in_lane(item as f64, lane)?;
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
    lanes.close().expect("close the held session");
}

#[test]
fn lane_panic_joins_and_the_session_is_reusable() {
    let backend = two_worker_backend();
    let mut lanes = TenferroLanes::open(&backend).expect("held tenferro session");
    let panic = catch_unwind(AssertUnwindSafe(|| {
        let _ = map_in_lanes(&mut lanes, vec![0_usize, 1, 2, 3], |item, lane| {
            double_in_lane(item as f64, lane)?;
            assert!(item < 2, "fixture panic on item {item}");
            Ok::<f64, String>(item as f64)
        });
    }));
    assert!(panic.is_err(), "the lane panic is resumed on the caller");

    // The phase joined every lane before resuming, and the held session is
    // still admitted and warm for the next fan-out.
    let lanes_run = AtomicUsize::new(0);
    let doubled = map_in_lanes(&mut lanes, vec![2_usize, 3], |item, lane| {
        lanes_run.fetch_add(1, Ordering::AcqRel);
        double_in_lane(item as f64, lane)
    })
    .expect("the session is reusable after a lane panic");
    assert_eq!(doubled, vec![4.0, 6.0]);
    assert_eq!(lanes_run.load(Ordering::Acquire), 2);
    lanes.close().expect("close the held session");
}

#[test]
fn one_worker_backend_rejects_a_coordinator_and_runs_one_inline_lane() {
    let backend = CpuBackend::with_threads(1).expect("one tenferro worker");
    let mut lanes = TenferroLanes::open(&backend).expect("held tenferro session");
    assert_eq!(lanes.lane_count(), 1);

    let coordinator_ran = AtomicUsize::new(0);
    let lane_ran = AtomicUsize::new(0);
    let error = lanes
        .run_with_coordinator(
            |_control| {
                coordinator_ran.fetch_add(1, Ordering::AcqRel);
                Ok::<(), String>(())
            },
            |_index, _lane, _control| {
                lane_ran.fetch_add(1, Ordering::AcqRel);
                Ok::<(), String>(())
            },
        )
        .expect_err("a one-lane phase cannot overlap a coordinator");
    assert!(matches!(error, LanesError::SingleLane), "{error}");
    assert_eq!(coordinator_ran.load(Ordering::Acquire), 0);
    assert_eq!(lane_ran.load(Ordering::Acquire), 0);

    // `map_in_lanes` goes through `Lanes::run`, so the single lane runs inline
    // on the calling thread.
    let caller = thread_name();
    let workers = Mutex::new(BTreeSet::new());
    let doubled = map_in_lanes(&mut lanes, vec![1_usize, 2, 3], |item, lane| {
        workers.lock().unwrap().insert(thread_name());
        double_in_lane(item as f64, lane)
    })
    .expect("the inline lane still evaluates every item");
    assert_eq!(doubled, vec![2.0, 4.0, 6.0]);
    let workers = workers.into_inner().unwrap();
    assert_eq!(workers, BTreeSet::from([caller]));
    lanes.close().expect("close the held session");
}

#[test]
fn coordinator_return_releases_the_lanes_without_an_explicit_cancel() {
    let backend = two_worker_backend();
    let mut lanes = TenferroLanes::open(&backend).expect("held tenferro session");
    // The coordinator plus both lanes; the coordinator returns normally only
    // after every lane is running. No callback cancels the control explicitly:
    // the executor's cancellation-on-exit and tenferro's own stop signal must
    // release the lanes.
    let started = Barrier::new(3);
    let exited = AtomicUsize::new(0);
    let tenferro_cancelled = AtomicUsize::new(0);
    let result = lanes
        .run_with_coordinator(
            |_control| {
                started.wait();
                Ok::<usize, String>(17)
            },
            |_index, lane, control| {
                started.wait();
                while !control.cancelled() {
                    std::thread::yield_now();
                }
                exited.fetch_add(1, Ordering::AcqRel);
                // The coordinator's exit also sets tenferro's own stop signal.
                while !lane.cancelled() {
                    std::thread::yield_now();
                }
                tenferro_cancelled.fetch_add(1, Ordering::AcqRel);
                Ok::<(), String>(())
            },
        )
        .expect("the coordinator result survives the lane release");
    assert_eq!(result, 17);
    assert_eq!(exited.load(Ordering::Acquire), 2);
    assert_eq!(tenferro_cancelled.load(Ordering::Acquire), 2);
    lanes.close().expect("close the held session");
}

#[test]
fn coordinator_error_releases_waiting_lanes() {
    let backend = two_worker_backend();
    let mut lanes = TenferroLanes::open(&backend).expect("held tenferro session");
    let started = Barrier::new(3);
    let exited = AtomicUsize::new(0);
    let error = lanes
        .run_with_coordinator(
            |_control| {
                started.wait();
                Err::<(), String>("coordinator failed".to_owned())
            },
            |_index, _lane, control| {
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
    lanes.close().expect("close the held session");
}

#[test]
fn coordinator_panic_releases_waiting_lanes_and_is_resumed() {
    let backend = two_worker_backend();
    let mut lanes = TenferroLanes::open(&backend).expect("held tenferro session");
    let started = Barrier::new(3);
    let exited = AtomicUsize::new(0);
    let panic = catch_unwind(AssertUnwindSafe(|| {
        let _ = lanes.run_with_coordinator(
            |_control| -> Result<(), String> {
                started.wait();
                panic!("fixture coordinator panic")
            },
            |_index, _lane, control| {
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
    lanes.close().expect("close the held session");
}

#[test]
fn lane_error_releases_waiting_peers_and_still_joins() {
    let backend = two_worker_backend();
    let mut lanes = TenferroLanes::open(&backend).expect("held tenferro session");
    let started = Barrier::new(2);
    let exited = AtomicUsize::new(0);
    let error = lanes
        .run(|index, _lane, control| {
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
    lanes.close().expect("close the held session");
}

#[test]
fn lane_unwind_releases_waiting_peers_and_is_resumed() {
    let backend = two_worker_backend();
    let mut lanes = TenferroLanes::open(&backend).expect("held tenferro session");
    let started = Barrier::new(2);
    let exited = AtomicUsize::new(0);
    let panic = catch_unwind(AssertUnwindSafe(|| {
        let _ = lanes.run(|index, _lane, control| {
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
    lanes.close().expect("close the held session");
}
