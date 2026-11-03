use super::{in_rayon_worker_context, map_in};
use crate::{Domain, DomainBusy, LocalMode, MapInError};
use rayon::ThreadPoolBuilder;
use std::error::Error;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, Once};

fn domain(worker_count: usize) -> (Domain, Arc<rayon::ThreadPool>) {
    let pool = Arc::new(
        ThreadPoolBuilder::new()
            .num_threads(worker_count)
            .build()
            .unwrap(),
    );
    let cpus = (0..worker_count).collect();
    let domain = Domain::external(Arc::clone(&pool), cpus, worker_count).unwrap();
    (domain, pool)
}

#[test]
fn rayon_worker_guard_matches_entry_contexts() {
    assert!(!in_rayon_worker_context());
    assert!(!std::thread::spawn(in_rayon_worker_context).join().unwrap());

    let pool = ThreadPoolBuilder::new().num_threads(1).build().unwrap();
    assert!(pool.install(in_rayon_worker_context));
    assert!(pool.scope(|_| in_rayon_worker_context()));
    assert!(!pool.in_place_scope(|_| in_rayon_worker_context()));

    let (sender, receiver) = mpsc::channel();
    rayon::scope(|scope| {
        assert!(in_rayon_worker_context());
        scope.spawn(|_| sender.send(in_rayon_worker_context()).unwrap());
    });
    assert!(receiver.recv().unwrap());
}

#[test]
fn all_modes_succeed_in_order() {
    let (domain, _) = domain(2);
    for mode in [LocalMode::Sequential, LocalMode::Inner, LocalMode::Outer] {
        let result = map_in(&domain, mode, vec![1, 2, 3], |item| {
            Ok::<_, &'static str>(item * 2)
        })
        .unwrap();
        assert_eq!(result, vec![2, 4, 6]);
    }
}

#[test]
fn sequential_and_inner_stop_at_first_error() {
    for mode in [LocalMode::Sequential, LocalMode::Inner] {
        let (domain, _) = domain(2);
        let calls = Arc::new(AtomicUsize::new(0));
        let seen = Arc::clone(&calls);
        let error = map_in(&domain, mode, vec![0, 1, 2, 3], move |item| {
            seen.fetch_add(1, Ordering::Relaxed);
            if item == 2 {
                Err::<usize, _>("callback failed")
            } else {
                Ok(item)
            }
        })
        .unwrap_err();
        assert!(matches!(error, MapInError::Callback(ref error) if error.index() == 2));
        assert_eq!(calls.load(Ordering::Relaxed), 3);
    }
}

#[test]
fn outer_evaluates_every_input_and_reports_lowest_error() {
    let (domain, _) = domain(2);
    let calls: Arc<Vec<AtomicUsize>> = Arc::new((0..6).map(|_| AtomicUsize::new(0)).collect());
    let seen = Arc::clone(&calls);
    let error = map_in(&domain, LocalMode::Outer, (0..6).collect(), move |item| {
        seen[item].fetch_add(1, Ordering::Relaxed);
        match item {
            1 => Err::<usize, _>("first"),
            4 => Err::<usize, _>("later"),
            _ => Ok(item),
        }
    })
    .unwrap_err();
    assert!(matches!(error, MapInError::Callback(ref error) if error.index() == 1));
    assert_eq!(error.to_string(), "map failed at index 1: first");
    assert!(calls.iter().all(|count| count.load(Ordering::Relaxed) == 1));
}

#[test]
fn preconditions_have_fixed_order_and_never_call_back() {
    let calls = Arc::new(AtomicUsize::new(0));
    let sequential = Domain::sequential();
    let admission = sequential.try_admit().unwrap();
    let seen = Arc::clone(&calls);
    assert!(matches!(
        map_in(&sequential, LocalMode::Sequential, vec![()], move |_| {
            seen.fetch_add(1, Ordering::Relaxed);
            Ok::<_, &'static str>(())
        }),
        Err(MapInError::MissingPool)
    ));
    drop(admission);

    let (domain, pool) = domain(1);
    let foreign_pool = ThreadPoolBuilder::new().num_threads(1).build().unwrap();
    let admission = domain.try_admit().unwrap();
    let seen = Arc::clone(&calls);
    let foreign_and_busy = foreign_pool.install(|| {
        map_in(&domain, LocalMode::Sequential, vec![()], move |_| {
            seen.fetch_add(1, Ordering::Relaxed);
            Ok::<_, &'static str>(())
        })
    });
    assert!(matches!(foreign_and_busy, Err(MapInError::ForeignPool)));
    drop(admission);

    let admission = domain.try_admit().unwrap();
    let seen = Arc::clone(&calls);
    let busy = pool.install(|| {
        map_in(&domain, LocalMode::Sequential, vec![()], move |_| {
            seen.fetch_add(1, Ordering::Relaxed);
            Ok::<_, &'static str>(())
        })
    });
    assert!(matches!(busy, Err(MapInError::DomainBusy(_))));
    drop(admission);
    assert_eq!(calls.load(Ordering::Relaxed), 0);
}

#[test]
fn foreign_custom_and_global_pools_are_rejected_but_same_pool_is_accepted() {
    let (domain, pool) = domain(1);
    let foreign_pool = ThreadPoolBuilder::new().num_threads(1).build().unwrap();
    let custom_result = foreign_pool.install(|| {
        map_in(&domain, LocalMode::Sequential, vec![1], |_| {
            Ok::<_, &'static str>(1)
        })
    });
    assert!(matches!(custom_result, Err(MapInError::ForeignPool)));

    let same_pool_result = pool.install(|| {
        map_in(&domain, LocalMode::Sequential, vec![1], |_| {
            Ok::<_, &'static str>(1)
        })
    });
    assert_eq!(same_pool_result.unwrap(), vec![1]);

    static GLOBAL: Once = Once::new();
    GLOBAL.call_once(|| {
        let _ = ThreadPoolBuilder::new().num_threads(2).build_global();
    });
    let (sender, receiver) = mpsc::channel();
    rayon::scope(|scope| {
        scope.spawn(|_| {
            sender
                .send(map_in(&domain, LocalMode::Sequential, vec![1], |_| {
                    Ok::<_, &'static str>(1)
                }))
                .unwrap();
        });
    });
    assert!(matches!(
        receiver.recv().unwrap(),
        Err(MapInError::ForeignPool)
    ));
}

#[test]
fn inner_callback_can_use_the_target_pool() {
    let (domain, _) = domain(2);
    let result = map_in(&domain, LocalMode::Inner, vec![()], |_| {
        let (thread_count, in_pool) = rayon::join(rayon::current_num_threads, || {
            rayon::current_thread_index().is_some()
        });
        Ok::<_, &'static str>((thread_count, in_pool))
    })
    .unwrap();
    assert_eq!(result, vec![(2, true)]);
}

#[test]
fn admission_is_released_after_success_error_and_unwind() {
    let (domain, _) = domain(1);
    map_in(&domain, LocalMode::Sequential, vec![()], |_| {
        Ok::<_, &'static str>(())
    })
    .unwrap();
    assert!(domain.try_admit().is_ok());

    let error = map_in(&domain, LocalMode::Sequential, vec![()], |_| {
        Err::<(), _>("callback failed")
    });
    assert!(error.is_err());
    assert!(domain.try_admit().is_ok());

    let result = catch_unwind(AssertUnwindSafe(|| {
        map_in(
            &domain,
            LocalMode::Sequential,
            vec![()],
            |_| -> Result<(), &'static str> {
                panic!("callback panic");
            },
        )
        .unwrap();
    }));
    assert!(result.is_err());
    assert!(domain.try_admit().is_ok());
}

#[test]
fn borrowed_inputs_and_sync_stack_capture_need_no_static_bound() {
    let (domain, _) = domain(2);
    let values = [1, 2, 3];
    let bias = 10;
    let captured = &bias;
    let items = values.iter().collect::<Vec<_>>();
    let result = map_in(&domain, LocalMode::Outer, items, |value| {
        Ok::<_, &'static str>(*value + *captured)
    })
    .unwrap();
    assert_eq!(result, vec![11, 12, 13]);
}

#[test]
fn error_sources_and_conversions_are_typed() {
    let busy: MapInError = DomainBusy.into();
    assert!(busy.source().is_some());
    let map_error = crate::map::map(vec![()], |_| Err::<(), _>("bad")).unwrap_err();
    let callback: MapInError = map_error.into();
    assert!(callback.source().is_some());
}
