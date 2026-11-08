use super::*;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{mpsc, Barrier};
use std::thread;

#[test]
fn domain_id_zero_conversion_and_nonzero_error() {
    let zero = DomainId::try_from(0).unwrap();
    assert_eq!(zero, DomainId::ZERO);
    assert_eq!(zero.get(), 0);

    let error = DomainId::try_from(1).unwrap_err();
    assert_eq!(error.requested(), 1);
    assert_eq!(error.to_string(), "unsupported domain ID: 1");
}

#[test]
fn place_valid_getters_and_negative_rank_error() {
    let place = Place::new(3, DomainId::ZERO).unwrap();
    assert_eq!(place.rank(), 3);
    assert_eq!(place.domain_id(), DomainId::ZERO);

    let error = Place::new(-1, DomainId::ZERO).unwrap_err();
    assert_eq!(error.requested(), -1);
    assert_eq!(error.to_string(), "rank must be non-negative: -1");
}

#[test]
fn sequential_domain_has_zero_id_and_one_worker() {
    let domain = Domain::sequential();
    assert_eq!(domain.id(), DomainId::ZERO);
    assert_eq!(domain.worker_count(), 1);
    #[cfg(feature = "rayon")]
    {
        assert_eq!(domain.cpu_set(), None);
        assert_eq!(domain.pool_ownership(), None);
        assert_eq!(domain.placement_status(), None);
        assert!(domain.pool.is_none());
    }
}

#[test]
fn nested_admission_is_busy_and_reacquires_after_drop() {
    let domain = Domain::sequential();
    let admission = domain.try_admit().unwrap();
    assert_eq!(domain.try_admit().unwrap_err(), DomainBusy);
    drop(admission);
    assert!(domain.try_admit().is_ok());
}

fn returns_error(domain: &Domain) -> Result<(), &'static str> {
    let _admission = domain.try_admit().unwrap();
    Err("ordinary error")
}

#[test]
fn admission_releases_after_helper_returns_error() {
    let domain = Domain::sequential();
    assert_eq!(returns_error(&domain), Err("ordinary error"));
    assert!(domain.try_admit().is_ok());
}

#[test]
fn admission_releases_during_unwind() {
    let domain = Domain::sequential();
    let result = catch_unwind(AssertUnwindSafe(|| {
        let _admission = domain.try_admit().unwrap();
        panic!("test panic");
    }));
    assert!(result.is_err());
    assert!(domain.try_admit().is_ok());
}

#[test]
fn scoped_cross_thread_admission_is_nonblocking_and_reacquires() {
    let domain = Domain::sequential();
    let barrier = Barrier::new(2);
    let (release_tx, release_rx) = mpsc::channel();
    let (busy_tx, busy_rx) = mpsc::channel();
    let (released_tx, released_rx) = mpsc::channel();
    let (success_tx, success_rx) = mpsc::channel();
    let domain_ref = &domain;

    thread::scope(|scope| {
        let holder_domain = domain_ref;
        let holder_barrier = &barrier;
        scope.spawn(move || {
            let admission = holder_domain.try_admit().unwrap();
            holder_barrier.wait();
            release_rx.recv().unwrap();
            drop(admission);
            released_tx.send(()).unwrap();
        });

        barrier.wait();
        let contender_domain = domain_ref;
        scope.spawn(move || {
            busy_tx
                .send(matches!(contender_domain.try_admit(), Err(DomainBusy)))
                .unwrap();
            release_tx.send(()).unwrap();
            released_rx.recv().unwrap();
            success_tx
                .send(contender_domain.try_admit().is_ok())
                .unwrap();
        });

        assert!(busy_rx.recv().unwrap());
        assert!(success_rx.recv().unwrap());
    });
}

#[cfg(feature = "rayon")]
struct FailingPinner;

#[cfg(feature = "rayon")]
impl Pinner for FailingPinner {
    fn pin_and_verify(&self, _cpu: usize) -> Result<(), String> {
        Err(String::from("injected failure"))
    }
}

#[cfg(feature = "rayon")]
#[test]
fn managed_rejects_invalid_worker_configuration() {
    assert!(matches!(
        Domain::managed(Vec::new(), 1),
        Err(DomainBuildError::EmptyCpuSet)
    ));
    assert!(matches!(
        Domain::managed(vec![0, 0], 1),
        Err(DomainBuildError::DuplicateCpu { cpu: 0 })
    ));
    assert!(matches!(
        Domain::managed(vec![0], 0),
        Err(DomainBuildError::ZeroWorkers)
    ));
    assert!(matches!(
        Domain::managed(vec![0], 2),
        Err(DomainBuildError::TooManyWorkers {
            worker_count: 2,
            cpu_count: 1
        })
    ));
}

#[cfg(feature = "rayon")]
#[test]
fn external_rejects_pool_size_mismatch() {
    let pool = std::sync::Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap(),
    );
    assert!(matches!(
        Domain::external(pool, vec![0], 1),
        Err(DomainBuildError::PoolSizeMismatch {
            expected: 1,
            actual: 2
        })
    ));
}

#[cfg(feature = "rayon")]
#[test]
fn external_does_not_own_or_pin_caller_pool() {
    let starts = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let starts_in_handler = std::sync::Arc::clone(&starts);
    let caller_pool = std::sync::Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .start_handler(move |_| {
                starts_in_handler.fetch_add(1, Ordering::Relaxed);
            })
            .build()
            .unwrap(),
    );
    let domain = Domain::external(std::sync::Arc::clone(&caller_pool), vec![0], 1).unwrap();
    assert_eq!(domain.pool_ownership(), Some(PoolOwnership::External));
    assert_eq!(
        domain.placement_status(),
        Some(PlacementStatus::CallerDeclared)
    );
    drop(domain);

    caller_pool.install(|| assert_eq!(rayon::current_num_threads(), 1));
    assert_eq!(starts.load(Ordering::Relaxed), 1);
}

#[cfg(all(feature = "rayon", target_os = "linux"))]
#[test]
fn managed_rejects_linux_cpu_bounds_and_process_disallowed_cpu() {
    let out_of_range = libc::CPU_SETSIZE as usize;
    assert!(matches!(
        Domain::managed(vec![out_of_range], 1),
        Err(DomainBuildError::CpuOutOfRange {
            cpu,
            cpu_set_size
        }) if cpu == out_of_range && cpu_set_size == out_of_range
    ));

    let allowed = process_allowed_cpus().unwrap();
    let disallowed = (0..libc::CPU_SETSIZE as usize).find(|cpu| !allowed.contains(cpu));
    let Some(disallowed) = disallowed else {
        return;
    };
    assert!(matches!(
        Domain::managed(vec![disallowed], 1),
        Err(DomainBuildError::CpuNotAllowed { cpu }) if cpu == disallowed
    ));
}

#[cfg(feature = "rayon")]
#[test]
fn managed_pin_failure_is_typed_and_does_not_panic() {
    let result = catch_unwind(AssertUnwindSafe(|| {
        build_managed(vec![0], 1, std::sync::Arc::new(FailingPinner))
    }))
    .unwrap();
    assert!(matches!(result, Err(DomainBuildError::PinFailed { .. })));
}

#[cfg(all(feature = "rayon", target_os = "linux"))]
#[test]
fn managed_workers_are_on_distinct_allowed_cpus() {
    let allowed = process_allowed_cpus().unwrap();
    assert!(!allowed.is_empty());
    let cpus: Vec<_> = allowed.iter().copied().take(2).collect();
    let worker_count = cpus.len().max(1);
    let domain = Domain::managed(cpus.clone(), worker_count).unwrap();
    assert_eq!(domain.placement_status(), Some(PlacementStatus::Verified));
    let observed = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let copy = std::sync::Arc::clone(&observed);
    domain.pool.as_ref().unwrap().broadcast(|_| {
        let cpu = unsafe { libc::sched_getcpu() } as usize;
        copy.lock().unwrap().push(cpu);
    });
    let observed = observed.lock().unwrap();
    assert_eq!(observed.len(), worker_count);
    assert!(observed.iter().all(|cpu| cpus.contains(cpu)));
    let distinct: std::collections::HashSet<_> = observed.iter().copied().collect();
    assert_eq!(distinct.len(), worker_count);
}

#[cfg(all(feature = "rayon", not(target_os = "linux")))]
#[test]
fn managed_non_linux_placement_is_caller_declared() {
    let domain = Domain::managed(vec![0], 1).unwrap();
    assert_eq!(
        domain.placement_status(),
        Some(PlacementStatus::CallerDeclared)
    );
}

#[cfg(feature = "rayon")]
#[test]
fn allowed_cpus_is_non_empty_and_sorted() {
    let cpus = allowed_cpus().unwrap();
    assert!(!cpus.is_empty());
    assert!(cpus.windows(2).all(|pair| pair[0] < pair[1]));
}

/// The mask `allowed_cpus` returns is the one `Domain::managed` validates,
/// against: a CPU outside it is rejected, and the returned ids are accepted.
/// A hardcoded `0..threads` fails this whenever the process mask omits cpu 0.
#[cfg(all(feature = "rayon", target_os = "linux"))]
#[test]
fn allowed_cpus_is_the_mask_managed_validates_against() {
    let cpus = allowed_cpus().unwrap();
    // Compared in full against the mask read here, not sampled: a mask that was
    // truncated to its first entries still contained the one cpu checked below
    // and still missed the one cpu rejected below.
    assert_eq!(
        cpus,
        process_mask(),
        "allowed_cpus must be the whole process affinity mask, ascending"
    );
    if let Some(outside) = (0..libc::CPU_SETSIZE as usize).find(|cpu| !cpus.contains(cpu)) {
        let error = Domain::managed(vec![outside], 1).unwrap_err();
        assert!(
            matches!(error, DomainBuildError::CpuNotAllowed { cpu } if cpu == outside),
            "cpu {outside} is outside the mask {cpus:?}, but managed domain did not reject it: {error}"
        );
    }
    // Whole mask, and more than one worker wherever the mask allows it: a
    // truncated mask rejects a request of three or more workers.
    let workers = cpus.len().min(4);
    let domain = Domain::managed(cpus, workers).unwrap();
    assert_eq!(domain.worker_count(), workers);
}

/// The process affinity mask, read directly: the reference the comparison above
/// needs, independent of `allowed_cpus`.
#[cfg(all(feature = "rayon", target_os = "linux"))]
fn process_mask() -> Vec<usize> {
    let mut set = unsafe { std::mem::zeroed::<libc::cpu_set_t>() };
    let rc =
        unsafe { libc::sched_getaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &mut set) };
    assert_eq!(rc, 0, "sched_getaffinity failed in the test itself");
    (0..libc::CPU_SETSIZE as usize)
        .filter(|cpu| unsafe { libc::CPU_ISSET(*cpu, &set) })
        .collect()
}
