use super::{BatchId, Coordinator, Dispatch, ItemResult, Lane, SchedulerError};
use std::num::NonZeroUsize;
use std::rc::Rc;

fn one() -> NonZeroUsize {
    NonZeroUsize::new(1).unwrap()
}

fn batch<T>(dispatch: Dispatch<T>) -> super::Batch<T> {
    match dispatch {
        Dispatch::Task(batch) => batch,
        Dispatch::Stop => panic!("expected task"),
    }
}

fn indices<T>(batch: super::Batch<T>) -> Vec<usize> {
    batch
        .into_items()
        .into_iter()
        .map(|item| item.original_index())
        .collect()
}

fn values<T>(batch: super::Batch<T>) -> Vec<T> {
    batch
        .into_items()
        .into_iter()
        .map(super::ScheduledItem::into_value)
        .collect()
}

fn result<T>(index: usize, value: T) -> ItemResult<T> {
    ItemResult::new(index, value)
}

#[test]
fn empty_world_size_one_and_more_ranks_than_items() {
    let mut empty = Coordinator::<(), i32>::new(0, vec![], vec![], one()).unwrap();
    assert!(empty.next_root_batch().unwrap().is_none());
    assert!(empty.is_finished());
    assert_eq!(empty.into_results().unwrap(), Vec::<i32>::new());

    let mut one_rank = Coordinator::new(0, vec![], vec![7, 8], one()).unwrap();
    let first = one_rank.next_root_batch().unwrap().unwrap();
    let first_id = first.id();
    assert_eq!(values(first), vec![7]);
    one_rank
        .on_root_success(first_id, vec![result(0, 70)])
        .unwrap();
    let second = one_rank.next_root_batch().unwrap().unwrap();
    let second_id = second.id();
    assert_eq!(values(second), vec![8]);
    one_rank
        .on_root_success(second_id, vec![result(1, 80)])
        .unwrap();
    assert!(one_rank.next_root_batch().unwrap().is_none());
    assert_eq!(one_rank.into_results().unwrap(), vec![70, 80]);

    let mut many = Coordinator::<i32, i32>::new(0, vec![1, 2, 3], vec![9], one()).unwrap();
    assert!(matches!(
        many.on_remote_ready(1).unwrap(),
        Dispatch::Task(_)
    ));
    assert!(matches!(many.on_remote_ready(2).unwrap(), Dispatch::Stop));
    assert!(matches!(many.on_remote_ready(3).unwrap(), Dispatch::Stop));
}

#[test]
fn fifo_batch_one_and_larger_batches() {
    let mut one_batch = Coordinator::<i32, ()>::new(0, vec![], vec![1, 2, 3], one()).unwrap();
    let first = one_batch.next_root_batch().unwrap().unwrap();
    assert_eq!(first.id().get(), 0);
    assert_eq!(values(first), vec![1]);
    let second = one_batch.next_root_batch();
    assert!(matches!(second, Err(SchedulerError::WrongLaneState)));

    let mut larger = Coordinator::<i32, ()>::new(
        0,
        vec![1],
        vec![10, 20, 30, 40, 50],
        NonZeroUsize::new(2).unwrap(),
    )
    .unwrap();
    assert_eq!(
        indices(batch(larger.on_remote_ready(1).unwrap())),
        vec![0, 1]
    );
    let root = larger.next_root_batch().unwrap().unwrap();
    assert_eq!(indices(root), vec![2, 3]);
    assert_eq!(larger.pending_count(), 1);
}

#[test]
fn reverse_completion_restores_order_and_metadata_is_exact() {
    let mut coordinator = Coordinator::new(0, vec![1], vec![10, 20, 30, 40], one()).unwrap();
    let root = coordinator.next_root_batch().unwrap().unwrap();
    let remote = batch(coordinator.on_remote_ready(1).unwrap());
    let remote_id = remote.id();
    let root_id = root.id();
    assert_eq!(coordinator.running_count(), 2);
    let remote_meta = coordinator
        .on_remote_success(1, remote_id, vec![result(1, 200)])
        .unwrap();
    assert_eq!(remote_meta.lane(), Lane::Remote(1));
    assert_eq!(remote_meta.batch_id(), remote_id);
    coordinator
        .on_root_success(root_id, vec![result(0, 100)])
        .unwrap();
    assert_eq!(coordinator.result_count(), 2);
    let next_remote = batch(coordinator.on_remote_ready(1).unwrap());
    let next_root = coordinator.next_root_batch().unwrap().unwrap();
    coordinator
        .on_remote_success(1, next_remote.id(), vec![result(2, 300)])
        .unwrap();
    coordinator
        .on_root_success(next_root.id(), vec![result(3, 400)])
        .unwrap();
    assert!(coordinator.next_root_batch().unwrap().is_none());
    assert!(matches!(
        coordinator.on_remote_ready(1).unwrap(),
        Dispatch::Stop
    ));
    coordinator.on_remote_drain(1).unwrap();
    assert_eq!(
        coordinator.into_results().unwrap(),
        vec![100, 200, 300, 400]
    );
}

#[test]
fn running_slots_are_capacity_one_without_prefetch() {
    let mut coordinator = Coordinator::new(0, vec![1], vec![1, 2, 3], one()).unwrap();
    let root = coordinator.next_root_batch().unwrap().unwrap();
    assert_eq!(coordinator.running_count(), 1);
    let remote = batch(coordinator.on_remote_ready(1).unwrap());
    assert_eq!(coordinator.running_count(), 2);
    assert_eq!(coordinator.pending_count(), 1);
    assert!(matches!(
        coordinator.next_root_batch(),
        Err(SchedulerError::WrongLaneState)
    ));
    assert_eq!(coordinator.pending_count(), 1);
    coordinator
        .on_remote_success(1, remote.id(), vec![result(1, 2)])
        .unwrap();
    assert_eq!(coordinator.running_count(), 1);
    assert_eq!(coordinator.pending_count(), 1);
    coordinator
        .on_root_success(root.id(), vec![result(0, 1)])
        .unwrap();
    assert_eq!(coordinator.running_count(), 0);
    assert_eq!(coordinator.pending_count(), 1);
}

#[test]
fn bounded_prefetch_promotes_in_order_and_never_exceeds_one() {
    let mut coordinator =
        Coordinator::new_with_prefetch(0, vec![1], vec![10, 20, 30], one(), true).unwrap();
    let current = batch(coordinator.on_remote_ready(1).unwrap());
    let prefetched = batch(coordinator.on_remote_ready(1).unwrap());
    assert_eq!(indices(current), vec![0]);
    assert_eq!(indices(prefetched), vec![1]);
    assert_eq!(coordinator.running_count(), 1);
    assert_eq!(coordinator.prefetched_count(), 1);
    assert_eq!(coordinator.pending_count(), 1);
    assert!(matches!(
        coordinator.on_remote_ready(1),
        Err(SchedulerError::WrongLaneState)
    ));

    coordinator
        .on_remote_success(1, BatchId(0), vec![result(0, 100)])
        .unwrap();
    assert_eq!(coordinator.prefetched_count(), 0);
    let third = batch(coordinator.on_remote_ready(1).unwrap());
    assert_eq!(indices(third), vec![2]);
    assert_eq!(coordinator.prefetched_count(), 1);
    coordinator
        .on_remote_success(1, BatchId(1), vec![result(1, 200)])
        .unwrap();
    assert_eq!(coordinator.prefetched_count(), 0);
    assert!(matches!(
        coordinator.on_remote_ready(1).unwrap(),
        Dispatch::Stop
    ));
    coordinator
        .on_remote_success(1, BatchId(2), vec![result(2, 300)])
        .unwrap();
    coordinator.on_remote_drain(1).unwrap();
    assert!(coordinator.next_root_batch().unwrap().is_none());
    assert_eq!(coordinator.into_results().unwrap(), vec![100, 200, 300]);
}

#[test]
fn bounded_prefetch_error_and_protocol_error_retain_next_batch_for_drain() {
    for protocol_error in [false, true] {
        let mut coordinator =
            Coordinator::<i32, i32>::new_with_prefetch(0, vec![1], vec![10, 20, 30], one(), true)
                .unwrap();
        let current = batch(coordinator.on_remote_ready(1).unwrap());
        let prefetched = batch(coordinator.on_remote_ready(1).unwrap());
        if !protocol_error {
            assert_eq!(
                coordinator
                    .on_remote_error(1, BatchId::from_raw(999))
                    .unwrap_err(),
                SchedulerError::BatchMismatch
            );
            assert_eq!(coordinator.running_count(), 1);
            assert_eq!(coordinator.prefetched_count(), 1);
            assert_eq!(coordinator.pending_count(), 1);
        }
        let completion = if protocol_error {
            coordinator.on_remote_protocol_error(1).unwrap()
        } else {
            coordinator.on_remote_error(1, current.id()).unwrap()
        };
        assert_eq!(completion.batch_id(), current.id());
        assert!(coordinator.failed());
        assert_eq!(coordinator.pending_count(), 0);
        assert_eq!(coordinator.running_count(), 1);
        assert_eq!(coordinator.prefetched_count(), 0);
        assert!(matches!(
            coordinator.on_remote_ready(1).unwrap(),
            Dispatch::Stop
        ));
        coordinator
            .on_remote_success(1, prefetched.id(), vec![result(1, 200)])
            .unwrap();
        assert_eq!(coordinator.result_count(), 0);
        coordinator.on_remote_drain(1).unwrap();
        assert!(coordinator.next_root_batch().unwrap().is_none());
        assert_eq!(
            coordinator.into_results().unwrap_err(),
            SchedulerError::Failed
        );
    }
}

#[test]
fn bounded_prefetch_stop_waits_for_current_result_before_drain() {
    let mut coordinator =
        Coordinator::new_with_prefetch(0, vec![1], vec![10], one(), true).unwrap();
    let current = batch(coordinator.on_remote_ready(1).unwrap());
    assert!(matches!(
        coordinator.on_remote_ready(1).unwrap(),
        Dispatch::Stop
    ));
    assert_eq!(
        coordinator.on_remote_drain(1).unwrap_err(),
        SchedulerError::WrongLaneState
    );
    coordinator
        .on_remote_success(1, current.id(), vec![result(0, 100)])
        .unwrap();
    coordinator.on_remote_drain(1).unwrap();
    assert!(coordinator.next_root_batch().unwrap().is_none());
    assert_eq!(coordinator.into_results().unwrap(), vec![100]);
}

#[test]
fn ready_stop_and_drain_are_each_once_and_ordered() {
    let mut coordinator = Coordinator::<(), i32>::new(0, vec![1], vec![], one()).unwrap();
    assert!(matches!(
        coordinator.on_remote_drain(1),
        Err(SchedulerError::WrongLaneState)
    ));
    assert!(matches!(
        coordinator.on_remote_ready(1).unwrap(),
        Dispatch::Stop
    ));
    assert!(matches!(
        coordinator.on_remote_ready(1),
        Err(SchedulerError::WrongLaneState)
    ));
    coordinator.on_remote_drain(1).unwrap();
    assert!(matches!(
        coordinator.on_remote_drain(1),
        Err(SchedulerError::WrongLaneState)
    ));
    assert!(matches!(
        coordinator.on_remote_ready(99),
        Err(SchedulerError::UnknownRank)
    ));
}

#[test]
fn malformed_completion_is_transactional_and_valid_retry_works() {
    let mut coordinator = Coordinator::new(0, vec![], vec![10], one()).unwrap();
    let batch = coordinator.next_root_batch().unwrap().unwrap();
    let id = batch.id();
    let before = (
        coordinator.pending_count(),
        coordinator.running_count(),
        coordinator.result_count(),
    );
    assert_eq!(
        coordinator.on_root_success(id, vec![]).unwrap_err(),
        SchedulerError::ResultCount
    );
    assert_eq!(
        (
            coordinator.pending_count(),
            coordinator.running_count(),
            coordinator.result_count()
        ),
        before
    );
    assert_eq!(
        coordinator
            .on_root_success(id, vec![result(1, 1)])
            .unwrap_err(),
        SchedulerError::IndexOutOfBounds
    );
    assert_eq!(
        (
            coordinator.pending_count(),
            coordinator.running_count(),
            coordinator.result_count()
        ),
        before
    );
    assert_eq!(
        coordinator
            .on_root_success(BatchId(99), vec![result(0, 1)])
            .unwrap_err(),
        SchedulerError::BatchMismatch
    );
    assert_eq!(
        (
            coordinator.pending_count(),
            coordinator.running_count(),
            coordinator.result_count()
        ),
        before
    );
    coordinator
        .on_root_success(id, vec![result(0, 100)])
        .unwrap();
    assert_eq!(coordinator.result_count(), 1);

    let mut already = Coordinator::new(0, vec![], vec![1], one()).unwrap();
    let batch = already.next_root_batch().unwrap().unwrap();
    already.ordered[0] = Some(10);
    assert_eq!(
        already
            .on_root_success(batch.id(), vec![result(0, 20)])
            .unwrap_err(),
        SchedulerError::AlreadyFilled
    );
    assert_eq!(already.running_count(), 1);
    already.ordered[0] = None;
    already
        .on_root_success(batch.id(), vec![result(0, 20)])
        .unwrap();
    assert_eq!(already.result_count(), 1);
}

#[test]
fn duplicate_and_out_of_order_results_are_transactional() {
    let mut coordinator =
        Coordinator::new(0, vec![], vec![10, 20], NonZeroUsize::new(2).unwrap()).unwrap();
    let batch = coordinator.next_root_batch().unwrap().unwrap();
    let id = batch.id();
    let before = (
        coordinator.pending_count(),
        coordinator.running_count(),
        coordinator.result_count(),
    );

    assert_eq!(
        coordinator
            .on_root_success(id, vec![result(1, 20), result(0, 10)])
            .unwrap_err(),
        SchedulerError::IndexSequence
    );
    assert_eq!(
        (
            coordinator.pending_count(),
            coordinator.running_count(),
            coordinator.result_count()
        ),
        before
    );

    assert_eq!(
        coordinator
            .on_root_success(id, vec![result(0, 10), result(0, 20)])
            .unwrap_err(),
        SchedulerError::DuplicateResultIndex
    );
    assert_eq!(
        (
            coordinator.pending_count(),
            coordinator.running_count(),
            coordinator.result_count()
        ),
        before
    );

    coordinator
        .on_root_success(id, vec![result(0, 10), result(1, 20)])
        .unwrap();
    assert_eq!(coordinator.result_count(), 2);
}

#[test]
fn failure_keeps_running_for_validation_then_discards_valid_success() {
    let mut coordinator = Coordinator::new(0, vec![], vec![1, 2], one()).unwrap();
    let batch = coordinator.next_root_batch().unwrap().unwrap();
    coordinator.fail();
    assert!(coordinator.failed());
    assert_eq!(coordinator.pending_count(), 0);
    assert_eq!(
        coordinator.on_root_success(batch.id(), vec![]).unwrap_err(),
        SchedulerError::ResultCount
    );
    assert_eq!(coordinator.running_count(), 1);
    coordinator
        .on_root_success(batch.id(), vec![result(0, 999)])
        .unwrap();
    assert_eq!(coordinator.running_count(), 0);
    assert_eq!(coordinator.result_count(), 0);
    assert!(coordinator.next_root_batch().unwrap().is_none());
    assert_eq!(
        coordinator.into_results().unwrap_err(),
        SchedulerError::Failed
    );
}

#[test]
fn protocol_error_uses_pinned_batch_and_preserves_invalid_state() {
    let mut coordinator = Coordinator::<i32, ()>::new(0, vec![1], vec![1], one()).unwrap();
    assert_eq!(
        coordinator.on_remote_protocol_error(1).unwrap_err(),
        SchedulerError::WrongLaneState
    );
    let batch = batch(coordinator.on_remote_ready(1).unwrap());
    let completion = coordinator.on_remote_protocol_error(1).unwrap();
    assert_eq!(completion.batch_id(), batch.id());
    assert!(coordinator.failed());
    assert_eq!(coordinator.pending_count(), 0);
    assert_eq!(
        coordinator.on_remote_protocol_error(1).unwrap_err(),
        SchedulerError::WrongLaneState
    );
    assert!(matches!(
        coordinator.on_remote_ready(1).unwrap(),
        Dispatch::Stop
    ));
}

#[test]
fn error_completion_clears_only_pending_and_allows_drain() {
    let mut coordinator = Coordinator::<i32, ()>::new(0, vec![1], vec![1, 2, 3], one()).unwrap();
    let batch = batch(coordinator.on_remote_ready(1).unwrap());
    assert_eq!(coordinator.pending_count(), 2);
    coordinator.on_remote_error(1, batch.id()).unwrap();
    assert_eq!(coordinator.pending_count(), 0);
    assert_eq!(coordinator.result_count(), 0);
    assert!(coordinator.failed());
    assert!(matches!(
        coordinator.on_remote_ready(1).unwrap(),
        Dispatch::Stop
    ));
    coordinator.on_remote_drain(1).unwrap();
}

#[test]
fn into_results_rejects_every_not_ready_or_failed_path() {
    let not_ready = Coordinator::<(), i32>::new(0, vec![], vec![()], one()).unwrap();
    assert_eq!(
        not_ready.into_results().unwrap_err(),
        SchedulerError::NotFinished
    );

    let mut no_result = Coordinator::<(), i32>::new(0, vec![], vec![()], one()).unwrap();
    let batch = no_result.next_root_batch().unwrap().unwrap();
    no_result
        .on_root_success(batch.id(), vec![result(0, 3)])
        .unwrap();
    assert!(no_result.next_root_batch().unwrap().is_none());
    no_result.ordered[0] = None;
    assert_eq!(
        no_result.into_results().unwrap_err(),
        SchedulerError::MissingResult
    );

    let mut failed = Coordinator::<(), i32>::new(0, vec![], vec![], one()).unwrap();
    failed.fail();
    assert_eq!(failed.into_results().unwrap_err(), SchedulerError::Failed);
}

#[test]
fn batch_id_overflow_does_not_consume_work() {
    let mut coordinator = Coordinator::<i32, ()>::new(0, vec![], vec![1], one()).unwrap();
    coordinator.next_batch_id = BatchId(u64::MAX);
    assert!(matches!(
        coordinator.next_root_batch(),
        Err(SchedulerError::BatchIdOverflow)
    ));
    assert_eq!(coordinator.pending_count(), 1);
    assert_eq!(coordinator.running_count(), 0);
}

#[test]
fn dynamic_skew_beats_static_contiguous_assignment() {
    let costs = [100, 1, 1, 1];
    let mut coordinator = Coordinator::new(0, vec![1], costs.to_vec(), one()).unwrap();

    let root = coordinator.next_root_batch().unwrap().unwrap();
    let root_id = root.id();
    let root_cost = values(root)[0];

    let mut remote_cost = 0;
    for index in 1..=3 {
        let remote = batch(coordinator.on_remote_ready(1).unwrap());
        let remote_id = remote.id();
        let cost = values(remote)[0];
        remote_cost += cost;
        coordinator
            .on_remote_success(1, remote_id, vec![result(index, cost)])
            .unwrap();
    }
    assert!(matches!(
        coordinator.on_remote_ready(1).unwrap(),
        Dispatch::Stop
    ));
    coordinator.on_remote_drain(1).unwrap();

    coordinator
        .on_root_success(root_id, vec![result(0, root_cost)])
        .unwrap();
    assert!(coordinator.next_root_batch().unwrap().is_none());

    let dynamic_max = root_cost.max(remote_cost);
    let static_max = (costs[0] + costs[1]).max(costs[2] + costs[3]);
    assert_eq!(dynamic_max, 100);
    assert_eq!(static_max, 101);
    assert!(dynamic_max < static_max);
    assert_eq!(coordinator.into_results().unwrap(), costs);
}

#[test]
fn root_values_need_not_be_serializable() {
    let value = Rc::new(7);
    let mut coordinator =
        Coordinator::<Rc<i32>, ()>::new(0, vec![], vec![Rc::clone(&value)], one()).unwrap();
    let batch = coordinator.next_root_batch().unwrap().unwrap();
    assert!(Rc::ptr_eq(&values(batch)[0], &value));
}

#[test]
fn constructor_validates_ranks() {
    assert!(matches!(
        Coordinator::<(), ()>::new(-1, vec![], vec![], one()),
        Err(SchedulerError::InvalidRoot)
    ));
    assert!(matches!(
        Coordinator::<(), ()>::new(0, vec![-1], vec![], one()),
        Err(SchedulerError::InvalidRemote)
    ));
    assert!(matches!(
        Coordinator::<(), ()>::new(0, vec![1, 1], vec![], one()),
        Err(SchedulerError::DuplicateRemote)
    ));
    assert!(matches!(
        Coordinator::<(), ()>::new(0, vec![0], vec![], one()),
        Err(SchedulerError::RootInRemotes)
    ));
}
