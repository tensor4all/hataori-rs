use crate::{
    action::{ActionValue, RegisteredAction},
    error::{ActionError, ResourceKind, RuntimeError},
};
use hataori_runtime_foundation::protocol::{
    ActionId, DomainId, LocalityId, ObjectId, RequestId, TraceId,
};
use rayon::{ThreadPool, ThreadPoolBuilder};
use std::{
    collections::BTreeMap,
    sync::{
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
        mpsc::{self, Receiver, SyncSender},
        Arc,
    },
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DomainConfig {
    pub id: DomainId,
    pub workers: usize,
    pub queue_capacity: usize,
}

impl Default for DomainConfig {
    fn default() -> Self {
        Self {
            id: DomainId::DEFAULT,
            workers: 1,
            queue_capacity: 64,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct DomainStats {
    pub queued: usize,
    pub running: usize,
    pub pending_completions: usize,
    pub completed: u64,
    pub cancelled_before_start: u64,
    pub total_queue_latency_ns: u64,
    pub max_queue_latency_ns: u64,
    pub total_execution_ns: u64,
    pub max_execution_ns: u64,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct ObjectJob {
    pub object: ObjectId,
    pub read: bool,
    pub placement: bool,
}

pub(crate) struct ActionJob {
    pub requester: LocalityId,
    pub request: RequestId,
    pub action_id: ActionId,
    pub domain: DomainId,
    pub trace_id: Option<TraceId>,
    pub input: ActionValue,
    pub handler: RegisteredAction,
    pub cancelled: Arc<AtomicBool>,
    pub local: bool,
    pub submitted_at: std::time::Instant,
    pub object: Option<ObjectJob>,
}

pub(crate) struct ActionCompletion {
    pub requester: LocalityId,
    pub request: RequestId,
    pub action_id: ActionId,
    pub domain: DomainId,
    pub trace_id: Option<TraceId>,
    pub result: Result<ActionValue, ActionError>,
    pub cancelled: bool,
    pub local: bool,
    pub object: Option<ObjectJob>,
}

struct DomainCounters {
    queued: AtomicUsize,
    running: AtomicUsize,
    inflight: AtomicUsize,
    completed: AtomicU64,
    cancelled: AtomicU64,
    queue_latency: AtomicU64,
    max_queue_latency: AtomicU64,
    execution: AtomicU64,
    max_execution: AtomicU64,
}

impl DomainCounters {
    fn snapshot(&self) -> DomainStats {
        let queued = self.queued.load(Ordering::Acquire);
        let running = self.running.load(Ordering::Acquire);
        DomainStats {
            queued,
            running,
            pending_completions: self
                .inflight
                .load(Ordering::Acquire)
                .saturating_sub(queued.saturating_add(running)),
            completed: self.completed.load(Ordering::Relaxed),
            cancelled_before_start: self.cancelled.load(Ordering::Relaxed),
            total_queue_latency_ns: self.queue_latency.load(Ordering::Relaxed),
            max_queue_latency_ns: self.max_queue_latency.load(Ordering::Relaxed),
            total_execution_ns: self.execution.load(Ordering::Relaxed),
            max_execution_ns: self.max_execution.load(Ordering::Relaxed),
        }
    }
}

struct DomainExecutor {
    pool: Arc<ThreadPool>,
    completions: SyncSender<ActionCompletion>,
    counters: Arc<DomainCounters>,
    queue_capacity: usize,
}

pub(crate) struct DomainRegistry {
    domains: BTreeMap<DomainId, DomainExecutor>,
    completions: Receiver<ActionCompletion>,
}

impl DomainRegistry {
    pub(crate) fn new(
        configs: &[DomainConfig],
        completion_capacity: usize,
    ) -> Result<Self, RuntimeError> {
        let (completion_tx, completions) = mpsc::sync_channel(completion_capacity);
        let mut domains = BTreeMap::new();
        for config in configs {
            if config.workers == 0 || config.queue_capacity < config.workers {
                return Err(RuntimeError::InvalidLimits(
                    "domain workers must be nonzero and fit in its queue capacity",
                ));
            }
            if domains.contains_key(&config.id) {
                return Err(RuntimeError::InvalidLimits("duplicate domain id"));
            }
            let domain = config.id;
            let pool = Arc::new(
                ThreadPoolBuilder::new()
                    .num_threads(config.workers)
                    .thread_name(move |worker| format!("hataori-domain-{}-{worker}", domain.get()))
                    .build()
                    .map_err(|_| RuntimeError::InvalidLimits("failed to start domain workers"))?,
            );
            domains.insert(
                config.id,
                DomainExecutor {
                    pool,
                    completions: completion_tx.clone(),
                    counters: Arc::new(DomainCounters {
                        queued: AtomicUsize::new(0),
                        running: AtomicUsize::new(0),
                        inflight: AtomicUsize::new(0),
                        completed: AtomicU64::new(0),
                        cancelled: AtomicU64::new(0),
                        queue_latency: AtomicU64::new(0),
                        max_queue_latency: AtomicU64::new(0),
                        execution: AtomicU64::new(0),
                        max_execution: AtomicU64::new(0),
                    }),
                    queue_capacity: config.queue_capacity,
                },
            );
        }
        Ok(Self {
            domains,
            completions,
        })
    }

    pub(crate) fn pools(&self) -> BTreeMap<DomainId, Arc<ThreadPool>> {
        self.domains
            .iter()
            .map(|(id, domain)| (*id, Arc::clone(&domain.pool)))
            .collect()
    }

    pub(crate) fn can_submit(&self, domain: DomainId) -> bool {
        self.domains.get(&domain).is_some_and(|entry| {
            entry.counters.queued.load(Ordering::Acquire) < entry.queue_capacity
        })
    }

    // `fetch_update` is deprecated on newer toolchains in favor of `try_update`,
    // which does not exist at the crate's MSRV (1.85).
    #[allow(deprecated)]
    pub(crate) fn submit(&self, job: ActionJob) -> Result<(), RuntimeError> {
        let domain = self
            .domains
            .get(&job.domain)
            .ok_or(RuntimeError::UnknownDomain(job.domain))?;
        domain
            .counters
            .queued
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |queued| {
                (queued < domain.queue_capacity).then_some(queued + 1)
            })
            .map_err(|_| RuntimeError::ResourceExhausted {
                resource: ResourceKind::ActionQueue,
                limit: domain.queue_capacity,
            })?;
        domain.counters.inflight.fetch_add(1, Ordering::AcqRel);
        let completions = domain.completions.clone();
        let counters = Arc::clone(&domain.counters);
        domain
            .pool
            .spawn(move || execute_job(job, completions, counters));
        Ok(())
    }

    pub(crate) fn try_completion(&self) -> Option<ActionCompletion> {
        let completion = self.completions.try_recv().ok()?;
        if let Some(domain) = self.domains.get(&completion.domain) {
            domain.counters.inflight.fetch_sub(1, Ordering::AcqRel);
        }
        Some(completion)
    }

    pub(crate) fn stats(&self) -> Vec<(DomainId, DomainStats)> {
        self.domains
            .iter()
            .map(|(id, domain)| (*id, domain.counters.snapshot()))
            .collect()
    }

    pub(crate) fn idle(&self) -> bool {
        self.domains
            .values()
            .all(|domain| domain.counters.inflight.load(Ordering::Acquire) == 0)
    }

    pub(crate) fn stop(&mut self) {
        debug_assert!(self.idle(), "domain stop requires drained work");
    }
}

fn execute_job(
    job: ActionJob,
    completions: SyncSender<ActionCompletion>,
    counters: Arc<DomainCounters>,
) {
    counters.queued.fetch_sub(1, Ordering::AcqRel);
    counters.running.fetch_add(1, Ordering::AcqRel);
    let started_at = std::time::Instant::now();
    let queue_latency = nanos(started_at.saturating_duration_since(job.submitted_at));
    counters
        .queue_latency
        .fetch_add(queue_latency, Ordering::Relaxed);
    counters
        .max_queue_latency
        .fetch_max(queue_latency, Ordering::Relaxed);
    let cancelled = job.cancelled.load(Ordering::Acquire);
    let result = if cancelled {
        counters.cancelled.fetch_add(1, Ordering::Relaxed);
        Err(ActionError::user("action cancelled before execution"))
    } else {
        job.handler.execute(job.input)
    };
    let execution = nanos(started_at.elapsed());
    counters.execution.fetch_add(execution, Ordering::Relaxed);
    counters
        .max_execution
        .fetch_max(execution, Ordering::Relaxed);
    counters.running.fetch_sub(1, Ordering::AcqRel);
    counters.completed.fetch_add(1, Ordering::Relaxed);
    let _ = completions.send(ActionCompletion {
        requester: job.requester,
        request: job.request,
        action_id: job.action_id,
        domain: job.domain,
        trace_id: job.trace_id,
        result,
        cancelled,
        local: job.local,
        object: job.object,
    });
}

fn nanos(duration: std::time::Duration) -> u64 {
    u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests;
