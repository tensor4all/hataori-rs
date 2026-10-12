use crate::atomic::AtomicTryUpdate;
use crate::{
    action::{Action, ActionRegistry, ActionValue, Segments},
    dedup::{DedupDisposition, DedupTable},
    domain::{ActionCompletion, ActionJob, DomainConfig, DomainRegistry, DomainStats},
    error::{cap, ResourceKind, RuntimeError, RuntimeState},
    object::{
        create_action_id, decode_location, encode_location, placement_action_id, ColocatedFuture,
        CreateFuture, DistributedObject, LeaseTransfer, LocalLeaseManager, MigrationFuture,
        MigrationPacket, MigrationReport, MobileObject, Mobility, ObjectCallFuture, ObjectEnvelope,
        ObjectLimits, ObjectReadAction, ObjectRegistry, ObjectService, ObjectStats,
        ObjectWriteAction, PlacementFallback, PlacementFuture, Remote, RemoteCallSpec,
        RemoteObjectCall, Retirement, RootedCreateFuture, TransferFuture, UpgradeFuture,
        WeakRemote, LEASE_ACQUIRE_ACTION_ID, LEASE_RELEASE_ACTION_ID, LEASE_RENEW_ACTION_ID,
        MIGRATION_FREEZE_ACTION_ID, MIGRATION_HEADER_BYTES, MIGRATION_RETIRE_ACTION_ID,
        ROOT_RELEASE_ACTION_ID, TRANSFER_ACK_ACTION_ID,
    },
    pending::{PendingEntry, PendingTable, Promise, RemoteFuture},
    wire::{self, RuntimeMessage, RuntimeMessageKind},
    WireValue,
};
use hataori_runtime_foundation::{
    protocol::{
        ActionId, Channel, DomainId, Hello, LocalityId, MessageId, ObjectId, ObjectTypeId, Parcel,
        ParcelKind, ProtocolLimits, RequestId, RunId, RuntimeVersion, TraceId,
    },
    transport::{SendTicket, TransportDriver, TransportEvent, TransportHandle, TransportStats},
};
use std::{
    collections::{BTreeMap, HashMap, VecDeque},
    future::Future,
    pin::pin,
    sync::{
        atomic::{AtomicU64, AtomicU8, AtomicUsize, Ordering},
        mpsc::{self, Receiver, SyncSender, TryRecvError, TrySendError},
        Arc, Mutex,
    },
    task::{Context, Poll, Wake, Waker},
    time::{Duration, Instant},
};

const RUNTIME_VERSION: RuntimeVersion = RuntimeVersion {
    major: 0,
    minor: 3,
    patch: 0,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Place {
    pub locality: LocalityId,
    pub domain: DomainId,
}

impl Place {
    pub const fn new(locality: LocalityId, domain: DomainId) -> Self {
        Self { locality, domain }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SpawnOptions {
    pub deadline: Duration,
    pub trace_id: Option<TraceId>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RuntimeLimits {
    pub max_pending_calls: usize,
    pub max_domains: usize,
    pub max_workers_per_domain: usize,
    pub max_action_queue_per_domain: usize,
    pub max_dedup_entries: usize,
    pub max_dedup_bytes: usize,
    pub max_progress_events: usize,
    pub default_deadline: Duration,
    pub dedup_ttl: Duration,
    pub shutdown_timeout: Duration,
}

impl Default for RuntimeLimits {
    fn default() -> Self {
        Self {
            max_pending_calls: 1024,
            max_domains: 16,
            max_workers_per_domain: 64,
            max_action_queue_per_domain: 256,
            max_dedup_entries: 1024,
            max_dedup_bytes: 16 * 1024 * 1024,
            max_progress_events: 256,
            default_deadline: Duration::from_secs(30),
            dedup_ttl: Duration::from_secs(60),
            shutdown_timeout: Duration::from_secs(10),
        }
    }
}

impl RuntimeLimits {
    pub fn validate(self) -> Result<Self, RuntimeError> {
        if self.max_pending_calls == 0
            || self.max_domains == 0
            || self.max_workers_per_domain == 0
            || self.max_action_queue_per_domain == 0
            || self.max_dedup_entries == 0
            || self.max_dedup_bytes == 0
            || self.max_progress_events == 0
            || self.default_deadline.is_zero()
            || self.dedup_ttl.is_zero()
            || self.shutdown_timeout.is_zero()
        {
            return Err(RuntimeError::InvalidLimits(
                "runtime limits and durations must be nonzero",
            ));
        }
        Ok(self)
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RuntimeProgress {
    pub events: usize,
    pub made_progress: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RuntimeStats {
    pub state: RuntimeState,
    pub pending_calls: usize,
    pub completed_calls: u64,
    pub cancelled_calls: u64,
    pub timed_out_calls: u64,
    pub late_results: u64,
    pub duplicate_requests: u64,
    pub action_failures: u64,
    pub local_typed_dispatches: u64,
    pub local_action_serializations: u64,
    pub dedup_entries: usize,
    pub dedup_bytes: usize,
    pub sent_by_channel: [u64; 3],
    pub received_by_channel: [u64; 3],
    pub domains: Vec<(DomainId, DomainStats)>,
    pub send_tickets: usize,
    pub queued_responses: usize,
    pub queued_response_bytes: usize,
    pub objects: ObjectStats,
    pub transport: TransportStats,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ShutdownReport {
    pub stats: RuntimeStats,
}

pub struct RuntimeBuilder {
    run_id: RunId,
    protocol_limits: ProtocolLimits,
    runtime_limits: RuntimeLimits,
    registry: ActionRegistry,
    object_registry: ObjectRegistry,
    object_limits: ObjectLimits,
    domains: Vec<DomainConfig>,
    sealed: bool,
}

impl RuntimeBuilder {
    pub fn object_limits(&mut self, limits: ObjectLimits) -> Result<&mut Self, RuntimeError> {
        if self.sealed {
            return Err(RuntimeError::BuilderSealed);
        }
        let limits = limits.validate()?;
        if limits.max_snapshot_bytes
            > self
                .protocol_limits
                .max_payload_bytes
                .saturating_sub(MIGRATION_HEADER_BYTES)
            || limits.max_snapshot_segments >= self.protocol_limits.max_segments
        {
            return Err(RuntimeError::InvalidLimits(
                "migration snapshot exceeds protocol payload limits",
            ));
        }
        self.object_limits = limits;
        Ok(self)
    }

    pub fn register_object_exclusive<T: DistributedObject>(
        &mut self,
    ) -> Result<&mut Self, RuntimeError> {
        if self.sealed {
            return Err(RuntimeError::BuilderSealed);
        }
        let object_type =
            ObjectTypeId::new(T::TYPE_ID).map_err(|_| RuntimeError::InvalidObjectTypeId)?;
        let create = create_action_id(object_type)?;
        if self.registry.get(create).is_some() {
            return Err(RuntimeError::DuplicateAction(create));
        }
        self.object_registry.register_exclusive::<T>()?;
        Ok(self)
    }

    pub fn register_object_read_write<T: DistributedObject + Sync>(
        &mut self,
    ) -> Result<&mut Self, RuntimeError> {
        if self.sealed {
            return Err(RuntimeError::BuilderSealed);
        }
        let object_type =
            ObjectTypeId::new(T::TYPE_ID).map_err(|_| RuntimeError::InvalidObjectTypeId)?;
        let create = create_action_id(object_type)?;
        if self.registry.get(create).is_some() {
            return Err(RuntimeError::DuplicateAction(create));
        }
        self.object_registry.register_read_write::<T>()?;
        Ok(self)
    }

    /// Registers an exclusive migratable or reconstructible object type.
    ///
    /// # Errors
    /// Returns a sealed-builder, duplicate ID/type, pinned mobility, invalid
    /// schema, or registry collision error.
    pub fn register_mobile_object_exclusive<T: MobileObject>(
        &mut self,
    ) -> Result<&mut Self, RuntimeError> {
        if self.sealed {
            return Err(RuntimeError::BuilderSealed);
        }
        let object_type =
            ObjectTypeId::new(T::TYPE_ID).map_err(|_| RuntimeError::InvalidObjectTypeId)?;
        let create = create_action_id(object_type)?;
        if self.registry.get(create).is_some() {
            return Err(RuntimeError::DuplicateAction(create));
        }
        self.object_registry.register_mobile_exclusive::<T>()?;
        Ok(self)
    }

    /// Registers a read/write migratable or reconstructible object type.
    ///
    /// # Errors
    /// Returns a sealed-builder, duplicate ID/type, pinned mobility, invalid
    /// schema, or registry collision error.
    pub fn register_mobile_object_read_write<T: MobileObject + Sync>(
        &mut self,
    ) -> Result<&mut Self, RuntimeError> {
        if self.sealed {
            return Err(RuntimeError::BuilderSealed);
        }
        let object_type =
            ObjectTypeId::new(T::TYPE_ID).map_err(|_| RuntimeError::InvalidObjectTypeId)?;
        let create = create_action_id(object_type)?;
        if self.registry.get(create).is_some() {
            return Err(RuntimeError::DuplicateAction(create));
        }
        self.object_registry.register_mobile_read_write::<T>()?;
        Ok(self)
    }

    pub fn register_object_read<T, A>(&mut self) -> Result<&mut Self, RuntimeError>
    where
        T: DistributedObject + Sync,
        A: ObjectReadAction<T>,
    {
        if self.sealed {
            return Err(RuntimeError::BuilderSealed);
        }
        let id = ActionId::new(A::ID).map_err(|_| RuntimeError::InvalidActionId)?;
        if self.registry.get(id).is_some() {
            return Err(RuntimeError::DuplicateAction(id));
        }
        self.object_registry.register_read::<T, A>()?;
        Ok(self)
    }

    pub fn register_object_write<T, A>(&mut self) -> Result<&mut Self, RuntimeError>
    where
        T: DistributedObject,
        A: ObjectWriteAction<T>,
    {
        if self.sealed {
            return Err(RuntimeError::BuilderSealed);
        }
        let id = ActionId::new(A::ID).map_err(|_| RuntimeError::InvalidActionId)?;
        if self.registry.get(id).is_some() {
            return Err(RuntimeError::DuplicateAction(id));
        }
        self.object_registry.register_write::<T, A>()?;
        Ok(self)
    }

    pub fn register<A, F>(&mut self, handler: F) -> Result<&mut Self, RuntimeError>
    where
        A: Action,
        F: Fn(A) -> Result<A::Output, crate::ActionError> + Send + Sync + 'static,
    {
        if self.sealed {
            return Err(RuntimeError::BuilderSealed);
        }
        let id = ActionId::new(A::ID).map_err(|_| RuntimeError::InvalidActionId)?;
        let placement = placement_action_id(id)?;
        if self.object_registry.reserves_action(id)
            || self.object_registry.reserves_action(placement)
            || self.registry.get(placement).is_some()
        {
            return Err(RuntimeError::DuplicateAction(id));
        }
        self.registry.register::<A, F>(handler)?;
        let handler = self.registry.get(id).cloned().unwrap();
        self.object_registry.register_placement::<A>(handler)?;
        Ok(self)
    }

    pub fn domain(&mut self, config: DomainConfig) -> Result<&mut Self, RuntimeError> {
        if self.sealed {
            return Err(RuntimeError::BuilderSealed);
        }
        if config.workers == 0
            || config.workers > self.runtime_limits.max_workers_per_domain
            || config.queue_capacity == 0
            || config.queue_capacity > self.runtime_limits.max_action_queue_per_domain
        {
            return Err(RuntimeError::InvalidLimits("invalid domain configuration"));
        }
        if self.domains.len() >= self.runtime_limits.max_domains {
            return Err(RuntimeError::InvalidLimits("too many domains"));
        }
        if self.domains.iter().any(|domain| domain.id == config.id) {
            return Err(RuntimeError::InvalidLimits("duplicate domain id"));
        }
        self.domains.push(config);
        Ok(self)
    }

    pub fn hello(&mut self) -> Result<Hello, RuntimeError> {
        self.sealed = true;
        Ok(Hello {
            run_id: self.run_id,
            runtime_version: RUNTIME_VERSION,
            action_registry_hash: self.registry.fingerprint(),
            object_registry_hash: self.object_registry.fingerprint(),
            capabilities: 0b11,
            limits: self.protocol_limits.validate().map_err(|error| {
                RuntimeError::Protocol(format!("invalid protocol limits: {error}"))
            })?,
        })
    }

    pub fn start<H, D>(mut self, handle: H, driver: D) -> Result<Runtime, RuntimeError>
    where
        H: TransportHandle + 'static,
        D: TransportDriver + 'static,
    {
        if !self.sealed {
            return Err(RuntimeError::HandshakeNotPrepared);
        }
        let local_id = driver.local_id();
        if !driver.members().contains(&local_id) {
            return Err(RuntimeError::InvalidMembership(local_id));
        }
        if self.domains.is_empty() {
            self.domains.push(DomainConfig {
                queue_capacity: self.runtime_limits.max_action_queue_per_domain,
                ..DomainConfig::default()
            });
        }
        let objects = Arc::new(ObjectService::new(
            self.run_id,
            local_id,
            self.object_limits,
            self.object_registry.clone(),
        ));
        self.object_registry.install(&objects, &mut self.registry)?;
        let completion_capacity = self
            .runtime_limits
            .max_pending_calls
            .checked_add(self.runtime_limits.max_dedup_entries)
            .ok_or(RuntimeError::InvalidLimits("completion capacity overflow"))?;
        let domains = DomainRegistry::new(&self.domains, completion_capacity)?;
        let domain_pools = domains.pools();
        let (local_tx, local_rx) = mpsc::sync_channel(self.runtime_limits.max_pending_calls);
        let state = Arc::new(AtomicU8::new(RuntimeState::Bootstrapping as u8));
        let shared = Arc::new(Shared {
            state: Arc::clone(&state),
            local_id,
            run_id: self.run_id,
            limits: self.runtime_limits,
            protocol_limits: self.protocol_limits,
            transport: Arc::new(handle),
            registry: Arc::new(self.registry),
            objects: Arc::clone(&objects),
            domain_pools,
            local_leases: LocalLeaseManager::new(
                self.object_limits.max_local_leases,
                self.object_limits.lease_renew_interval,
            ),
            pending: Arc::new(PendingTable::new(self.runtime_limits.max_pending_calls)),
            local_tx,
            next_request: AtomicU64::new(1),
            next_object: AtomicU64::new(1),
            next_message: AtomicU64::new(1),
            tickets: Mutex::new(HashMap::new()),
            work_gate: Mutex::new(()),
            local_object_calls: AtomicUsize::new(0),
            counters: SharedCounters::default(),
        });
        state.store(RuntimeState::Running as u8, Ordering::Release);
        Ok(Runtime {
            shared,
            driver: Box::new(driver),
            domains,
            local_rx,
            dedup: DedupTable::new(
                self.runtime_limits.max_dedup_entries,
                self.runtime_limits.max_dedup_bytes,
                self.runtime_limits.dedup_ttl,
            ),
            responses: VecDeque::new(),
            response_bytes: 0,
            objects,
        })
    }
}

#[derive(Default)]
struct SharedCounters {
    completed: AtomicU64,
    cancelled: AtomicU64,
    timed_out: AtomicU64,
    late_results: AtomicU64,
    duplicate_requests: AtomicU64,
    action_failures: AtomicU64,
    local_typed_dispatches: AtomicU64,
    local_action_serializations: AtomicU64,
    sent: [AtomicU64; 3],
    received: [AtomicU64; 3],
}

struct LocalObjectCall<'a>(&'a AtomicUsize);

impl Drop for LocalObjectCall<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::AcqRel);
    }
}

enum TicketPurpose {
    Request(RequestId),
    RequiredResponse,
    BestEffort,
}

struct LocalRequest {
    message: RuntimeMessage,
    input: ActionValue,
    cancel_token: Arc<std::sync::atomic::AtomicBool>,
    trace_id: Option<TraceId>,
}

struct Shared {
    state: Arc<AtomicU8>,
    local_id: LocalityId,
    run_id: RunId,
    limits: RuntimeLimits,
    protocol_limits: ProtocolLimits,
    transport: Arc<dyn TransportHandle>,
    registry: Arc<ActionRegistry>,
    objects: Arc<ObjectService>,
    domain_pools: BTreeMap<DomainId, Arc<rayon::ThreadPool>>,
    local_leases: LocalLeaseManager,
    pending: Arc<PendingTable>,
    local_tx: SyncSender<LocalRequest>,
    next_request: AtomicU64,
    next_object: AtomicU64,
    next_message: AtomicU64,
    tickets: Mutex<HashMap<SendTicket, TicketPurpose>>,
    work_gate: Mutex<()>,
    local_object_calls: AtomicUsize,
    counters: SharedCounters,
}

impl Shared {
    fn state(&self) -> RuntimeState {
        decode_state(self.state.load(Ordering::Acquire))
    }

    fn ensure_running(&self) -> Result<(), RuntimeError> {
        let state = self.state();
        if state == RuntimeState::Running {
            Ok(())
        } else if matches!(state, RuntimeState::Draining | RuntimeState::Stopped) {
            Err(RuntimeError::Shutdown)
        } else {
            Err(RuntimeError::InvalidState {
                expected: RuntimeState::Running,
                actual: state,
            })
        }
    }

    fn next_request(&self) -> Result<RequestId, RuntimeError> {
        let sequence = self
            .next_request
            .try_update_compat(Ordering::AcqRel, Ordering::Acquire, |value| {
                value.checked_add(1).filter(|next| *next != 0)
            })
            .map_err(|_| RuntimeError::ResourceExhausted {
                resource: ResourceKind::PendingCalls,
                limit: self.limits.max_pending_calls,
            })?;
        RequestId::new(self.local_id, sequence)
            .map_err(|_| RuntimeError::Protocol("request id overflow".into()))
    }

    fn next_object(&self) -> Result<ObjectId, RuntimeError> {
        let sequence = self
            .next_object
            .try_update_compat(Ordering::AcqRel, Ordering::Acquire, |value| {
                value.checked_add(1).filter(|next| *next != 0)
            })
            .map_err(|_| RuntimeError::ResourceExhausted {
                resource: ResourceKind::Objects,
                limit: self.objects.limits.max_objects,
            })?;
        let unique = (u128::from(self.local_id.get()) << 64) | u128::from(sequence);
        ObjectId::new(self.run_id, unique)
            .map_err(|_| RuntimeError::Protocol("object id overflow".into()))
    }

    fn next_message(&self) -> Result<MessageId, RuntimeError> {
        let value = self
            .next_message
            .try_update_compat(Ordering::AcqRel, Ordering::Acquire, |value| {
                value.checked_add(1).filter(|next| *next != 0)
            })
            .map_err(|_| RuntimeError::Protocol("message id overflow".into()))?;
        MessageId::new(u128::from(value))
            .map_err(|_| RuntimeError::Protocol("zero message id".into()))
    }

    fn send(
        &self,
        destination: LocalityId,
        message: RuntimeMessage,
        purpose: TicketPurpose,
        trace_id: Option<TraceId>,
    ) -> Result<SendTicket, RuntimeError> {
        let channel = if message.kind == RuntimeMessageKind::Cancel {
            Channel::Control
        } else {
            Channel::Action
        };
        let parcel_kind = if channel == Channel::Control {
            ParcelKind::Control
        } else {
            ParcelKind::Data
        };
        let parcel = Parcel {
            run_id: self.run_id,
            message_id: self.next_message()?,
            channel,
            kind: parcel_kind,
            source: self.local_id,
            destination,
            trace_id,
            segments: wire::encode(message),
        };
        parcel.validate(self.protocol_limits).map_err(|error| {
            RuntimeError::Protocol(format!("runtime parcel exceeds limits: {error}"))
        })?;
        let mut tickets = self.tickets.lock().unwrap();
        let action_limit = self
            .limits
            .max_pending_calls
            .saturating_add(self.limits.max_dedup_entries)
            .saturating_add(self.limits.max_action_queue_per_domain);
        let total_limit = action_limit.saturating_add(self.limits.max_pending_calls);
        let limit = if channel == Channel::Control {
            total_limit
        } else {
            action_limit
        };
        if tickets.len() >= limit {
            return Err(RuntimeError::ResourceExhausted {
                resource: ResourceKind::SendTickets,
                limit,
            });
        }
        let ticket = self.transport.try_send(parcel)?;
        tickets.insert(ticket, purpose);
        self.counters.sent[channel as usize].fetch_add(1, Ordering::Relaxed);
        Ok(ticket)
    }

    fn cancel(&self, request: RequestId) {
        let Some(entry) = self.pending.remove(request) else {
            return;
        };
        self.counters.cancelled.fetch_add(1, Ordering::Relaxed);
        if let Some(token) = entry.cancel_token {
            token.store(true, Ordering::Release);
        }
        if entry.destination != self.local_id {
            let message = RuntimeMessage {
                kind: RuntimeMessageKind::Cancel,
                request,
                action: entry.action,
                domain: entry.domain,
                deadline_ms: 0,
                payload: Vec::new(),
            };
            let _ = self.send(
                entry.destination,
                message,
                TicketPurpose::BestEffort,
                entry.trace_id,
            );
        }
    }
}

#[derive(Clone)]
pub struct RuntimeClient {
    shared: Arc<Shared>,
}

impl std::fmt::Debug for RuntimeClient {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("RuntimeClient")
            .field("local_id", &self.shared.local_id)
            .field("state", &self.shared.state())
            .finish_non_exhaustive()
    }
}

impl RuntimeClient {
    pub fn local_id(&self) -> LocalityId {
        self.shared.local_id
    }

    pub fn spawn_on<A: Action>(
        &self,
        place: Place,
        action: A,
    ) -> Result<RemoteFuture<A::Output>, RuntimeError> {
        self.spawn_on_with(
            place,
            action,
            SpawnOptions {
                deadline: self.shared.limits.default_deadline,
                trace_id: None,
            },
        )
    }

    pub fn spawn_colocated<T: DistributedObject, A: Action>(
        &self,
        remote: &Remote<T>,
        action: A,
    ) -> Result<ColocatedFuture<A::Output>, RuntimeError> {
        let location = self
            .shared
            .objects
            .resolve(remote.object)
            .unwrap_or(remote.location);
        let action_id = ActionId::new(A::ID).map_err(|_| RuntimeError::InvalidActionId)?;
        let action_payload = action
            .encode()
            .map_err(|error| RuntimeError::Protocol(format!("action encode failed: {error}")))?;
        let payload = ObjectEnvelope {
            object: remote.object,
            object_type: remote.object_type,
            epoch: location.epoch(),
            lease_locality: self.shared.local_id,
            domain: location.domain(),
            rooted: false,
        }
        .encode(action_payload.clone());
        let placed_action = placement_action_id(action_id)?;
        let deadline = self.shared.limits.default_deadline;
        let future = self.spawn_encoded(
            Place::new(location.locality(), location.domain()),
            placed_action,
            payload,
            SpawnOptions {
                deadline,
                trace_id: None,
            },
        )?;
        Ok(ColocatedFuture::new(
            RemoteObjectCall::new(
                future,
                self.clone(),
                RemoteCallSpec {
                    object: remote.object,
                    object_type: remote.object_type,
                    action: placed_action,
                    payload: action_payload,
                    last_epoch: location.epoch(),
                    max_redirects: self.shared.objects.limits.max_redirects,
                    deadline,
                },
            ),
            Arc::clone(&remote.lease),
        ))
    }

    pub fn spawn_preferred_colocated<T, A>(
        &self,
        remote: &Remote<T>,
        fallback: PlacementFallback,
        action: A,
    ) -> Result<PlacementFuture<A::Output>, RuntimeError>
    where
        T: DistributedObject,
        A: Action,
    {
        let location = self
            .shared
            .objects
            .resolve(remote.object)
            .unwrap_or(remote.location);
        let action_id = ActionId::new(A::ID).map_err(|_| RuntimeError::InvalidActionId)?;
        let payload = action
            .encode()
            .map_err(|error| RuntimeError::Protocol(format!("action encode failed: {error}")))?;
        let options = SpawnOptions {
            deadline: self.shared.limits.default_deadline,
            trace_id: None,
        };
        let preferred_payload = ObjectEnvelope {
            object: remote.object,
            object_type: remote.object_type,
            epoch: location.epoch(),
            lease_locality: self.shared.local_id,
            domain: location.domain(),
            rooted: false,
        }
        .encode(payload.clone());
        let placed_action = placement_action_id(action_id)?;
        let preferred = self.spawn_encoded(
            Place::new(location.locality(), location.domain()),
            placed_action,
            preferred_payload,
            options,
        )?;
        let preferred = RemoteObjectCall::new(
            preferred,
            self.clone(),
            RemoteCallSpec {
                object: remote.object,
                object_type: remote.object_type,
                action: placed_action,
                payload: payload.clone(),
                last_epoch: location.epoch(),
                max_redirects: self.shared.objects.limits.max_redirects,
                deadline: options.deadline,
            },
        );
        Ok(PlacementFuture::new(
            ColocatedFuture::new(preferred, Arc::clone(&remote.lease)),
            fallback,
            self.clone(),
            action_id,
            payload,
            options,
        ))
    }

    pub fn spawn_on_with<A: Action>(
        &self,
        place: Place,
        action: A,
        options: SpawnOptions,
    ) -> Result<RemoteFuture<A::Output>, RuntimeError> {
        let action_id = ActionId::new(A::ID).map_err(|_| RuntimeError::InvalidActionId)?;
        if place.locality == self.shared.local_id {
            self.spawn_inner(
                place,
                action_id,
                ActionValue::Typed(Box::new(action)),
                options,
                None,
            )
        } else {
            let payload = action.encode().map_err(|error| {
                RuntimeError::Protocol(format!("action encode failed: {error}"))
            })?;
            self.spawn_encoded(place, action_id, payload, options)
        }
    }

    pub(crate) fn spawn_encoded<T: WireValue>(
        &self,
        place: Place,
        action_id: ActionId,
        payload: Segments,
        options: SpawnOptions,
    ) -> Result<RemoteFuture<T>, RuntimeError> {
        self.spawn_inner(
            place,
            action_id,
            ActionValue::Encoded(payload),
            options,
            None,
        )
    }

    pub(crate) fn spawn_encoded_retry<T: WireValue>(
        &self,
        place: Place,
        action_id: ActionId,
        payload: Segments,
        options: SpawnOptions,
        request: RequestId,
    ) -> Result<RemoteFuture<T>, RuntimeError> {
        self.spawn_inner(
            place,
            action_id,
            ActionValue::Encoded(payload),
            options,
            Some(request),
        )
    }

    fn spawn_inner<T: WireValue>(
        &self,
        place: Place,
        action_id: ActionId,
        input: ActionValue,
        options: SpawnOptions,
        request: Option<RequestId>,
    ) -> Result<RemoteFuture<T>, RuntimeError> {
        self.shared.ensure_running()?;
        if options.deadline.is_zero() {
            return Err(RuntimeError::InvalidDeadline);
        }
        if self.shared.registry.get(action_id).is_none() {
            return Err(RuntimeError::UnknownAction(action_id));
        }
        let local = place.locality == self.shared.local_id;
        if !local && matches!(&input, ActionValue::Typed(_)) {
            return Err(RuntimeError::Protocol(
                "typed action cannot cross a locality boundary".into(),
            ));
        }
        let deadline_ms = u64::try_from(options.deadline.as_millis().max(1))
            .map_err(|_| RuntimeError::InvalidDeadline)?;
        let _gate = self.shared.work_gate.lock().unwrap();
        self.shared.ensure_running()?;
        let request = match request {
            Some(request) => request,
            None => self.shared.next_request()?,
        };
        let promise = Arc::new(Promise::new());
        let expires_at = Instant::now()
            .checked_add(options.deadline)
            .ok_or(RuntimeError::InvalidDeadline)?;
        let cancel_token = (place.locality == self.shared.local_id)
            .then(|| Arc::new(std::sync::atomic::AtomicBool::new(false)));
        self.shared.pending.insert(
            request,
            PendingEntry {
                promise: Arc::clone(&promise),
                destination: place.locality,
                action: action_id,
                domain: place.domain,
                trace_id: options.trace_id,
                expires_at,
                deadline: options.deadline,
                cancel_token: cancel_token.clone(),
            },
        )?;
        let mut input = Some(input);
        let payload = if local {
            Vec::new()
        } else {
            match input.take().unwrap() {
                ActionValue::Encoded(payload) => payload,
                ActionValue::Typed(_) => unreachable!("typed remote input rejected above"),
            }
        };
        let message = RuntimeMessage {
            kind: RuntimeMessageKind::Request,
            request,
            action: action_id,
            domain: place.domain,
            deadline_ms,
            payload,
        };
        let submitted = if local {
            self.shared
                .local_tx
                .try_send(LocalRequest {
                    message,
                    input: input.take().unwrap(),
                    cancel_token: cancel_token.unwrap(),
                    trace_id: options.trace_id,
                })
                .map_err(|error| match error {
                    TrySendError::Full(_) => RuntimeError::ResourceExhausted {
                        resource: ResourceKind::ActionQueue,
                        limit: self.shared.limits.max_pending_calls,
                    },
                    TrySendError::Disconnected(_) => RuntimeError::Shutdown,
                })
                .map(|_| ())
        } else {
            self.shared
                .send(
                    place.locality,
                    message,
                    TicketPurpose::Request(request),
                    options.trace_id,
                )
                .map(|_| ())
        };
        if let Err(error) = submitted {
            self.shared.pending.remove(request);
            return Err(error);
        }
        let shared = Arc::clone(&self.shared);
        Ok(RemoteFuture::new(
            request,
            promise,
            Arc::new(move |request| shared.cancel(request)),
        ))
    }

    pub fn create_at<T: DistributedObject>(
        &self,
        place: Place,
        state: T,
    ) -> Result<CreateFuture<T>, RuntimeError> {
        self.create_at_inner(place, state, false)
    }

    pub fn create_rooted_at<T: DistributedObject>(
        &self,
        place: Place,
        state: T,
    ) -> Result<RootedCreateFuture<T>, RuntimeError> {
        Ok(RootedCreateFuture::new(
            self.create_at_inner(place, state, true)?,
        ))
    }

    fn create_at_inner<T: DistributedObject>(
        &self,
        place: Place,
        state: T,
        rooted: bool,
    ) -> Result<CreateFuture<T>, RuntimeError> {
        let object_type =
            ObjectTypeId::new(T::TYPE_ID).map_err(|_| RuntimeError::InvalidObjectTypeId)?;
        let object = self.shared.next_object()?;
        if place.locality == self.shared.local_id {
            let location = self.shared.objects.create_local(
                object,
                object_type,
                place.domain,
                state,
                self.shared.local_id,
                rooted,
            )?;
            let remote = self.attach_object(object, object_type, location);
            return Ok(CreateFuture::ready(
                remote,
                object,
                object_type,
                self.clone(),
            ));
        }
        let payload = state
            .encode()
            .map_err(|error| RuntimeError::Protocol(format!("object encode failed: {error}")))?;
        let payload = ObjectEnvelope {
            object,
            object_type,
            epoch: 1,
            lease_locality: self.shared.local_id,
            domain: place.domain,
            rooted,
        }
        .encode(payload);
        let future = self.spawn_encoded::<Vec<u8>>(
            place,
            create_action_id(object_type)?,
            payload,
            SpawnOptions {
                deadline: self.shared.limits.default_deadline,
                trace_id: None,
            },
        )?;
        Ok(CreateFuture::new(future, object, object_type, self.clone()))
    }

    fn run_local_object<O: Send>(
        &self,
        domain: DomainId,
        call: impl FnOnce() -> Result<O, RuntimeError> + Send,
    ) -> Result<O, RuntimeError> {
        let gate = self.shared.work_gate.lock().unwrap();
        self.shared.ensure_running()?;
        self.shared
            .local_object_calls
            .fetch_add(1, Ordering::AcqRel);
        let _active = LocalObjectCall(&self.shared.local_object_calls);
        let pool = self
            .shared
            .domain_pools
            .get(&domain)
            .cloned()
            .ok_or(RuntimeError::UnknownDomain(domain))?;
        drop(gate);
        pool.install(call)
    }

    pub fn call_object_read<T, A>(
        &self,
        remote: &Remote<T>,
        action: A,
    ) -> Result<ObjectCallFuture<A::Output>, RuntimeError>
    where
        T: DistributedObject + Sync,
        A: ObjectReadAction<T>,
    {
        let location = self
            .shared
            .objects
            .resolve(remote.object)
            .unwrap_or(remote.location);
        if location.locality() == self.shared.local_id
            && self
                .shared
                .objects
                .is_local_resident(remote.object, location)
        {
            let object = remote.object;
            let object_type = remote.object_type;
            let objects = Arc::clone(&self.shared.objects);
            return Ok(ObjectCallFuture::local(
                self.run_local_object(location.domain(), move || {
                    objects.call_read_local(object, object_type, location, action)
                }),
            ));
        }
        let payload = action.encode().map_err(|error| {
            RuntimeError::Protocol(format!("object action encode failed: {error}"))
        })?;
        self.call_object(remote, A::ID, payload)
            .map(ObjectCallFuture::remote)
    }

    pub fn call_object_write<T, A>(
        &self,
        remote: &Remote<T>,
        action: A,
    ) -> Result<ObjectCallFuture<A::Output>, RuntimeError>
    where
        T: DistributedObject,
        A: ObjectWriteAction<T>,
    {
        let location = self
            .shared
            .objects
            .resolve(remote.object)
            .unwrap_or(remote.location);
        if location.locality() == self.shared.local_id
            && self
                .shared
                .objects
                .is_local_resident(remote.object, location)
        {
            let object = remote.object;
            let object_type = remote.object_type;
            let objects = Arc::clone(&self.shared.objects);
            return Ok(ObjectCallFuture::local(
                self.run_local_object(location.domain(), move || {
                    objects.call_write_local(object, object_type, location, action)
                }),
            ));
        }
        let payload = action.encode().map_err(|error| {
            RuntimeError::Protocol(format!("object action encode failed: {error}"))
        })?;
        self.call_object(remote, A::ID, payload)
            .map(ObjectCallFuture::remote)
    }

    fn call_object<T: DistributedObject, O: WireValue>(
        &self,
        remote: &Remote<T>,
        raw_action: u128,
        action_payload: Segments,
    ) -> Result<RemoteObjectCall<O>, RuntimeError> {
        let action = ActionId::new(raw_action).map_err(|_| RuntimeError::InvalidActionId)?;
        let location = self
            .shared
            .objects
            .resolve(remote.object)
            .unwrap_or(remote.location);
        let payload = ObjectEnvelope {
            object: remote.object,
            object_type: remote.object_type,
            epoch: location.epoch(),
            lease_locality: self.shared.local_id,
            domain: location.domain(),
            rooted: false,
        }
        .encode(action_payload.clone());
        let deadline = self.shared.limits.default_deadline;
        let future = self.spawn_encoded(
            Place::new(location.locality(), location.domain()),
            action,
            payload,
            SpawnOptions {
                deadline,
                trace_id: None,
            },
        )?;
        Ok(RemoteObjectCall::new(
            future,
            self.clone(),
            RemoteCallSpec {
                object: remote.object,
                object_type: remote.object_type,
                action,
                payload: action_payload,
                last_epoch: location.epoch(),
                max_redirects: self.shared.objects.limits.max_redirects,
                deadline,
            },
        ))
    }

    pub(crate) fn resolve_object(
        &self,
        object: ObjectId,
    ) -> Option<hataori_runtime_foundation::protocol::ObjectLocation> {
        self.shared.objects.resolve(object)
    }

    pub(crate) fn cache_object_location(
        &self,
        object: ObjectId,
        location: hataori_runtime_foundation::protocol::ObjectLocation,
    ) {
        self.shared.objects.cache_location(object, location);
    }

    pub fn clear_object_resolver(&self) {
        self.shared.objects.clear_resolver();
    }

    /// Starts explicit migration to an exact logical place.
    ///
    /// # Errors
    /// Returns typed placement, mobility, snapshot, resource, transport,
    /// rollback, post-commit, or shutdown errors.
    pub fn migrate<T: MobileObject>(
        &self,
        remote: &Remote<T>,
        destination: Place,
    ) -> Result<MigrationFuture<T>, RuntimeError> {
        {
            let _gate = self.shared.work_gate.lock().unwrap();
            self.shared.ensure_running()?;
        }
        let from = self
            .shared
            .objects
            .resolve(remote.object)
            .unwrap_or(remote.location);
        if from.locality() == destination.locality && from.domain() == destination.domain {
            return Ok(MigrationFuture::ready(
                MigrationReport {
                    object: remote.object,
                    from,
                    to: from,
                    snapshot_bytes: 0,
                },
                remote.object_type,
            ));
        }
        let (mobility, snapshot_version, snapshot_schema) =
            self.shared.objects.migration_metadata(remote.object_type)?;
        if mobility == Mobility::Pinned {
            return Err(RuntimeError::PinnedObjectType(remote.object_type));
        }
        let migration = self.shared.next_request()?;
        self.shared.objects.begin_migration(migration)?;
        let packet = MigrationPacket {
            migration,
            object: remote.object,
            object_type: remote.object_type,
            authority: remote.lease.location.locality(),
            from,
            destination,
            snapshot_version,
            snapshot_schema,
            snapshot: Vec::new(),
        };
        let future = self.migration_step(
            Place::new(from.locality(), from.domain()),
            MIGRATION_FREEZE_ACTION_ID,
            packet.clone(),
        );
        match future {
            Ok(future) => Ok(MigrationFuture::new(
                self.clone(),
                Arc::clone(&self.shared.objects),
                Arc::clone(&remote.lease),
                packet,
                remote.lease.location,
                future,
            )),
            Err(error) => {
                self.shared.objects.finish_migration(
                    migration,
                    crate::object::MigrationOutcome::RolledBack,
                    0,
                );
                Err(error)
            }
        }
    }

    pub(crate) fn migration_step(
        &self,
        place: Place,
        action: u128,
        packet: MigrationPacket,
    ) -> Result<RemoteFuture<MigrationPacket>, RuntimeError> {
        self.spawn_encoded(
            place,
            ActionId::new(action).map_err(|_| RuntimeError::InvalidActionId)?,
            packet
                .encode()
                .map_err(|error| RuntimeError::ObjectAction(error.message()))?,
            SpawnOptions {
                deadline: self.shared.limits.default_deadline,
                trace_id: None,
            },
        )
    }

    pub(crate) fn fail_committed_migration(&self) {
        self.shared
            .state
            .store(RuntimeState::Failed as u8, Ordering::Release);
    }

    pub fn transfer_object<T: DistributedObject>(
        &self,
        remote: &Remote<T>,
        destination: LocalityId,
    ) -> Result<TransferFuture<T>, RuntimeError> {
        let payload = ObjectEnvelope {
            object: remote.object,
            object_type: remote.object_type,
            epoch: remote.location.epoch(),
            lease_locality: destination,
            domain: remote.location.domain(),
            rooted: true,
        }
        .encode(Vec::new());
        let inner = self.spawn_encoded::<()>(
            Place::new(remote.location.locality(), remote.location.domain()),
            ActionId::new(LEASE_ACQUIRE_ACTION_ID).unwrap(),
            payload,
            SpawnOptions {
                deadline: self.shared.limits.default_deadline,
                trace_id: None,
            },
        )?;
        Ok(TransferFuture::new(
            inner,
            remote.object,
            remote.object_type,
            remote.location,
            destination,
            Instant::now() + self.shared.objects.limits.lease_ttl,
            self.clone(),
        ))
    }

    pub fn import_transfer<T: DistributedObject>(
        &self,
        mut transfer: LeaseTransfer<T>,
    ) -> Result<Remote<T>, RuntimeError> {
        if transfer.destination != self.shared.local_id {
            return Err(RuntimeError::InvalidMembership(transfer.destination));
        }
        if transfer.expires_at <= Instant::now() {
            return Err(RuntimeError::LeaseTransferExpired(transfer.object));
        }
        let remote =
            self.attach_object(transfer.object, transfer.object_type, transfer.location)?;
        transfer.active = false;
        self.object_control(
            transfer.object,
            transfer.object_type,
            transfer.location,
            transfer.destination,
            TRANSFER_ACK_ACTION_ID,
        );
        Ok(remote)
    }

    pub fn upgrade_object<T: DistributedObject>(
        &self,
        weak: &WeakRemote<T>,
    ) -> Result<UpgradeFuture<T>, RuntimeError> {
        let payload = ObjectEnvelope {
            object: weak.object,
            object_type: weak.object_type,
            epoch: weak.location.epoch(),
            lease_locality: self.shared.local_id,
            domain: weak.location.domain(),
            rooted: false,
        }
        .encode(Vec::new());
        let inner = self.spawn_encoded::<()>(
            Place::new(weak.location.locality(), weak.location.domain()),
            ActionId::new(LEASE_ACQUIRE_ACTION_ID).unwrap(),
            payload,
            SpawnOptions {
                deadline: self.shared.limits.default_deadline,
                trace_id: None,
            },
        )?;
        Ok(UpgradeFuture::new(
            inner,
            weak.object,
            weak.object_type,
            weak.location,
            self.clone(),
        ))
    }

    pub(crate) fn attach_object<T: DistributedObject>(
        &self,
        object: ObjectId,
        object_type: ObjectTypeId,
        location: hataori_runtime_foundation::protocol::ObjectLocation,
    ) -> Result<Remote<T>, RuntimeError> {
        self.shared.objects.cache_location(object, location);
        self.shared
            .local_leases
            .attach(object, object_type, location, self.clone())
    }

    pub(crate) fn renew_object(
        &self,
        object: ObjectId,
        object_type: ObjectTypeId,
        location: hataori_runtime_foundation::protocol::ObjectLocation,
    ) {
        self.object_control(
            object,
            object_type,
            location,
            self.shared.local_id,
            LEASE_RENEW_ACTION_ID,
        );
    }

    pub(crate) fn release_object(
        &self,
        object: ObjectId,
        object_type: ObjectTypeId,
        location: hataori_runtime_foundation::protocol::ObjectLocation,
    ) {
        self.object_control(
            object,
            object_type,
            location,
            self.shared.local_id,
            LEASE_RELEASE_ACTION_ID,
        );
    }

    pub(crate) fn release_root(
        &self,
        object: ObjectId,
        object_type: ObjectTypeId,
        location: hataori_runtime_foundation::protocol::ObjectLocation,
    ) {
        self.object_control(
            object,
            object_type,
            location,
            self.shared.local_id,
            ROOT_RELEASE_ACTION_ID,
        );
    }

    pub(crate) fn release_for_locality(
        &self,
        object: ObjectId,
        object_type: ObjectTypeId,
        location: hataori_runtime_foundation::protocol::ObjectLocation,
        locality: LocalityId,
    ) {
        self.object_control(
            object,
            object_type,
            location,
            locality,
            LEASE_RELEASE_ACTION_ID,
        );
    }

    fn object_control(
        &self,
        object: ObjectId,
        object_type: ObjectTypeId,
        location: hataori_runtime_foundation::protocol::ObjectLocation,
        lease_locality: LocalityId,
        raw_action: u128,
    ) {
        if location.locality() == self.shared.local_id {
            match raw_action {
                LEASE_RENEW_ACTION_ID => {
                    self.shared
                        .objects
                        .renew(object, lease_locality, Instant::now());
                }
                ROOT_RELEASE_ACTION_ID => {
                    self.shared.objects.release_root(object);
                }
                TRANSFER_ACK_ACTION_ID => {
                    self.shared.objects.ack_transfer(object, lease_locality);
                }
                _ => {
                    self.shared.objects.release(object, lease_locality);
                }
            }
            return;
        }
        let Ok(request) = self.shared.next_request() else {
            return;
        };
        let Ok(action) = ActionId::new(raw_action) else {
            return;
        };
        let message = RuntimeMessage {
            kind: RuntimeMessageKind::Request,
            request,
            action,
            domain: location.domain(),
            deadline_ms: u64::try_from(self.shared.limits.default_deadline.as_millis().max(1))
                .unwrap_or(u64::MAX),
            payload: ObjectEnvelope {
                object,
                object_type,
                epoch: location.epoch(),
                lease_locality,
                domain: location.domain(),
                rooted: false,
            }
            .encode(Vec::new()),
        };
        let _ = self.shared.send(
            location.locality(),
            message,
            TicketPurpose::BestEffort,
            None,
        );
    }

    pub(crate) fn cancel(&self, request: RequestId) {
        self.shared.cancel(request);
    }
}

struct QueuedResponse {
    destination: LocalityId,
    message: RuntimeMessage,
    trace_id: Option<TraceId>,
    cache: bool,
    bytes: usize,
}

pub struct Runtime {
    shared: Arc<Shared>,
    driver: Box<dyn TransportDriver>,
    domains: DomainRegistry,
    local_rx: Receiver<LocalRequest>,
    dedup: DedupTable,
    responses: VecDeque<QueuedResponse>,
    response_bytes: usize,
    objects: Arc<ObjectService>,
}

impl std::fmt::Debug for Runtime {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let stats = self.stats();
        formatter
            .debug_struct("Runtime")
            .field("local_id", &self.shared.local_id)
            .field("state", &stats.state)
            .field("pending_calls", &stats.pending_calls)
            .field("dedup_entries", &stats.dedup_entries)
            .field("send_tickets", &stats.send_tickets)
            .field("queued_responses", &stats.queued_responses)
            .field("queued_response_bytes", &stats.queued_response_bytes)
            .finish_non_exhaustive()
    }
}

impl Runtime {
    pub fn builder(
        run_id: RunId,
        protocol_limits: ProtocolLimits,
        runtime_limits: RuntimeLimits,
    ) -> Result<RuntimeBuilder, RuntimeError> {
        protocol_limits
            .validate()
            .map_err(|error| RuntimeError::Protocol(format!("invalid protocol limits: {error}")))?;
        let largest_runtime_frame = protocol_limits
            .max_payload_bytes
            .checked_add(protocol_limits.max_segments.saturating_mul(8))
            .and_then(|bytes| bytes.checked_add(128))
            .ok_or(RuntimeError::InvalidLimits(
                "runtime frame capacity overflow",
            ))?;
        let action_byte_limit = protocol_limits
            .max_inflight_bytes_per_peer
            .checked_sub(protocol_limits.control_reserved_bytes_per_peer)
            .ok_or(RuntimeError::InvalidLimits(
                "control reservation exceeds in-flight capacity",
            ))?;
        if largest_runtime_frame > action_byte_limit {
            return Err(RuntimeError::InvalidLimits(
                "one maximum runtime parcel must fit the action byte reservation",
            ));
        }
        let mut object_limits = ObjectLimits::default();
        object_limits.max_snapshot_bytes = object_limits.max_snapshot_bytes.min(
            protocol_limits
                .max_payload_bytes
                .saturating_sub(MIGRATION_HEADER_BYTES),
        );
        object_limits.max_snapshot_segments = object_limits
            .max_snapshot_segments
            .min(protocol_limits.max_segments.saturating_sub(1));
        let object_limits = object_limits.validate()?;
        Ok(RuntimeBuilder {
            run_id,
            protocol_limits,
            runtime_limits: runtime_limits.validate()?,
            registry: ActionRegistry::default(),
            object_registry: ObjectRegistry::default(),
            object_limits,
            domains: Vec::new(),
            sealed: false,
        })
    }

    pub fn client(&self) -> RuntimeClient {
        RuntimeClient {
            shared: Arc::clone(&self.shared),
        }
    }

    pub fn local_id(&self) -> LocalityId {
        self.shared.local_id
    }

    pub fn localities(&self) -> &[LocalityId] {
        self.driver.members()
    }

    pub fn state(&self) -> RuntimeState {
        self.shared.state()
    }

    pub fn spawn_on<A: Action>(
        &self,
        place: Place,
        action: A,
    ) -> Result<RemoteFuture<A::Output>, RuntimeError> {
        self.client().spawn_on(place, action)
    }

    pub fn spawn_on_with<A: Action>(
        &self,
        place: Place,
        action: A,
        options: SpawnOptions,
    ) -> Result<RemoteFuture<A::Output>, RuntimeError> {
        self.client().spawn_on_with(place, action, options)
    }

    /// Creates a pinned object at an exact logical place.
    ///
    /// # Errors
    /// Returns a typed registry, placement, codec, resource, transport, or
    /// shutdown error before retaining an unowned object.
    pub fn create_at<T: DistributedObject>(
        &self,
        place: Place,
        state: T,
    ) -> Result<CreateFuture<T>, RuntimeError> {
        self.client().create_at(place, state)
    }

    /// Creates an exact pinned object and one runtime-owned root.
    ///
    /// # Errors
    /// Returns a typed registry, placement, codec, resource, transport, or
    /// shutdown error before retaining an unowned root.
    pub fn create_rooted_at<T: DistributedObject>(
        &self,
        place: Place,
        state: T,
    ) -> Result<RootedCreateFuture<T>, RuntimeError> {
        self.client().create_rooted_at(place, state)
    }

    /// Spawns an independent action at an object's pinned location.
    ///
    /// # Errors
    /// Returns a typed resolver, placement-ticket, queue, transport, codec, or
    /// shutdown error.
    pub fn spawn_colocated<T: DistributedObject, A: Action>(
        &self,
        remote: &Remote<T>,
        action: A,
    ) -> Result<ColocatedFuture<A::Output>, RuntimeError> {
        self.client().spawn_colocated(remote, action)
    }

    pub fn clear_object_resolver(&self) {
        self.shared.objects.clear_resolver();
    }

    /// Starts explicit migration to an exact logical place.
    ///
    /// # Errors
    /// Returns typed placement, mobility, snapshot, resource, transport,
    /// rollback, post-commit, or shutdown errors.
    pub fn migrate<T: MobileObject>(
        &self,
        remote: &Remote<T>,
        destination: Place,
    ) -> Result<MigrationFuture<T>, RuntimeError> {
        self.client().migrate(remote, destination)
    }

    fn send_retirement(&self, retirement: &Retirement) -> Result<(), RuntimeError> {
        let request = self.shared.next_request()?;
        let packet = MigrationPacket {
            migration: request,
            object: retirement.object,
            object_type: retirement.object_type,
            authority: retirement.authority,
            from: retirement.location,
            destination: Place::new(retirement.location.locality(), retirement.location.domain()),
            snapshot_version: retirement.snapshot_version,
            snapshot_schema: retirement.snapshot_schema,
            snapshot: Vec::new(),
        };
        let payload = match packet.encode() {
            Ok(payload) => payload,
            Err(error) => return Err(RuntimeError::ObjectAction(error.message())),
        };
        let message = RuntimeMessage {
            kind: RuntimeMessageKind::Request,
            request,
            action: ActionId::new(MIGRATION_RETIRE_ACTION_ID).unwrap(),
            domain: retirement.location.domain(),
            deadline_ms: u64::try_from(self.shared.limits.default_deadline.as_millis().max(1))
                .unwrap_or(u64::MAX),
            payload,
        };
        self.shared
            .send(
                retirement.location.locality(),
                message,
                TicketPurpose::BestEffort,
                None,
            )
            .map(|_| ())
    }

    pub fn progress(&mut self, max_events: usize) -> Result<RuntimeProgress, RuntimeError> {
        let state = self.state();
        if !matches!(state, RuntimeState::Running | RuntimeState::Draining) {
            return Err(RuntimeError::InvalidState {
                expected: RuntimeState::Running,
                actual: state,
            });
        }
        let limit = max_events.min(self.shared.limits.max_progress_events);
        if limit == 0 {
            return Ok(RuntimeProgress::default());
        }
        self.expire_pending();
        self.dedup.expire(Instant::now());
        let now = Instant::now();
        for lease in self.shared.local_leases.due(now) {
            lease.renew();
        }
        self.objects.expire(now);
        for retirement in self.objects.take_retirements(limit) {
            let object = retirement.object;
            match self.send_retirement(&retirement) {
                Ok(()) => self.objects.finish_retirement(object),
                Err(_) => {
                    self.objects.requeue_retirement(retirement);
                    break;
                }
            }
        }
        let mut events = Vec::new();
        let transport = self.driver.progress(&mut events, limit).map_err(|error| {
            self.fail(RuntimeError::Transport(error.clone()));
            RuntimeError::Transport(error)
        })?;
        let mut handled = 0;
        for event in events {
            if let Err(error) = self.handle_transport_event(event) {
                self.fail(error.clone());
                return Err(error);
            }
            handled += 1;
        }
        while handled < limit {
            let Some(completion) = self.domains.try_completion() else {
                break;
            };
            if let Err(error) = self.handle_completion(completion) {
                self.fail(error.clone());
                return Err(error);
            }
            handled += 1;
        }
        if handled < limit {
            let ready = self
                .objects
                .retry_ready(limit - handled, |domain| self.domains.can_submit(domain));
            for job in ready {
                self.domains.submit(job)?;
                handled += 1;
            }
        }
        while handled < limit {
            match self.local_rx.try_recv() {
                Ok(request) => {
                    self.dispatch_local(request)?;
                    handled += 1;
                }
                Err(TryRecvError::Empty | TryRecvError::Disconnected) => break,
            }
        }
        let sent = match self.drain_responses(limit - handled) {
            Ok(sent) => sent,
            Err(error) => {
                self.fail(error.clone());
                return Err(error);
            }
        };
        handled += sent;
        Ok(RuntimeProgress {
            events: handled,
            made_progress: transport.made_progress || handled > 0,
        })
    }

    pub fn block_on<F, T, E>(&mut self, future: F) -> Result<T, E>
    where
        F: Future<Output = Result<T, E>>,
        E: From<RuntimeError>,
    {
        struct ThreadWake(std::thread::Thread);
        impl Wake for ThreadWake {
            fn wake(self: Arc<Self>) {
                self.0.unpark();
            }
        }
        let waker = Waker::from(Arc::new(ThreadWake(std::thread::current())));
        let mut context = Context::from_waker(&waker);
        let mut future = pin!(future);
        loop {
            if let Poll::Ready(output) = future.as_mut().poll(&mut context) {
                return output;
            }
            match self.progress(self.shared.limits.max_progress_events) {
                Ok(progress) if progress.made_progress => {}
                Ok(_) => std::thread::park_timeout(Duration::from_millis(1)),
                Err(error) => {
                    self.fail(error.clone());
                    return Err(error.into());
                }
            }
        }
    }

    pub fn stats(&self) -> RuntimeStats {
        RuntimeStats {
            state: self.state(),
            pending_calls: self.shared.pending.len(),
            completed_calls: self.shared.counters.completed.load(Ordering::Relaxed),
            cancelled_calls: self.shared.counters.cancelled.load(Ordering::Relaxed),
            timed_out_calls: self.shared.counters.timed_out.load(Ordering::Relaxed),
            late_results: self.shared.counters.late_results.load(Ordering::Relaxed),
            duplicate_requests: self
                .shared
                .counters
                .duplicate_requests
                .load(Ordering::Relaxed),
            action_failures: self.shared.counters.action_failures.load(Ordering::Relaxed),
            local_typed_dispatches: self
                .shared
                .counters
                .local_typed_dispatches
                .load(Ordering::Relaxed),
            local_action_serializations: self
                .shared
                .counters
                .local_action_serializations
                .load(Ordering::Relaxed),
            dedup_entries: self.dedup.len(),
            dedup_bytes: self.dedup.retained_bytes(),
            sent_by_channel: std::array::from_fn(|index| {
                self.shared.counters.sent[index].load(Ordering::Relaxed)
            }),
            received_by_channel: std::array::from_fn(|index| {
                self.shared.counters.received[index].load(Ordering::Relaxed)
            }),
            domains: self.domains.stats(),
            send_tickets: self.shared.tickets.lock().unwrap().len(),
            queued_responses: self.responses.len(),
            queued_response_bytes: self.response_bytes,
            objects: self.objects.stats(Instant::now()),
            transport: self.driver.stats(),
        }
    }

    pub fn shutdown(&mut self) -> Result<ShutdownReport, RuntimeError> {
        let state = self.state();
        if state == RuntimeState::Stopped {
            return Ok(ShutdownReport {
                stats: self.stats(),
            });
        }
        if !matches!(state, RuntimeState::Running | RuntimeState::Failed) {
            return Err(RuntimeError::InvalidState {
                expected: RuntimeState::Running,
                actual: state,
            });
        }
        let work_gate = self.shared.work_gate.lock().unwrap();
        self.shared
            .state
            .store(RuntimeState::Draining as u8, Ordering::Release);
        self.shared.local_leases.clear();
        for (request, entry) in self.shared.pending.drain() {
            entry.promise.complete(Err(RuntimeError::Shutdown));
            if let Some(token) = entry.cancel_token {
                token.store(true, Ordering::Release);
            }
            if entry.destination != self.shared.local_id {
                let message = RuntimeMessage {
                    kind: RuntimeMessageKind::Cancel,
                    request,
                    action: entry.action,
                    domain: entry.domain,
                    deadline_ms: 0,
                    payload: Vec::new(),
                };
                let _ = self.shared.send(
                    entry.destination,
                    message,
                    TicketPurpose::BestEffort,
                    entry.trace_id,
                );
            }
        }
        while let Ok(request) = self.local_rx.try_recv() {
            request.cancel_token.store(true, Ordering::Release);
        }
        drop(work_gate);
        let deadline = Instant::now() + self.shared.limits.shutdown_timeout;
        while {
            let objects = self.objects.stats(Instant::now());
            !self.domains.idle()
                || objects.queued_calls != 0
                || objects.in_flight_calls != 0
                || self.shared.local_object_calls.load(Ordering::Acquire) != 0
        } {
            self.progress(self.shared.limits.max_progress_events)?;
            if Instant::now() >= deadline {
                self.shared
                    .state
                    .store(RuntimeState::Failed as u8, Ordering::Release);
                return Err(RuntimeError::ShutdownTimeout);
            }
            std::thread::yield_now();
        }
        while let Some(completion) = self.domains.try_completion() {
            self.handle_completion(completion)?;
        }
        while !self.responses.is_empty() {
            self.progress(self.shared.limits.max_progress_events)?;
            if Instant::now() >= deadline {
                self.shared
                    .state
                    .store(RuntimeState::Failed as u8, Ordering::Release);
                return Err(RuntimeError::ShutdownTimeout);
            }
        }
        self.domains.stop();
        self.dedup.clear();
        self.objects.clear_all();
        let mut events = Vec::new();
        self.driver.shutdown(
            &mut events,
            deadline.saturating_duration_since(Instant::now()),
        )?;
        for event in events {
            if let TransportEvent::LocalSendComplete { ticket }
            | TransportEvent::SendFailed { ticket, .. } = event
            {
                self.shared.tickets.lock().unwrap().remove(&ticket);
            }
        }
        self.shared
            .state
            .store(RuntimeState::Stopped as u8, Ordering::Release);
        let report = ShutdownReport {
            stats: self.stats(),
        };
        if report.stats.pending_calls != 0
            || report.stats.send_tickets != 0
            || report.stats.dedup_entries != 0
            || report.stats.queued_responses != 0
            || report.stats.queued_response_bytes != 0
            || report.stats.objects.live_objects != 0
            || report.stats.objects.leases != 0
            || report.stats.objects.roots != 0
            || report.stats.objects.placement_tickets != 0
            || report.stats.objects.transfers != 0
            || report.stats.objects.active_migrations != 0
            || report.stats.objects.prepared_migrations != 0
            || report.stats.objects.forwarding_entries != 0
            || report.stats.objects.migration_snapshot_bytes != 0
            || report.stats.transport.retained_bytes() != 0
        {
            return Err(RuntimeError::RetainedResources);
        }
        Ok(report)
    }

    fn dispatch_local(&mut self, request: LocalRequest) -> Result<(), RuntimeError> {
        match &request.input {
            ActionValue::Typed(_) => self
                .shared
                .counters
                .local_typed_dispatches
                .fetch_add(1, Ordering::Relaxed),
            ActionValue::Encoded(_) => self
                .shared
                .counters
                .local_action_serializations
                .fetch_add(1, Ordering::Relaxed),
        };
        let request_id = request.message.request;
        let action_id = request.message.action;
        let result = self
            .shared
            .registry
            .get(action_id)
            .cloned()
            .ok_or(RuntimeError::UnknownAction(action_id))
            .and_then(|handler| {
                let domain = request.message.domain;
                let job = ActionJob {
                    requester: self.shared.local_id,
                    request: request_id,
                    action_id,
                    domain,
                    trace_id: request.trace_id,
                    input: request.input,
                    handler,
                    cancelled: request.cancel_token,
                    local: true,
                    submitted_at: Instant::now(),
                    object: None,
                };
                let Some(job) =
                    self.objects
                        .admit(action_id, job, self.domains.can_submit(domain))?
                else {
                    return Ok(());
                };
                let admitted = job.object;
                if let Err(error) = self.domains.submit(job) {
                    if let Some(admitted) = admitted {
                        self.objects.rollback(admitted);
                    }
                    return Err(error);
                }
                Ok(())
            });
        if let Err(error) = result {
            if let Some(entry) = self.shared.pending.remove(request_id) {
                entry.promise.complete(Err(error));
            }
        }
        Ok(())
    }

    fn handle_transport_event(&mut self, event: TransportEvent) -> Result<(), RuntimeError> {
        match event {
            TransportEvent::LocalSendComplete { ticket } => {
                self.shared.tickets.lock().unwrap().remove(&ticket);
                Ok(())
            }
            TransportEvent::SendFailed { ticket, error } => {
                match self.shared.tickets.lock().unwrap().remove(&ticket) {
                    Some(TicketPurpose::Request(request)) => {
                        if let Some(entry) = self.shared.pending.remove(request) {
                            entry.promise.complete(Err(RuntimeError::Transport(error)));
                        }
                        Ok(())
                    }
                    Some(TicketPurpose::BestEffort) | None => Ok(()),
                    Some(TicketPurpose::RequiredResponse) => Err(RuntimeError::Transport(error)),
                }
            }
            TransportEvent::Incoming { parcel } => {
                self.shared.counters.received[parcel.channel as usize]
                    .fetch_add(1, Ordering::Relaxed);
                let source = parcel.source;
                let trace_id = parcel.trace_id;
                let message = wire::decode(parcel)
                    .map_err(|error| RuntimeError::Protocol(error.to_string()))?;
                self.handle_message(source, trace_id, message)
            }
            TransportEvent::PeerFailed { peer, .. } => {
                self.fail(RuntimeError::PeerFailed(peer));
                Err(RuntimeError::PeerFailed(peer))
            }
            TransportEvent::ShutdownComplete => Ok(()),
        }
    }

    fn handle_message(
        &mut self,
        source: LocalityId,
        trace_id: Option<TraceId>,
        message: RuntimeMessage,
    ) -> Result<(), RuntimeError> {
        match message.kind {
            RuntimeMessageKind::Request => self.handle_request(source, trace_id, message),
            RuntimeMessageKind::Cancel => self.dedup.cancel(
                message.request,
                message.action,
                message.domain,
                Instant::now(),
            ),
            RuntimeMessageKind::Success => {
                self.complete_pending(source, message, |message| Ok(message.payload))
            }
            RuntimeMessageKind::Failure => self.complete_pending(source, message, |message| {
                let text = String::from_utf8(message.payload.into_iter().next().unwrap())
                    .map_err(|_| RuntimeError::Protocol("remote error is not UTF-8".into()))?;
                let message_text = cap(text);
                let category = message_text.to_ascii_lowercase();
                let error = if category.contains("placement")
                    || category.contains("admission")
                    || category.contains("actionqueue")
                    || category.contains("objectmailbox")
                {
                    RuntimeError::Placement {
                        request: message.request,
                        message: message_text,
                    }
                } else if category.contains("object") {
                    RuntimeError::RemoteObject {
                        request: message.request,
                        message: message_text,
                    }
                } else if category.contains("lease") || category.contains("transfer") {
                    RuntimeError::Lease {
                        request: message.request,
                        message: message_text,
                    }
                } else {
                    RuntimeError::RemoteAction {
                        request: message.request,
                        message: message_text,
                    }
                };
                Err(error)
            }),
            RuntimeMessageKind::Cancelled => self.complete_pending(source, message, |message| {
                Err(RuntimeError::Cancelled(message.request))
            }),
            RuntimeMessageKind::DuplicateResultUnavailable => {
                self.complete_pending(source, message, |message| {
                    Err(RuntimeError::DuplicateResultUnavailable(message.request))
                })
            }
            RuntimeMessageKind::Moved => self.complete_pending(source, message, |message| {
                if message.payload.len() != 2 || message.payload[0].len() != 32 {
                    return Err(RuntimeError::Protocol("invalid moved payload".into()));
                }
                let identity = &message.payload[0];
                let run = RunId::new(u128::from_le_bytes(identity[0..16].try_into().unwrap()))
                    .map_err(|_| RuntimeError::Protocol("invalid moved run".into()))?;
                let object = ObjectId::new(
                    run,
                    u128::from_le_bytes(identity[16..32].try_into().unwrap()),
                )
                .map_err(|_| RuntimeError::Protocol("invalid moved object".into()))?;
                let location = decode_location(message.payload[1].clone())?;
                Err(RuntimeError::Moved { object, location })
            }),
        }
    }

    fn handle_request(
        &mut self,
        source: LocalityId,
        trace_id: Option<TraceId>,
        message: RuntimeMessage,
    ) -> Result<(), RuntimeError> {
        if self.state() != RuntimeState::Running {
            return self.send_failure(source, &message, "runtime is shutting down", trace_id);
        }
        if self
            .objects
            .request_targets_current(message.action, &message.payload)
        {
            self.dedup
                .release_redirect(message.request, message.action, message.domain)?;
        }
        match self.dedup.begin(
            message.request,
            message.action,
            message.domain,
            Instant::now(),
        ) {
            Ok(DedupDisposition::New(cancelled)) => {
                let Some(handler) = self.shared.registry.get(message.action).cloned() else {
                    return self.send_and_cache_failure(
                        source,
                        message,
                        "unknown action",
                        trace_id,
                    );
                };
                let job = ActionJob {
                    requester: source,
                    request: message.request,
                    action_id: message.action,
                    domain: message.domain,
                    trace_id,
                    input: ActionValue::Encoded(message.payload.clone()),
                    handler,
                    cancelled,
                    local: false,
                    submitted_at: Instant::now(),
                    object: None,
                };
                let admitted = self.objects.admit(
                    message.action,
                    job,
                    self.domains.can_submit(message.domain),
                );
                let Some(job) = (match admitted {
                    Ok(job) => job,
                    Err(RuntimeError::Moved { object, location }) => {
                        return self
                            .send_and_cache_moved(source, message, object, location, trace_id);
                    }
                    Err(error) => {
                        return self.send_and_cache_failure(
                            source,
                            message,
                            &error.to_string(),
                            trace_id,
                        );
                    }
                }) else {
                    return Ok(());
                };
                let admitted = job.object;
                if let Err(error) = self.domains.submit(job) {
                    if let Some(admitted) = admitted {
                        self.objects.rollback(admitted);
                    }
                    return self.send_and_cache_failure(
                        source,
                        message,
                        &error.to_string(),
                        trace_id,
                    );
                }
                Ok(())
            }
            Ok(DedupDisposition::Running) => {
                self.shared
                    .counters
                    .duplicate_requests
                    .fetch_add(1, Ordering::Relaxed);
                Ok(())
            }
            Ok(
                DedupDisposition::Replay(response)
                | DedupDisposition::Unavailable(response)
                | DedupDisposition::Cancelled(response),
            ) => {
                self.shared
                    .counters
                    .duplicate_requests
                    .fetch_add(1, Ordering::Relaxed);
                self.queue_response(source, response, trace_id, false)
            }
            Err(error) => self.send_failure(source, &message, &error.to_string(), trace_id),
        }
    }

    fn handle_completion(&mut self, completion: ActionCompletion) -> Result<(), RuntimeError> {
        if let Some(object) = completion.object {
            for job in self.objects.complete(object) {
                let admitted = job.object;
                if let Err(error) = self.domains.submit(job) {
                    if let Some(admitted) = admitted {
                        self.objects.rollback(admitted);
                    }
                    return Err(error);
                }
            }
        }
        if completion.local {
            let entry = self.shared.pending.take_checked(
                completion.request,
                self.shared.local_id,
                completion.action_id,
            )?;
            if let Some(entry) = entry {
                let result = if completion.cancelled {
                    Err(RuntimeError::Cancelled(completion.request))
                } else {
                    completion
                        .result
                        .map_err(|error| RuntimeError::RemoteAction {
                            request: completion.request,
                            message: error.message(),
                        })
                };
                entry.promise.complete(result);
                self.shared
                    .counters
                    .completed
                    .fetch_add(1, Ordering::Relaxed);
            } else {
                self.shared
                    .counters
                    .late_results
                    .fetch_add(1, Ordering::Relaxed);
            }
            return Ok(());
        }
        let (kind, payload) = if completion.cancelled {
            (RuntimeMessageKind::Cancelled, Vec::new())
        } else {
            match completion.result {
                Ok(ActionValue::Encoded(payload)) => (RuntimeMessageKind::Success, payload),
                Ok(ActionValue::Typed(_)) => {
                    return Err(RuntimeError::Protocol(
                        "remote action produced typed output".into(),
                    ));
                }
                Err(error) => {
                    self.shared
                        .counters
                        .action_failures
                        .fetch_add(1, Ordering::Relaxed);
                    (
                        RuntimeMessageKind::Failure,
                        vec![error.message().into_bytes()],
                    )
                }
            }
        };
        let message = RuntimeMessage {
            kind,
            request: completion.request,
            action: completion.action_id,
            domain: completion.domain,
            deadline_ms: 0,
            payload,
        };
        self.queue_response(completion.requester, message, completion.trace_id, true)
    }

    fn complete_pending<F>(
        &mut self,
        source: LocalityId,
        message: RuntimeMessage,
        map: F,
    ) -> Result<(), RuntimeError>
    where
        F: FnOnce(RuntimeMessage) -> Result<Segments, RuntimeError>,
    {
        match self
            .shared
            .pending
            .take_checked(message.request, source, message.action)?
        {
            Some(entry) => {
                entry
                    .promise
                    .complete(map(message).map(ActionValue::Encoded));
                self.shared
                    .counters
                    .completed
                    .fetch_add(1, Ordering::Relaxed);
            }
            None => {
                self.shared
                    .counters
                    .late_results
                    .fetch_add(1, Ordering::Relaxed);
            }
        }
        Ok(())
    }

    fn send_failure(
        &mut self,
        destination: LocalityId,
        request: &RuntimeMessage,
        message: &str,
        trace_id: Option<TraceId>,
    ) -> Result<(), RuntimeError> {
        self.queue_response(
            destination,
            RuntimeMessage {
                kind: RuntimeMessageKind::Failure,
                request: request.request,
                action: request.action,
                domain: request.domain,
                deadline_ms: 0,
                payload: vec![cap(message.to_owned()).into_bytes()],
            },
            trace_id,
            false,
        )
    }

    fn send_and_cache_moved(
        &mut self,
        destination: LocalityId,
        request: RuntimeMessage,
        object: ObjectId,
        location: hataori_runtime_foundation::protocol::ObjectLocation,
        trace_id: Option<TraceId>,
    ) -> Result<(), RuntimeError> {
        let mut identity = Vec::with_capacity(32);
        identity.extend_from_slice(&object.run().get().to_le_bytes());
        identity.extend_from_slice(&object.unique().to_le_bytes());
        let location = encode_location(location)
            .map_err(|error| RuntimeError::Protocol(error.message()))?
            .pop()
            .ok_or_else(|| RuntimeError::Protocol("missing moved location".into()))?;
        let response = RuntimeMessage {
            kind: RuntimeMessageKind::Moved,
            request: request.request,
            action: request.action,
            domain: request.domain,
            deadline_ms: 0,
            payload: vec![identity, location],
        };
        self.queue_response(destination, response, trace_id, true)
    }

    fn send_and_cache_failure(
        &mut self,
        destination: LocalityId,
        request: RuntimeMessage,
        error: &str,
        trace_id: Option<TraceId>,
    ) -> Result<(), RuntimeError> {
        let response = RuntimeMessage {
            kind: RuntimeMessageKind::Failure,
            request: request.request,
            action: request.action,
            domain: request.domain,
            deadline_ms: 0,
            payload: vec![cap(error.to_owned()).into_bytes()],
        };
        self.queue_response(destination, response, trace_id, true)
    }

    fn queue_response(
        &mut self,
        destination: LocalityId,
        message: RuntimeMessage,
        trace_id: Option<TraceId>,
        cache: bool,
    ) -> Result<(), RuntimeError> {
        let bytes = message
            .payload
            .iter()
            .try_fold(0_usize, |total, segment| total.checked_add(segment.len()));
        let bytes = bytes.ok_or(RuntimeError::ResourceExhausted {
            resource: ResourceKind::DedupBytes,
            limit: self.shared.limits.max_dedup_bytes,
        })?;
        let next_bytes =
            self.response_bytes
                .checked_add(bytes)
                .ok_or(RuntimeError::ResourceExhausted {
                    resource: ResourceKind::DedupBytes,
                    limit: self.shared.limits.max_dedup_bytes,
                })?;
        if self.responses.len() >= self.shared.limits.max_dedup_entries {
            return Err(RuntimeError::ResourceExhausted {
                resource: ResourceKind::DedupEntries,
                limit: self.shared.limits.max_dedup_entries,
            });
        }
        if next_bytes > self.shared.limits.max_dedup_bytes {
            return Err(RuntimeError::ResourceExhausted {
                resource: ResourceKind::DedupBytes,
                limit: self.shared.limits.max_dedup_bytes,
            });
        }
        self.response_bytes = next_bytes;
        self.responses.push_back(QueuedResponse {
            destination,
            message,
            trace_id,
            cache,
            bytes,
        });
        Ok(())
    }

    fn drain_responses(&mut self, max_responses: usize) -> Result<usize, RuntimeError> {
        let mut sent = 0;
        while sent < max_responses {
            let Some(response) = self.responses.pop_front() else {
                break;
            };
            self.response_bytes = self.response_bytes.saturating_sub(response.bytes);
            match self.shared.send(
                response.destination,
                response.message.clone(),
                TicketPurpose::RequiredResponse,
                response.trace_id,
            ) {
                Ok(_) => {
                    if response.cache {
                        self.dedup.complete(response.message, Instant::now())?;
                    }
                    sent += 1;
                }
                Err(
                    RuntimeError::Transport(
                        hataori_runtime_foundation::transport::TransportError::QueueFull { .. }
                        | hataori_runtime_foundation::transport::TransportError::ByteLimit { .. },
                    )
                    | RuntimeError::ResourceExhausted {
                        resource: ResourceKind::SendTickets,
                        ..
                    },
                ) => {
                    self.response_bytes = self.response_bytes.saturating_add(response.bytes);
                    self.responses.push_front(response);
                    break;
                }
                Err(error) => return Err(error),
            }
        }
        Ok(sent)
    }

    fn expire_pending(&self) {
        for (request, entry) in self.shared.pending.expired(Instant::now()) {
            entry.promise.complete(Err(RuntimeError::DeadlineExceeded {
                request,
                deadline: entry.deadline,
            }));
            self.shared
                .counters
                .timed_out
                .fetch_add(1, Ordering::Relaxed);
            if let Some(token) = entry.cancel_token {
                token.store(true, Ordering::Release);
            }
            if entry.destination != self.shared.local_id {
                let message = RuntimeMessage {
                    kind: RuntimeMessageKind::Cancel,
                    request,
                    action: entry.action,
                    domain: entry.domain,
                    deadline_ms: 0,
                    payload: Vec::new(),
                };
                let _ = self.shared.send(
                    entry.destination,
                    message,
                    TicketPurpose::BestEffort,
                    entry.trace_id,
                );
            }
        }
    }

    fn fail(&self, error: RuntimeError) {
        self.shared
            .state
            .store(RuntimeState::Failed as u8, Ordering::Release);
        for (_, entry) in self.shared.pending.drain() {
            entry.promise.complete(Err(error.clone()));
            if let Some(token) = entry.cancel_token {
                token.store(true, Ordering::Release);
            }
        }
    }
}

impl Drop for Runtime {
    fn drop(&mut self) {
        if !matches!(self.state(), RuntimeState::Stopped | RuntimeState::Failed) {
            let _work_gate = self.shared.work_gate.lock().unwrap();
            self.shared
                .state
                .store(RuntimeState::Draining as u8, Ordering::Release);
            self.shared.local_leases.clear();
            for (_, entry) in self.shared.pending.drain() {
                entry.promise.complete(Err(RuntimeError::Shutdown));
                if let Some(token) = entry.cancel_token {
                    token.store(true, Ordering::Release);
                }
            }
        }
    }
}

fn decode_state(value: u8) -> RuntimeState {
    match value {
        value if value == RuntimeState::Created as u8 => RuntimeState::Created,
        value if value == RuntimeState::Bootstrapping as u8 => RuntimeState::Bootstrapping,
        value if value == RuntimeState::Running as u8 => RuntimeState::Running,
        value if value == RuntimeState::Draining as u8 => RuntimeState::Draining,
        value if value == RuntimeState::Stopped as u8 => RuntimeState::Stopped,
        _ => RuntimeState::Failed,
    }
}

#[cfg(test)]
mod tests;
